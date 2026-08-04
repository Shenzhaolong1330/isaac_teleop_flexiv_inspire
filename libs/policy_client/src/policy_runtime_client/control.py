"""Synchronous guarded v2 action client using the existing v1 control lease."""

from __future__ import annotations

import queue
import threading
import time
from typing import Iterable, Sequence

import numpy as np

from policy_contracts import policy_action24_to_native30

from .generated import policy_data_v2_pb2 as data_pb
from .generated import policy_data_v2_pb2_grpc as data_pb_grpc
from .generated import policy_service_v1_pb2 as control_pb
from .generated import policy_service_v1_pb2_grpc as control_pb_grpc
from .runtime import ProfileRuntimeError, SyncPolicyProfileClient


class PolicyActionError(ProfileRuntimeError):
    pass


_END = object()


class SyncPolicyActionClient:
    """One persistent action stream with fail-closed lease/TTL semantics.

    Lease acquisition is lazy so a policy process may connect and inspect data
    before the operator presses the local pedal.  The first action requires the
    complete local authorization state.  A rejected action, response timeout or
    broken stream drops the lease locally; the server TTL/heartbeat path stops
    motion independently of this client.
    """

    def __init__(
        self,
        profile_client: SyncPolicyProfileClient,
        *,
        client_id: str,
        action_ttl_ms: float = 250.0,
        requested_lease_ms: int = 2_000,
    ) -> None:
        if (
            profile_client.channel is None
            or profile_client.mapper is None
            or profile_client.stub is None
        ):
            raise PolicyActionError("profile client must be connected first")
        if not str(client_id).strip():
            raise PolicyActionError("client_id must be non-empty")
        ttl_ns = int(float(action_ttl_ms) * 1e6)
        if not 0 < ttl_ns <= 1_000_000_000:
            raise PolicyActionError("action_ttl_ms must be in (0,1000]")
        if not 0 < int(requested_lease_ms) <= 10_000:
            raise PolicyActionError("requested_lease_ms must be in (0,10000]")

        self.profile_client = profile_client
        self.client_id = str(client_id).strip()
        self.action_ttl_ns = ttl_ns
        self.requested_lease_ms = int(requested_lease_ms)
        self._control_stub = control_pb_grpc.PolicyServiceStub(profile_client.channel)
        self._data_stub = data_pb_grpc.PolicyDataServiceStub(profile_client.channel)
        self._lock = threading.Lock()
        self._lease_id: str | None = None
        self._sequence = 0
        self._request_queue: queue.Queue | None = None
        self._response_queue: queue.Queue | None = None
        self._call = None
        self._response_thread: threading.Thread | None = None
        self._closed = False

    @staticmethod
    def _request_iterator(items: queue.Queue) -> Iterable[object]:
        while True:
            item = items.get()
            if item is _END:
                return
            yield item

    def _response_worker(self, call, responses: queue.Queue) -> None:
        try:
            for response in call:
                responses.put(response)
        except Exception as exc:  # grpc.RpcError plus transport shutdown errors
            responses.put(exc)

    def _drop_stream(self) -> None:
        call, self._call = self._call, None
        requests, self._request_queue = self._request_queue, None
        self._response_queue = None
        thread, self._response_thread = self._response_thread, None
        if call is not None:
            call.cancel()
        if requests is not None:
            requests.put(_END)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.2)

    def _acquire_lease(self) -> None:
        mapper = self.profile_client.mapper
        if mapper is None:
            raise PolicyActionError("profile client is disconnected")
        try:
            result = self._control_stub.AcquireControlLease(
                control_pb.AcquireControlLeaseRequest(
                    client_id=self.client_id,
                    session_id=mapper.session_id,
                    requested_duration_ms=self.requested_lease_ms,
                ),
                timeout=self.profile_client.request_timeout_s,
            )
        except Exception as exc:
            raise PolicyActionError(f"control lease request failed: {exc}") from exc
        if not result.granted or not result.lease_id:
            raise PolicyActionError(
                "control lease was not granted: " + (result.reason or "unknown reason")
            )
        self._lease_id = str(result.lease_id)
        self._sequence = 0

    def _start_stream(self) -> None:
        requests: queue.Queue = queue.Queue()
        responses: queue.Queue = queue.Queue()
        call = self._data_stub.StreamActions(self._request_iterator(requests))
        thread = threading.Thread(
            target=self._response_worker,
            args=(call, responses),
            name="policy-action-responses",
            daemon=True,
        )
        self._request_queue = requests
        self._response_queue = responses
        self._call = call
        self._response_thread = thread
        thread.start()

    def _ensure_ready(self) -> None:
        if self._closed:
            raise PolicyActionError("policy action client is closed")
        if self.profile_client.mapper is None or self.profile_client.channel is None:
            raise PolicyActionError("profile client is disconnected")
        if self._lease_id is None:
            self._acquire_lease()
        if self._call is None:
            self._start_stream()

    def send(
        self,
        action: Sequence[float],
        *,
        execute_after_ns: int = 0,
    ):
        return self.send_chunk((action,), execute_after_ns=(execute_after_ns,))

    def send_chunk(
        self,
        actions: Sequence[Sequence[float]],
        *,
        execute_after_ns: Sequence[int],
    ):
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        offsets = tuple(int(item) for item in execute_after_ns)
        if values.ndim != 2 or values.shape[1] != 24 or values.shape[0] != len(offsets):
            raise PolicyActionError("action chunk must have shape [N,24] with N offsets")
        if not 1 <= values.shape[0] <= 32:
            raise PolicyActionError("action chunk must contain 1..32 points")
        if any(item < 0 for item in offsets) or any(
            current < previous for previous, current in zip(offsets, offsets[1:])
        ):
            raise PolicyActionError("execute_after_ns must be non-negative and monotonic")
        if offsets[-1] >= self.action_ttl_ns:
            raise PolicyActionError("action point must execute before the action TTL")
        # Run the exact canonical mapping locally as an early finite/range/SO(3)
        # validation.  The server repeats validation and owns the authoritative
        # conversion; no native vector is transmitted from this policy client.
        for point in values:
            policy_action24_to_native30(point)

        with self._lock:
            self._ensure_ready()
            mapper = self.profile_client.mapper
            assert mapper is not None
            assert self._lease_id is not None
            assert self._request_queue is not None
            assert self._response_queue is not None
            self._sequence += 1
            sequence = self._sequence
            request = data_pb.PolicyActionChunk(
                schema_version=2,
                schema_hash=mapper.schema_hash,
                action_schema_id="cartesian_delta_rotvec_v1",
                lease_id=self._lease_id,
                session_id=mapper.session_id,
                sequence=sequence,
                client_issued_monotonic_ns=time.monotonic_ns(),
                ttl_from_server_receive_ns=self.action_ttl_ns,
                deadman=True,
            )
            for offset_ns, point in zip(offsets, values):
                packed = np.asarray(point, dtype="<f4")
                target = request.points.add(execute_after_ns=offset_ns)
                target.action.CopyFrom(
                    data_pb.TensorPayload(
                        dtype=data_pb.FLOAT32,
                        shape=(24,),
                        data=packed.tobytes(),
                    )
                )
            self._request_queue.put(request)
            try:
                response = self._response_queue.get(
                    timeout=self.profile_client.request_timeout_s
                )
            except queue.Empty as exc:
                self._drop_stream()
                self._lease_id = None
                raise PolicyActionError("timed out waiting for action acknowledgement") from exc
            if isinstance(response, Exception):
                self._drop_stream()
                self._lease_id = None
                raise PolicyActionError(f"policy action stream failed: {response}")
            if int(response.sequence) != sequence:
                self._drop_stream()
                self._lease_id = None
                raise PolicyActionError("action acknowledgement sequence mismatch")
            if not response.accepted:
                reason = response.reason or "server rejected action"
                self._drop_stream()
                self._lease_id = None
                raise PolicyActionError(reason)
            return response

    def close(self, *, reason: str = "policy-client-disconnect") -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            lease_id = self._lease_id
            if lease_id is not None:
                try:
                    self._control_stub.Stop(
                        control_pb.StopRequest(lease_id=lease_id, reason=str(reason)),
                        timeout=self.profile_client.request_timeout_s,
                    )
                except Exception:
                    try:
                        self._control_stub.ReleaseControlLease(
                            control_pb.ReleaseControlLeaseRequest(lease_id=lease_id),
                            timeout=self.profile_client.request_timeout_s,
                        )
                    except Exception:
                        pass
            self._lease_id = None
            self._drop_stream()
