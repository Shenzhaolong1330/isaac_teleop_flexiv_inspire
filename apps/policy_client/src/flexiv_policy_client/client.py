"""Synchronous multi-rate PolicyData v2 client with a persistent action stream.

The package is intentionally independent of ROS, Flexiv RDK, Inspire drivers
and the server repository.  One gRPC channel may carry multiple independent
sample subscriptions plus the bidirectional action stream.
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path
import queue
import threading
import time
from typing import Iterable, Mapping, Sequence

import grpc
import numpy as np

from .generated import policy_data_v2_pb2 as data_pb
from .generated import policy_data_v2_pb2_grpc as data_grpc
from .generated import policy_service_v1_pb2 as control_pb
from .generated import policy_service_v1_pb2_grpc as control_grpc


class PolicyClientError(RuntimeError):
    pass


_PROTO_DTYPES = {
    data_pb.BOOL: np.dtype("?"),
    data_pb.UINT8: np.dtype("u1"),
    data_pb.UINT16: np.dtype("<u2"),
    data_pb.UINT32: np.dtype("<u4"),
    data_pb.UINT64: np.dtype("<u8"),
    data_pb.INT8: np.dtype("i1"),
    data_pb.INT16: np.dtype("<i2"),
    data_pb.INT32: np.dtype("<i4"),
    data_pb.INT64: np.dtype("<i8"),
    data_pb.FLOAT16: np.dtype("<f2"),
    data_pb.FLOAT32: np.dtype("<f4"),
    data_pb.FLOAT64: np.dtype("<f8"),
}


@dataclass(frozen=True)
class ImageValue:
    encoding: str
    data: bytes
    width: int
    height: int
    channels: int


@dataclass(frozen=True)
class Sample:
    channel_id: str
    value: np.ndarray | ImageValue | None
    source_time_ns: int
    host_receive_time_ns: int
    mapped_host_time_ns: int
    sequence: int
    valid: bool
    age_ns: int
    invalid_reason: str
    timing_valid: bool


@dataclass(frozen=True)
class SnapshotSpec:
    alignment: str = "latest_causal"
    tolerance_ms: float = 0.0
    max_age_ms: float = 0.0


def _decode_tensor(payload) -> np.ndarray:
    try:
        dtype = _PROTO_DTYPES[int(payload.dtype)]
    except KeyError as exc:
        raise PolicyClientError(
            f"unsupported tensor dtype enum: {payload.dtype}"
        ) from exc
    shape = tuple(int(value) for value in payload.shape)
    if not shape or any(value <= 0 for value in shape):
        raise PolicyClientError(f"invalid tensor shape: {shape}")
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(payload.data) != expected:
        raise PolicyClientError(
            f"tensor has {len(payload.data)} bytes, expected {expected}"
        )
    value = np.frombuffer(payload.data, dtype=dtype).reshape(shape)
    value.setflags(write=False)
    return value


def decode_sample(message) -> Sample:
    payload = message.WhichOneof("payload")
    if payload == "tensor":
        value: np.ndarray | ImageValue | None = _decode_tensor(message.tensor)
    elif payload == "image":
        value = ImageValue(
            encoding=str(message.image.encoding),
            data=bytes(message.image.data),
            width=int(message.image.width),
            height=int(message.image.height),
            channels=int(message.image.channels),
        )
    else:
        value = None
    timing = message.timing
    return Sample(
        channel_id=str(message.channel_id),
        value=value,
        source_time_ns=int(timing.source_time_ns),
        host_receive_time_ns=int(timing.host_receive_time_ns),
        mapped_host_time_ns=int(timing.mapped_host_time_ns),
        sequence=int(timing.sequence),
        valid=bool(timing.valid),
        age_ns=int(timing.age_ns),
        invalid_reason=str(timing.invalid_reason),
        timing_valid=bool(timing.timing_valid),
    )


def _target_host(target: str) -> str:
    return str(target).rsplit(":", 1)[0].strip("[]")


def _is_loopback(target: str) -> bool:
    host = _target_host(target)
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


_END = object()


class _ActionStream:
    def __init__(
        self,
        owner: "MultiRatePolicyClient",
        *,
        client_id: str,
        ttl_ms: float,
        lease_ms: int,
    ) -> None:
        if not client_id.strip():
            raise ValueError("action client_id cannot be empty")
        if not 0.0 < float(ttl_ms) <= 1000.0:
            raise ValueError("action ttl_ms must be in (0,1000]")
        if not 0 < int(lease_ms) <= 10_000:
            raise ValueError("action lease_ms must be in (0,10000]")
        self.owner = owner
        self.client_id = client_id.strip()
        self.ttl_ns = int(float(ttl_ms) * 1e6)
        self.lease_ms = int(lease_ms)
        self.lease_id = ""
        self.sequence = 0
        self.requests: queue.Queue | None = None
        self.responses: queue.Queue | None = None
        self.call = None
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

    @staticmethod
    def _request_iterator(requests: queue.Queue):
        while True:
            item = requests.get()
            if item is _END:
                return
            yield item

    @staticmethod
    def _read_responses(call, responses: queue.Queue) -> None:
        try:
            for response in call:
                responses.put(response)
        except Exception as exc:  # grpc transport errors are surfaced to send()
            responses.put(exc)

    def _drop(self) -> None:
        call, self.call = self.call, None
        requests, self.requests = self.requests, None
        self.responses = None
        thread, self.thread = self.thread, None
        if call is not None and hasattr(call, "cancel"):
            call.cancel()
        if requests is not None:
            requests.put(_END)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.2)

    def _ensure_ready(self) -> None:
        if self.owner.description is None:
            raise PolicyClientError("client is not connected")
        if not self.lease_id:
            response = self.owner.control_stub.AcquireControlLease(
                control_pb.AcquireControlLeaseRequest(
                    client_id=self.client_id,
                    session_id=self.owner.description.session_id,
                    requested_duration_ms=self.lease_ms,
                ),
                timeout=self.owner.request_timeout_s,
            )
            if not response.granted or not response.lease_id:
                raise PolicyClientError(
                    "control lease rejected: "
                    + (response.reason or "unknown reason")
                )
            self.lease_id = str(response.lease_id)
            self.sequence = 0
        if self.call is None:
            requests: queue.Queue = queue.Queue()
            responses: queue.Queue = queue.Queue()
            call = self.owner.data_stub.StreamActions(
                self._request_iterator(requests)
            )
            thread = threading.Thread(
                target=self._read_responses,
                args=(call, responses),
                name="policy-action-responses",
                daemon=True,
            )
            self.requests = requests
            self.responses = responses
            self.call = call
            self.thread = thread
            thread.start()

    def send(
        self,
        actions: Sequence[Sequence[float]] | Sequence[float],
        *,
        execute_after_ns: Sequence[int] | None = None,
    ):
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != 24:
            raise PolicyClientError("policy action must have shape [24] or [N,24]")
        if not 1 <= values.shape[0] <= 32:
            raise PolicyClientError("action chunk must contain 1..32 points")
        if not np.all(np.isfinite(values)):
            raise PolicyClientError("policy action contains NaN/Inf")
        if np.any(values[:, 12:24] < 0.0) or np.any(values[:, 12:24] > 1.0):
            raise PolicyClientError("hand targets must be normalized to [0,1]")
        offsets = (
            (0,) * values.shape[0]
            if execute_after_ns is None
            else tuple(int(item) for item in execute_after_ns)
        )
        if len(offsets) != values.shape[0]:
            raise PolicyClientError("one execute_after_ns is required per point")
        if any(value < 0 for value in offsets) or any(
            current < previous
            for previous, current in zip(offsets, offsets[1:])
        ):
            raise PolicyClientError(
                "execute_after_ns must be non-negative and monotonic"
            )
        if offsets[-1] >= self.ttl_ns:
            raise PolicyClientError("last action point must execute before TTL")

        with self.lock:
            try:
                self._ensure_ready()
                assert self.owner.description is not None
                assert self.requests is not None
                assert self.responses is not None
                self.sequence += 1
                request = data_pb.PolicyActionChunk(
                    schema_version=2,
                    schema_hash=self.owner.description.schema_hash,
                    action_schema_id="cartesian_delta_rotvec_v1",
                    lease_id=self.lease_id,
                    session_id=self.owner.description.session_id,
                    sequence=self.sequence,
                    client_issued_monotonic_ns=time.monotonic_ns(),
                    ttl_from_server_receive_ns=self.ttl_ns,
                    deadman=True,
                )
                for offset, point in zip(offsets, values):
                    packed = np.asarray(point, dtype="<f4")
                    target = request.points.add(execute_after_ns=offset)
                    target.action.CopyFrom(
                        data_pb.TensorPayload(
                            dtype=data_pb.FLOAT32,
                            shape=(24,),
                            data=packed.tobytes(),
                        )
                    )
                self.requests.put(request)
                response = self.responses.get(timeout=self.owner.request_timeout_s)
                if isinstance(response, Exception):
                    raise PolicyClientError(f"action stream failed: {response}")
                if int(response.sequence) != self.sequence:
                    raise PolicyClientError("action acknowledgement sequence mismatch")
                if not response.accepted:
                    raise PolicyClientError(
                        response.reason or "server rejected policy action"
                    )
                return response
            except Exception:
                self._drop()
                self.lease_id = ""
                raise

    def close(self) -> None:
        with self.lock:
            if self.lease_id:
                try:
                    self.owner.control_stub.Stop(
                        control_pb.StopRequest(
                            lease_id=self.lease_id,
                            reason="remote-policy-client-close",
                        ),
                        timeout=self.owner.request_timeout_s,
                    )
                except Exception:
                    pass
            self.lease_id = ""
            self._drop()


class MultiRatePolicyClient:
    """Latest-value cache fed by independent per-rate subscriptions."""

    def __init__(
        self,
        *,
        target: str,
        server_ca: str | Path | None = None,
        client_cert: str | Path | None = None,
        client_key: str | Path | None = None,
        insecure_loopback: bool = False,
        connect_timeout_s: float = 5.0,
        request_timeout_s: float = 1.0,
        action_client_id: str = "remote-policy",
        action_ttl_ms: float = 250.0,
        action_lease_ms: int = 2000,
    ) -> None:
        self.target = str(target)
        self.server_ca = None if not server_ca else Path(server_ca).expanduser()
        self.client_cert = (
            None if not client_cert else Path(client_cert).expanduser()
        )
        self.client_key = None if not client_key else Path(client_key).expanduser()
        self.insecure_loopback = bool(insecure_loopback)
        self.connect_timeout_s = float(connect_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self.channel = None
        self.data_stub = None
        self.control_stub = None
        self.description = None
        self._samples: dict[str, Sample] = {}
        self._received_count: dict[str, int] = {}
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._subscription_calls: dict[str, object] = {}
        self._subscription_threads: dict[str, threading.Thread] = {}
        self._subscription_errors: dict[str, Exception] = {}
        self._action_options = (
            str(action_client_id),
            float(action_ttl_ms),
            int(action_lease_ms),
        )
        self._action: _ActionStream | None = None

    def connect(self):
        if self.channel is not None:
            raise PolicyClientError("client is already connected")
        if self.insecure_loopback:
            if not _is_loopback(self.target):
                raise PolicyClientError("insecure transport is loopback-only")
            channel = grpc.insecure_channel(self.target)
        else:
            if self.server_ca is None:
                raise PolicyClientError("server_ca is required for TLS")
            cert = None if self.client_cert is None else self.client_cert.read_bytes()
            key = None if self.client_key is None else self.client_key.read_bytes()
            if (cert is None) != (key is None):
                raise PolicyClientError(
                    "client_cert and client_key must be provided together"
                )
            credentials = grpc.ssl_channel_credentials(
                root_certificates=self.server_ca.read_bytes(),
                private_key=key,
                certificate_chain=cert,
            )
            channel = grpc.secure_channel(self.target, credentials)
        try:
            grpc.channel_ready_future(channel).result(timeout=self.connect_timeout_s)
            data_stub = data_grpc.PolicyDataServiceStub(channel)
            description = data_stub.DescribeSystem(
                data_pb.DescribeSystemRequest(), timeout=self.request_timeout_s
            )
        except Exception:
            channel.close()
            raise
        if int(description.schema_version) != 2:
            channel.close()
            raise PolicyClientError("server does not expose PolicyData v2")
        self.channel = channel
        self.data_stub = data_stub
        self.control_stub = control_grpc.PolicyServiceStub(channel)
        self.description = description
        self._stop.clear()
        return description

    @property
    def descriptors(self) -> dict[str, object]:
        if self.description is None:
            raise PolicyClientError("client is not connected")
        return {item.channel_id: item for item in self.description.channels}

    def subscribe(self, name: str, channels: Mapping[str, float]) -> None:
        if self.description is None or self.data_stub is None:
            raise PolicyClientError("client is not connected")
        selected_name = str(name).strip()
        if not selected_name or selected_name in self._subscription_threads:
            raise PolicyClientError("subscription name is empty or already active")
        if not channels:
            raise PolicyClientError("subscription requires at least one channel")
        descriptors = self.descriptors
        unknown = sorted(set(channels).difference(descriptors))
        if unknown:
            raise PolicyClientError("unknown channels: " + ", ".join(unknown))
        invalid_rates = {
            channel_id: rate_hz
            for channel_id, rate_hz in channels.items()
            if not np.isfinite(float(rate_hz)) or float(rate_hz) <= 0.0
        }
        if invalid_rates:
            raise PolicyClientError(
                "subscription rates must be finite and positive: "
                + ", ".join(
                    f"{channel_id}={rate_hz}"
                    for channel_id, rate_hz in invalid_rates.items()
                )
            )
        request = data_pb.SubscribeRequest(
            client_id=f"remote-policy-{selected_name}",
            expected_schema_hash=self.description.schema_hash,
            expected_session_id=self.description.session_id,
            channels=[
                data_pb.ChannelSubscription(
                    channel_id=channel_id,
                    max_rate_hz=float(rate_hz),
                    drop_policy=data_pb.DROP_OLDEST,
                )
                for channel_id, rate_hz in channels.items()
            ],
        )
        call = self.data_stub.SubscribeSamples(request)
        thread = threading.Thread(
            target=self._subscription_worker,
            args=(selected_name, call),
            name=f"policy-subscription-{selected_name}",
            daemon=True,
        )
        self._subscription_calls[selected_name] = call
        self._subscription_threads[selected_name] = thread
        thread.start()

    def _subscription_worker(self, name: str, call) -> None:
        try:
            for message in call:
                sample = decode_sample(message)
                with self._condition:
                    self._samples[sample.channel_id] = sample
                    self._received_count[sample.channel_id] = (
                        self._received_count.get(sample.channel_id, 0) + 1
                    )
                    self._condition.notify_all()
                if self._stop.is_set():
                    return
        except Exception as exc:
            if not self._stop.is_set():
                with self._condition:
                    self._subscription_errors[name] = exc
                    self._condition.notify_all()

    def latest(self, channel_id: str) -> Sample | None:
        with self._condition:
            return self._samples.get(str(channel_id))

    def latest_many(self, channel_ids: Iterable[str]) -> dict[str, Sample]:
        selected = tuple(str(value) for value in channel_ids)
        with self._condition:
            return {
                channel_id: self._samples[channel_id]
                for channel_id in selected
                if channel_id in self._samples
            }

    def received_counts(self) -> dict[str, int]:
        with self._condition:
            return dict(self._received_count)

    def wait_for_channels(
        self,
        channel_ids: Iterable[str],
        *,
        timeout_s: float,
        require_valid: bool = True,
    ) -> dict[str, Sample]:
        selected = tuple(str(value) for value in channel_ids)
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while True:
                if self._subscription_errors:
                    name, error = next(iter(self._subscription_errors.items()))
                    raise PolicyClientError(
                        f"subscription {name} failed: {error}"
                    ) from error
                available = {
                    channel_id: self._samples[channel_id]
                    for channel_id in selected
                    if channel_id in self._samples
                }
                if len(available) == len(selected) and (
                    not require_valid
                    or all(
                        sample.valid and sample.timing_valid
                        for sample in available.values()
                    )
                ):
                    return available
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    missing = sorted(set(selected).difference(available))
                    invalid = sorted(
                        channel_id
                        for channel_id, sample in available.items()
                        if require_valid
                        and (not sample.valid or not sample.timing_valid)
                    )
                    raise TimeoutError(
                        f"timed out waiting for channels; missing={missing}, "
                        f"invalid={invalid}"
                    )
                self._condition.wait(remaining)

    def wait_for_update(
        self,
        channel_id: str,
        *,
        after_sequence: int,
        timeout_s: float,
    ) -> Sample:
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while True:
                sample = self._samples.get(str(channel_id))
                if sample is not None and sample.sequence > int(after_sequence):
                    return sample
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(f"no new sample for {channel_id}")
                self._condition.wait(remaining)

    def snapshot(
        self,
        channels: Mapping[str, SnapshotSpec] | Iterable[str],
        *,
        target_monotonic_ns: int = 0,
    ) -> dict[str, Sample]:
        if self.description is None or self.data_stub is None:
            raise PolicyClientError("client is not connected")
        if isinstance(channels, Mapping):
            specs = dict(channels)
        else:
            specs = {str(value): SnapshotSpec() for value in channels}
        alignments = {
            "latest_causal": data_pb.LATEST_CAUSAL,
            "nearest": data_pb.NEAREST,
            "interpolate": data_pb.INTERPOLATE,
        }
        request = data_pb.SnapshotRequest(
            expected_schema_hash=self.description.schema_hash,
            expected_session_id=self.description.session_id,
            target_monotonic_ns=int(target_monotonic_ns),
        )
        for channel_id, spec in specs.items():
            try:
                alignment = alignments[spec.alignment]
            except KeyError as exc:
                raise PolicyClientError(
                    f"unknown alignment mode: {spec.alignment}"
                ) from exc
            request.channels.add(
                channel_id=channel_id,
                alignment=alignment,
                tolerance_ns=int(spec.tolerance_ms * 1e6),
                max_age_ns=int(spec.max_age_ms * 1e6),
            )
        response = self.data_stub.GetSnapshot(
            request, timeout=self.request_timeout_s
        )
        result = {item.channel_id: decode_sample(item) for item in response.samples}
        if not response.complete:
            raise PolicyClientError(
                "snapshot incomplete: " + "; ".join(response.errors)
            )
        return result

    def send_action(
        self,
        action: Sequence[float],
        *,
        execute_after_ns: int = 0,
    ):
        return self.send_action_chunk(
            (action,), execute_after_ns=(execute_after_ns,)
        )

    def send_action_chunk(
        self,
        actions: Sequence[Sequence[float]],
        *,
        execute_after_ns: Sequence[int],
    ):
        if self.control_stub is None or self.data_stub is None:
            raise PolicyClientError("client is not connected")
        if self._action is None:
            client_id, ttl_ms, lease_ms = self._action_options
            self._action = _ActionStream(
                self,
                client_id=client_id,
                ttl_ms=ttl_ms,
                lease_ms=lease_ms,
            )
        return self._action.send(actions, execute_after_ns=execute_after_ns)

    def close(self) -> None:
        self._stop.set()
        action, self._action = self._action, None
        if action is not None:
            action.close()
        calls = tuple(self._subscription_calls.values())
        self._subscription_calls.clear()
        for call in calls:
            call.cancel()
        threads = tuple(self._subscription_threads.values())
        self._subscription_threads.clear()
        for thread in threads:
            thread.join(timeout=1.0)
        channel, self.channel = self.channel, None
        self.data_stub = None
        self.control_stub = None
        self.description = None
        if channel is not None:
            channel.close()

    def __enter__(self) -> "MultiRatePolicyClient":
        self.connect()
        return self

    def __exit__(self, *_args) -> None:
        self.close()
