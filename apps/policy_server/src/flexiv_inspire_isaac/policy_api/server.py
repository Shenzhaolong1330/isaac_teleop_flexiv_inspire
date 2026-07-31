"""TLS grpc.aio adapter for the canonical libs/rpc_interfaces/proto PolicyService v1."""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
import ipaddress
from pathlib import Path
import time
from typing import Callable

from .broker import LatestActionBuffer, ObservationBroker, PolicyStreamLiveness
from .lease import ControlLeaseManager, LocalControlState
from .models import ActionChunk, ActionPoint, capabilities_v1, validate_action_chunk


@dataclass(frozen=True)
class TlsFiles:
    server_cert: Path
    server_key: Path
    client_ca: Path | None = None

    def load(self, bind_host: str) -> tuple[bytes, bytes, bytes | None, bool]:
        certificate = self.server_cert.read_bytes()
        key = self.server_key.read_bytes()
        client_ca = None if self.client_ca is None else self.client_ca.read_bytes()
        is_loopback = ipaddress.ip_address(bind_host).is_loopback
        if not is_loopback and client_ca is None:
            raise ValueError("non-loopback PolicyService requires a client CA (mTLS)")
        return certificate, key, client_ca, not is_loopback


def _peer_identity(context) -> str:
    common_names = context.auth_context().get("x509_common_name", ())
    common_name = (
        common_names[0].decode("utf-8", errors="replace") if common_names else ""
    )
    return f"{context.peer()}|{common_name}"


async def serve(
    *,
    bind_host: str,
    port: int,
    tls: TlsFiles,
    lease_manager: ControlLeaseManager,
    local_state: Callable[[], LocalControlState],
    observation_broker: ObservationBroker,
    action_buffer: LatestActionBuffer[ActionChunk],
    stop_callback: Callable[[str], None],
    action_liveness: PolicyStreamLiveness | None = None,
) -> None:
    import grpc
    from .generated import policy_service_v1_pb2 as pb
    from .generated import policy_service_v1_pb2_grpc as pb_grpc

    if action_liveness is None:
        action_liveness = PolicyStreamLiveness()

    class Servicer(pb_grpc.PolicyServiceServicer):
        def __init__(self) -> None:
            self._last_sequence_by_lease: dict[str, int] = {}

        async def GetCapabilities(self, request, context):
            values = capabilities_v1()
            return pb.Capabilities(
                schema_version=values["schema_version"],
                rotation_representation=pb.ROT6D_FIRST_TWO_COLUMNS,
                rotation_element_order=values["rotation_element_order"],
                default_action_dimension=values["default_action_dimension"],
                action_layout=values["action_layout"],
                frame_id=values["frame_id"],
                linear_unit=values["linear_unit"],
                angular_unit=values["angular_unit"],
                max_chunk_points=values["max_chunk_points"],
                max_chunk_duration_s=values["max_chunk_duration_s"],
                supported_control_representations=values[
                    "supported_control_representations"
                ],
                action_clock_semantics=values["action_clock_semantics"],
                observation_layout=values["observation_layout"],
            )

        async def AcquireControlLease(self, request, context):
            state = local_state()
            if not state.policy_lease_allowed:
                lease_manager.invalidate()
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, "local policy authorization lost")
            if request.session_id != state.session_id:
                return pb.ControlLease(granted=False, reason="session mismatch")
            try:
                lease = lease_manager.acquire(
                    client_id=request.client_id,
                    peer=_peer_identity(context),
                    requested_ms=request.requested_duration_ms,
                    local_state=state,
                )
            except PermissionError as exc:
                return pb.ControlLease(granted=False, reason=str(exc))
            return pb.ControlLease(
                granted=True,
                lease_id=lease.token,
                expires_monotonic_ns=lease.expires_ns,
            )

        async def ReleaseControlLease(self, request, context):
            lease_id = request.lease_id
            released = lease_manager.release(
                request.lease_id, _peer_identity(context)
            )
            if released:
                self._last_sequence_by_lease.pop(lease_id, None)
                action_liveness.clear()
            return pb.LeaseResult(
                success=released,
                reason="" if released else "lease not owned by authenticated peer",
            )

        async def StreamObservations(self, request, context):
            peer = _peer_identity(context)
            sequence = 0
            requested_hz = int(request.max_rate_hz or 30)
            rate_hz = min(30, max(1, requested_hz))
            minimum_period_s = 1.0 / rate_hz
            last_send_s = 0.0
            while True:
                state = local_state()
                if not state.policy_lease_allowed:
                    lease_manager.invalidate()
                    action_liveness.clear()
                    await context.abort(
                        grpc.StatusCode.PERMISSION_DENIED,
                        "local policy authorization lost",
                    )
                try:
                    lease_manager.validate(
                        request.lease_id, peer, state.session_id
                    )
                except PermissionError as exc:
                    action_liveness.clear()
                    await context.abort(
                        grpc.StatusCode.PERMISSION_DENIED, str(exc)
                    )
                try:
                    sequence, observation = await asyncio.wait_for(
                        observation_broker.next_after(sequence), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue
                delay = minimum_period_s - (time.monotonic() - last_send_s)
                if delay > 0.0:
                    await asyncio.sleep(delay)
                    latest_sequence, latest_observation = await observation_broker.latest()
                    if latest_observation is not None and latest_sequence > sequence:
                        sequence, observation = latest_sequence, latest_observation
                filtered = pb.Observation()
                filtered.CopyFrom(observation)
                if not request.include_images:
                    filtered.ClearField("camera_images")
                if not request.include_tactile:
                    for field in ("left_tactile", "right_tactile"):
                        filtered.ClearField(field)
                last_send_s = time.monotonic()
                yield filtered

        async def StreamActions(self, request_iterator, context):
            peer = _peer_identity(context)
            stream_id = secrets.token_urlsafe(16)
            try:
                async for request in request_iterator:
                    receive_ns = time.monotonic_ns()
                    try:
                        state = local_state()
                        if request.session_id != state.session_id:
                            lease_manager.invalidate()
                            raise PermissionError("hardware session changed")
                        if not state.policy_lease_allowed:
                            lease_manager.invalidate()
                            raise PermissionError("local policy authorization lost")
                        lease_manager.validate(
                            request.lease_id, peer, state.session_id, refresh_ms=10000
                        )
                        previous = self._last_sequence_by_lease.get(request.lease_id, -1)
                        if request.sequence <= previous:
                            raise ValueError("action sequence must strictly increase per lease")
                        chunk = ActionChunk(
                            schema_version=request.schema_version,
                            lease_id=request.lease_id,
                            session_id=request.session_id,
                            sequence=request.sequence,
                            client_issued_monotonic_ns=request.client_issued_monotonic_ns,
                            server_receive_monotonic_ns=receive_ns,
                            ttl_from_server_receive_ns=request.ttl_from_server_receive_ns,
                            frame_id=request.frame_id,
                            deadman=request.deadman,
                            points=tuple(
                                ActionPoint(
                                    execute_after_s=point.execute_after_s,
                                    values=tuple(point.values),
                                )
                                for point in request.points
                            ),
                        )
                        validate_action_chunk(chunk, now_ns=time.monotonic_ns())
                        self._last_sequence_by_lease[request.lease_id] = request.sequence
                        action_buffer.put(chunk)
                        action_liveness.arm(
                            stream_id=stream_id,
                            lease_id=request.lease_id,
                            session_id=request.session_id,
                            sequence=request.sequence,
                            action_deadline_ns=(
                                receive_ns + request.ttl_from_server_receive_ns
                            ),
                        )
                        yield pb.ActionResult(
                            sequence=request.sequence, accepted=True, control_state=state.state
                        )
                    except (PermissionError, ValueError) as exc:
                        if action_liveness.disarm(stream_id):
                            stop_callback(f"policy-action-rejected:{exc}")
                        yield pb.ActionResult(
                            sequence=request.sequence,
                            accepted=False,
                            reason=str(exc),
                            control_state=local_state().state,
                        )
            finally:
                if action_liveness.disarm(stream_id):
                    stop_callback("policy-action-stream-ended")

        async def Stop(self, request, context):
            state = local_state()
            lease_manager.validate(
                request.lease_id,
                _peer_identity(context),
                state.session_id,
            )
            # The only remote state-changing method is a one-way safe stop.
            stop_callback(request.reason or "remote-policy-stop")
            lease_manager.invalidate()
            action_liveness.clear()
            return pb.StopResult(
                latched=False,
                control_state=state.state,
                stop_published=True,
                status="stop published; supervisor HOLD_LATCHED acknowledgement pending",
            )

    certificate, key, client_ca, require_client_auth = tls.load(bind_host)
    credentials = grpc.ssl_server_credentials(
        [(key, certificate)],
        root_certificates=client_ca,
        require_client_auth=require_client_auth,
    )
    server = grpc.aio.server(
        options=[
            ("grpc.max_receive_message_length", 32 * 1024 * 1024),
            ("grpc.max_send_message_length", 32 * 1024 * 1024),
        ]
    )
    pb_grpc.add_PolicyServiceServicer_to_server(Servicer(), server)
    bound_port = server.add_secure_port(f"{bind_host}:{port}", credentials)
    if bound_port == 0:
        raise RuntimeError(f"failed to bind PolicyService on {bind_host}:{port}")
    await server.start()
    try:
        await server.wait_for_termination()
    finally:
        try:
            await server.stop(grace=0)
        except asyncio.CancelledError:
            pass
