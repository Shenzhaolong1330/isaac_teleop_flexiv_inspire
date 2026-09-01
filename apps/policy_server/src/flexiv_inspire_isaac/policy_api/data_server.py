"""gRPC PolicyDataService v2 multi-rate data and guarded action plane."""

from __future__ import annotations

import asyncio
import math
import secrets
import time
from typing import Callable

import numpy as np
from policy_contracts import (
    ActionMappingRegistry,
    SystemSchema,
    TensorDescriptor,
    flexiv_inspire_action_mappings,
)

from .broker import LatestActionBuffer, PolicyStreamLiveness
from .channel_broker import (
    ChannelBroker,
    ChannelDataError,
    ChannelSample,
    SampleTiming,
)
from .generated import policy_data_v2_pb2 as pb
from .generated import policy_data_v2_pb2_grpc as pb_grpc
from .lease import ControlLeaseManager, LocalControlState
from .models import ActionChunk, ActionPoint, validate_action_chunk


_DTYPE_TO_PROTO = {
    "bool": pb.BOOL,
    "uint8": pb.UINT8,
    "uint16": pb.UINT16,
    "uint32": pb.UINT32,
    "uint64": pb.UINT64,
    "int8": pb.INT8,
    "int16": pb.INT16,
    "int32": pb.INT32,
    "int64": pb.INT64,
    "float16": pb.FLOAT16,
    "float32": pb.FLOAT32,
    "float64": pb.FLOAT64,
    "bytes": pb.BYTES,
}

_DTYPE_TO_NUMPY = {
    "float32": np.dtype("<f4"),
}


def _peer_identity(context) -> str:
    common_names = context.auth_context().get("x509_common_name", ())
    common_name = (
        common_names[0].decode("utf-8", errors="replace") if common_names else ""
    )
    return f"{context.peer()}|{common_name}"


def decode_policy_action_chunk(
    request,
    *,
    schema: SystemSchema,
    receive_ns: int,
    action_mappings: ActionMappingRegistry | None = None,
) -> ActionChunk:
    """Validate a v2 wire chunk and convert it once to canonical v1 30D.

    Metadata remains outside the policy tensor.  The only currently registered
    policy action representation is 24D world-frame XYZ + rotation-vector +
    normalized absolute hand targets.  The native 30D representation is also
    accepted for existing PolicyService clients.
    """

    if int(request.schema_version) != schema.schema_version:
        raise ValueError("unsupported PolicyData schema version")
    if not request.schema_hash or request.schema_hash != schema.schema_hash:
        raise ValueError("system schema hash changed")
    try:
        descriptor = schema.action(request.action_schema_id)
    except ValueError as exc:
        raise ValueError(f"unknown action schema: {request.action_schema_id}") from exc
    if descriptor.frame_id != "world" or not descriptor.relative:
        raise ValueError("only relative world-frame actions are supported")
    try:
        numpy_dtype = _DTYPE_TO_NUMPY[descriptor.tensor.dtype]
    except KeyError as exc:
        raise ValueError(
            f"action dtype is not executable: {descriptor.tensor.dtype}"
        ) from exc
    expected_proto_dtype = _DTYPE_TO_PROTO[descriptor.tensor.dtype]
    expected_shape = tuple(descriptor.tensor.shape)
    expected_count = int(math.prod(expected_shape))

    mappings = action_mappings or flexiv_inspire_action_mappings()
    points: list[ActionPoint] = []
    for point in request.points:
        payload = point.action
        if int(payload.dtype) != expected_proto_dtype:
            raise ValueError("action tensor dtype differs from its descriptor")
        if tuple(payload.shape) != expected_shape:
            raise ValueError("action tensor shape differs from its descriptor")
        if len(payload.data) != expected_count * numpy_dtype.itemsize:
            raise ValueError("action tensor byte length differs from its descriptor")
        policy_values = np.frombuffer(payload.data, dtype=numpy_dtype).astype(
            np.float64, copy=False
        )
        native_values = mappings.map(descriptor.schema_id, policy_values)
        points.append(
            ActionPoint(
                execute_after_s=int(point.execute_after_ns) / 1e9,
                values=tuple(float(value) for value in native_values),
            )
        )

    return ActionChunk(
        schema_version=1,
        lease_id=str(request.lease_id),
        session_id=str(request.session_id),
        sequence=int(request.sequence),
        client_issued_monotonic_ns=int(request.client_issued_monotonic_ns),
        server_receive_monotonic_ns=int(receive_ns),
        ttl_from_server_receive_ns=int(request.ttl_from_server_receive_ns),
        frame_id=descriptor.frame_id,
        deadman=bool(request.deadman),
        points=tuple(points),
        valid_mask=(10 if descriptor.schema_id == "right_cartesian_delta_rotvec_v1" else 15),
    )


def tensor_descriptor_message(descriptor: TensorDescriptor) -> pb.TensorDescriptor:
    return pb.TensorDescriptor(
        dtype=_DTYPE_TO_PROTO[descriptor.dtype],
        shape=descriptor.shape,
        element_names=descriptor.element_names,
        unit=descriptor.unit,
    )


def system_description_message(
    schema: SystemSchema, state: LocalControlState
) -> pb.SystemDescription:
    result = pb.SystemDescription(
        schema_version=schema.schema_version,
        schema_hash=schema.schema_hash,
        system_id=schema.system_id,
        robot_type=schema.robot_type,
        session_id=state.session_id,
        control_state=state.state,
        metadata=dict(schema.metadata or {}),
    )
    for channel in schema.channels:
        target = result.channels.add(
            channel_id=channel.channel_id,
            semantic=channel.semantic,
            native_rate_hz=channel.native_rate_hz,
            frame_id=channel.frame_id,
            clock_domain=channel.clock_domain,
            encodings=channel.encodings,
        )
        target.tensor.CopyFrom(tensor_descriptor_message(channel.tensor))
    for action in schema.action_schemas:
        target = result.action_schemas.add(
            schema_id=action.schema_id,
            frame_id=action.frame_id,
            representation=action.representation,
            rate_hz=action.rate_hz,
            relative=action.relative,
        )
        target.tensor.CopyFrom(tensor_descriptor_message(action.tensor))
    return result


def sample_envelope_message(
    sample: ChannelSample,
    *,
    schema: SystemSchema,
    reference_ns: int | None = None,
) -> pb.SampleEnvelope:
    now_ns = time.monotonic_ns() if reference_ns is None else int(reference_ns)
    timing = sample.timing
    age_ns = max(0, now_ns - timing.alignment_time_ns)
    result = pb.SampleEnvelope(
        schema_version=schema.schema_version,
        schema_hash=schema.schema_hash,
        channel_id=sample.channel_id,
        timing=pb.SampleMetadata(
            source_time_ns=timing.source_time_ns,
            host_receive_time_ns=timing.host_receive_time_ns,
            mapped_host_time_ns=timing.mapped_host_time_ns,
            acquisition_start_ns=timing.acquisition_start_ns,
            acquisition_end_ns=timing.acquisition_end_ns,
            sequence=timing.sequence,
            valid=timing.valid,
            age_ns=age_ns,
            invalid_reason=timing.invalid_reason,
            source_clock_domain=timing.source_clock_domain,
            host_clock_domain=timing.host_clock_domain,
            timing_valid=timing.timing_valid,
        ),
    )
    if sample.is_image:
        result.image.CopyFrom(
            pb.ImagePayload(
                encoding=sample.image_encoding,
                data=sample.image_data,
                width=sample.image_width,
                height=sample.image_height,
                channels=sample.image_channels,
            )
        )
    elif sample.tensor is not None:
        descriptor = schema.channel(sample.channel_id).tensor
        result.tensor.CopyFrom(
            pb.TensorPayload(
                dtype=_DTYPE_TO_PROTO[descriptor.dtype],
                shape=sample.tensor.shape,
                data=sample.tensor.tobytes(order="C"),
            )
        )
    return result


def invalid_envelope_message(
    channel_id: str,
    reason: str,
    *,
    schema: SystemSchema,
    reference_ns: int,
) -> pb.SampleEnvelope:
    sample = ChannelSample(
        channel_id=channel_id,
        timing=SampleTiming(
            source_time_ns=0,
            host_receive_time_ns=reference_ns,
            mapped_host_time_ns=0,
            acquisition_start_ns=0,
            acquisition_end_ns=0,
            sequence=0,
            valid=False,
            invalid_reason=reason,
            source_clock_domain="",
            host_clock_domain="host_monotonic",
            timing_valid=False,
        ),
    )
    return sample_envelope_message(sample, schema=schema, reference_ns=reference_ns)


class PolicyDataServicer(pb_grpc.PolicyDataServiceServicer):
    def __init__(
        self,
        *,
        schema: SystemSchema,
        broker: ChannelBroker,
        local_state: Callable[[], LocalControlState],
        lease_manager: ControlLeaseManager | None = None,
        action_buffer: LatestActionBuffer[ActionChunk] | None = None,
        stop_callback: Callable[[str], None] | None = None,
        action_liveness: PolicyStreamLiveness | None = None,
        last_sequence_by_lease: dict[str, int] | None = None,
        action_mappings: ActionMappingRegistry | None = None,
    ) -> None:
        self.schema = schema
        self.broker = broker
        self.local_state = local_state
        self.lease_manager = lease_manager
        self.action_buffer = action_buffer
        self.stop_callback = stop_callback
        self.action_liveness = action_liveness
        self.last_sequence_by_lease = last_sequence_by_lease
        self.action_mappings = action_mappings or flexiv_inspire_action_mappings()

    def _validate_expectations(self, schema_hash: str, session_id: str) -> None:
        if schema_hash and schema_hash != self.schema.schema_hash:
            raise ChannelDataError("system schema hash changed")
        current = self.local_state()
        if session_id and session_id != current.session_id:
            raise ChannelDataError("hardware session changed")

    async def DescribeSystem(self, request, context):
        return system_description_message(self.schema, self.local_state())

    async def SubscribeSamples(self, request, context):
        import grpc

        try:
            self._validate_expectations(
                request.expected_schema_hash, request.expected_session_id
            )
            if not request.client_id or not request.channels:
                raise ChannelDataError("client_id and at least one channel are required")
            subscriptions = {}
            reliable = set()
            for item in request.channels:
                if item.channel_id in subscriptions:
                    raise ChannelDataError(f"duplicate channel: {item.channel_id}")
                descriptor = self.broker.descriptor(item.channel_id)
                requested_rate = float(item.max_rate_hz or descriptor.native_rate_hz)
                if not math.isfinite(requested_rate) or requested_rate <= 0.0:
                    raise ChannelDataError("max_rate_hz must be finite and positive")
                requested_rate = min(requested_rate, descriptor.native_rate_hz)
                if item.drop_policy == pb.RELIABLE:
                    if requested_rate + 1e-9 < descriptor.native_rate_hz:
                        raise ChannelDataError(
                            f"reliable channel {item.channel_id} cannot be downsampled"
                        )
                    reliable.add(item.channel_id)
                subscriptions[item.channel_id] = requested_rate
            positions = await self.broker.current_positions(subscriptions)
            next_send_ns = {channel_id: 0 for channel_id in subscriptions}
            while True:
                self._validate_expectations(
                    request.expected_schema_hash, request.expected_session_id
                )
                try:
                    samples, positions = await asyncio.wait_for(
                        self.broker.next_after(
                            positions, reliable_channels=frozenset(reliable)
                        ),
                        timeout=0.25,
                    )
                except asyncio.TimeoutError:
                    if context.cancelled():
                        return
                    continue
                now_ns = time.monotonic_ns()
                for stored in samples:
                    channel_id = stored.sample.channel_id
                    period_ns = int(round(1e9 / subscriptions[channel_id]))
                    sample_time_ns = stored.sample.timing.alignment_time_ns
                    if channel_id not in reliable:
                        deadline = next_send_ns[channel_id]
                        if deadline and sample_time_ns < deadline:
                            continue
                        if not deadline:
                            next_send_ns[channel_id] = sample_time_ns + period_ns
                        else:
                            while deadline <= sample_time_ns:
                                deadline += period_ns
                            next_send_ns[channel_id] = deadline
                    yield sample_envelope_message(
                        stored.sample, schema=self.schema, reference_ns=now_ns
                    )
        except ChannelDataError as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

    async def GetSnapshot(self, request, context):
        import grpc

        try:
            self._validate_expectations(
                request.expected_schema_hash, request.expected_session_id
            )
            if not request.channels:
                raise ChannelDataError("snapshot requires at least one channel")
            target_ns = int(request.target_monotonic_ns or time.monotonic_ns())
            result = pb.Snapshot(
                schema_version=self.schema.schema_version,
                schema_hash=self.schema.schema_hash,
                session_id=self.local_state().session_id,
                target_monotonic_ns=target_ns,
                complete=True,
            )
            seen: set[str] = set()
            alignment_names = {
                pb.ALIGNMENT_MODE_UNSPECIFIED: "latest_causal",
                pb.LATEST_CAUSAL: "latest_causal",
                pb.NEAREST: "nearest",
                pb.INTERPOLATE: "interpolate",
            }
            for item in request.channels:
                if item.channel_id in seen:
                    raise ChannelDataError(f"duplicate channel: {item.channel_id}")
                seen.add(item.channel_id)
                descriptor = self.broker.descriptor(item.channel_id)
                default_tolerance = int(math.ceil(2e9 / descriptor.native_rate_hz))
                try:
                    sample = await self.broker.sample_at(
                        item.channel_id,
                        target_ns,
                        alignment=alignment_names[item.alignment],
                        tolerance_ns=int(item.tolerance_ns or default_tolerance),
                        max_age_ns=int(item.max_age_ns or default_tolerance),
                    )
                    result.samples.add().CopyFrom(
                        sample_envelope_message(
                            sample, schema=self.schema, reference_ns=target_ns
                        )
                    )
                    if not sample.timing.valid:
                        result.complete = False
                        result.errors.append(
                            f"{item.channel_id}:{sample.timing.invalid_reason or 'invalid'}"
                        )
                except (ChannelDataError, KeyError) as exc:
                    result.complete = False
                    reason = str(exc)
                    result.errors.append(f"{item.channel_id}:{reason}")
                    result.samples.add().CopyFrom(
                        invalid_envelope_message(
                            item.channel_id,
                            reason,
                            schema=self.schema,
                            reference_ns=target_ns,
                        )
                    )
            return result
        except ChannelDataError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

    async def StreamActions(self, request_iterator, context):
        """Accept v2 policy actions through the existing v1 safety ingress."""

        import grpc

        if (
            self.lease_manager is None
            or self.action_buffer is None
            or self.stop_callback is None
            or self.action_liveness is None
            or self.last_sequence_by_lease is None
        ):
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "PolicyData action plane is not configured",
            )
            return

        peer = _peer_identity(context)
        stream_id = secrets.token_urlsafe(16)
        accepted_lease_id = ""
        try:
            async for request in request_iterator:
                receive_ns = time.monotonic_ns()
                try:
                    state = self.local_state()
                    if request.session_id != state.session_id:
                        self.lease_manager.invalidate()
                        raise PermissionError("hardware session changed")
                    if not state.policy_lease_allowed:
                        self.lease_manager.invalidate()
                        raise PermissionError("local policy authorization lost")
                    self.lease_manager.validate(
                        request.lease_id,
                        peer,
                        state.session_id,
                        refresh_ms=10_000,
                    )
                    previous = self.last_sequence_by_lease.get(request.lease_id, -1)
                    if request.sequence <= previous:
                        raise ValueError(
                            "action sequence must strictly increase per lease"
                        )
                    chunk = decode_policy_action_chunk(
                        request,
                        schema=self.schema,
                        receive_ns=receive_ns,
                        action_mappings=self.action_mappings,
                    )
                    validate_action_chunk(chunk, now_ns=time.monotonic_ns())
                    self.last_sequence_by_lease[request.lease_id] = request.sequence
                    accepted_lease_id = str(request.lease_id)
                    self.action_buffer.put(chunk)
                    self.action_liveness.arm(
                        stream_id=stream_id,
                        lease_id=request.lease_id,
                        session_id=request.session_id,
                        sequence=request.sequence,
                        action_deadline_ns=(
                            receive_ns + request.ttl_from_server_receive_ns
                        ),
                    )
                    yield pb.PolicyActionResult(
                        sequence=request.sequence,
                        accepted=True,
                        control_state=state.state,
                    )
                except (PermissionError, ValueError) as exc:
                    self.lease_manager.invalidate()
                    self.last_sequence_by_lease.pop(request.lease_id, None)
                    if self.action_liveness.disarm(stream_id):
                        self.stop_callback(f"policy-action-v2-rejected:{exc}")
                    yield pb.PolicyActionResult(
                        sequence=request.sequence,
                        accepted=False,
                        reason=str(exc),
                        control_state=self.local_state().state,
                    )
        finally:
            if self.action_liveness.disarm(stream_id):
                self.lease_manager.invalidate()
                self.last_sequence_by_lease.pop(accepted_lease_id, None)
                self.stop_callback("policy-action-v2-stream-ended")


def add_policy_data_servicer(
    server,
    *,
    schema: SystemSchema,
    broker: ChannelBroker,
    local_state: Callable[[], LocalControlState],
    lease_manager: ControlLeaseManager | None = None,
    action_buffer: LatestActionBuffer[ActionChunk] | None = None,
    stop_callback: Callable[[str], None] | None = None,
    action_liveness: PolicyStreamLiveness | None = None,
    last_sequence_by_lease: dict[str, int] | None = None,
    action_mappings: ActionMappingRegistry | None = None,
) -> None:
    pb_grpc.add_PolicyDataServiceServicer_to_server(
        PolicyDataServicer(
            schema=schema,
            broker=broker,
            local_state=local_state,
            lease_manager=lease_manager,
            action_buffer=action_buffer,
            stop_callback=stop_callback,
            action_liveness=action_liveness,
            last_sequence_by_lease=last_sequence_by_lease,
            action_mappings=action_mappings,
        ),
        server,
    )
