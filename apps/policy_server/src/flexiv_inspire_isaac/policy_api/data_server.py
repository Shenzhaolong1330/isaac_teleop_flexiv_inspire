"""gRPC PolicyDataService v2 read-only data-plane implementation."""

from __future__ import annotations

import asyncio
import math
import time
from typing import Callable

from policy_contracts import SystemSchema, TensorDescriptor

from .channel_broker import (
    ChannelBroker,
    ChannelDataError,
    ChannelSample,
    SampleTiming,
)
from .generated import policy_data_v2_pb2 as pb
from .generated import policy_data_v2_pb2_grpc as pb_grpc
from .lease import LocalControlState


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
    ) -> None:
        self.schema = schema
        self.broker = broker
        self.local_state = local_state

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


def add_policy_data_servicer(
    server,
    *,
    schema: SystemSchema,
    broker: ChannelBroker,
    local_state: Callable[[], LocalControlState],
) -> None:
    pb_grpc.add_PolicyDataServiceServicer_to_server(
        PolicyDataServicer(schema=schema, broker=broker, local_state=local_state),
        server,
    )
