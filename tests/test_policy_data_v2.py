from __future__ import annotations

import asyncio

import numpy as np
import pytest

from flexiv_inspire_isaac.policy_api.channel_broker import (
    ChannelBroker,
    ChannelDataError,
    SampleTiming,
    image_sample,
    tensor_sample,
)
from flexiv_inspire_isaac.policy_api.data_server import (
    PolicyDataServicer,
    decode_policy_action_chunk,
    sample_envelope_message,
    system_description_message,
)
from flexiv_inspire_isaac.policy_api.data_client import ChannelStats, decode_tensor
from flexiv_inspire_isaac.policy_api.generated import policy_data_v2_pb2 as pb
from flexiv_inspire_isaac.policy_api.generated import policy_data_v2_pb2_grpc as pb_grpc
from flexiv_inspire_isaac.policy_api.broker import (
    LatestActionBuffer,
    PolicyStreamLiveness,
)
from flexiv_inspire_isaac.policy_api.lease import (
    ControlLeaseManager,
    LocalControlState,
)
from flexiv_inspire_isaac.policy_api.models import validate_action_chunk
from flexiv_inspire_isaac.policy_api.system_schema import build_system_schema


def _timing(timestamp_ns: int, sequence: int, *, valid: bool = True) -> SampleTiming:
    return SampleTiming(
        source_time_ns=timestamp_ns - 100,
        host_receive_time_ns=timestamp_ns + 100,
        mapped_host_time_ns=timestamp_ns,
        acquisition_start_ns=timestamp_ns - 50,
        acquisition_end_ns=timestamp_ns + 50,
        sequence=sequence,
        valid=valid,
        invalid_reason="" if valid else "source-invalid",
        source_clock_domain="robot",
        host_clock_domain="host_monotonic",
        timing_valid=True,
    )


def _state() -> LocalControlState:
    return LocalControlState(
        session_id="session",
        ft_zeroed=False,
        local_policy_authorized=False,
        pedal_valid=False,
        arms_online=True,
        hands_online=True,
        state="READY",
    )


def _armed_state() -> LocalControlState:
    return LocalControlState(
        session_id="session",
        ft_zeroed=True,
        local_policy_authorized=True,
        pedal_valid=True,
        arms_online=True,
        hands_online=True,
        state="POLICY_ARMED",
    )


def _policy_action(
    schema,
    *,
    sequence: int = 1,
    values: np.ndarray | None = None,
) -> pb.PolicyActionChunk:
    if values is None:
        values = np.concatenate(
            (np.zeros(12, dtype=np.float32), np.full(12, 0.5, dtype=np.float32))
        )
    values = np.asarray(values, dtype="<f4")
    return pb.PolicyActionChunk(
        schema_version=2,
        schema_hash=schema.schema_hash,
        action_schema_id="cartesian_delta_rotvec_v1",
        lease_id="lease",
        session_id="session",
        sequence=sequence,
        client_issued_monotonic_ns=1,
        ttl_from_server_receive_ns=500_000_000,
        deadman=True,
        points=[
            pb.PolicyActionPoint(
                execute_after_ns=0,
                action=pb.TensorPayload(
                    dtype=pb.FLOAT32,
                    shape=values.shape,
                    data=values.tobytes(),
                ),
            )
        ],
    )


def test_system_description_exposes_independent_rates_and_minimal_action() -> None:
    schema = build_system_schema(
        arm_rate_hz=200.0,
        hand_rate_hz=15.0,
        tactile_rate_hz=15.0,
        camera_rate_hz=15.0,
        action_rate_hz=30.0,
    )
    message = system_description_message(schema, _state())
    channels = {item.channel_id: item for item in message.channels}
    actions = {item.schema_id: item for item in message.action_schemas}

    assert message.schema_hash == schema.schema_hash
    assert channels["arm.left.q"].native_rate_hz == 200.0
    assert channels["camera.head.rgb"].native_rate_hz == 15.0
    assert actions["cartesian_delta_rotvec_v1"].rate_hz == 30.0
    assert tuple(actions["cartesian_delta_rotvec_v1"].tensor.shape) == (24,)
    assert tuple(actions["flexiv_inspire_native_rot6d_v1"].tensor.shape) == (30,)


def test_v2_policy_action_converts_once_to_canonical_30d() -> None:
    schema = build_system_schema()
    request = _policy_action(schema)

    chunk = decode_policy_action_chunk(request, schema=schema, receive_ns=100)
    validate_action_chunk(chunk, now_ns=101)

    values = np.asarray(chunk.points[0].values)
    assert values.shape == (30,)
    assert np.allclose(values[0:3], 0.0)
    assert np.allclose(values[3:9], [1, 0, 0, 0, 1, 0])
    assert np.allclose(values[9:12], 0.0)
    assert np.allclose(values[12:18], [1, 0, 0, 0, 1, 0])
    assert np.allclose(values[18:30], 500.0)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        (lambda request: setattr(request, "schema_hash", "wrong"), "schema hash"),
        (
            lambda request: setattr(request.points[0].action, "dtype", pb.FLOAT64),
            "dtype",
        ),
        (
            lambda request: request.points[0].action.shape.__setitem__(0, 23),
            "shape",
        ),
        (
            lambda request: setattr(request, "action_schema_id", "unknown"),
            "unknown action schema",
        ),
    ),
)
def test_v2_policy_action_rejects_schema_tensor_mismatch(mutation, reason) -> None:
    schema = build_system_schema()
    request = _policy_action(schema)
    mutation(request)

    with pytest.raises(ValueError, match=reason):
        decode_policy_action_chunk(request, schema=schema, receive_ns=100)


def test_v2_action_stream_reuses_lease_ttl_and_liveness() -> None:
    class Context:
        @staticmethod
        def peer():
            return "test-peer"

        @staticmethod
        def auth_context():
            return {}

    async def requests(item):
        yield item

    async def scenario() -> None:
        schema = build_system_schema()
        lease = ControlLeaseManager()
        granted = lease.acquire(
            client_id="client",
            peer="test-peer|",
            requested_ms=2_000,
            local_state=_armed_state(),
        )
        request = _policy_action(schema)
        request.lease_id = granted.token
        actions = LatestActionBuffer()
        liveness = PolicyStreamLiveness()
        stops: list[str] = []
        servicer = PolicyDataServicer(
            schema=schema,
            broker=ChannelBroker(schema),
            local_state=_armed_state,
            lease_manager=lease,
            action_buffer=actions,
            stop_callback=stops.append,
            action_liveness=liveness,
            last_sequence_by_lease={},
        )

        results = [
            item
            async for item in servicer.StreamActions(requests(request), Context())
        ]

        assert len(results) == 1 and results[0].accepted
        chunk = actions.take()
        assert chunk is not None and chunk.sequence == 1
        assert liveness.current() is None
        assert stops == ["policy-action-v2-stream-ended"]

    asyncio.run(scenario())


def test_v2_action_stream_fails_closed_when_pedal_is_released() -> None:
    class Context:
        @staticmethod
        def peer():
            return "test-peer"

        @staticmethod
        def auth_context():
            return {}

    async def requests(item):
        yield item

    async def scenario() -> None:
        schema = build_system_schema()
        lease = ControlLeaseManager()
        granted = lease.acquire(
            client_id="client",
            peer="test-peer|",
            requested_ms=2_000,
            local_state=_armed_state(),
        )
        request = _policy_action(schema)
        request.lease_id = granted.token
        actions = LatestActionBuffer()
        servicer = PolicyDataServicer(
            schema=schema,
            broker=ChannelBroker(schema),
            local_state=_state,
            lease_manager=lease,
            action_buffer=actions,
            stop_callback=lambda reason: None,
            action_liveness=PolicyStreamLiveness(),
            last_sequence_by_lease={},
        )

        results = [
            item
            async for item in servicer.StreamActions(requests(request), Context())
        ]

        assert len(results) == 1 and not results[0].accepted
        assert "authorization" in results[0].reason
        assert actions.take() is None
        assert not lease.current_valid(granted.token, "session")

    asyncio.run(scenario())


def test_broker_drop_old_keeps_latest_while_reliable_keeps_each_sample() -> None:
    async def scenario() -> None:
        schema = build_system_schema(arm_rate_hz=20.0)
        broker = ChannelBroker(schema)
        channel = "arm.left.q"
        descriptor = broker.descriptor(channel)
        positions = await broker.current_positions((channel,))
        for sequence in range(1, 4):
            await broker.publish(
                tensor_sample(
                    channel,
                    np.full(7, sequence, dtype=np.float64),
                    _timing(sequence * 1_000, sequence),
                    descriptor,
                )
            )
        dropped, dropped_positions = await broker.next_after(positions)
        reliable, reliable_positions = await broker.next_after(
            positions, reliable_channels=frozenset((channel,))
        )

        assert len(dropped) == 1
        assert dropped[0].sample.timing.sequence == 3
        assert [item.sample.timing.sequence for item in reliable] == [1, 2, 3]
        assert dropped_positions[channel] == reliable_positions[channel] == 3

    asyncio.run(scenario())


def test_snapshot_alignment_is_causal_nearest_and_so3_interpolated() -> None:
    async def scenario() -> None:
        schema = build_system_schema(arm_rate_hz=10.0)
        broker = ChannelBroker(schema)
        channel = "arm.left.tcp_pose"
        descriptor = broker.descriptor(channel)
        first = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float64)
        # 180 deg about Z in xyzw.
        second = np.asarray([2, 4, 6, 0, 0, 1, 0], dtype=np.float64)
        await broker.publish_many(
            (
                tensor_sample(channel, first, _timing(1_000, 1), descriptor),
                tensor_sample(channel, second, _timing(3_000, 2), descriptor),
            )
        )

        causal = await broker.sample_at(
            channel, 2_200, alignment="latest_causal", tolerance_ns=2_000, max_age_ns=2_000
        )
        nearest = await broker.sample_at(
            channel, 2_200, alignment="nearest", tolerance_ns=2_000, max_age_ns=2_000
        )
        interpolated = await broker.sample_at(
            channel, 2_000, alignment="interpolate", tolerance_ns=2_000, max_age_ns=2_000
        )

        assert causal.timing.sequence == 1
        assert nearest.timing.sequence == 2
        assert np.allclose(interpolated.tensor[:3], [1, 2, 3])
        expected = np.sqrt(0.5)
        assert np.allclose(np.abs(interpolated.tensor[5:7]), [expected, expected])
        assert interpolated.timing.alignment_time_ns == 2_000

    asyncio.run(scenario())


def test_images_are_not_interpolated_and_invalid_snapshot_has_no_fake_payload() -> None:
    async def scenario() -> None:
        schema = build_system_schema(camera_rate_hz=15.0)
        broker = ChannelBroker(schema)
        channel = "camera.head.rgb"
        await broker.publish(
            image_sample(
                channel,
                b"jpeg",
                encoding="jpeg",
                width=424,
                height=240,
                channels=3,
                timing=_timing(1_000, 1),
            )
        )
        with pytest.raises(ChannelDataError, match="bracket|interpolated"):
            await broker.sample_at(
                channel,
                1_500,
                alignment="interpolate",
                tolerance_ns=1_000,
                max_age_ns=1_000,
            )

    asyncio.run(scenario())


def test_tensor_envelope_is_little_endian_packed_with_metadata_outside_tensor() -> None:
    schema = build_system_schema()
    descriptor = schema.channel("arm.left.q")
    sample = tensor_sample(
        descriptor.channel_id,
        np.arange(7, dtype=np.float64),
        _timing(10_000, 7),
        descriptor,
    )

    message = sample_envelope_message(sample, schema=schema, reference_ns=10_500)

    assert message.tensor.dtype == pb.FLOAT64
    assert tuple(message.tensor.shape) == (7,)
    assert np.array_equal(np.frombuffer(message.tensor.data, dtype="<f8"), np.arange(7))
    assert message.timing.age_ns == 500
    assert len(message.tensor.data) == 7 * 8
    assert np.array_equal(decode_tensor(message.tensor), np.arange(7))


def test_channel_stats_reports_rate_age_and_invalid_count() -> None:
    stats = ChannelStats()
    for index, valid in enumerate((True, True, False)):
        stats.observe(
            pb.SampleEnvelope(
                timing=pb.SampleMetadata(valid=valid, age_ns=(index + 1) * 1_000_000)
            ),
            receive_ns=1_000_000_000 + index * 10_000_000,
        )

    report = stats.report()
    assert report["count"] == 3
    assert report["rate_hz"] == pytest.approx(100.0)
    assert report["mean_age_ms"] == pytest.approx(2.0)
    assert report["invalid"] == 1


def test_grpc_v2_describe_snapshot_and_async_subscription() -> None:
    async def scenario() -> None:
        import grpc

        schema = build_system_schema(arm_rate_hz=200.0)
        broker = ChannelBroker(schema)
        server = grpc.aio.server()
        pb_grpc.add_PolicyDataServiceServicer_to_server(
            PolicyDataServicer(schema=schema, broker=broker, local_state=_state),
            server,
        )
        try:
            port = server.add_insecure_port("127.0.0.1:0")
        except RuntimeError as exc:
            pytest.skip(f"loopback sockets are unavailable in this sandbox: {exc}")
        await server.start()
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        stub = pb_grpc.PolicyDataServiceStub(channel)
        try:
            description = await stub.DescribeSystem(pb.DescribeSystemRequest())
            assert description.schema_hash == schema.schema_hash

            descriptor = broker.descriptor("arm.left.q")
            first = tensor_sample(
                descriptor.channel_id,
                np.arange(7, dtype=np.float64),
                _timing(10_000, 1),
                descriptor,
            )
            await broker.publish(first)
            snapshot = await stub.GetSnapshot(
                pb.SnapshotRequest(
                    expected_schema_hash=schema.schema_hash,
                    expected_session_id="session",
                    target_monotonic_ns=10_100,
                    channels=[
                        pb.SnapshotChannelRequest(
                            channel_id=descriptor.channel_id,
                            alignment=pb.LATEST_CAUSAL,
                            tolerance_ns=1_000,
                            max_age_ns=1_000,
                        )
                    ],
                )
            )
            assert snapshot.complete
            assert snapshot.samples[0].timing.sequence == 1

            call = stub.SubscribeSamples(
                pb.SubscribeRequest(
                    client_id="test",
                    expected_schema_hash=schema.schema_hash,
                    expected_session_id="session",
                    channels=[
                        pb.ChannelSubscription(
                            channel_id=descriptor.channel_id,
                            max_rate_hz=100.0,
                            drop_policy=pb.DROP_OLDEST,
                        )
                    ],
                )
            )
            pending = asyncio.create_task(call.read())
            await asyncio.sleep(0.02)
            await broker.publish(
                tensor_sample(
                    descriptor.channel_id,
                    np.full(7, 2.0),
                    _timing(20_000, 2),
                    descriptor,
                )
            )
            streamed = await asyncio.wait_for(pending, timeout=1.0)
            assert streamed.channel_id == descriptor.channel_id
            assert streamed.timing.sequence == 2
            call.cancel()
        finally:
            await channel.close()
            await server.stop(grace=0)

    asyncio.run(scenario())
