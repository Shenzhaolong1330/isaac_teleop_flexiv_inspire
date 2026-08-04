import pytest

from flexiv_inspire_isaac.policy_api.broker import PolicyStreamLiveness

from flexiv_inspire_isaac.policy_api.lease import (
    ControlLeaseManager,
    LocalControlState,
)
from flexiv_inspire_isaac.policy_api.ros_adapter import (
    initial_policy_command_sequence,
    policy_heartbeat_sequence,
)
from flexiv_inspire_isaac.policy_api.models import (
    ActionChunk,
    ActionPoint,
    capabilities_v1,
    validate_action_chunk,
)


def local_state(**overrides):
    values = dict(
        session_id="session",
        ft_zeroed=True,
        local_policy_authorized=True,
        pedal_valid=True,
        arms_online=True,
        hands_online=True,
        state="POLICY_ARMED",
    )
    values.update(overrides)
    return LocalControlState(**values)


def action(sequence=1, *, receive=1000, ttl=1000, offsets=(0.0000001,)):
    values = [0.0] * 30
    values[3:9] = [1, 0, 0, 0, 1, 0]
    values[12:18] = [1, 0, 0, 0, 1, 0]
    values[18:30] = [500] * 12
    return ActionChunk(
        schema_version=1,
        lease_id="lease",
        session_id="session",
        sequence=sequence,
        client_issued_monotonic_ns=999_999_999_999,
        server_receive_monotonic_ns=receive,
        ttl_from_server_receive_ns=ttl,
        frame_id="world",
        deadman=True,
        points=tuple(ActionPoint(offset, tuple(values)) for offset in offsets),
    )


def test_capabilities_fix_rotation_order_30d_layout_and_relative_time():
    capabilities = capabilities_v1()
    assert capabilities["rotation_element_order"] == [
        "R00",
        "R10",
        "R20",
        "R01",
        "R11",
        "R21",
    ]
    assert capabilities["default_action_dimension"] == 30
    assert "robot-host receipt" in capabilities["action_clock_semantics"]


def test_policy_ros_sequence_uses_process_independent_monotonic_epoch():
    previous = initial_policy_command_sequence(100_000_000_000)
    replacement = initial_policy_command_sequence(101_000_000_000)

    assert previous == 100_000_000
    assert replacement > previous
    assert replacement <= ((1 << 64) - 1) >> 6


def test_client_monotonic_epoch_is_never_compared_with_server_epoch():
    validate_action_chunk(action(), now_ns=1100)


def test_action_validation_rejects_zero_rot6d():
    base = action()
    broken = list(base.points[0].values)
    broken[3:9] = [0] * 6
    chunk = ActionChunk(
        1,
        "lease",
        "session",
        2,
        10**18,
        1000,
        1000,
        "world",
        True,
        (ActionPoint(0.0000001, tuple(broken)),),
    )
    with pytest.raises(ValueError, match="norm"):
        validate_action_chunk(chunk, now_ns=1100)


def test_chunk_rejects_expired_unordered_and_over_horizon_offsets():
    with pytest.raises(ValueError, match="expired"):
        validate_action_chunk(action(ttl=100), now_ns=1100)
    with pytest.raises(ValueError, match="monotonic"):
        validate_action_chunk(
            action(ttl=1000, offsets=(0.0000002, 0.0000001)), now_ns=1100
        )
    with pytest.raises(ValueError, match="horizon"):
        validate_action_chunk(
            action(ttl=1_000_000_000, offsets=(1.000000001,)), now_ns=1100
        )


def test_one_second_chunk_keeps_policy_heartbeat_alive_until_lease_or_gate_loss():
    now = [1_000_000_000]
    manager = ControlLeaseManager(max_lease_ms=2000, clock_ns=lambda: now[0])
    lease = manager.acquire(
        client_id="a", peer="peer-a", requested_ms=2000, local_state=local_state()
    )
    liveness = PolicyStreamLiveness()
    liveness.arm(
        stream_id="stream",
        lease_id=lease.token,
        session_id="session",
        sequence=7,
        action_deadline_ns=2_000_000_000,
    )
    for elapsed_ms in range(0, 1000, 50):
        now[0] = 1_000_000_000 + elapsed_ms * 1_000_000
        assert policy_heartbeat_sequence(liveness, manager, local_state()) == 7
    assert liveness.current() is not None
    now[0] = 2_000_000_000
    assert policy_heartbeat_sequence(liveness, manager, local_state()) is None
    liveness.arm(
        stream_id="stream-2",
        lease_id=lease.token,
        session_id="session",
        sequence=8,
        action_deadline_ns=2_500_000_000,
    )
    assert policy_heartbeat_sequence(
        liveness, manager, local_state(pedal_valid=False)
    ) is None
    assert liveness.current() is None


def test_remote_lease_requires_every_local_gate_and_is_single_owner():
    now = [1_000_000]
    manager = ControlLeaseManager(clock_ns=lambda: now[0])
    with pytest.raises(PermissionError):
        manager.acquire(
            client_id="a",
            peer="peer-a",
            requested_ms=1000,
            local_state=local_state(ft_zeroed=False),
        )
    lease = manager.acquire(
        client_id="a",
        peer="peer-a",
        requested_ms=1000,
        local_state=local_state(),
    )
    with pytest.raises(PermissionError):
        manager.acquire(
            client_id="b",
            peer="peer-b",
            requested_ms=1000,
            local_state=local_state(),
        )
    assert manager.validate(lease.token, "peer-a", "session") == lease
    assert manager.release(lease.token, "peer-a")
