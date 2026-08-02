from types import SimpleNamespace

import pytest

from flexiv_inspire_isaac.dftp.command_gate import (
    GateSnapshot,
    SafeCommandRejected,
    validate_safe_command_message,
)


def duration(ns):
    return SimpleNamespace(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


class Message:
    LEFT_HAND_VALID = 4
    RIGHT_HAND_VALID = 8

    def __init__(self):
        self.schema_version = 1
        self.session_id = "session"
        self.source = "policy"
        self.sequence = 7
        self.ttl = duration(100_000_000)
        self.header = SimpleNamespace(stamp=duration(1_000_000_000))
        self.valid_mask = 12
        self.deadman = True
        self.trajectory = [
            SimpleNamespace(
                execute_after=duration(0),
                left_hand_targets=[100] * 6,
                right_hand_targets=[200] * 6,
            )
        ]


def snapshot(**changes):
    values = dict(
        configured_session="session",
        control_session="session",
        control_active=True,
        fault_latched=False,
        control_state_received_ns=10,
        control_state_timeout_ns=200,
        last_hand_state_ns={"left": 10, "right": 10},
        hand_state_timeout_ns=200,
        last_sequence=6,
    )
    values.update(changes)
    return GateSnapshot(**values)


def test_safe_sent_command_yields_atomic_six_axis_targets():
    targets = validate_safe_command_message(
        Message(), snapshot(), now_monotonic_ns=20, now_ros_ns=1_050_000_000
    )
    assert [(target.side, target.angles) for target in targets] == [
        ("left", (100,) * 6),
        ("right", (200,) * 6),
    ]
    assert targets[0].remaining_ttl_ns == 50_000_000


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"control_active": False}, "not ACTIVE"),
        ({"fault_latched": True}, "not ACTIVE"),
        ({"control_session": "other"}, "session"),
        ({"last_sequence": 7}, "sequence"),
        ({"control_state_received_ns": -1000}, "stale"),
        ({"last_hand_state_ns": {"left": -1000, "right": 10}}, "hand state"),
    ],
)
def test_gate_rejects_missing_local_safety_condition(changes, reason):
    with pytest.raises(SafeCommandRejected, match=reason):
        validate_safe_command_message(
            Message(),
            snapshot(**changes),
            now_monotonic_ns=20,
            now_ros_ns=1_050_000_000,
        )


def test_execute_after_at_ttl_boundary_is_rejected():
    message = Message()
    message.trajectory[0].execute_after = duration(100_000_000)
    with pytest.raises(SafeCommandRejected, match="execute_after"):
        validate_safe_command_message(
            message, snapshot(), now_monotonic_ns=20, now_ros_ns=1_050_000_000
        )


def test_negative_execute_after_is_rejected_at_hand_boundary():
    message = Message()
    message.trajectory[0].execute_after = duration(-1)
    with pytest.raises(SafeCommandRejected, match="execute_after"):
        validate_safe_command_message(
            message, snapshot(), now_monotonic_ns=20, now_ros_ns=1_050_000_000
        )


def test_expired_and_multichunk_commands_are_rejected():
    message = Message()
    with pytest.raises(SafeCommandRejected, match="expired"):
        validate_safe_command_message(
            message, snapshot(), now_monotonic_ns=20, now_ros_ns=1_200_000_000
        )
    message.trajectory.append(message.trajectory[0])
    with pytest.raises(SafeCommandRejected, match="exactly one"):
        validate_safe_command_message(
            message, snapshot(), now_monotonic_ns=20, now_ros_ns=1_050_000_000
        )
