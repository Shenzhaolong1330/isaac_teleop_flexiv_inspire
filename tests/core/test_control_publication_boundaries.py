from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

pytest.importorskip("flexiv_inspire_interfaces")

from flexiv_inspire_control.node import ControlBridge
from flexiv_inspire_control.frames import BaseTransform
from isaac_teleop_core.command import (
    BimanualCommand,
    CommandPoint,
    CommandSource,
    ControlRepresentation,
    SCHEMA_VERSION,
    ValidMask,
)
from isaac_teleop_core.control import ControlState


class _Parameter:
    def __init__(self, value):
        self.value = value


class _Publisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


class _Clock:
    @staticmethod
    def now():
        return SimpleNamespace(to_msg=lambda: SimpleNamespace(sec=0, nanosec=0))


class _Arbiter:
    def __init__(self) -> None:
        self.state = ControlState.ACTIVE
        self.marked_sent = []
        self.deadman_releases = []
        self.heartbeats = []

    @property
    def snapshot(self):
        return SimpleNamespace(state=self.state)

    def tick(self, *, now_monotonic_ns: int):
        return self.snapshot

    def mark_sent(self, command) -> None:
        self.marked_sent.append(command)

    def heartbeat(self, source: CommandSource, *, now_monotonic_ns: int) -> None:
        self.heartbeats.append((source, now_monotonic_ns))

    def reject_invalid_command(
        self, source: CommandSource, *, now_monotonic_ns: int
    ) -> None:
        self.state = ControlState.HOLD_LATCHED

    def require_reauthorization(
        self, source: CommandSource, *, now_monotonic_ns: int
    ) -> None:
        self.state = ControlState.HOLD_LATCHED

    def fault(self) -> None:
        self.state = ControlState.FAULT

    def observe_deadman_released(self, source: CommandSource) -> None:
        self.deadman_releases.append(source)


class _IPC:
    def __init__(self, *, accepted: bool) -> None:
        self.accepted = accepted
        self.requests = []

    def request(self, kind: str, payload: dict):
        self.requests.append((kind, payload))
        return "command_ack", {
            "accepted": self.accepted,
            "reason": "" if self.accepted else "mock_reject",
        }


def _wire_message(sequence: int = 7):
    return SimpleNamespace(
        sequence=sequence,
        trajectory=[
            SimpleNamespace(
                execute_after=SimpleNamespace(sec=0, nanosec=0),
            )
        ],
        header=SimpleNamespace(stamp=None),
        ttl=SimpleNamespace(sec=0, nanosec=900_000_000),
    )


def _command(point: CommandPoint) -> BimanualCommand:
    return BimanualCommand(
        schema_version=SCHEMA_VERSION,
        session_id="test-session",
        source=CommandSource.TELEOP,
        sequence=7,
        issued_monotonic_ns=time.monotonic_ns(),
        ttl_ns=900_000_000,
        representation=ControlRepresentation.CARTESIAN_ROT6D,
        frame_id="world",
        points=(point,),
        valid_mask=ValidMask.LEFT_ARM,
        deadman=True,
    )


def _bridge(*, token: str | None, rdk_accepts: bool) -> ControlBridge:
    bridge = ControlBridge.__new__(ControlBridge)
    bridge._chunk_lock = threading.Lock()
    bridge._hardware_command_lock = threading.Lock()
    bridge._chunk_generation = 1
    bridge._state_lock = threading.RLock()
    bridge._arbiter = _Arbiter()
    bridge._safe_pub = _Publisher()
    bridge._sent_pub = _Publisher()
    bridge._safe_pose_rdk = {
        "left": np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
    }
    bridge._previous_output_quaternion = {"left": None, "right": None}
    bridge._world_from_base = {
        side: BaseTransform(
            np.zeros(3, dtype=float),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        )
        for side in ("left", "right")
    }
    bridge._pending_arm_token = token
    bridge._pending_arm_token_expiry_ns = (
        time.monotonic_ns() + 1_000_000_000 if token else 0
    )
    bridge._rdk_control_lease_active = False
    bridge._local_permission = True
    bridge._physical_pedal = True
    bridge._ipc_command = _IPC(accepted=rdk_accepts)
    bridge._update_gates = lambda now: None
    bridge._send_hold_once = lambda reason: bridge.holds.append(reason)
    bridge._publish_trace = lambda *args: bridge.traces.append(args)
    bridge._publish_control_state = lambda: None
    bridge.get_clock = lambda: _Clock()
    bridge.holds = []
    bridge.traces = []
    values = {
        "software_safety_limits_enabled": True,
        "max_translation_step_m": 0.01,
        "max_rotation_step_rad": 0.10,
        "max_linear_velocity_m_s": 0.05,
        "max_angular_velocity_rad_s": 0.15,
        "max_linear_acceleration_m_s2": 0.25,
        "max_angular_acceleration_rad_s2": 0.50,
        "cartesian_control_mode": "impedance",
        "cartesian_position_stiffness": [3000.0, 3000.0, 3000.0, 200.0, 200.0, 200.0],
        "cartesian_impedance_stiffness": [1200.0, 1200.0, 1200.0, 80.0, 80.0, 80.0],
        "cartesian_damping_ratio": [0.7] * 6,
    }
    bridge.get_parameter = lambda name: _Parameter(values[name])
    return bridge


@pytest.mark.parametrize(
    ("point", "token", "rdk_accepts", "safe_count", "ipc_count", "state"),
    [
        (
            replace(
                CommandPoint.identity(),
                left_delta_xyz=np.array([0.011, 0.0, 0.0]),
            ),
            "local-token",
            True,
            0,
            0,
            ControlState.HOLD_LATCHED,
        ),
        (
            CommandPoint.identity(),
            None,
            True,
            0,
            0,
            ControlState.HOLD_LATCHED,
        ),
        (
            CommandPoint.identity(),
            "local-token",
            False,
            1,
            1,
            ControlState.HOLD_LATCHED,
        ),
    ],
    ids=["over-step", "no-local-token", "rdk-reject-recoverable"],
)
def test_failed_points_preserve_requested_safe_sent_boundaries(
    point,
    token,
    rdk_accepts,
    safe_count,
    ipc_count,
    state,
) -> None:
    bridge = _bridge(token=token, rdk_accepts=rdk_accepts)
    command = _command(point)

    bridge._execute_chunk(1, command, _wire_message(), time.monotonic_ns())

    assert len(bridge._safe_pub.messages) == safe_count
    assert len(bridge._sent_pub.messages) == 0
    assert len(bridge._ipc_command.requests) == ipc_count
    assert bridge._arbiter.state is state
    assert len(bridge.traces) == 1
    assert bridge.traces[0][1] is (
        bridge._safe_pub.messages[0] if safe_count else None
    )
    assert bridge.traces[0][2] is None
    assert bridge.holds
    if token is None:
        assert bridge.holds == ["control_authorization_expired"]


def test_mapper_clamped_boundary_survives_floating_point_roundoff() -> None:
    bridge = _bridge(token="local-token", rdk_accepts=True)
    # This is the exact value captured from the 2026-08-02 hardware MCAP after
    # normalizing a Quest step to the configured 0.01 m boundary.
    point = replace(
        CommandPoint.identity(),
        left_delta_xyz=np.array(
            [0.005904499750351698, -0.007826379128638596, 0.0019709572377164704]
        ),
    )

    bridge._execute_chunk(
        1, _command(point), _wire_message(), time.monotonic_ns()
    )

    assert len(bridge._safe_pub.messages) == 1
    assert len(bridge._sent_pub.messages) == 1
    assert len(bridge._ipc_command.requests) == 1
    assert bridge._arbiter.state is ControlState.ACTIVE
    assert bridge.traces[0][1] is bridge._safe_pub.messages[0]
    assert bridge.traces[0][2] is bridge._sent_pub.messages[0]
    assert bridge.holds == []


def test_practical_mode_does_not_reject_a_bridge_step_over_local_limit() -> None:
    bridge = _bridge(token="local-token", rdk_accepts=True)
    original_get_parameter = bridge.get_parameter
    bridge.get_parameter = lambda name: (
        _Parameter(False)
        if name == "software_safety_limits_enabled"
        else original_get_parameter(name)
    )
    point = replace(
        CommandPoint.identity(),
        left_delta_xyz=np.array([0.011, 0.0, 0.0]),
    )

    bridge._execute_chunk(
        1, _command(point), _wire_message(), time.monotonic_ns()
    )

    assert len(bridge._sent_pub.messages) == 1
    assert len(bridge._ipc_command.requests) == 1
    assert bridge._arbiter.state is ControlState.ACTIVE


def test_sent_is_published_only_after_positive_rdk_ack() -> None:
    bridge = _bridge(token="local-token", rdk_accepts=True)
    command = _command(CommandPoint.identity())

    bridge._execute_chunk(1, command, _wire_message(), time.monotonic_ns())

    assert len(bridge._safe_pub.messages) == 1
    assert len(bridge._ipc_command.requests) == 1
    assert len(bridge._sent_pub.messages) == 1
    assert bridge._sent_pub.messages[0] is bridge._safe_pub.messages[0]
    assert bridge._arbiter.marked_sent == [command]
    assert bridge._arbiter.state is ControlState.ACTIVE
    assert not bridge.holds


def test_pedal_release_cancels_safe_point_before_hardware_send() -> None:
    bridge = _bridge(token="local-token", rdk_accepts=True)

    class ReleaseBeforeHardwareSend:
        def __enter__(self):
            bridge._chunk_generation += 1
            bridge._physical_pedal = False

        def __exit__(self, *_args):
            return False

    bridge._hardware_command_lock = ReleaseBeforeHardwareSend()

    bridge._execute_chunk(
        1,
        _command(CommandPoint.identity()),
        _wire_message(),
        time.monotonic_ns(),
    )

    assert len(bridge._safe_pub.messages) == 1
    assert len(bridge._ipc_command.requests) == 0
    assert len(bridge._sent_pub.messages) == 0
    assert bridge._arbiter.state is ControlState.ACTIVE
    assert bridge.traces[-1][3] == "clutch released before hardware send"


def test_neutral_teleop_packet_does_not_latch_hold() -> None:
    bridge = _bridge(token="local-token", rdk_accepts=True)
    bridge._requested_pub = _Publisher()
    bridge._emit_deviceio = lambda *args, **kwargs: True
    message = SimpleNamespace(deadman=False, valid_mask=0)

    bridge._on_command(CommandSource.TELEOP, message)

    assert bridge._arbiter.state is ControlState.ACTIVE
    assert bridge._arbiter.deadman_releases == [CommandSource.TELEOP]
    assert bridge._requested_pub.messages == [message]
    assert bridge._arbiter.heartbeats[0][0] is CommandSource.TELEOP
    assert bridge.traces[0][3] == "teleop_clutch_released"
    assert not bridge.holds


def test_neutral_teleop_packet_is_best_effort_before_recorder_ingress() -> None:
    bridge = _bridge(token="local-token", rdk_accepts=True)
    bridge._requested_pub = _Publisher()
    priorities = []
    bridge._emit_deviceio = lambda *args, **kwargs: priorities.append(
        kwargs["critical"]
    ) or False
    message = SimpleNamespace(deadman=False, valid_mask=0)

    bridge._on_command(CommandSource.TELEOP, message)

    assert priorities == [False]
    assert bridge._requested_pub.messages == [message]
    assert bridge._arbiter.state is ControlState.ACTIVE
    assert not bridge.holds


def test_only_deadman_authorized_nonempty_command_requires_critical_capture() -> None:
    assert not ControlBridge._command_may_actuate(
        SimpleNamespace(deadman=False, valid_mask=0)
    )
    assert not ControlBridge._command_may_actuate(
        SimpleNamespace(deadman=True, valid_mask=0)
    )
    assert ControlBridge._command_may_actuate(
        SimpleNamespace(deadman=True, valid_mask=int(ValidMask.LEFT_ARM))
    )
