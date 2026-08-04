from __future__ import annotations

import json
import math
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

pytest.importorskip("flexiv_inspire_interfaces")

from flexiv_inspire_control.node import ControlBridge
from flexiv_inspire_control.frames import BaseTransform
from flexiv_inspire_control.teleop_input_node import (
    TeleopInput,
    _initial_command_sequence,
    _validated_button,
    _validated_squeezes,
)
from isaac_teleop_core.command import (
    CommandSource,
    CommandPoint,
    ControlRepresentation,
    ValidMask,
)
from isaac_teleop_core.control import (
    ControlArbiter,
    ControlState,
    GateInputs,
    HoldReason,
)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -0.01, 1.01, 999])
def test_quest_squeeze_rejects_nonfinite_and_out_of_range(bad: float) -> None:
    with pytest.raises(ValueError):
        _validated_squeezes(
            {"left_squeeze_value": bad, "right_squeeze_value": 0.5}
        )


def test_quest_squeeze_accepts_only_normalized_values() -> None:
    assert _validated_squeezes(
        {"left_squeeze_value": 0.25, "right_squeeze_value": 1.0}
    ) == {"left": 0.25, "right": 1.0}


@pytest.mark.parametrize(
    ("raw", "expected"),
    ((False, False), (True, True), (0, False), (1, True), (0.0, False), (1.0, True)),
)
def test_quest_button_accepts_boolean_and_normalized_transport_values(
    raw: object, expected: bool
) -> None:
    assert _validated_button(raw) is expected


@pytest.mark.parametrize("raw", (-1, 0.5, 2, math.nan, "true", None))
def test_quest_button_rejects_ambiguous_values(raw: object) -> None:
    with pytest.raises(ValueError, match="Quest home button"):
        _validated_button(raw)


def test_middle_pedal_directly_gates_teleop_mapping() -> None:
    node = TeleopInput.__new__(TeleopInput)
    node._pedal_pressed = False
    node.get_parameter = lambda name: Parameter("pedal")
    assert node._deadman(time.monotonic_ns()) is False

    node._pedal_pressed = True
    assert node._deadman(time.monotonic_ns()) is True


def test_teleop_sequence_epoch_survives_publisher_restart() -> None:
    previous_start = _initial_command_sequence(100_000_000_000)
    previous_last = previous_start + 10_000

    replacement_start = _initial_command_sequence(101_000_000_000)

    assert replacement_start > previous_last
    assert replacement_start <= ((1 << 64) - 1) >> 6


def test_expired_daemon_control_token_is_recoverable() -> None:
    assert ControlBridge._is_expired_control_authorization(
        RuntimeError("RDK rejected command: authorization is missing, expired or consumed")
    )
    assert ControlBridge._is_expired_control_authorization(
        PermissionError("authorization expired")
    )
    assert not ControlBridge._is_expired_control_authorization(
        RuntimeError("RDK rejected command: invalid_arm_mask")
    )


def test_deviceio_is_bypassed_when_no_foreground_recorder_exists(tmp_path) -> None:
    class Emitter:
        socket_path = tmp_path / "deviceio.sock"

        @staticmethod
        def emit(*_args, **_kwargs) -> None:
            raise AssertionError("no recorder socket means emit must not be called")

    bridge = ControlBridge.__new__(ControlBridge)
    bridge._deviceio = Emitter()

    assert ControlBridge._emit_deviceio(
        bridge, "/control/sent_command", SimpleNamespace(sequence=1), 1, critical=True
    )


def test_pedal_release_acknowledges_replay_deadman_for_later_home() -> None:
    arbiter = ControlArbiter()
    arbiter.begin_hardware_session("session")
    arbiter.mark_ft_zeroed(session_id="session", connection_generation=1)
    arbiter.declare_ready(connection_generation=1)
    arbiter.update_gates(
        GateInputs(
            local_permission=True,
            physical_pedal=True,
            arms_online=True,
            hands_online=True,
            limits_ok=True,
            collision_clear=True,
        )
    )
    arbiter.arm(CommandSource.REPLAY)
    arbiter.stop()

    bridge = ControlBridge.__new__(ControlBridge)
    bridge._arbiter = arbiter
    bridge._physical_pedal = True
    bridge._chunk_lock = threading.Lock()
    bridge._chunk_generation = 0
    bridge._update_gates = lambda _now: None
    bridge._publish_control_state = lambda: None
    bridge._send_hold_once = lambda _reason: None

    bridge._on_pedal_state(False)
    arbiter.clear_hold(local_acknowledged=True)
    assert arbiter.snapshot.state is ControlState.READY


def test_daemon_clutch_hold_is_not_classified_as_hardware_fault() -> None:
    assert ControlBridge._routine_daemon_hold_reason(
        RuntimeError(
            "RDK rejected command: hold_latched:physical_pedal_released"
        )
    ) is HoldReason.PEDAL_RELEASED
    assert ControlBridge._routine_daemon_hold_reason(
        RuntimeError(
            "RDK rejected command: hold_latched:source_deadman_released"
        )
    ) is HoldReason.DEADMAN_RELEASED
    assert ControlBridge._routine_daemon_hold_reason(
        RuntimeError("RDK rejected command: hold_latched:hardware_fault")
    ) is None


class Parameter:
    def __init__(self, value):
        self.value = value


def test_practical_mode_bypasses_duplicate_observation_limits_only() -> None:
    bridge = ControlBridge.__new__(ControlBridge)
    bridge.get_parameter = lambda name: Parameter(False)
    wire = {
        "connected": True,
        "fault": "",
        "q": [99.0] * 7,
        "dq": [99.0] * 7,
        "tcp_velocity": [99.0] * 6,
        "external_wrench": [999.0] * 6,
        "temperature": [999.0] * 7,
    }

    assert bridge._arm_wire_safe(wire) == (True, "")

    wire["connected"] = False
    assert bridge._arm_wire_safe(wire)[0] is False
    wire["connected"] = True
    wire["q"][0] = math.nan
    assert bridge._arm_wire_safe(wire)[0] is False


def test_practical_mode_does_not_promote_reported_fault_to_bridge_fault() -> None:
    class Arbiter:
        def update_gates(self, gates, *, now_monotonic_ns):
            self.gates = gates

    bridge = ControlBridge.__new__(ControlBridge)
    now = time.monotonic_ns()
    bridge.get_parameter = lambda _name: Parameter(False)
    bridge._last_arm_observation_ns = {"left": now, "right": now}
    bridge._latest_wire = {
        "left": {"connected": True, "fault": "Minor fault occurred"},
        "right": {"connected": True, "fault": ""},
    }
    bridge._last_hand_observation_ns = {"left": now, "right": now}
    bridge._hand_connected = {"left": True, "right": True}
    bridge._local_permission = True
    bridge._physical_pedal = True
    bridge._limits_ok = False
    bridge._collision_clear = False
    bridge._state_lock = threading.RLock()
    bridge._arbiter = Arbiter()
    bridge._try_arm_pending_authorization = lambda _now: None

    bridge._update_gates(now)

    assert bridge._arbiter.gates.arms_online is True
    assert bridge._arbiter.gates.hardware_fault is False
    assert bridge._arbiter.gates.limits_ok is True
    assert bridge._arbiter.gates.collision_clear is True


def test_home_keepalive_allows_blocking_hardware_mode_transition() -> None:
    """A lift-to-joint mode switch takes seconds on the real Flexiv arms."""

    class MaintenanceClient:
        def __init__(self) -> None:
            self.timeouts: list[float] = []
            self.keepalive_ttls_ns: list[int] = []

        def request(self, kind, payload, *, timeout_s):
            assert kind == "home_command"
            self.timeouts.append(timeout_s)
            self.keepalive_ttls_ns.append(
                int(payload["expires_monotonic_ns"]) - time.monotonic_ns()
            )
            if len(self.timeouts) == 1:
                return "home_result", {
                    "accepted": True,
                    "completed": False,
                    "reason": "home_lift_in_progress",
                    "max_position_error_rad": 0.1,
                }
            return "home_result", {
                "accepted": True,
                "completed": True,
                "reason": "home_complete",
                "max_position_error_rad": 0.001,
            }

    values = {
        "home_timeout_s": 20.0,
        "home_lift_enabled": True,
        "home_lift_parallel": False,
        "home_lift_timeout_s": 12.0,
        "home_left_joints_rad": [0.0] * 7,
        "home_right_joints_rad": [0.0] * 7,
        "home_max_velocity_rad_s": 0.5,
        "home_max_acceleration_rad_s2": 1.0,
        "home_tolerance_rad": 0.01,
        "home_lift_left_safe_z_m": -0.3,
        "home_lift_right_safe_z_m": -0.3,
        "home_lift_max_linear_velocity_m_s": 0.12,
        "home_lift_max_angular_velocity_rad_s": 0.5,
        "home_lift_max_linear_acceleration_m_s2": 0.5,
        "home_lift_max_angular_acceleration_rad_s2": 1.0,
        "home_lift_tolerance_m": 0.005,
        "cartesian_position_stiffness": [3000.0] * 3 + [200.0] * 3,
        "cartesian_damping_ratio": [0.7] * 6,
    }
    bridge = ControlBridge.__new__(ControlBridge)
    bridge._stop = threading.Event()
    bridge._state_lock = threading.RLock()
    bridge._home_sequence = 0
    bridge._session_id = "session"
    bridge._pending_home_token = "token"
    bridge._pending_home_token_expiry_ns = time.monotonic_ns() + 1_000_000_000
    bridge._home_authorization_lease_active = False
    bridge._home_inflight = True
    bridge._local_permission = True
    bridge._collision_clear = True
    bridge._ipc_maintenance = MaintenanceClient()
    bridge._home_gate_failure = lambda _now: ""
    bridge.get_parameter = lambda name: Parameter(values[name])
    statuses = []
    bridge._publish_home_status = lambda *args, **kwargs: statuses.append(
        (args, kwargs)
    )
    bridge.get_logger = lambda: SimpleNamespace(
        info=lambda _message: None,
        error=lambda _message: None,
    )

    ControlBridge._run_home(bridge, "reset-1")

    assert bridge._ipc_maintenance.timeouts == [20.0, 10.0]
    assert all(
        200_000_000 <= ttl_ns <= 240_000_000
        for ttl_ns in bridge._ipc_maintenance.keepalive_ttls_ns
    )
    assert statuses[-1][0][0] == "complete"


class Command:
    valid_mask = ValidMask.LEFT_ARM | ValidMask.RIGHT_ARM
    representation = ControlRepresentation.CARTESIAN_ROT6D


class EpochClient:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class EpochArbiter:
    def __init__(self) -> None:
        self.reconnect_count = 0

    def on_rdk_reconnect(self) -> None:
        self.reconnect_count += 1


class EpochMapper:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


def test_daemon_instance_change_closes_every_other_persistent_ipc_client() -> None:
    import threading

    bridge = ControlBridge.__new__(ControlBridge)
    bridge._state_lock = threading.RLock()
    bridge._daemon_instance_id = "daemon-old"
    bridge._arbiter = EpochArbiter()
    bridge._safe_pose_rdk = {"left": np.ones(7), "right": np.ones(7)}
    bridge._pending_arm_token = "arm-token"
    bridge._pending_arm_token_expiry_ns = 1
    bridge._rdk_control_lease_active = True
    bridge._pending_home_token = "home-token"
    bridge._pending_home_token_expiry_ns = 1
    bridge._home_authorization_lease_active = True
    bridge._clock_mappers = {"left": EpochMapper(), "right": EpochMapper()}
    bridge._ipc_command = EpochClient()
    bridge._ipc_maintenance = EpochClient()
    bridge._ipc_hand = EpochClient()

    assert bridge._accept_daemon_instance("daemon-new") is True
    assert bridge._daemon_instance_id == "daemon-new"
    assert bridge._arbiter.reconnect_count == 1
    assert bridge._safe_pose_rdk == {}
    assert bridge._pending_arm_token is None
    assert bridge._pending_home_token is None
    assert bridge._rdk_control_lease_active is False
    assert bridge._home_authorization_lease_active is False
    assert all(mapper.reset_count == 1 for mapper in bridge._clock_mappers.values())
    assert bridge._ipc_command.close_count == 1
    assert bridge._ipc_maintenance.close_count == 1
    assert bridge._ipc_hand.close_count == 1

    # Repeated observations from the same daemon must keep live clients open.
    assert bridge._accept_daemon_instance("daemon-new") is False
    assert bridge._ipc_command.close_count == 1


def test_fresh_home_token_forces_next_request_to_refresh_daemon_lease() -> None:
    import threading

    bridge = ControlBridge.__new__(ControlBridge)
    bridge._state_lock = threading.RLock()
    bridge._session_id = "session"
    bridge._home_inflight = False
    bridge._home_authorization_lease_active = True
    bridge._pending_home_token = None
    bridge._pending_home_token_expiry_ns = 0

    ControlBridge._on_home_authorization(
        bridge,
        SimpleNamespace(
            data=json.dumps(
                {
                    "session_id": "session",
                    "one_time_token": "fresh-token",
                    "expires_monotonic_ns": time.monotonic_ns() + 1_000_000_000,
                }
            )
        ),
    )

    assert bridge._pending_home_token == "fresh-token"
    assert bridge._home_authorization_lease_active is False


def test_home_authorization_can_clear_released_invalid_command_hold() -> None:
    import threading

    class HoldArbiter:
        def __init__(self) -> None:
            self.state = ControlState.HOLD_LATCHED
            self.cleared = []

        @property
        def snapshot(self):
            return SimpleNamespace(state=self.state)

        def clear_hold(self, *, local_acknowledged: bool) -> None:
            self.cleared.append(local_acknowledged)
            self.state = ControlState.READY

    bridge = ControlBridge.__new__(ControlBridge)
    bridge._state_lock = threading.RLock()
    bridge._session_id = "session"
    bridge._home_inflight = False
    bridge._home_authorization_lease_active = False
    bridge._pending_home_token = None
    bridge._pending_home_token_expiry_ns = 0
    bridge._arbiter = HoldArbiter()
    bridge._publish_home_status = lambda *args, **kwargs: None

    ControlBridge._on_home_authorization(
        bridge,
        SimpleNamespace(data=json.dumps({
            "session_id": "session",
            "one_time_token": "fresh-token",
            "expires_monotonic_ns": time.monotonic_ns() + 1_000_000_000,
            "clear_hold_latched": True,
        })),
    )

    assert bridge._arbiter.cleared == [True]
    assert bridge._pending_home_token == "fresh-token"


def test_same_source_arm_authorization_refresh_keeps_existing_lease() -> None:
    import threading

    bridge = ControlBridge.__new__(ControlBridge)
    bridge._state_lock = threading.RLock()
    bridge._session_id = "session"
    bridge._arbiter = ControlArbiter()
    bridge._arbiter.begin_hardware_session("session")
    bridge._arbiter.mark_ft_zeroed(session_id="session", connection_generation=1)
    bridge._arbiter.declare_ready(connection_generation=1)
    bridge._arbiter.update_gates(
        GateInputs(
            local_permission=True,
            physical_pedal=True,
            arms_online=True,
            hands_online=True,
            limits_ok=True,
            collision_clear=True,
        )
    )
    from isaac_teleop_core.command import CommandSource

    bridge._arbiter.arm(CommandSource.TELEOP)
    bridge._pending_arm_source = CommandSource.TELEOP
    bridge._pending_arm_token = "old-token"
    bridge._pending_arm_token_expiry_ns = time.monotonic_ns() + 1
    bridge._rdk_control_lease_active = True
    bridge._hold_sent_for_latch = True
    bridge._publish_control_state = lambda: None
    bridge.get_logger = lambda: SimpleNamespace(error=lambda _message: None)

    ControlBridge._on_arm_authorization(
        bridge,
        SimpleNamespace(
            data=json.dumps(
                {
                    "session_id": "session",
                    "source": "teleop",
                    "one_time_token": "fresh-token",
                    "expires_monotonic_ns": time.monotonic_ns() + 1_000_000_000,
                }
            )
        ),
    )

    assert bridge._arbiter.snapshot.state is ControlState.TELEOP_ARMED
    assert bridge._pending_arm_token == "fresh-token"
    assert bridge._rdk_control_lease_active is True


def test_pending_arm_waits_for_fresh_arm_and_hand_observations() -> None:
    bridge = ControlBridge.__new__(ControlBridge)
    bridge._state_lock = threading.RLock()
    bridge._arbiter = ControlArbiter()
    bridge._arbiter.begin_hardware_session("session")
    bridge._arbiter.mark_ft_zeroed(
        session_id="session", connection_generation=1
    )
    bridge._arbiter.declare_ready(connection_generation=1)
    bridge._pending_arm_source = CommandSource.TELEOP
    bridge._pending_arm_token = "fresh-token"
    bridge._pending_arm_token_expiry_ns = time.monotonic_ns() + 1_000_000_000
    bridge._last_gate_inputs = GateInputs(
        local_permission=True,
        physical_pedal=True,
        arms_online=False,
        hands_online=False,
        limits_ok=True,
        collision_clear=True,
    )

    assert bridge._try_arm_pending_authorization(time.monotonic_ns()) is False
    assert bridge._arbiter.snapshot.state is ControlState.READY

    bridge._last_gate_inputs = GateInputs(
        local_permission=True,
        physical_pedal=True,
        arms_online=True,
        hands_online=True,
        limits_ok=True,
        collision_clear=True,
    )
    bridge._arbiter.update_gates(bridge._last_gate_inputs)
    assert bridge._try_arm_pending_authorization(time.monotonic_ns()) is True
    assert bridge._arbiter.snapshot.state is ControlState.TELEOP_ARMED


def test_hardware_hold_waits_for_single_synchronous_stop() -> None:
    class IPC:
        def __init__(self) -> None:
            self.timeout = None

        def request(self, kind, payload, *, timeout_s=None):
            assert kind == "hold"
            assert payload == {"reason": "physical_pedal_released", "latch": True}
            self.timeout = timeout_s
            return "command_ack", {"accepted": True, "reason": "held"}

    bridge = ControlBridge.__new__(ControlBridge)
    bridge._state_lock = threading.RLock()
    bridge._hardware_command_lock = threading.Lock()
    bridge._ipc_command = IPC()
    bridge._hold_sent_for_latch = False
    bridge._hold_request_inflight = False
    bridge.get_logger = lambda: SimpleNamespace(error=lambda *args, **kwargs: None)

    bridge._send_hold_once("physical_pedal_released")

    assert bridge._ipc_command.timeout == 5.0
    assert bridge._hold_sent_for_latch is True


def test_minor_fault_hold_does_not_flood_stop_retries() -> None:
    class IPC:
        def __init__(self) -> None:
            self.calls = 0

        def request(self, kind, payload, *, timeout_s=None):
            self.calls += 1
            raise RuntimeError(
                "right:stop: Robot is not operational: Minor fault occurred"
            )

    errors = []
    bridge = ControlBridge.__new__(ControlBridge)
    bridge._state_lock = threading.RLock()
    bridge._hardware_command_lock = threading.Lock()
    bridge._ipc_command = IPC()
    bridge._hold_sent_for_latch = False
    bridge._hold_request_inflight = False
    bridge.get_logger = lambda: SimpleNamespace(
        error=lambda *args, **kwargs: errors.append(args[0])
    )

    bridge._send_hold_once("hardware_fault")
    bridge._send_hold_once("hardware_fault")

    assert bridge._ipc_command.calls == 1
    assert bridge._hold_sent_for_latch is True
    assert len(errors) == 1
    assert "robot reset" in errors[0]


def test_candidate_targets_do_not_advance_safe_pose_before_ack() -> None:
    bridge = ControlBridge.__new__(ControlBridge)
    import threading

    bridge._state_lock = threading.RLock()
    initial = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
    bridge._safe_pose_rdk = {"left": initial.copy(), "right": initial.copy()}
    bridge._previous_output_quaternion = {"left": None, "right": None}
    bridge._world_from_base = {
        "left": BaseTransform([-0.25, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        "right": BaseTransform([0.25, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
    }
    values = {
        "software_safety_limits_enabled": True,
        "max_translation_step_m": 0.01,
        "max_rotation_step_rad": 0.1,
        "max_linear_velocity_m_s": 0.05,
        "max_angular_velocity_rad_s": 0.15,
        "max_linear_acceleration_m_s2": 0.25,
        "max_angular_acceleration_rad_s2": 0.5,
        "cartesian_control_mode": "impedance",
        "cartesian_position_stiffness": [3000.0, 3000.0, 3000.0, 200.0, 200.0, 200.0],
        "cartesian_impedance_stiffness": [1200.0, 1200.0, 1200.0, 80.0, 80.0, 80.0],
        "cartesian_damping_ratio": [0.7] * 6,
    }
    bridge.get_parameter = lambda name: Parameter(values[name])
    targets, candidates, quaternions = bridge._targets_for_point(
        Command(), CommandPoint.identity()
    )
    assert set(targets) == {"left", "right"}
    assert targets["left"]["control_mode"] == "impedance"
    assert targets["right"]["control_mode"] == "impedance"
    assert set(candidates) == {"left", "right"}
    assert set(quaternions) == {"left", "right"}
    np.testing.assert_array_equal(bridge._safe_pose_rdk["left"], initial)
    np.testing.assert_array_equal(bridge._safe_pose_rdk["right"], initial)
    assert bridge._previous_output_quaternion == {"left": None, "right": None}


def test_home_gate_does_not_require_physical_pedal() -> None:
    bridge = ControlBridge.__new__(ControlBridge)
    now = time.monotonic_ns()
    bridge._local_permission = True
    bridge._physical_pedal = False
    bridge._collision_clear = True
    bridge._limits_ok = True
    bridge._arm_safety_ok = {"left": True, "right": True}
    bridge._last_arm_observation_ns = {"left": now, "right": now}
    bridge._latest_wire = {
        side: {"connected": True, "fault": ""}
        for side in ("left", "right")
    }
    bridge._hand_connected = {"left": True, "right": True}
    bridge._last_hand_observation_ns = {"left": now, "right": now}

    assert bridge._home_gate_failure(now) == ""


def test_home_gate_does_not_require_fresh_hand_observations() -> None:
    bridge = ControlBridge.__new__(ControlBridge)
    now = time.monotonic_ns()
    bridge._local_permission = True
    bridge._collision_clear = True
    bridge._limits_ok = True
    bridge._arm_safety_ok = {"left": True, "right": True}
    bridge._last_arm_observation_ns = {"left": now, "right": now}
    bridge._latest_wire = {
        side: {"connected": True, "fault": ""}
        for side in ("left", "right")
    }
    bridge._hand_connected = {"left": False, "right": False}
    bridge._last_hand_observation_ns = {"left": 0, "right": 0}

    assert bridge._home_gate_failure(now) == ""


def test_one_observation_ipc_timeout_keeps_last_good_arm_sample() -> None:
    class StopAfterOneIteration:
        def __init__(self) -> None:
            self.calls = 0

        def is_set(self) -> bool:
            self.calls += 1
            return self.calls > 1

        @staticmethod
        def wait(_duration: float) -> bool:
            return True

    class IPC:
        @staticmethod
        def request(_kind, _payload):
            raise TimeoutError("mock scheduling hiccup")

    class Logger:
        @staticmethod
        def warning(_message, **_kwargs) -> None:
            pass

    bridge = ControlBridge.__new__(ControlBridge)
    bridge._stop = StopAfterOneIteration()
    bridge._ipc_observation = IPC()
    bridge._last_arm_observation_ns = {"left": 11, "right": 12}
    bridge._limits_ok = True
    bridge._arm_safety_ok = {"left": True, "right": True}
    bridge.get_parameter = lambda _name: SimpleNamespace(value=200.0)
    bridge.get_logger = lambda: Logger()

    bridge._observation_loop()

    assert bridge._last_arm_observation_ns == {"left": 11, "right": 12}
    assert bridge._limits_ok is True
    assert bridge._arm_safety_ok == {"left": True, "right": True}


def test_hand_monitor_ipc_timeout_does_not_offline_valid_ros_hand() -> None:
    class IPC:
        @staticmethod
        def request(_kind, _payload):
            raise TimeoutError("mock scheduling hiccup")

    class Logger:
        @staticmethod
        def warning(_message, **_kwargs) -> None:
            pass

    now = time.monotonic_ns()
    bridge = ControlBridge.__new__(ControlBridge)
    bridge._state_lock = threading.RLock()
    bridge._ipc_hand = IPC()
    bridge._hand_connected = {"left": True, "right": True}
    bridge._last_hand_observation_ns = {"left": now, "right": now}
    bridge._hand_angles = {
        "left": np.zeros(6, dtype=np.float64),
        "right": np.zeros(6, dtype=np.float64),
    }
    bridge._update_gates = lambda _now: None
    bridge.get_logger = lambda: Logger()
    message = SimpleNamespace(
        connected=True,
        fault=False,
        angle=[1.0] * 6,
        sequence=7,
        acquisition=SimpleNamespace(
            source_time=SimpleNamespace(sec=1, nanosec=2),
            source_clock_domain="hand_controller",
        ),
    )

    bridge._on_hand_state("left", message)

    assert bool(bridge._hand_connected["left"]) is True
    np.testing.assert_array_equal(bridge._hand_angles["left"], np.ones(6))
