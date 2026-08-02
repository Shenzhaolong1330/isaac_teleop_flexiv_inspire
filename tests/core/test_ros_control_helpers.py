from __future__ import annotations

import json
import math
from types import SimpleNamespace
import time

import numpy as np
import pytest

pytest.importorskip("flexiv_inspire_interfaces")

from flexiv_inspire_control.node import ControlBridge
from flexiv_inspire_control.frames import BaseTransform
from flexiv_inspire_control.teleop_input_node import (
    TeleopInput,
    _validated_button,
    _validated_squeezes,
)
from isaac_teleop_core.command import (
    CommandPoint,
    ControlRepresentation,
    ValidMask,
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


class Parameter:
    def __init__(self, value):
        self.value = value


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
