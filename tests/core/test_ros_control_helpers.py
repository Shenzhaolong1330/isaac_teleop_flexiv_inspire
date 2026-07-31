from __future__ import annotations

import math
import time

import numpy as np
import pytest

pytest.importorskip("flexiv_inspire_interfaces")

from flexiv_inspire_control.node import ControlBridge
from flexiv_inspire_control.frames import BaseTransform
from flexiv_inspire_control.teleop_input_node import _validated_squeezes
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


class Parameter:
    def __init__(self, value):
        self.value = value


class Command:
    valid_mask = ValidMask.LEFT_ARM | ValidMask.RIGHT_ARM
    representation = ControlRepresentation.CARTESIAN_ROT6D


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
