from __future__ import annotations

import math

import numpy as np
import pytest

from isaac_teleop_core.command import (
    ACTION_DIM,
    BimanualCommand,
    CommandPoint,
    CommandSource,
    ControlRepresentation,
    ROTATION_ORDER,
    SCHEMA_VERSION,
    ValidMask,
)
from isaac_teleop_core.rotation6d import (
    IDENTITY_ROT6D,
    RotationError,
    compose_world_delta_rot6d,
    geodesic_distance_rad,
    matrix_to_quaternion_xyzw,
    matrix_to_rotation6d,
    matrix_to_rotvec,
    quaternion_xyzw_to_matrix,
    quaternion_xyzw_to_rotation6d,
    rotation6d_to_matrix,
    rotvec_to_matrix,
)


def random_rotvec(rng: np.random.Generator) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return axis * rng.uniform(0.0, math.pi)


def test_random_matrix_rot6d_roundtrip() -> None:
    rng = np.random.default_rng(4)
    for _ in range(1000):
        matrix = rotvec_to_matrix(random_rotvec(rng))
        decoded = rotation6d_to_matrix(matrix_to_rotation6d(matrix))
        assert geodesic_distance_rad(matrix, decoded) < 2.0e-8


def test_quaternion_sign_maps_to_identical_rotation6d() -> None:
    quaternion = np.array([0.4, -0.1, 0.2, 0.88])
    quaternion /= np.linalg.norm(quaternion)
    assert np.allclose(
        quaternion_xyzw_to_rotation6d(quaternion),
        quaternion_xyzw_to_rotation6d(-quaternion),
        atol=1e-14,
    )


def test_quaternion_output_uses_previous_hemisphere() -> None:
    matrix = quaternion_xyzw_to_matrix([0.0, 0.0, 1.0, 0.0])
    positive = matrix_to_quaternion_xyzw(matrix, previous=[0.0, 0.0, 1.0, 0.0])
    negative = matrix_to_quaternion_xyzw(matrix, previous=[0.0, 0.0, -1.0, 0.0])
    assert np.dot(positive, [0.0, 0.0, 1.0, 0.0]) > 0.0
    assert np.dot(negative, [0.0, 0.0, -1.0, 0.0]) > 0.0
    assert np.allclose(positive, -negative)


def test_rotation6d_is_continuous_across_pi() -> None:
    epsilon = 1e-7
    before = matrix_to_rotation6d(rotvec_to_matrix([0.0, 0.0, math.pi - epsilon]))
    after = matrix_to_rotation6d(rotvec_to_matrix([0.0, 0.0, math.pi + epsilon]))
    assert np.linalg.norm(after - before) < 5.0e-7


def test_identity_delta_has_no_drift_and_left_composes() -> None:
    previous = rotvec_to_matrix([0.2, -0.1, 0.4])
    assert np.allclose(
        compose_world_delta_rot6d(IDENTITY_ROT6D, previous),
        previous,
        atol=1e-12,
    )
    delta = rotvec_to_matrix([0.0, 0.1, 0.0])
    composed = compose_world_delta_rot6d(matrix_to_rotation6d(delta), previous)
    assert np.allclose(composed, delta @ previous, atol=1e-12)


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(6),
        [1, 0, 0, 2, 0, 0],
        [1, 0, 0, 1, 1e-12, 0],
        [np.nan, 0, 0, 0, 1, 0],
        [np.inf, 0, 0, 0, 1, 0],
    ],
)
def test_invalid_rotation6d_is_rejected(value: object) -> None:
    with pytest.raises(RotationError):
        rotation6d_to_matrix(value)


def test_reflection_and_nonorthogonal_matrix_are_rejected() -> None:
    with pytest.raises(RotationError):
        matrix_to_rotation6d(np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(RotationError):
        matrix_to_rotvec(np.ones((3, 3)))


def test_default_policy_layout_is_exactly_30_and_uses_identity_rot6d() -> None:
    point = CommandPoint.identity()
    vector = point.to_policy_vector()
    assert vector.shape == (ACTION_DIM,) == (30,)
    assert np.array_equal(vector[3:9], IDENTITY_ROT6D)
    assert np.array_equal(vector[12:18], IDENTITY_ROT6D)
    reconstructed = CommandPoint.from_policy_vector(vector)
    assert np.array_equal(reconstructed.to_policy_vector(), vector)
    with pytest.raises(ValueError):
        CommandPoint.from_policy_vector(np.zeros(30))
    with pytest.raises(ValueError):
        CommandPoint.from_policy_vector(np.zeros(29))


def test_public_model_retains_quaternion_and_joint_modes() -> None:
    quat_point = CommandPoint.from_quaternion_cartesian(
        left_delta_xyz=np.zeros(3),
        left_delta_quaternion_xyzw=[0, 0, 0, 1],
        right_delta_xyz=np.zeros(3),
        right_delta_quaternion_xyzw=[0, 0, 0, 1],
        left_hand_targets=np.zeros(6),
        right_hand_targets=np.zeros(6),
    )
    common = dict(
        schema_version=SCHEMA_VERSION,
        session_id="s",
        source=CommandSource.TELEOP,
        sequence=1,
        issued_monotonic_ns=1,
        ttl_ns=1_000_000,
        frame_id="world",
        valid_mask=ValidMask.LEFT_ARM,
        deadman=True,
    )
    quaternion_command = BimanualCommand(
        **common,
        representation=ControlRepresentation.CARTESIAN_QUATERNION,
        rotation_order="QUATERNION_XYZW",
        points=(quat_point,),
    )
    assert quaternion_command.representation is ControlRepresentation.CARTESIAN_QUATERNION

    joint_point = CommandPoint.from_joint_positions(
        left_arm_joint_positions=np.zeros(7),
        right_arm_joint_positions=np.zeros(7),
        left_hand_targets=np.zeros(6),
        right_hand_targets=np.zeros(6),
    )
    joint_command = BimanualCommand(
        **{**common, "sequence": 2},
        representation=ControlRepresentation.JOINT_POSITION,
        rotation_order="NONE",
        points=(joint_point,),
    )
    assert joint_command.points[0].left_arm_joint_positions.shape == (7,)


def test_rotvec_roundtrip_near_pi() -> None:
    matrix = rotvec_to_matrix([math.pi - 1e-8, 0.0, 0.0])
    decoded = matrix_to_rotvec(matrix)
    assert geodesic_distance_rad(matrix, rotvec_to_matrix(decoded)) < 2e-8
