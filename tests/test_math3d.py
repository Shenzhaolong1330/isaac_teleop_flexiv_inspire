from __future__ import annotations

import math

import numpy as np

from flexiv_inspire_isaac.math3d import (
    quat_multiply_xyzw,
    quat_to_rotvec,
    relative_quaternion_xyzw,
    rotvec_to_quat,
)


def test_quaternion_sign_is_equivalent() -> None:
    quaternion = rotvec_to_quat([0.2, -0.1, 0.3])
    np.testing.assert_allclose(
        quat_to_rotvec(quaternion),
        quat_to_rotvec(-quaternion),
        atol=1e-9,
    )


def test_wraparound_179_to_minus_179_is_two_degrees() -> None:
    before = rotvec_to_quat([0.0, 0.0, math.radians(179.0)])
    after = rotvec_to_quat([0.0, 0.0, math.radians(-179.0)])
    delta = quat_to_rotvec(relative_quaternion_xyzw(after, before))
    assert abs(np.linalg.norm(delta) - math.radians(2.0)) < 1e-8


def test_world_left_multiplication_order() -> None:
    base = rotvec_to_quat([math.pi / 2, 0.0, 0.0])
    world_step = rotvec_to_quat([0.0, math.pi / 2, 0.0])
    result = quat_multiply_xyzw(world_step, base)
    expected = quat_multiply_xyzw(world_step, base)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_invalid_quaternion_rejected() -> None:
    with np.testing.assert_raises(ValueError):
        quat_to_rotvec([0.0, 0.0, 0.0, 0.0])
    with np.testing.assert_raises(ValueError):
        quat_to_rotvec([float("nan"), 0.0, 0.0, 1.0])

