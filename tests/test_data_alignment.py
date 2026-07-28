import math

import numpy as np

from flexiv_inspire_isaac.data_pipeline.alignment import (
    Pose,
    TimedSample,
    causal_nearest,
    interpolate_pose,
    matrix_to_rot6d,
    quaternion_xyzw_to_matrix,
    slerp_xyzw,
)


def sample(value, timestamp, sequence=0, valid=True):
    return TimedSample(value, timestamp, timestamp + 1, sequence, valid)


def test_causal_nearest_never_uses_future_sample():
    samples = [sample("old", 90), sample("future", 110)]
    aligned = causal_nearest(samples, 100, tolerance_ns=20)
    assert aligned.valid and aligned.value == "old" and aligned.age_ns == 10
    assert not causal_nearest(samples, 100, tolerance_ns=5).valid


def test_quaternion_sign_has_identical_rotation_and_rot6d():
    q = np.array([0.2, -0.3, 0.1, 0.9])
    first = matrix_to_rot6d(quaternion_xyzw_to_matrix(q))
    second = matrix_to_rot6d(quaternion_xyzw_to_matrix(-q))
    assert np.allclose(first, second, atol=1e-12)


def test_pose_interpolation_uses_so3_then_derives_rot6d():
    left = sample(Pose((0, 0, 0), (0, 0, 0, 1)), 0, 1)
    right = sample(Pose((2, 0, 0), (0, 0, 1, 0)), 100, 2)
    result = interpolate_pose(left, right, 50, max_bracket_ns=100)
    assert result.valid
    assert result.value.pose.xyz == (1.0, 0.0, 0.0)
    rotation = quaternion_xyzw_to_matrix(result.value.pose.quaternion_xyzw)
    assert np.allclose(rotation @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-8)
    assert np.allclose(
        result.value.rotation6d, matrix_to_rot6d(rotation), atol=1e-12
    )


def test_matrix_to_rot6d_uses_first_two_columns_order():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    assert matrix_to_rot6d(rotation) == (0.0, 1.0, 0.0, -1.0, 0.0, 0.0)
