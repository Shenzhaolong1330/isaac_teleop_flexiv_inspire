import numpy as np

from flexiv_inspire_isaac.cameras.calibration import (
    _inverse,
    _matrix,
    _pose,
    _rotation_distance_deg,
    solve_samples,
)


def _rotated_pose(xyz, quaternion):
    return _matrix(xyz, quaternion)


def test_eye_in_hand_recovers_tcp_to_camera_transform():
    tcp_camera = _rotated_pose([0.03, -0.02, 0.08], [0.0, 0.0, 0.0, 1.0])
    world_target = _rotated_pose([0.5, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0])
    grippers = [
        _rotated_pose([0.1, 0.0, 0.2], [0.0, 0.0, 0.0, 1.0]),
        _rotated_pose([0.2, 0.1, 0.25], [0.0, 0.0, 0.38268343, 0.92387953]),
        _rotated_pose([0.0, -0.1, 0.3], [0.0, 0.25881905, 0.0, 0.96592583]),
        _rotated_pose([0.15, -0.05, 0.4], [0.17364818, 0.0, 0.0, 0.98480775]),
    ]
    samples = [{"world_T_tcp": _pose(gripper), "camera_T_target": _pose(_inverse(tcp_camera) @ _inverse(gripper) @ world_target)} for gripper in grippers]
    solved = solve_samples("eye_in_hand", samples)
    np.testing.assert_allclose(solved["tcp_T_camera"]["xyz"], [0.03, -0.02, 0.08], atol=1e-6)


def test_eye_to_hand_recovers_world_to_camera_transform():
    world_camera = _rotated_pose([0.4, -0.2, 0.7], [0.0, 0.0, 0.0, 1.0])
    tcp_target = _rotated_pose([0.02, 0.01, 0.12], [0.0, 0.0, 0.0, 1.0])
    grippers = [
        _rotated_pose([0.1, 0.0, 0.2], [0.0, 0.0, 0.0, 1.0]),
        _rotated_pose([0.2, 0.1, 0.25], [0.0, 0.0, 0.38268343, 0.92387953]),
        _rotated_pose([0.0, -0.1, 0.3], [0.0, 0.25881905, 0.0, 0.96592583]),
        _rotated_pose([0.15, -0.05, 0.4], [0.17364818, 0.0, 0.0, 0.98480775]),
    ]
    samples = [{"world_T_tcp": _pose(gripper), "camera_T_target": _pose(_inverse(world_camera) @ gripper @ tcp_target)} for gripper in grippers]
    solved = solve_samples("eye_to_hand", samples)
    np.testing.assert_allclose(solved["world_T_camera"]["xyz"], [0.4, -0.2, 0.7], atol=1e-6)


def test_rotation_diversity_accepts_in_place_camera_calibration_motion():
    identity = _pose(_matrix([0, 0, 0], [0, 0, 0, 1]))
    quarter_turn = _pose(
        _matrix([0, 0, 0], [0, 0, 0.38268343, 0.92387953])
    )

    assert np.isclose(_rotation_distance_deg(identity, quarter_turn), 45.0)
