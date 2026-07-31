from pathlib import Path

import numpy as np

from flexiv_inspire_control.frames import (
    BaseTransform,
    load_base_transforms,
    rdk_pose_base_to_world,
    world_delta_target_to_rdk,
)
from isaac_teleop_core.rotation6d import rdk_pose_to_ros_pose


def test_default_frame_config_places_bases_about_midpoint() -> None:
    path = Path(__file__).parents[1] / "config" / "dual_arm_frames.yaml"
    world, bases = load_base_transforms(path)
    assert world == "world"
    np.testing.assert_allclose(bases["left"].translation_m, [-0.25, 0.0, 0.0])
    np.testing.assert_allclose(bases["right"].translation_m, [0.25, 0.0, 0.0])


def test_base_pose_and_world_delta_are_transformed() -> None:
    transform = BaseTransform([-0.25, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    raw_pose = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    world_pose = rdk_pose_base_to_world(raw_pose, transform)
    np.testing.assert_allclose(world_pose[:3], [-0.15, 0.0, 0.0])
    target, _ = world_delta_target_to_rdk(
        raw_pose, np.array([0.01, 0.0, 0.0]), np.eye(3), transform,
        previous_output_quaternion_xyzw=None,
    )
    position, _ = rdk_pose_to_ros_pose(target)
    np.testing.assert_allclose(position, [0.11, 0.0, 0.0])
