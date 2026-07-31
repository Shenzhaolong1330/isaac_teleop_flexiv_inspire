"""Rigid transforms between each Flexiv controller base and the shared world."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from isaac_teleop_core.rotation6d import (
    matrix_to_quaternion_xyzw,
    quaternion_xyzw_to_matrix,
    rdk_pose_to_ros_pose,
    ros_pose_to_rdk_pose,
)


@dataclass(frozen=True)
class BaseTransform:
    """``world_T_base``; translation is metres and quaternion is ROS xyzw."""

    translation_m: np.ndarray
    rotation_xyzw: np.ndarray

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation_m, dtype=np.float64).reshape(-1)
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError("base translation_m must be three finite values")
        rotation = np.asarray(self.rotation_xyzw, dtype=np.float64).reshape(-1)
        if rotation.shape != (4,):
            raise ValueError("base rotation_xyzw must have four values")
        object.__setattr__(self, "translation_m", translation)
        # Decode eagerly: invalid/non-normalized values fail while loading config.
        quaternion_xyzw_to_matrix(rotation)
        object.__setattr__(self, "rotation_xyzw", rotation / np.linalg.norm(rotation))

    @property
    def rotation(self) -> np.ndarray:
        return quaternion_xyzw_to_matrix(self.rotation_xyzw)


def load_base_transforms(path: str | Path) -> tuple[str, dict[str, BaseTransform]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or int(raw.get("schema_version", 0)) != 1:
        raise ValueError("frame config must be schema_version: 1")
    world_frame = str(raw.get("world_frame", "")).strip()
    bases = raw.get("bases")
    if not world_frame or not isinstance(bases, Mapping) or set(bases) != {"left", "right"}:
        raise ValueError("frame config requires world_frame and left/right bases")
    parsed: dict[str, BaseTransform] = {}
    for side in ("left", "right"):
        entry = bases[side]
        if not isinstance(entry, Mapping):
            raise ValueError(f"frame config bases.{side} must be a mapping")
        parsed[side] = BaseTransform(entry.get("translation_m"), entry.get("rotation_xyzw"))
    return world_frame, parsed


def rdk_pose_base_to_world(pose_rdk: np.ndarray, world_from_base: BaseTransform) -> np.ndarray:
    position_base, quaternion_base = rdk_pose_to_ros_pose(pose_rdk)
    rotation_world = world_from_base.rotation @ quaternion_xyzw_to_matrix(quaternion_base)
    position_world = world_from_base.rotation @ position_base + world_from_base.translation_m
    return ros_pose_to_rdk_pose(position_world, matrix_to_quaternion_xyzw(rotation_world))


def world_delta_target_to_rdk(
    pose_rdk: np.ndarray,
    delta_xyz_world: np.ndarray,
    delta_rotation_world: np.ndarray,
    world_from_base: BaseTransform,
    *,
    previous_output_quaternion_xyzw: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a world-frame Cartesian delta, returning the RDK base-frame target."""

    position_base, quaternion_base = rdk_pose_to_ros_pose(pose_rdk)
    base_rotation = quaternion_xyzw_to_matrix(quaternion_base)
    world_rotation = world_from_base.rotation @ base_rotation
    target_world_rotation = delta_rotation_world @ world_rotation
    target_base_rotation = world_from_base.rotation.T @ target_world_rotation
    target_base_position = position_base + world_from_base.rotation.T @ delta_xyz_world
    quaternion = matrix_to_quaternion_xyzw(
        target_base_rotation, previous=previous_output_quaternion_xyzw
    )
    return ros_pose_to_rdk_pose(target_base_position, quaternion), quaternion
