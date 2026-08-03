"""Conversion and safety checks between ROS messages, core commands and RDK."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from isaac_teleop_core.command import (
    BimanualCommand,
    CommandPoint,
    CommandSource,
    ControlRepresentation,
    ROTATION_ORDER,
    SCHEMA_VERSION,
    ValidMask,
)
from isaac_teleop_core.rotation6d import (
    compose_world_delta_rot6d,
    geodesic_distance_rad,
    matrix_to_quaternion_xyzw,
    quaternion_xyzw_to_matrix,
    rdk_pose_to_ros_pose,
    ros_pose_to_rdk_pose,
)
from .frames import BaseTransform, world_delta_target_to_rdk


def duration_ns(message: Any) -> int:
    return int(message.sec) * 1_000_000_000 + int(message.nanosec)


def command_from_ros(
    message: Any,
    *,
    expected_source: CommandSource,
    received_monotonic_ns: int | None = None,
) -> BimanualCommand:
    if message.source != expected_source.value:
        raise ValueError("command source field does not match its ROS topic")
    received = time.monotonic_ns() if received_monotonic_ns is None else received_monotonic_ns
    ttl = duration_ns(message.ttl)
    representation = {
        1: ControlRepresentation.CARTESIAN_ROT6D,
        2: ControlRepresentation.CARTESIAN_QUATERNION,
        3: ControlRepresentation.JOINT_POSITION,
    }.get(int(message.representation))
    if representation is None:
        raise ValueError("unknown control representation")
    points = []
    for incoming in message.trajectory:
        points.append(
            CommandPoint(
                execute_after_s=duration_ns(incoming.execute_after) / 1e9,
                left_delta_xyz=incoming.left_delta_xyz,
                left_delta_rotation6d=incoming.left_delta_rotation6d,
                right_delta_xyz=incoming.right_delta_xyz,
                right_delta_rotation6d=incoming.right_delta_rotation6d,
                left_hand_targets=incoming.left_hand_targets,
                right_hand_targets=incoming.right_hand_targets,
                left_delta_quaternion_xyzw=incoming.left_delta_quaternion_xyzw,
                right_delta_quaternion_xyzw=incoming.right_delta_quaternion_xyzw,
                left_arm_joint_positions=incoming.left_arm_joint_positions,
                right_arm_joint_positions=incoming.right_arm_joint_positions,
            )
        )
    return BimanualCommand(
        schema_version=int(message.schema_version),
        session_id=message.session_id,
        source=expected_source,
        sequence=int(message.sequence),
        issued_monotonic_ns=received,
        ttl_ns=ttl,
        representation=representation,
        frame_id=message.frame_id,
        points=tuple(points),
        valid_mask=ValidMask(int(message.valid_mask)),
        deadman=bool(message.deadman),
        rotation_order=message.rotation_order,
        metadata=dict(zip(message.metadata_keys, message.metadata_values, strict=True)),
    )


def cartesian_target_from_point(
    point: CommandPoint,
    *,
    side: str,
    representation: ControlRepresentation,
    previous_safe_pose_rdk: np.ndarray,
    max_translation_step_m: float,
    max_rotation_step_rad: float,
    previous_output_quaternion_xyzw: np.ndarray | None,
    world_from_base: BaseTransform | None = None,
    enforce_step_limits: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    delta_xyz = point.left_delta_xyz if side == "left" else point.right_delta_xyz
    # The Quest mapper clamps a long-but-valid sample to this exact boundary.
    # Floating-point normalization can produce e.g. 0.010000000000000002 for a
    # configured 0.01 m limit.  Treat only a materially larger value as an
    # over-step; otherwise the first clamped frame permanently latches the
    # control bridge in ``invalid_command``.
    translation_tolerance = max(1.0e-12, abs(max_translation_step_m) * 1.0e-12)
    if enforce_step_limits and (
        float(np.linalg.norm(delta_xyz))
        > max_translation_step_m + translation_tolerance
    ):
        raise ValueError(f"{side} translation step exceeds safety limit")
    position, previous_quaternion = rdk_pose_to_ros_pose(previous_safe_pose_rdk)
    previous_rotation = quaternion_xyzw_to_matrix(previous_quaternion)
    if representation is ControlRepresentation.CARTESIAN_ROT6D:
        delta = (
            point.left_delta_rotation6d
            if side == "left"
            else point.right_delta_rotation6d
        )
        new_rotation = compose_world_delta_rot6d(delta, previous_rotation)
    elif representation is ControlRepresentation.CARTESIAN_QUATERNION:
        delta_quaternion = (
            point.left_delta_quaternion_xyzw
            if side == "left"
            else point.right_delta_quaternion_xyzw
        )
        new_rotation = quaternion_xyzw_to_matrix(delta_quaternion) @ previous_rotation
    else:
        raise ValueError("initial hardware controller supports Cartesian modes only")
    rotation_tolerance = max(1.0e-12, abs(max_rotation_step_rad) * 1.0e-12)
    if enforce_step_limits and (
        geodesic_distance_rad(previous_rotation, new_rotation)
        > max_rotation_step_rad + rotation_tolerance
    ):
        raise ValueError(f"{side} rotation step exceeds safety limit")
    output_quaternion = matrix_to_quaternion_xyzw(
        new_rotation,
        previous=(previous_quaternion if previous_output_quaternion_xyzw is None else previous_output_quaternion_xyzw),
    )
    if world_from_base is None:
        target = ros_pose_to_rdk_pose(position + delta_xyz, output_quaternion)
        return target, output_quaternion
    return world_delta_target_to_rdk(
        previous_safe_pose_rdk, delta_xyz, new_rotation @ previous_rotation.T,
        world_from_base, previous_output_quaternion_xyzw=previous_output_quaternion_xyzw,
    )
