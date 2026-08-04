"""Pure mappings between native Flexiv/Inspire and policy tensors."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .rotation import (
    matrix_to_rotation6d,
    matrix_to_rotvec,
    rotation6d_to_matrix,
    rotvec_to_matrix,
)
from .registry import ActionMappingRegistry


class MappingError(ValueError):
    pass


def _finite(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise MappingError(f"{name} must contain {size} finite values")
    return result


def _normalized_hands(value: Sequence[float]) -> np.ndarray:
    result = _finite(value, 12, "hand angle")
    if np.any(result < 0.0) or np.any(result > 1000.0):
        raise MappingError("native hand angle must be in [0,1000]")
    return result / 1000.0


def _native_hands(value: Sequence[float]) -> np.ndarray:
    result = _finite(value, 12, "normalized hand target")
    if np.any(result < 0.0) or np.any(result > 1.0):
        raise MappingError("normalized hand target must be in [0,1]")
    return result * 1000.0


def native_action30_to_policy24(value: Sequence[float]) -> np.ndarray:
    """Convert canonical world delta Rotation-6D action to minimal rotvec action."""

    native = _finite(value, 30, "native action")
    left = np.concatenate(
        (native[0:3], matrix_to_rotvec(rotation6d_to_matrix(native[3:9])))
    )
    right = np.concatenate(
        (native[9:12], matrix_to_rotvec(rotation6d_to_matrix(native[12:18])))
    )
    return np.concatenate((left, right, _normalized_hands(native[18:30])))


def policy_action24_to_native30(value: Sequence[float]) -> np.ndarray:
    """Convert one minimal policy delta to canonical 30D without repeating it."""

    policy = _finite(value, 24, "policy action")
    left = np.concatenate(
        (policy[0:3], matrix_to_rotation6d(rotvec_to_matrix(policy[3:6])))
    )
    right = np.concatenate(
        (policy[6:9], matrix_to_rotation6d(rotvec_to_matrix(policy[9:12])))
    )
    return np.concatenate((left, right, _native_hands(policy[12:24])))


def flexiv_inspire_action_mappings() -> ActionMappingRegistry:
    """Build the station mapping registry without importing RPC or hardware code."""

    return (
        ActionMappingRegistry(canonical_dimension=30)
        .register(
            "cartesian_delta_rotvec_v1",
            input_dimension=24,
            transform=policy_action24_to_native30,
        )
        .register(
            "flexiv_inspire_native_rot6d_v1",
            input_dimension=30,
            transform=lambda value: value,
        )
    )


def _pose_rotvecs(arm_pose18: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    pose = _finite(arm_pose18, 18, "arm pose")
    left = np.concatenate((pose[0:3], matrix_to_rotvec(rotation6d_to_matrix(pose[3:9]))))
    right = np.concatenate(
        (pose[9:12], matrix_to_rotvec(rotation6d_to_matrix(pose[12:18])))
    )
    return left, right


def legacy_state38(
    arm_q14: Sequence[float],
    arm_pose18: Sequence[float],
    hand_angle12: Sequence[float],
) -> np.ndarray:
    joints = _finite(arm_q14, 14, "arm q")
    left_pose, right_pose = _pose_rotvecs(arm_pose18)
    hands = _normalized_hands(hand_angle12)
    return np.concatenate(
        (joints[:7], left_pose, hands[:6], joints[7:], right_pose, hands[6:])
    )


def joint_minimal_state(
    arm_q14: Sequence[float], hand_angle12: Sequence[float]
) -> np.ndarray:
    return np.concatenate(
        (_finite(arm_q14, 14, "arm q"), _normalized_hands(hand_angle12))
    )


def cartesian_minimal_state(
    arm_pose18: Sequence[float], hand_angle12: Sequence[float]
) -> np.ndarray:
    left_pose, right_pose = _pose_rotvecs(arm_pose18)
    return np.concatenate((left_pose, right_pose, _normalized_hands(hand_angle12)))
