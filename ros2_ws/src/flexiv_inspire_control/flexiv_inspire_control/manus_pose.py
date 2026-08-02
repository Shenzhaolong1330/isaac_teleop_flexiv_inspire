"""Strict Isaac Teleop ``/xr_teleop/hand`` PoseArray feature extraction.

Isaac publishes 50 poses: left OpenXR joints 1..25 followed by right 1..25.
OpenXR PALM (index 0) is deliberately absent. Invalid joints are represented by
the publisher as a zero-position, identity-orientation pose.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from isaac_teleop_core.rotation6d import (
    geodesic_distance_rad,
    quaternion_xyzw_to_matrix,
)


TRANSPORT_JOINT_NAMES = (
    "wrist",
    "thumb_metacarpal",
    "thumb_proximal",
    "thumb_distal",
    "thumb_tip",
    "index_metacarpal",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "index_tip",
    "middle_metacarpal",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "middle_tip",
    "ring_metacarpal",
    "ring_proximal",
    "ring_intermediate",
    "ring_distal",
    "ring_tip",
    "little_metacarpal",
    "little_proximal",
    "little_intermediate",
    "little_distal",
    "little_tip",
)
TRANSPORT_JOINT_COUNT_PER_SIDE = 25
TRANSPORT_POSE_COUNT = 50

FEATURE_NAMES = frozenset(
    {
        "little_mcp_flexion",
        "little_pip",
        "little_dip",
        "ring_mcp_flexion",
        "ring_pip",
        "ring_dip",
        "middle_mcp_flexion",
        "middle_pip",
        "middle_dip",
        "index_mcp_flexion",
        "index_pip",
        "index_dip",
        "thumb_cmc_flexion",
        "thumb_mcp",
        "thumb_ip",
        "thumb_cmc_abduction",
    }
)


class ManusPoseError(ValueError):
    pass


def _validated_hand(poses: object) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values = np.asarray(poses, dtype=np.float64)
    if values.shape != (TRANSPORT_JOINT_COUNT_PER_SIDE, 7):
        raise ManusPoseError(
            "each hand must contain exactly 25 poses (OpenXR indices 1..25)"
        )
    if not np.all(np.isfinite(values)):
        raise ManusPoseError("MANUS PoseArray contains NaN or Inf")
    rotations: dict[str, np.ndarray] = {}
    for index, name in enumerate(TRANSPORT_JOINT_NAMES):
        position = values[index, :3]
        quaternion = values[index, 3:]
        if np.allclose(position, 0.0, atol=1e-12) and np.allclose(
            quaternion, [0.0, 0.0, 0.0, 1.0], atol=1e-12
        ):
            raise ManusPoseError(f"MANUS joint {name} is publisher-invalid")
        try:
            rotations[name] = quaternion_xyzw_to_matrix(quaternion)
        except Exception as exc:
            raise ManusPoseError(f"MANUS joint {name} quaternion is invalid") from exc
    return values, rotations


def _relative_angle(
    rotations: dict[str, np.ndarray], parent: str, child: str
) -> float:
    relative = rotations[parent].T @ rotations[child]
    value = geodesic_distance_rad(relative, np.eye(3))
    if not math.isfinite(value):
        raise ManusPoseError(f"non-finite relative angle {parent}->{child}")
    return float(value)


def hand_features(poses: object) -> dict[str, float]:
    """Derive calibration features from one 25-pose OpenXR hand transport.

    Flexion features are magnitudes of parent-to-child relative rotations.
    Thumb abduction is a signed wrist-frame direction angle and therefore must
    be calibrated independently for left and right hands at the actual site.
    """

    values, rotations = _validated_hand(poses)
    features: dict[str, float] = {}
    for finger in ("index", "middle", "ring", "little"):
        features[f"{finger}_mcp_flexion"] = _relative_angle(
            rotations, f"{finger}_metacarpal", f"{finger}_proximal"
        )
        features[f"{finger}_pip"] = _relative_angle(
            rotations, f"{finger}_proximal", f"{finger}_intermediate"
        )
        features[f"{finger}_dip"] = _relative_angle(
            rotations, f"{finger}_intermediate", f"{finger}_distal"
        )
    features["thumb_cmc_flexion"] = _relative_angle(
        rotations, "wrist", "thumb_metacarpal"
    )
    features["thumb_mcp"] = _relative_angle(
        rotations, "thumb_metacarpal", "thumb_proximal"
    )
    features["thumb_ip"] = _relative_angle(
        rotations, "thumb_proximal", "thumb_distal"
    )
    # The wrist-to-thumb-metacarpal vector points at the CMC joint location,
    # which is effectively fixed in the hand skeleton and therefore cannot
    # measure thumb opposition.  Use the metacarpal bone direction instead:
    # its proximal endpoint moves as the thumb abducts/rotates.
    thumb_metacarpal_index = TRANSPORT_JOINT_NAMES.index("thumb_metacarpal")
    thumb_proximal_index = TRANSPORT_JOINT_NAMES.index("thumb_proximal")
    thumb_direction_world = (
        values[thumb_proximal_index, :3]
        - values[thumb_metacarpal_index, :3]
    )
    thumb_direction_local = rotations["wrist"].T @ thumb_direction_world
    planar_norm = float(np.linalg.norm(thumb_direction_local[:2]))
    if planar_norm < 1e-8:
        raise ManusPoseError("thumb metacarpal direction is degenerate")
    features["thumb_cmc_abduction"] = float(
        math.atan2(thumb_direction_local[1], thumb_direction_local[0])
    )
    if set(features) != FEATURE_NAMES or not all(
        math.isfinite(value) for value in features.values()
    ):
        raise ManusPoseError("derived MANUS feature set is incomplete or non-finite")
    return features


def split_bimanual_pose_array(poses: object) -> dict[str, dict[str, float]]:
    values = np.asarray(poses, dtype=np.float64)
    if values.shape != (TRANSPORT_POSE_COUNT, 7):
        raise ManusPoseError(
            "Isaac /xr_teleop/hand must contain exactly 50 poses: left25+right25"
        )
    return {
        "left": hand_features(values[:25]),
        "right": hand_features(values[25:]),
    }


def pose_message_values(poses: Sequence[object]) -> np.ndarray:
    if len(poses) != TRANSPORT_POSE_COUNT:
        raise ManusPoseError(
            "Isaac /xr_teleop/hand must contain exactly 50 poses: left25+right25"
        )
    return np.asarray(
        [
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
            for pose in poses
        ],
        dtype=np.float64,
    )
