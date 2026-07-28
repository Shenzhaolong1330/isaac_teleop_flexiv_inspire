from __future__ import annotations

import math

import numpy as np
import pytest

from flexiv_inspire_control.manus_pose import (
    FEATURE_NAMES,
    ManusPoseError,
    TRANSPORT_JOINT_NAMES,
    hand_features,
    split_bimanual_pose_array,
)


def quaternion_x(angle: float) -> np.ndarray:
    return np.array([math.sin(angle / 2), 0, 0, math.cos(angle / 2)])


def valid_hand() -> np.ndarray:
    result = np.zeros((25, 7), dtype=float)
    result[:, 0] = np.linspace(0.1, 0.35, 25)
    result[:, 1] = np.linspace(0.2, 0.3, 25)
    result[:, 6] = 1.0
    return result


def test_transport_order_is_explicit_wrist_through_little_tip() -> None:
    assert len(TRANSPORT_JOINT_NAMES) == 25
    assert TRANSPORT_JOINT_NAMES[0] == "wrist"
    assert TRANSPORT_JOINT_NAMES[1] == "thumb_metacarpal"
    assert TRANSPORT_JOINT_NAMES[-1] == "little_tip"


def test_exact_left25_right25_transport_and_complete_features() -> None:
    both = np.vstack([valid_hand(), valid_hand()])
    features = split_bimanual_pose_array(both)
    assert set(features) == {"left", "right"}
    assert set(features["left"]) == FEATURE_NAMES
    assert set(features["right"]) == FEATURE_NAMES


@pytest.mark.parametrize("count", [0, 25, 49, 51])
def test_wrong_pose_count_is_rejected(count: int) -> None:
    with pytest.raises(ManusPoseError, match="exactly 50"):
        split_bimanual_pose_array(np.zeros((count, 7)))


def test_publisher_zero_pose_invalidates_entire_hand() -> None:
    hand = valid_hand()
    hand[6, :3] = 0.0
    hand[6, 3:] = [0, 0, 0, 1]
    with pytest.raises(ManusPoseError, match="publisher-invalid"):
        hand_features(hand)


def test_relative_bone_rotation_produces_named_flexion() -> None:
    hand = valid_hand()
    proximal = TRANSPORT_JOINT_NAMES.index("index_proximal")
    hand[proximal, 3:] = quaternion_x(0.4)
    features = hand_features(hand)
    assert features["index_mcp_flexion"] == pytest.approx(0.4)
