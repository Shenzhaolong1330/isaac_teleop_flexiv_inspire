"""Versioned minimal and compatibility policy feature profiles."""

from __future__ import annotations

from dataclasses import dataclass


SIDES = ("left", "right")
HAND_ACTUATORS = (
    "little",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotate",
)
POSE_AXES = ("x", "y", "z", "rx", "ry", "rz")
IMAGE_KEYS = (
    "observation.images.left_wrist_image",
    "observation.images.right_wrist_image",
    "observation.images.head_image",
)


@dataclass(frozen=True)
class PolicyProfile:
    profile_id: str
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    image_keys: tuple[str, ...]
    state_semantics: str
    action_semantics: str
    compatibility_only: bool = False

    @property
    def state_dimension(self) -> int:
        return len(self.state_names)

    @property
    def action_dimension(self) -> int:
        return len(self.action_names)


def _legacy_state_names() -> tuple[str, ...]:
    names: list[str] = []
    for side in SIDES:
        names.extend(f"{side}_joint_{index}.pos" for index in range(1, 8))
        names.extend(f"{side}_ee_pose.{axis}" for axis in POSE_AXES)
        names.extend(f"{side}_hand_state_{index}" for index in range(6))
    return tuple(names)


def _legacy_action_names() -> tuple[str, ...]:
    names: list[str] = []
    for side in SIDES:
        names.extend(f"{side}_delta_ee_pose.{axis}" for axis in POSE_AXES)
    for side in SIDES:
        names.extend(f"{side}_hand_cmd_{index}" for index in range(6))
    return tuple(names)


def _minimal_action_names() -> tuple[str, ...]:
    names: list[str] = []
    for side in SIDES:
        names.extend(f"{side}_delta_ee_pose.{axis}" for axis in POSE_AXES)
    for side in SIDES:
        names.extend(f"{side}_hand_cmd.{name}" for name in HAND_ACTUATORS)
    return tuple(names)


DUAL_ARM_LEROBOT_V1_PROFILE = PolicyProfile(
    profile_id="dual_arm_lerobot_v1",
    state_names=_legacy_state_names(),
    action_names=_legacy_action_names(),
    image_keys=IMAGE_KEYS,
    state_semantics="legacy q+world_tcp_rotvec+normalized_hand, side grouped",
    action_semantics="world relative xyz+rotvec, normalized absolute hand target",
    compatibility_only=True,
)

JOINT_MINIMAL_PROFILE = PolicyProfile(
    profile_id="joint_proprio_cartesian_v1",
    state_names=tuple(
        [
            f"{side}_joint_{index}.pos"
            for side in SIDES
            for index in range(1, 8)
        ]
        + [
            f"{side}_hand_state.{name}"
            for side in SIDES
            for name in HAND_ACTUATORS
        ]
    ),
    action_names=_minimal_action_names(),
    image_keys=IMAGE_KEYS,
    state_semantics="q+normalized_hand; no duplicate TCP pose",
    action_semantics="world relative xyz+rotvec, normalized absolute hand target",
)

CARTESIAN_MINIMAL_PROFILE = PolicyProfile(
    profile_id="cartesian_proprio_v1",
    state_names=tuple(
        [f"{side}_ee_pose.{axis}" for side in SIDES for axis in POSE_AXES]
        + [
            f"{side}_hand_state.{name}"
            for side in SIDES
            for name in HAND_ACTUATORS
        ]
    ),
    action_names=_minimal_action_names(),
    image_keys=IMAGE_KEYS,
    state_semantics="world TCP xyz+rotvec+normalized_hand; no duplicate q",
    action_semantics="world relative xyz+rotvec, normalized absolute hand target",
)

_PROFILES = {
    item.profile_id: item
    for item in (
        DUAL_ARM_LEROBOT_V1_PROFILE,
        JOINT_MINIMAL_PROFILE,
        CARTESIAN_MINIMAL_PROFILE,
    )
}


def get_profile(profile_id: str) -> PolicyProfile:
    try:
        return _PROFILES[str(profile_id)]
    except KeyError as exc:
        raise ValueError(f"unsupported policy profile: {profile_id}") from exc
