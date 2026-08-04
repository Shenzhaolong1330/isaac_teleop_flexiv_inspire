"""Strict multi-episode LeRobot writer for versioned policy profiles.

This module is intentionally separate from :mod:`lerobot_v3`: the existing
rich/native exporter remains the default and therefore stays compatible with
all previously recorded datasets and consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from policy_contracts import (
    FeatureContract,
    cartesian_minimal_state,
    get_profile,
    joint_minimal_state,
    legacy_state38,
    native_action30_to_policy24,
)

from .export_spec import ActionView
from .lerobot_v3 import (
    _decode_image,
    _field_vector,
    _pose_vector,
    _validated_action,
)


SOURCE_IMAGE_KEYS = {
    "observation.images.left_wrist_image": "observation.images.left_wrist",
    "observation.images.right_wrist_image": "observation.images.right_wrist",
    "observation.images.head_image": "observation.images.head",
}


@dataclass(frozen=True)
class PolicyEpisode:
    rows: Sequence[Mapping[str, Any]]
    task: str
    source_manifest: str = ""


@dataclass(frozen=True)
class ProfileExportResult:
    output_root: str
    profile: str
    source_episodes: int
    episodes_written: int
    frames_written: int
    frames_dropped_invalid_action: int
    frames_dropped_invalid_image: int
    reload_length: int


def profile_features(
    profile_id: str,
    image_shapes: Mapping[str, tuple[int, int, int]],
) -> dict[str, dict[str, Any]]:
    """Return the exact feature set owned by one policy profile."""

    profile = get_profile(profile_id)
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (profile.state_dimension,),
            "names": list(profile.state_names),
        },
        "action": {
            "dtype": "float32",
            "shape": (profile.action_dimension,),
            "names": list(profile.action_names),
        },
    }
    for target in profile.image_keys:
        features[target] = {
            "dtype": "video",
            "shape": image_shapes[target],
            "names": ["height", "width", "channel"],
        }
    # Compile our own output just like a policy consumer will do. This catches
    # accidental state/action additions before any frames are written.
    FeatureContract.from_info(
        {"fps": 1, "robot_type": "profile-validation", "features": features}
    ).validate_profile(profile)
    return features


def aligned_row_to_policy_frame(
    row: Mapping[str, Any],
    *,
    task: str,
    profile_id: str,
    image_shapes: Mapping[str, tuple[int, int, int]],
) -> dict[str, Any]:
    """Map one native aligned row to one exact policy frame."""

    if not task.strip():
        raise ValueError("LeRobot frames require a non-empty task")
    profile = get_profile(profile_id)
    images: dict[str, np.ndarray] = {}
    for target in profile.image_keys:
        source = SOURCE_IMAGE_KEYS[target]
        if not row.get(f"{source}.valid", False):
            raise ValueError(f"required image {source} is invalid")
        images[target] = _decode_image(row.get(source), image_shapes[target])

    arm_states = [
        row.get(f"observation.{side}_arm.state")
        for side in ("left", "right")
    ]
    arm_q = np.concatenate(
        [_field_vector(state, "q", 7) for state in arm_states]
    )
    arm_pose = np.concatenate(
        [
            _pose_vector(row.get(f"observation.{side}_arm.pose"))
            for side in ("left", "right")
        ]
    )
    hand_angle = np.concatenate(
        [
            _field_vector(
                row.get(f"observation.{side}_hand.state"), "angle", 6
            )
            for side in ("left", "right")
        ]
    )
    if profile_id == "dual_arm_lerobot_v1":
        state = legacy_state38(arm_q, arm_pose, hand_angle)
    elif profile_id == "joint_proprio_cartesian_v1":
        state = joint_minimal_state(arm_q, hand_angle)
    elif profile_id == "cartesian_proprio_v1":
        state = cartesian_minimal_state(arm_pose, hand_angle)
    else:  # get_profile() above should make this unreachable.
        raise ValueError(f"unsupported policy profile: {profile_id}")

    native_action = _validated_action(row.get("action"), ActionView())
    action = native_action30_to_policy24(native_action)
    if state.shape != (profile.state_dimension,):
        raise ValueError("profile state mapper returned the wrong dimension")
    if action.shape != (profile.action_dimension,):
        raise ValueError("profile action mapper returned the wrong dimension")
    return {
        "task": task,
        "observation.state": state.astype(np.float32, copy=False),
        "action": action.astype(np.float32, copy=False),
        **images,
    }


def _infer_image_shapes(
    episodes: Sequence[PolicyEpisode], profile_id: str
) -> dict[str, tuple[int, int, int]]:
    profile = get_profile(profile_id)
    for episode in episodes:
        for row in episode.rows:
            candidate: dict[str, tuple[int, int, int]] = {}
            try:
                for target in profile.image_keys:
                    source = SOURCE_IMAGE_KEYS[target]
                    if not row.get(f"{source}.valid", False):
                        raise ValueError("invalid image")
                    candidate[target] = tuple(
                        int(value) for value in _decode_image(row.get(source)).shape
                    )
            except ValueError:
                continue
            return candidate
    raise ValueError("no fully valid aligned image frames to export")


def export_policy_episodes(
    episodes: Sequence[PolicyEpisode],
    *,
    output_root: str | Path,
    repo_id: str,
    profile_id: str,
    fps: float,
    dataset_class=None,
) -> ProfileExportResult:
    """Write multiple source episodes to one strict LeRobot dataset."""

    if not episodes:
        raise ValueError("at least one source episode is required")
    profile = get_profile(profile_id)
    image_shapes = _infer_image_shapes(episodes, profile_id)
    features = profile_features(profile_id, image_shapes)

    if dataset_class is None:
        try:
            import lerobot
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot is unavailable; run with envs/data-py312 where "
                "lerobot==0.6.0 is pinned"
            ) from exc
        version = str(getattr(lerobot, "__version__", ""))
        if version != "0.6.0":
            raise RuntimeError(
                f"expected lerobot==0.6.0, loaded {version or 'unknown'}"
            )
        dataset_class = LeRobotDataset

    root = Path(output_root)
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite existing dataset root: {root}"
            )
        root.rmdir()
    robot_type = (
        "flexiv_dual_arm"
        if profile.compatibility_only
        else "flexiv_rizon4s_dual_inspire"
    )
    dataset = dataset_class.create(
        repo_id=repo_id,
        fps=int(fps),
        features=features,
        root=root,
        robot_type=robot_type,
        use_videos=True,
    )
    written = 0
    episodes_written = 0
    dropped_action = 0
    dropped_image = 0
    try:
        for episode in episodes:
            episode_written = 0
            for row in episode.rows:
                validity = row.get("valid", {})
                explicitly_invalid = (
                    isinstance(validity, Mapping)
                    and "action" in validity
                    and not bool(validity["action"])
                )
                if row.get("action") is None or explicitly_invalid:
                    dropped_action += 1
                    continue
                try:
                    frame = aligned_row_to_policy_frame(
                        row,
                        task=episode.task,
                        profile_id=profile_id,
                        image_shapes=image_shapes,
                    )
                except ValueError as exc:
                    if "required image" in str(exc):
                        dropped_image += 1
                        continue
                    raise
                dataset.add_frame(frame)
                episode_written += 1
                written += 1
            if episode_written == 0:
                provenance = episode.source_manifest or "<unknown manifest>"
                raise ValueError(
                    f"source episode has no fully valid frames: {provenance}"
                )
            dataset.save_episode()
            episodes_written += 1
        dataset.finalize()
    except Exception:
        dataset.finalize()
        raise

    reloaded = dataset_class(repo_id=repo_id, root=root)
    reload_length = len(reloaded)
    if reload_length != written:
        raise RuntimeError(
            f"LeRobot reload validation failed: wrote {written}, "
            f"loaded {reload_length}"
        )
    sample = reloaded[0]
    expected_keys = {"task", "observation.state", "action", *profile.image_keys}
    # Real LeRobot adds indexing/timestamp columns on reload. Validate profile
    # tensor/image keys, while allowing those framework-owned metadata fields.
    if not expected_keys.issubset(sample):
        raise RuntimeError("reloaded dataset is missing profile features")
    if tuple(np.asarray(sample["observation.state"]).shape) != (
        profile.state_dimension,
    ):
        raise RuntimeError("reloaded observation.state has the wrong shape")
    if tuple(np.asarray(sample["action"]).shape) != (
        profile.action_dimension,
    ):
        raise RuntimeError("reloaded action has the wrong shape")
    return ProfileExportResult(
        output_root=str(root.resolve()),
        profile=profile_id,
        source_episodes=len(episodes),
        episodes_written=episodes_written,
        frames_written=written,
        frames_dropped_invalid_action=dropped_action,
        frames_dropped_invalid_image=dropped_image,
        reload_length=reload_length,
    )
