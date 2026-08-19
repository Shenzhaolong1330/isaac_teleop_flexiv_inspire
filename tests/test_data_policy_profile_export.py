from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flexiv_inspire_isaac.data_pipeline.alignment import InterpolatedPose, Pose
from flexiv_inspire_isaac.data_pipeline.profile_export import (
    PolicyEpisode,
    aligned_row_to_policy_frame,
    export_policy_episodes,
    profile_features,
)
from policy_contracts import FeatureContract, get_profile


IMAGE_SHAPES = {
    "observation.images.left_wrist_image": (12, 16, 3),
    "observation.images.right_wrist_image": (12, 16, 3),
    "observation.images.head_image": (12, 16, 3),
}


def _row() -> dict:
    action = np.zeros(30, dtype=np.float32)
    action[0:3] = (0.01, 0.02, 0.03)
    action[3:9] = (1, 0, 0, 0, 1, 0)
    action[9:12] = (-0.01, -0.02, -0.03)
    action[12:18] = (1, 0, 0, 0, 1, 0)
    action[18:30] = np.arange(100, 1300, 100, dtype=np.float32)
    # Keep hand targets inside the native [0,1000] domain.
    action[18:30] = np.linspace(0, 1000, 12, dtype=np.float32)
    row = {
        "action": action,
        "action.valid": True,
        "observation.left_arm.pose": InterpolatedPose(
            Pose((0.1, 0.2, 0.3), (0, 0, 0, 1)),
            (1, 0, 0, 0, 1, 0),
            1,
            2,
        ),
        "observation.right_arm.pose": InterpolatedPose(
            Pose((-0.1, -0.2, -0.3), (0, 0, 0, 1)),
            (1, 0, 0, 0, 1, 0),
            1,
            2,
        ),
        "observation.left_arm.state": {"q": np.arange(1, 8)},
        "observation.right_arm.state": {"q": np.arange(8, 15)},
        "observation.left_hand.state": {"angle": np.full(6, 250)},
        "observation.right_hand.state": {"angle": np.full(6, 750)},
    }
    for camera in ("head", "left_wrist", "right_wrist"):
        key = f"observation.images.{camera}"
        row[key] = np.zeros((12, 16, 3), dtype=np.uint8)
        row[f"{key}.valid"] = True
    for side in ("left", "right"):
        for suffix in ("arm.pose", "arm.state", "hand.state"):
            row[f"observation.{side}_{suffix}.valid"] = True
    return row


class CountingDataset:
    instances = []

    @classmethod
    def create(cls, **kwargs):
        Path(kwargs["root"]).mkdir(parents=True, exist_ok=False)
        instance = cls(**kwargs)
        cls.instances.append(instance)
        return instance

    def __init__(self, **kwargs):
        self.create_kwargs = kwargs
        self.frames = []
        self.episodes_saved = 0
        self.finalized = False
        if self.instances:
            self.frames = list(self.instances[-1].frames)

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.episodes_saved += 1

    def finalize(self):
        self.finalized = True

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]


class FailingReloadDataset(CountingDataset):
    def __init__(self, **kwargs):
        if "fps" not in kwargs:
            raise RuntimeError("reload failed")
        super().__init__(**kwargs)


def test_profile_features_are_exact_and_compile_against_contract():
    for profile_id in (
        "dual_arm_lerobot_v1",
        "joint_proprio_cartesian_v1",
        "cartesian_proprio_v1",
    ):
        profile = get_profile(profile_id)
        features = profile_features(profile_id, IMAGE_SHAPES)
        assert set(features) == {
            "observation.state",
            "action",
            *profile.image_keys,
        }
        contract = FeatureContract.from_info(
            {"fps": 15, "robot_type": "test", "features": features}
        )
        contract.validate_profile(profile)


def test_legacy_profile_matches_old_38d_state_and_24d_action_order():
    frame = aligned_row_to_policy_frame(
        _row(),
        task="pick object",
        profile_id="dual_arm_lerobot_v1",
        image_shapes=IMAGE_SHAPES,
    )

    state = frame["observation.state"]
    action = frame["action"]
    assert state.shape == (38,)
    assert state[0:7].tolist() == list(range(1, 8))
    assert np.allclose(state[7:10], (0.1, 0.2, 0.3))
    assert np.allclose(state[13:19], 0.25)
    assert state[19:26].tolist() == list(range(8, 15))
    assert np.allclose(state[32:38], 0.75)
    assert action.shape == (24,)
    assert np.allclose(action[0:3], (0.01, 0.02, 0.03))
    assert np.allclose(action[6:9], (-0.01, -0.02, -0.03))
    assert np.allclose(action[12:], np.linspace(0, 1, 12))


def test_joint_minimal_profile_does_not_require_unused_tcp_pose():
    row = _row()
    for side in ("left", "right"):
        row.pop(f"observation.{side}_arm.pose")
        row.pop(f"observation.{side}_arm.pose.valid")

    frame = aligned_row_to_policy_frame(
        row,
        task="pick object",
        profile_id="joint_proprio_cartesian_v1",
        image_shapes=IMAGE_SHAPES,
    )

    assert frame["observation.state"].shape == (26,)


def test_minimal_joint_profile_has_no_duplicate_tcp_state():
    frame = aligned_row_to_policy_frame(
        _row(),
        task="pick object",
        profile_id="joint_proprio_cartesian_v1",
        image_shapes=IMAGE_SHAPES,
    )

    state = frame["observation.state"]
    assert state.shape == (26,)
    assert state[:14].tolist() == list(range(1, 15))
    assert np.allclose(state[14:20], 0.25)
    assert np.allclose(state[20:26], 0.75)


def test_multiple_sources_become_multiple_episodes_in_one_dataset(tmp_path):
    CountingDataset.instances.clear()
    result = export_policy_episodes(
        [
            PolicyEpisode([_row()], "task one", "one/manifest.json"),
            PolicyEpisode([_row(), _row()], "task two", "two/manifest.json"),
        ],
        output_root=tmp_path / "merged",
        repo_id="local/test",
        profile_id="joint_proprio_cartesian_v1",
        fps=15,
        dataset_class=CountingDataset,
    )

    created = CountingDataset.instances[0]
    assert created.episodes_saved == 2
    assert created.finalized
    assert len(created.frames) == 3
    assert result.source_episodes == result.episodes_written == 2
    assert result.frames_written == result.reload_length == 3
    assert result.frames_dropped_invalid_observation == 0
    assert (tmp_path / "merged").is_dir()
    assert not list(tmp_path.glob(".merged.incomplete-*"))


def test_invalid_required_hand_frame_is_dropped_without_fabricating_state(tmp_path):
    CountingDataset.instances.clear()
    invalid = _row()
    invalid["observation.left_hand.state"] = None
    invalid["observation.left_hand.state.valid"] = False
    invalid["observation.left_hand.state.invalid_reason"] = "no-causal-sample"

    result = export_policy_episodes(
        [PolicyEpisode([invalid, _row()], "task", "manifest.json")],
        output_root=tmp_path / "merged",
        repo_id="local/test-invalid-observation",
        profile_id="joint_proprio_cartesian_v1",
        fps=15,
        dataset_class=CountingDataset,
    )

    assert result.frames_written == 1
    assert result.frames_dropped_invalid_observation == 1


def test_reload_failure_removes_staging_dataset_and_never_publishes(tmp_path):
    FailingReloadDataset.instances.clear()

    with pytest.raises(RuntimeError, match="reload failed"):
        export_policy_episodes(
            [PolicyEpisode([_row()], "task", "manifest.json")],
            output_root=tmp_path / "merged",
            repo_id="local/test-reload-failure",
            profile_id="joint_proprio_cartesian_v1",
            fps=15,
            dataset_class=FailingReloadDataset,
        )

    assert not (tmp_path / "merged").exists()
    assert not list(tmp_path.glob(".merged.incomplete-*"))
