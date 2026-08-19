from __future__ import annotations

import json

import numpy as np
import zarr

from flexiv_inspire_isaac.data_pipeline.rl100_zarr import (
    RL100SourceEpisode,
    RL100ZarrSpec,
    aligned_row_to_rl100_frame,
    export_rl100_zarr,
)
from flexiv_inspire_isaac.data_pipeline.rl100_reward_cli import (
    apply_episode_reward_labels,
)


def _row(timestamp_ns: int, *, segment: int = 0) -> dict:
    action = np.zeros(30, dtype=np.float32)
    action[0:3] = (0.01, 0.02, 0.03)
    action[3:9] = (1, 0, 0, 0, 1, 0)
    action[9:12] = (-0.01, -0.02, -0.03)
    action[12:18] = (1, 0, 0, 0, 1, 0)
    action[18:30] = 500
    row = {
        "timestamp_ns": timestamp_ns,
        "observation.capture_segment": segment,
        "action": action,
        "action.valid": True,
        "observation.left_arm.state": {"q": np.arange(1, 8)},
        "observation.left_arm.state.valid": True,
        "observation.right_arm.state": {"q": np.arange(8, 15)},
        "observation.right_arm.state.valid": True,
        "observation.left_hand.state": {"angle": np.full(6, 250)},
        "observation.left_hand.state.valid": True,
        "observation.right_hand.state": {"angle": np.full(6, 750)},
        "observation.right_hand.state.valid": True,
    }
    for camera in ("head", "left_wrist", "right_wrist"):
        key = f"observation.images.{camera}"
        row[key] = np.zeros((12, 16, 3), dtype=np.uint8)
        row[f"{key}.valid"] = True
    return row


def test_rl100_frame_has_exact_mvp_dimensions_and_normalized_hands():
    frame = aligned_row_to_rl100_frame(_row(0))

    assert frame["state"].shape == (26,)
    assert frame["action"].shape == (24,)
    assert frame["rgb_head"].shape == (3, 12, 16)
    assert frame["rgb_head"].dtype == np.uint8
    assert np.allclose(frame["state"][14:20], 0.25)
    assert np.allclose(frame["state"][20:26], 0.75)
    assert np.allclose(frame["action"][12:24], 0.5)


def test_export_writes_rl100_schema_and_splits_command_gaps(tmp_path):
    period = 33_333_333
    first = [_row(index * period, segment=0) for index in range(9)]
    second_start = first[-1]["timestamp_ns"] + 200_000_000
    second = [
        _row(second_start + index * period, segment=1) for index in range(9)
    ]
    output = tmp_path / "demo.zarr"

    result = export_rl100_zarr(
        [RL100SourceEpisode([*first, *second], "episode/manifest.json", "task")],
        output_root=output,
        spec=RL100ZarrSpec(),
    )

    root = zarr.open(str(output), mode="r")
    assert root.attrs["schema_id"] == "flexiv_rl100_dp_rgb_v1"
    assert root.attrs["observation_alignment"] == "latest_causal"
    assert root["data/state"].shape == (18, 26)
    assert root["data/action"].shape == (18, 24)
    assert root["data/rgb_head"].shape == (18, 3, 12, 16)
    assert root["meta/episode_ends"][:].tolist() == [9, 18]
    assert result.episodes_written == 2
    report = json.loads((output / "export_validation.json").read_text())
    assert report["reward_available"] is False
    assert report["offline_rl_ready"] is False


def test_invalid_frame_breaks_sequence_and_short_runs_are_dropped(tmp_path):
    rows = [_row(index * 33_333_333) for index in range(8)]
    rows[4]["observation.images.head.valid"] = False

    result = export_rl100_zarr(
        [
            RL100SourceEpisode(
                [*rows, *[_row(1_000_000_000 + index * 33_333_333, segment=1) for index in range(9)]],
                "episode/manifest.json",
            )
        ],
        output_root=tmp_path / "demo.zarr",
        spec=RL100ZarrSpec(),
    )

    assert result.frames_written == 9
    assert result.frames_dropped_invalid_image == 1
    assert result.frames_dropped_short_segment == 7


def test_explicit_complete_success_labels_make_zarr_offline_rl_ready(tmp_path):
    output = tmp_path / "demo.zarr"
    export_rl100_zarr(
        [
            RL100SourceEpisode(
                [_row(index * 33_333_333) for index in range(9)],
                "episode/manifest.json",
            )
        ],
        output_root=output,
        spec=RL100ZarrSpec(),
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episodes": [{"episode_index": 0, "success": True}],
            }
        )
    )

    apply_episode_reward_labels(output, labels)

    root = zarr.open(str(output), mode="r")
    assert root.attrs["reward_available"] is True
    assert root.attrs["offline_rl_ready"] is True
    assert root["data/reward"][-1, 0] == 1.0
    assert root["data/done"][-1, 0] == 1.0
    assert root["data/return"][0, 0] > 0.0
