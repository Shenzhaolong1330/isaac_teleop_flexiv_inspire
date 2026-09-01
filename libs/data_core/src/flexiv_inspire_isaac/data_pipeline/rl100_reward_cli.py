"""Attach explicit episode-level sparse rewards to a derived RL-100 Zarr."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

import numcodecs
import numpy as np
import zarr

from .rl100_zarr import RIGHT_SCHEMA_ID, SCHEMA_ID


def apply_episode_reward_labels(
    zarr_path: str | Path,
    labels_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict:
    dataset_path = Path(zarr_path).expanduser().resolve(strict=True)
    labels_file = Path(labels_path).expanduser().resolve(strict=True)
    labels = json.loads(labels_file.read_text(encoding="utf-8"))
    if not isinstance(labels, dict) or int(labels.get("schema_version", 0)) != 1:
        raise ValueError("reward labels must use schema_version: 1")

    root = zarr.open(str(dataset_path), mode="a")
    if str(root.attrs.get("schema_id", "")) not in {SCHEMA_ID, RIGHT_SCHEMA_ID}:
        raise ValueError("expected a supported Flexiv RL-100 Zarr schema")
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    raw_episodes = labels.get("episodes")
    if not isinstance(raw_episodes, list):
        raise ValueError("reward labels episodes must be a list")
    by_index: dict[int, bool] = {}
    for item in raw_episodes:
        if not isinstance(item, dict) or not isinstance(item.get("success"), bool):
            raise ValueError("each reward label requires episode_index and boolean success")
        index = int(item.get("episode_index", -1))
        if index in by_index:
            raise ValueError(f"duplicate reward label for episode {index}")
        by_index[index] = bool(item["success"])
    expected = set(range(len(episode_ends)))
    if set(by_index) != expected:
        raise ValueError(
            "reward labels must cover every exported episode exactly; "
            f"missing={sorted(expected.difference(by_index))}, "
            f"unexpected={sorted(set(by_index).difference(expected))}"
        )

    gamma = float(labels.get("gamma", 0.99))
    success_reward = float(labels.get("terminal_success_reward", 1.0))
    failure_reward = float(labels.get("terminal_failure_reward", 0.0))
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0,1]")
    if not all(np.isfinite(value) for value in (success_reward, failure_reward)):
        raise ValueError("terminal rewards must be finite")

    data = root["data"]
    names = ("reward", "done", "return")
    existing = [name for name in names if name in data]
    if existing and not overwrite:
        raise FileExistsError(
            f"reward arrays already exist: {existing}; pass --overwrite to replace"
        )
    count = int(episode_ends[-1])
    reward = np.zeros((count, 1), dtype=np.float32)
    done = np.zeros((count, 1), dtype=np.float32)
    returns = np.zeros((count, 1), dtype=np.float32)
    start = 0
    for episode_index, end in enumerate(episode_ends):
        end = int(end)
        done[end - 1, 0] = 1.0
        reward[end - 1, 0] = (
            success_reward if by_index[episode_index] else failure_reward
        )
        running = 0.0
        for index in range(end - 1, start - 1, -1):
            running = float(reward[index, 0]) + gamma * running
            returns[index, 0] = running
        start = end

    compressor = numcodecs.Blosc(
        cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE
    )
    suffix = uuid.uuid4().hex
    temporary_names = {}
    try:
        for name, values in (("reward", reward), ("done", done), ("return", returns)):
            temporary = f"_{name}_incomplete_{suffix}"
            temporary_names[name] = temporary
            data.array(
                temporary,
                values,
                chunks=(min(4096, max(1, count)), 1),
                compressor=compressor,
            )
        for name in names:
            if name in data:
                del data[name]
            data.move(temporary_names[name], name)
    except Exception:
        for temporary in temporary_names.values():
            if temporary in data:
                del data[temporary]
        raise

    root.attrs.update(
        {
            "reward_available": True,
            "offline_rl_ready": True,
            "reward_semantics": "episode-terminal sparse success reward",
            "reward_gamma": gamma,
            "reward_labels_file": str(labels_file),
            "successful_episodes": int(sum(by_index.values())),
        }
    )
    validation_path = dataset_path / "export_validation.json"
    validation = (
        json.loads(validation_path.read_text(encoding="utf-8"))
        if validation_path.is_file()
        else {}
    )
    validation.update(
        {
            "reward_available": True,
            "offline_rl_ready": True,
            "offline_rl_blocker": "",
            "reward_semantics": root.attrs["reward_semantics"],
            "reward_gamma": gamma,
            "successful_episodes": int(sum(by_index.values())),
            "reward_labels_file": str(labels_file),
        }
    )
    validation_path.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return validation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Attach complete human success/failure labels to RL-100 Zarr"
    )
    parser.add_argument("--zarr", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = apply_episode_reward_labels(
        args.zarr, args.labels, overwrite=args.overwrite
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
