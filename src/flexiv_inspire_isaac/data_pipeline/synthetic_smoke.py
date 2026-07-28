"""Synthetic, hardware-free LeRobot 0.6.0 create/finalize/reload smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from .lerobot_v3 import VALIDITY_FIELDS, export_rows


def make_row(index: int) -> dict:
    image = np.full((240, 424, 3), index, dtype=np.uint8)
    identity_rot6d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    action = np.zeros(30, dtype=np.float32)
    action[3:9] = identity_rot6d
    action[12:18] = identity_rot6d
    row = {
        "timestamp_ns": 1_000_000_000 + index * 33_333_333,
        "action": action,
    }
    for camera in ("head", "left_wrist", "right_wrist"):
        row[f"observation.images.{camera}"] = image
    for side in ("left", "right"):
        row[f"observation.{side}_arm.pose"] = {
            "xyz": [0.1 * index, 0.0, 0.5],
            "rotation6d": identity_rot6d,
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0 if side == "left" else -1.0],
        }
        row[f"observation.{side}_arm.state"] = {
            "q": [0.01 * index] * 7,
            "dq": [0.0] * 7,
            "tau": [1.0] * 7,
            "tau_des": [1.1] * 7,
            "tau_ext": [0.1] * 7,
            "tau_interact": [0.2] * 7,
            "temperature": [32.0] * 7,
            "tcp_twist": [0.0] * 6,
        }
        row[f"observation.{side}_arm.tcp_twist"] = [0.0] * 6
        row[f"observation.{side}_arm.raw_ft"] = [0.0] * 6
        row[f"observation.{side}_arm.tcp_wrench"] = [0.0] * 6
        row[f"observation.{side}_hand.state"] = {
            "angle": [500.0] * 6,
            "position": [500.0] * 6,
            "actual_force": [10.0] * 6,
            "current": [20.0] * 6,
            "temperature": [30.0] * 6,
            "error": [0] * 6,
            "status": [1] * 6,
        }
        row[f"observation.{side}_hand.tactile"] = np.arange(
            1062, dtype=np.uint16
        )
    for name in VALIDITY_FIELDS:
        key = "action" if name == "action" else f"observation.{name}"
        row[f"{key}.valid"] = True
        row[f"{key}.age_ns"] = 0
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root")
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = Path(tempfile.mkdtemp(prefix="lerobot-smoke-")) / "dataset"
    result = export_rows(
        [make_row(index) for index in range(args.frames)],
        output_root=output_root,
        repo_id="local/flexiv-inspire-synthetic-smoke",
        task="synthetic validation only",
    )
    print(json.dumps(result.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
