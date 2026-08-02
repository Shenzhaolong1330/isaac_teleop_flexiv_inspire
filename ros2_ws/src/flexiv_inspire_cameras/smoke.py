"""Read-only three-camera RGB/depth/point-cloud acquisition smoke test."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import threading
import time

from .capture import TripleRealSenseCapture
from .config import load_camera_configs

DEFAULT_CONFIG = Path(__file__).resolve().with_name("realsense_rgb.yaml")


def run_smoke(
    config_path: str, *, frames_per_camera: int = 3, timeout_s: float = 15.0
) -> dict:
    if frames_per_camera <= 0:
        raise ValueError("frames_per_camera must be positive")
    configs = load_camera_configs(config_path)
    counts: Counter[str] = Counter()
    last_status = {}
    modalities = {}
    lock = threading.Lock()

    def on_frame(frame) -> None:
        with lock:
            counts[frame.camera_name] += 1
            modalities[frame.camera_name] = {
                "rgb_bytes": len(frame.jpeg),
                "depth_bytes": len(frame.depth_z16 or b""),
                "pointcloud_bytes": len(frame.pointcloud_xyz_f32 or b""),
                "pointcloud_width": frame.pointcloud_width,
                "pointcloud_height": frame.pointcloud_height,
                "pointcloud_frame_id": frame.pointcloud_frame_id,
            }

    def on_status(status) -> None:
        with lock:
            last_status[status.camera_name] = asdict(status)

    capture = TripleRealSenseCapture(
        configs, on_frame=on_frame, on_status=on_status
    )
    started_ns = time.monotonic_ns()
    capture.start()
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with lock:
                complete = all(
                    counts[name] >= frames_per_camera for name in configs
                )
            if complete:
                break
            time.sleep(0.02)
        else:
            with lock:
                detail = {
                    "counts": dict(counts),
                    "last_status": dict(last_status),
                }
            raise TimeoutError(
                f"did not receive {frames_per_camera} frames from every camera: "
                + json.dumps(detail, ensure_ascii=False)
            )
    finally:
        capture.stop()
    ended_ns = time.monotonic_ns()
    with lock:
        missing_modalities = []
        for name, config in configs.items():
            actual = modalities.get(name, {})
            if int(actual.get("rgb_bytes", 0)) <= 0:
                missing_modalities.append(f"{name}:rgb")
            if config.record_depth and int(actual.get("depth_bytes", 0)) <= 0:
                missing_modalities.append(f"{name}:depth")
            if (
                config.pointcloud_enabled
                and int(actual.get("pointcloud_bytes", 0)) <= 0
            ):
                missing_modalities.append(f"{name}:pointcloud")
        if missing_modalities:
            raise RuntimeError(
                "configured camera modalities produced no data: "
                + ", ".join(missing_modalities)
            )
        document = {
            "mode": "read-only-no-firmware-change",
            "frames_per_camera_required": frames_per_camera,
            "counts": dict(counts),
            "modalities": dict(modalities),
            "last_status": dict(last_status),
            "elapsed_s": (ended_ns - started_ns) / 1e9,
        }
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
    )
    parser.add_argument("--frames-per-camera", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    args = parser.parse_args()
    print(
        json.dumps(
            run_smoke(
                args.config,
                frames_per_camera=args.frames_per_camera,
                timeout_s=args.timeout_s,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
