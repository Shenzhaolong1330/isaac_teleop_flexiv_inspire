"""Read-only three-camera acquisition smoke test."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import threading
import time

from .capture import TripleRealSenseCapture
from .config import load_camera_configs


def run_smoke(
    config_path: str, *, frames_per_camera: int = 3, timeout_s: float = 15.0
) -> dict:
    if frames_per_camera <= 0:
        raise ValueError("frames_per_camera must be positive")
    configs = load_camera_configs(config_path)
    counts: Counter[str] = Counter()
    last_status = {}
    lock = threading.Lock()

    def on_frame(frame) -> None:
        with lock:
            counts[frame.camera_name] += 1

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
        document = {
            "mode": "read-only-rgb-no-firmware-change",
            "frames_per_camera_required": frames_per_camera,
            "counts": dict(counts),
            "last_status": dict(last_status),
            "elapsed_s": (ended_ns - started_ns) / 1e9,
        }
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="src/flexiv_inspire_isaac/cameras/realsense_rgb.yaml",
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
