"""Read-only camera/library compatibility check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PINNED_LIBREALSENSE, load_camera_configs

DEFAULT_CONFIG = Path(__file__).resolve().with_name("realsense_rgb.yaml")


def verify(config_path: str) -> dict:
    configs = load_camera_configs(config_path)
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 for librealsense 2.57.7 is not installed") from exc
    runtime_version = getattr(rs, "__version__", None)
    if runtime_version is not None and not str(runtime_version).startswith(
        PINNED_LIBREALSENSE
    ):
        raise RuntimeError(
            f"loaded pyrealsense2 {runtime_version}, expected {PINNED_LIBREALSENSE}"
        )
    context = rs.context()
    attached = {device.get_info(rs.camera_info.serial_number) for device in context.devices}
    expected = {camera.serial for camera in configs.values()}
    return {
        "pinned_librealsense": PINNED_LIBREALSENSE,
        "runtime_version": runtime_version or "extension-does-not-report-version",
        "expected_serials": sorted(expected),
        "attached_serials": sorted(attached),
        "all_attached": expected <= attached,
        "modalities": {
            name: {
                "rgb": True,
                "depth": camera.record_depth,
                "pointcloud": camera.pointcloud_enabled,
                "resolution": f"{camera.width}x{camera.height}@{camera.fps}",
                "jpeg_quality": camera.jpeg_quality,
            }
            for name, camera in configs.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
    )
    args = parser.parse_args()
    result = verify(args.config)
    print(json.dumps(result, indent=2))
    return 0 if result["all_attached"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
