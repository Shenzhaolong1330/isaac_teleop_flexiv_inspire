from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml


PINNED_LIBREALSENSE = "2.57.7"
MIN_WIDTH = 160
MAX_WIDTH = 1920
MIN_HEIGHT = 120
MAX_HEIGHT = 1080
MIN_FPS = 1
MAX_FPS = 90


@dataclass(frozen=True)
class CameraConfig:
    name: str
    serial: str
    width: int
    height: int
    fps: int
    pixel_format: str
    jpeg_quality: int
    depth_enabled: bool

    def __post_init__(self) -> None:
        if not MIN_WIDTH <= self.width <= MAX_WIDTH:
            raise ValueError(f"camera width must be in {MIN_WIDTH}..{MAX_WIDTH}")
        if not MIN_HEIGHT <= self.height <= MAX_HEIGHT:
            raise ValueError(f"camera height must be in {MIN_HEIGHT}..{MAX_HEIGHT}")
        if not MIN_FPS <= self.fps <= MAX_FPS:
            raise ValueError(f"camera fps must be in {MIN_FPS}..{MAX_FPS}")
        if self.pixel_format.lower() not in {"rgb8", "bgr8"}:
            raise ValueError("camera stream must be RGB")
        if self.depth_enabled:
            raise ValueError("depth is intentionally disabled in this system")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("JPEG quality must be in 1..100")


def load_camera_configs(path: str | Path) -> Mapping[str, CameraConfig]:
    document = yaml.safe_load(Path(path).read_text())
    if str(document.get("librealsense_version")) != PINNED_LIBREALSENSE:
        raise ValueError(
            f"librealsense must be pinned to {PINNED_LIBREALSENSE}; mixing 2.57/2.58 "
            "is forbidden"
        )
    cameras = {
        name: CameraConfig(name=name, **config)
        for name, config in document["cameras"].items()
    }
    if set(cameras) != {"head", "left_wrist", "right_wrist"}:
        raise ValueError("exactly head, left_wrist and right_wrist cameras are required")
    serials = [camera.serial for camera in cameras.values()]
    if len(set(serials)) != 3:
        raise ValueError("camera serials must be unique")
    return cameras
