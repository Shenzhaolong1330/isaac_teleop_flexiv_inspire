from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml


PINNED_LIBREALSENSE = "2.57.7"


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
        if (self.width, self.height, self.fps) != (424, 240, 30):
            raise ValueError("first release is fixed to 424x240 RGB at 30 Hz")
        if self.pixel_format.lower() not in {"rgb8", "bgr8"}:
            raise ValueError("camera stream must be RGB")
        if self.depth_enabled:
            raise ValueError("depth is intentionally disabled in the first release")
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
