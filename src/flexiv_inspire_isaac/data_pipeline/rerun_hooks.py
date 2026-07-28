"""Non-blocking Rerun hooks for tactile and control trace visualization."""

from __future__ import annotations

import numpy as np

from flexiv_inspire_isaac.dftp.models import TactileFrame


def tactile_atlas(frame: TactileFrame) -> np.ndarray:
    surfaces = {surface.name: surface for surface in frame.surfaces}
    canvas = np.zeros((48, 48), dtype=np.uint16)
    finger_x = {"little": 0, "ring": 10, "middle": 20, "index": 30}
    for finger, x in finger_x.items():
        y = 0
        for suffix in ("end", "tip", "pad"):
            surface = surfaces[f"{finger}_{suffix}"]
            image = np.asarray(surface.values, dtype=np.uint16).reshape(
                surface.rows, surface.cols
            )
            canvas[y : y + surface.rows, x : x + surface.cols] = image
            y += surface.rows + 1
    x, y = 40, 0
    for suffix in ("end", "tip", "middle", "pad"):
        surface = surfaces[f"thumb_{suffix}"]
        image = np.asarray(surface.values, dtype=np.uint16).reshape(
            surface.rows, surface.cols
        )
        canvas[y : y + surface.rows, x : x + surface.cols] = image
        y += surface.rows + 1
    palm = surfaces["palm"]
    palm_image = np.asarray(palm.values, dtype=np.uint16).reshape(
        palm.rows, palm.cols, order="F"
    )
    canvas[34 : 34 + palm.rows, 16 : 16 + palm.cols] = palm_image
    return canvas


class RerunHooks:
    def __init__(self, recording_stream=None) -> None:
        try:
            import rerun as rr
        except ImportError as exc:
            raise RuntimeError("install 'rerun-sdk' in envs/ros-py312") from exc
        self.rr = rr
        self.stream = recording_stream

    def _log(self, path: str, archetype) -> None:
        if self.stream is None:
            self.rr.log(path, archetype)
        else:
            self.stream.log(path, archetype)

    def log_tactile(self, frame: TactileFrame) -> None:
        atlas = tactile_atlas(frame)
        self._log(f"robot/{frame.side}_hand/tactile", self.rr.Image(atlas))

    def log_command_trace(
        self,
        timestamp_ns: int,
        requested: np.ndarray,
        safe: np.ndarray,
        sent: np.ndarray,
    ) -> None:
        self.rr.set_time("host_monotonic", timestamp=timestamp_ns * 1e-9)
        for name, values in (
            ("requested", requested),
            ("safe", safe),
            ("sent", sent),
        ):
            self._log(f"control/{name}", self.rr.Scalars(np.asarray(values)))
