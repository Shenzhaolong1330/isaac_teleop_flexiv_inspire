"""Configurable, timestamp-aligned LeRobot export view definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
import yaml


class ExportSpecError(ValueError):
    pass


@dataclass(frozen=True)
class ActionView:
    name: str = "sent_command"

    @property
    def shape(self) -> int:
        return {"sent_command": 30, "absolute_joint_position": 14, "absolute_cartesian_pose": 18}[self.name]

    @property
    def names(self) -> list[str]:
        if self.name == "sent_command":
            return [*[
                f"{side}_{axis}" for side in ("left", "right")
                for axis in ("dx", "dy", "dz", "dR00", "dR10", "dR20", "dR01", "dR11", "dR21")
            ], *[
                f"{side}_hand_{joint}" for side in ("left", "right")
                for joint in ("little", "ring", "middle", "index", "thumb_bend", "thumb_rotate")
            ]]
        if self.name == "absolute_joint_position":
            return [f"{side}_j{joint}" for side in ("left", "right") for joint in range(7)]
        return [f"{side}_{axis}" for side in ("left", "right") for axis in ("x", "y", "z", "R00", "R10", "R20", "R01", "R11", "R21")]


@dataclass(frozen=True)
class ExportSpec:
    timeline_source: str = "camera/head/jpeg"
    fps: float = 30.0
    action: ActionView = field(default_factory=ActionView)
    # canonical exporter stream -> recorded stream; lets a future site remap
    # topics without mutating raw MCAP or exporter code.
    channels: Mapping[str, str] = field(default_factory=dict)


def load_export_spec(path: str | Path | None) -> ExportSpec:
    if path is None:
        return ExportSpec()
    raw = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "lerobot_export" in raw:
        raw = raw["lerobot_export"]
    if not isinstance(raw, dict) or int(raw.get("schema_version", 0)) != 1:
        raise ExportSpecError("export config must be schema_version: 1")
    timeline = raw.get("timeline", {})
    action = raw.get("action", {})
    if not isinstance(timeline, dict) or not isinstance(action, dict):
        raise ExportSpecError("timeline and action must be mappings")
    source = str(timeline.get("source", "")).lstrip("/")
    fps = float(timeline.get("fps", 0.0))
    view = str(action.get("view", ""))
    if not source or not 0.0 < fps <= 1000.0:
        raise ExportSpecError("timeline.source and timeline.fps are required")
    if view not in {"sent_command", "absolute_joint_position", "absolute_cartesian_pose"}:
        raise ExportSpecError("action.view is unsupported")
    raw_channels = raw.get("channels", {})
    if not isinstance(raw_channels, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in raw_channels.items()):
        raise ExportSpecError("channels must map canonical stream names to recorded stream names")
    return ExportSpec(source, fps, ActionView(view), {k.lstrip("/"): v.lstrip("/") for k, v in raw_channels.items()})


def remap_streams(streams: Mapping[str, object], spec: ExportSpec) -> dict[str, object]:
    result = dict(streams)
    for canonical, source in spec.channels.items():
        if source not in streams:
            raise ExportSpecError(f"configured source stream is absent: {source}")
        result[canonical] = streams[source]
    return result
