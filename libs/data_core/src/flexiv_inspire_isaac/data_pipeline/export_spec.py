"""Configurable, timestamp-aligned LeRobot export view definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
import yaml


class ExportSpecError(ValueError):
    pass


def _find_export_sections(path: Path, stack: tuple[Path, ...] = ()) -> list[object]:
    path = path.expanduser().resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ExportSpecError(f"export config include cycle: {chain}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ExportSpecError(f"export config must be a mapping: {path}")
    found: list[object] = []
    if "lerobot_export" in raw:
        found.append(raw["lerobot_export"])
    includes = raw.get("includes", [])
    if includes:
        if not isinstance(includes, list) or not all(
            isinstance(item, str) and item.strip() for item in includes
        ):
            raise ExportSpecError("export config includes must be non-empty paths")
        for include in includes:
            found.extend(
                _find_export_sections(path.parent / include, (*stack, path))
            )
    # Also accept a standalone export document rather than a system fragment.
    if not found and not includes and ("timeline" in raw or "action" in raw):
        found.append(raw)
    return found


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
    sections = _find_export_sections(Path(path))
    if not sections:
        raise ExportSpecError("config does not define lerobot_export")
    if len(sections) != 1:
        raise ExportSpecError("config defines lerobot_export more than once")
    raw = sections[0]
    if not isinstance(raw, dict) or int(raw.get("schema_version", 0)) != 1:
        raise ExportSpecError("export config must be schema_version: 1")
    timeline = raw.get("timeline", {})
    action = raw.get("action", {})
    if not isinstance(timeline, dict) or not isinstance(action, dict):
        raise ExportSpecError("timeline and action must be mappings")
    source = str(timeline.get("source", "")).lstrip("/")
    fps = float(timeline.get("fps", 0.0))
    view = str(action.get("view", ""))
    if not source or not 0.0 < fps <= 1000.0 or not fps.is_integer():
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
