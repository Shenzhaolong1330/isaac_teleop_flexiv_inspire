"""Configurable, timestamp-aligned LeRobot export view definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
import yaml


class ExportSpecError(ValueError):
    pass


CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
CORE_LEROBOT_FIELDS = (
    "observation.images.head",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.arm_pose",
    "observation.arm_quaternion_xyzw",
    "observation.arm_q",
    "observation.arm_dq",
    "observation.arm_tau",
    "observation.arm_tau_des",
    "observation.arm_tau_ext",
    "observation.arm_tau_interact",
    "observation.arm_temperature",
    "observation.tcp_twist",
    "observation.hand_state",
    "observation.hand_position",
    "observation.hand_actual_force",
    "observation.hand_current",
    "observation.hand_temperature",
    "observation.hand_error",
    "observation.hand_status",
    "observation.hand_field_valid",
    "observation.hand_field_age_s",
    "observation.force_torque",
    "observation.tactile",
    "observation.valid",
    "observation.age_s",
    "observation.arm_high_rate",
    "observation.arm_high_rate_valid",
    "observation.arm_high_rate_age_s",
    "action",
)
DEPTH_LEROBOT_FIELDS = tuple(
    field
    for camera in CAMERA_NAMES
    for field in (
        f"observation.depth.{camera}",
        f"observation.depth_scale_m.{camera}",
        f"observation.depth_intrinsics.{camera}",
    )
)
SEGMENT_LEROBOT_FIELDS = (
    "observation.source_timestamp_ns",
    "observation.source_gap_s",
    "observation.capture_segment",
    "observation.frame_in_segment",
    "observation.capture_segment_start",
)
SUPPORTED_LEROBOT_FIELDS = frozenset(
    (*CORE_LEROBOT_FIELDS, *DEPTH_LEROBOT_FIELDS, *SEGMENT_LEROBOT_FIELDS)
)


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
class DepthExport:
    enabled: bool = False
    cameras: tuple[str, ...] = ()
    representation: str = "z16"
    storage: str = "parquet"


@dataclass(frozen=True)
class SegmentExport:
    gap_threshold_s: float = 0.25
    split_episodes: bool = False


@dataclass(frozen=True)
class ExportSpec:
    timeline_source: str = "camera/head/jpeg"
    fps: float = 30.0
    resample_timeline: bool = True
    action: ActionView = field(default_factory=ActionView)
    high_rate_arm_samples_per_frame: int = 0
    # canonical exporter stream -> recorded stream; lets a future site remap
    # topics without mutating raw MCAP or exporter code.
    channels: Mapping[str, str] = field(default_factory=dict)
    fields: tuple[str, ...] = CORE_LEROBOT_FIELDS
    depth: DepthExport = field(default_factory=DepthExport)
    segments: SegmentExport = field(default_factory=SegmentExport)


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
    resample = timeline.get("resample", True)
    view = str(action.get("view", ""))
    high_rate_samples = int(raw.get("high_rate_arm_samples_per_frame", 0))
    if not source or not 0.0 < fps <= 1000.0 or not fps.is_integer():
        raise ExportSpecError("timeline.source and timeline.fps are required")
    if not isinstance(resample, bool):
        raise ExportSpecError("timeline.resample must be boolean")
    if view not in {"sent_command", "absolute_joint_position", "absolute_cartesian_pose"}:
        raise ExportSpecError("action.view is unsupported")
    if not 0 <= high_rate_samples <= 128:
        raise ExportSpecError(
            "high_rate_arm_samples_per_frame must be in [0,128]"
        )
    raw_channels = raw.get("channels", {})
    if not isinstance(raw_channels, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in raw_channels.items()):
        raise ExportSpecError("channels must map canonical stream names to recorded stream names")
    raw_fields = raw.get("fields")
    if raw_fields is None:
        fields = CORE_LEROBOT_FIELDS
    elif not isinstance(raw_fields, list) or not all(
        isinstance(item, str) and item.strip() for item in raw_fields
    ):
        raise ExportSpecError("fields must be a list of LeRobot field names")
    else:
        fields = tuple(item.strip() for item in raw_fields)
        if len(fields) != len(set(fields)):
            raise ExportSpecError("fields must not contain duplicates")
        unsupported = sorted(set(fields).difference(SUPPORTED_LEROBOT_FIELDS))
        if unsupported:
            raise ExportSpecError(
                "unsupported LeRobot fields: " + ", ".join(unsupported)
            )
    if "action" not in fields:
        raise ExportSpecError("fields must include action")

    raw_depth = raw.get("depth", {})
    if not isinstance(raw_depth, dict):
        raise ExportSpecError("depth must be a mapping")
    depth_enabled = raw_depth.get("enabled", False)
    if not isinstance(depth_enabled, bool):
        raise ExportSpecError("depth.enabled must be boolean")
    raw_depth_cameras = raw_depth.get("cameras", [])
    if not isinstance(raw_depth_cameras, list) or not all(
        isinstance(item, str) and item in CAMERA_NAMES
        for item in raw_depth_cameras
    ):
        raise ExportSpecError(
            "depth.cameras must contain head/left_wrist/right_wrist"
        )
    depth_cameras = tuple(raw_depth_cameras)
    if len(depth_cameras) != len(set(depth_cameras)):
        raise ExportSpecError("depth.cameras must not contain duplicates")
    representation = str(raw_depth.get("representation", "z16"))
    storage = str(raw_depth.get("storage", "parquet"))
    if representation != "z16" or storage != "parquet":
        raise ExportSpecError(
            "depth currently supports representation=z16 and storage=parquet"
        )
    configured_depth_fields = {
        field_name
        for field_name in fields
        if field_name in DEPTH_LEROBOT_FIELDS
    }
    expected_depth_fields = {
        field_name
        for camera in depth_cameras
        for field_name in (
            f"observation.depth.{camera}",
            f"observation.depth_scale_m.{camera}",
            f"observation.depth_intrinsics.{camera}",
        )
    }
    if depth_enabled and not depth_cameras:
        raise ExportSpecError("depth.cameras is required when depth is enabled")
    if not depth_enabled and configured_depth_fields:
        raise ExportSpecError("depth fields require depth.enabled: true")
    if depth_enabled and configured_depth_fields != expected_depth_fields:
        raise ExportSpecError(
            "fields must include depth, depth_scale_m and depth_intrinsics "
            "for every configured depth camera"
        )

    raw_segments = raw.get("segments", {})
    if not isinstance(raw_segments, dict):
        raise ExportSpecError("segments must be a mapping")
    gap_threshold_s = float(raw_segments.get("gap_threshold_s", 0.25))
    split_episodes = raw_segments.get("split_episodes", False)
    if not 0.05 <= gap_threshold_s <= 60.0:
        raise ExportSpecError("segments.gap_threshold_s must be in [0.05,60]")
    if not isinstance(split_episodes, bool):
        raise ExportSpecError("segments.split_episodes must be boolean")
    high_rate_fields = {
        "observation.arm_high_rate",
        "observation.arm_high_rate_valid",
        "observation.arm_high_rate_age_s",
    }
    selected_high_rate = set(fields).intersection(high_rate_fields)
    if selected_high_rate and selected_high_rate != high_rate_fields:
        raise ExportSpecError("all three arm_high_rate fields must be selected together")
    if raw_fields is not None and selected_high_rate and high_rate_samples == 0:
        raise ExportSpecError(
            "arm_high_rate fields require high_rate_arm_samples_per_frame > 0"
        )
    return ExportSpec(
        timeline_source=source,
        fps=fps,
        resample_timeline=resample,
        action=ActionView(view),
        high_rate_arm_samples_per_frame=high_rate_samples,
        channels={k.lstrip("/"): v.lstrip("/") for k, v in raw_channels.items()},
        fields=fields,
        depth=DepthExport(
            enabled=depth_enabled,
            cameras=depth_cameras,
            representation=representation,
            storage=storage,
        ),
        segments=SegmentExport(
            gap_threshold_s=gap_threshold_s,
            split_episodes=split_episodes,
        ),
    )


def remap_streams(streams: Mapping[str, object], spec: ExportSpec) -> dict[str, object]:
    result = dict(streams)
    for canonical, source in spec.channels.items():
        if source not in streams:
            raise ExportSpecError(f"configured source stream is absent: {source}")
        result[canonical] = streams[source]
    return result
