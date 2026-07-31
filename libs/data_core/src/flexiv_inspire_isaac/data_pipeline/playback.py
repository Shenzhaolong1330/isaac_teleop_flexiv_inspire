"""Shared, side-effect-free dataset selection and playback validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import yaml

from isaac_teleop_core.command import (
    BimanualCommand,
    CommandSource,
    ROTATION_ORDER,
    ValidMask,
)


class PlaybackConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetSpec:
    root: Path
    episode: str
    require_completed: bool


@dataclass(frozen=True)
class VisualizeSpec:
    sink: str
    save_path: Path | None
    viewer_port: int
    speed: float
    realtime: bool


@dataclass(frozen=True)
class ReplaySpec:
    enabled: bool
    operator_confirmation: str
    speed: float
    ttl_s: float
    max_inter_command_gap_s: float
    max_schedule_lateness_s: float
    start_delay_s: float
    home_before_start: bool
    include_hands: bool


@dataclass(frozen=True)
class PlaybackSpec:
    path: Path
    project_root: Path
    site_config: Path
    dataset: DatasetSpec
    visualize: VisualizeSpec
    replay: ReplaySpec


@dataclass(frozen=True)
class EpisodeSelection:
    directory: Path
    manifest_path: Path
    manifest: dict[str, Any]
    deviceio_mcap: Path


@dataclass(frozen=True)
class DeviceIORecord:
    topic: str
    timestamp_ns: int
    valid: bool
    timing_valid: bool
    sequence: int
    payload: Any
    envelope: dict[str, Any]


@dataclass(frozen=True)
class RecordedCommand:
    timestamp_ns: int
    original_sequence: int
    valid_mask: int
    action: tuple[float, ...]


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlaybackConfigError(f"{name} must be a mapping")
    return value


def _resolve(root: Path, value: Any, name: str) -> Path:
    raw = str(value).strip()
    if not raw:
        raise PlaybackConfigError(f"{name} is required")
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_playback_config(path: str | Path) -> PlaybackSpec:
    source = Path(path).expanduser().resolve(strict=True)
    raw = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "playback config")
    if int(raw.get("schema_version", 0)) != 1:
        raise PlaybackConfigError("playback schema_version must be 1")
    project_root = source.parent.parent
    if not (project_root / "pyproject.toml").is_file():
        project_root = source.parent
    site_config = _resolve(
        project_root, raw.get("site_config", "config/site.yaml"), "site_config"
    )

    dataset_raw = _mapping(raw.get("dataset"), "dataset")
    episode = str(dataset_raw.get("episode", "")).strip()
    if not episode:
        raise PlaybackConfigError("dataset.episode is required")
    dataset = DatasetSpec(
        root=_resolve(project_root, dataset_raw.get("root"), "dataset.root"),
        episode=episode,
        require_completed=bool(dataset_raw.get("require_completed", True)),
    )

    visualize_raw = _mapping(raw.get("visualize"), "visualize")
    sink = str(visualize_raw.get("sink", "spawn")).strip()
    if sink not in {"spawn", "save"}:
        raise PlaybackConfigError("visualize.sink must be spawn or save")
    save_raw = str(visualize_raw.get("save_path", "")).strip()
    save_path = _resolve(project_root, save_raw, "visualize.save_path") if save_raw else None
    if sink == "save" and save_path is None:
        raise PlaybackConfigError("visualize.save_path is required for sink=save")
    viewer_port = int(visualize_raw.get("viewer_port", 9876))
    if not 1024 <= viewer_port <= 65535:
        raise PlaybackConfigError("visualize.viewer_port is invalid")
    visualize_speed = float(visualize_raw.get("speed", 1.0))
    if not 0.05 <= visualize_speed <= 16.0:
        raise PlaybackConfigError("visualize.speed must be in [0.05,16]")
    visualize = VisualizeSpec(
        sink=sink,
        save_path=save_path,
        viewer_port=viewer_port,
        speed=visualize_speed,
        realtime=bool(visualize_raw.get("realtime", False)),
    )

    replay_raw = _mapping(raw.get("replay"), "replay")
    replay_speed = float(replay_raw.get("speed", 1.0))
    if not 0.05 <= replay_speed <= 1.0:
        raise PlaybackConfigError("replay.speed must be in [0.05,1.0]")
    ttl_s = float(replay_raw.get("ttl_s", 0.2))
    if not 0.02 <= ttl_s <= 1.0:
        raise PlaybackConfigError("replay.ttl_s must be in [0.02,1.0]")
    max_gap = float(replay_raw.get("max_inter_command_gap_s", 0.1))
    if not 0.01 <= max_gap <= 0.2:
        raise PlaybackConfigError(
            "replay.max_inter_command_gap_s must be in [0.01,0.2]"
        )
    max_lateness = float(replay_raw.get("max_schedule_lateness_s", 0.02))
    if not 0.001 <= max_lateness <= 0.05:
        raise PlaybackConfigError(
            "replay.max_schedule_lateness_s must be in [0.001,0.05]"
        )
    start_delay = float(replay_raw.get("start_delay_s", 2.0))
    if not 0.0 <= start_delay <= 30.0:
        raise PlaybackConfigError("replay.start_delay_s must be in [0,30]")
    replay = ReplaySpec(
        enabled=bool(replay_raw.get("enabled", False)),
        operator_confirmation=str(
            replay_raw.get("operator_confirmation", "")
        ).strip(),
        speed=replay_speed,
        ttl_s=ttl_s,
        max_inter_command_gap_s=max_gap,
        max_schedule_lateness_s=max_lateness,
        start_delay_s=start_delay,
        home_before_start=bool(replay_raw.get("home_before_start", True)),
        include_hands=bool(replay_raw.get("include_hands", True)),
    )
    return PlaybackSpec(
        path=source,
        project_root=project_root,
        site_config=site_config,
        dataset=dataset,
        visualize=visualize,
        replay=replay,
    )


def _candidate_key(path: Path, manifest: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(manifest.get("episode_index", 0)),
        int(manifest.get("attempt", 0)),
        path.stat().st_mtime_ns,
    )


def resolve_episode(spec: PlaybackSpec, *, for_hardware: bool = False) -> EpisodeSelection:
    root = spec.dataset.root.resolve(strict=True)
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for child in root.iterdir():
        manifest_path = child / "manifest.json"
        if not child.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(manifest, dict):
            candidates.append((child, manifest))
    if not candidates:
        raise FileNotFoundError(f"no episode manifests under {root}")

    selector = spec.dataset.episode
    if selector == "latest":
        selected = max(candidates, key=lambda item: _candidate_key(*item))
    elif selector.isdigit():
        index = int(selector)
        matching = [
            item for item in candidates if int(item[1].get("episode_index", -1)) == index
        ]
        if not matching:
            raise FileNotFoundError(f"episode index {index} is absent under {root}")
        selected = max(matching, key=lambda item: _candidate_key(*item))
    else:
        selected_path = (root / selector).resolve(strict=True)
        try:
            selected_path.relative_to(root)
        except ValueError as exc:
            raise PlaybackConfigError("dataset.episode escapes dataset.root") from exc
        matching = [item for item in candidates if item[0] == selected_path]
        if not matching:
            raise FileNotFoundError(f"episode manifest is absent: {selected_path}")
        selected = matching[0]

    directory, manifest = selected
    completed_required = spec.dataset.require_completed or for_hardware
    if completed_required and not bool(manifest.get("completed", False)):
        raise PlaybackConfigError(
            f"episode is incomplete: {manifest.get('completion_reason', '')}"
        )
    if for_hardware and int(manifest.get("pause_count", 0)) > 1:
        raise PlaybackConfigError(
            "hardware replay rejects episodes paused mid-demonstration"
        )
    raw_mcap = str(manifest.get("deviceio_mcap", "")).strip()
    if not raw_mcap:
        raise PlaybackConfigError("manifest.deviceio_mcap is missing")
    deviceio = Path(raw_mcap)
    if not deviceio.is_absolute():
        deviceio = directory / deviceio
    deviceio = deviceio.resolve(strict=True)
    return EpisodeSelection(
        directory=directory,
        manifest_path=directory / "manifest.json",
        manifest=manifest,
        deviceio_mcap=deviceio,
    )


def read_deviceio_records(path: str | Path) -> list[DeviceIORecord]:
    from mcap.reader import make_reader

    result: list[DeviceIORecord] = []
    with Path(path).open("rb") as handle:
        for _schema, channel, message in make_reader(handle).iter_messages():
            if channel.message_encoding != "json":
                continue
            document = json.loads(message.data)
            if not isinstance(document, dict) or "payload" not in document:
                continue
            timing_valid = bool(document.get("timing_valid", False))
            mapped = document.get("mapped_host_time_ns")
            timestamp = (
                int(mapped)
                if timing_valid and mapped is not None
                else int(document.get("host_receive_time_ns", message.log_time))
            )
            result.append(
                DeviceIORecord(
                    topic=str(document.get("topic", channel.topic)),
                    timestamp_ns=timestamp,
                    valid=bool(document.get("valid", False)),
                    timing_valid=timing_valid and mapped is not None,
                    sequence=int(document.get("sequence", 0)),
                    payload=document["payload"],
                    envelope=document,
                )
            )
    result.sort(key=lambda item: (item.timestamp_ns, item.sequence, item.topic))
    if not result:
        raise PlaybackConfigError("deviceio MCAP contains no JSON records")
    return result


def _command_action(payload: Any) -> tuple[int, int, tuple[float, ...]]:
    command = _mapping(payload, "recorded sent command")
    if int(command.get("schema_version", 0)) != 1:
        raise PlaybackConfigError("recorded sent command schema is not v1")
    if int(command.get("representation", 0)) != 1:
        raise PlaybackConfigError(
            "hardware replay currently requires CARTESIAN_ROT6D sent commands"
        )
    if str(command.get("frame_id", "")) != "world":
        raise PlaybackConfigError("recorded replay command frame must be world")
    if str(command.get("rotation_order", "")) != ROTATION_ORDER:
        raise PlaybackConfigError("recorded replay Rotation-6D order is invalid")
    if not bool(command.get("deadman", False)):
        raise PlaybackConfigError("recorded sent command has no motion authority")
    valid_mask = int(command.get("valid_mask", 0))
    if valid_mask <= 0 or valid_mask & ~0xF:
        raise PlaybackConfigError("recorded sent command valid_mask is invalid")
    points = command.get("trajectory")
    if not isinstance(points, list) or len(points) != 1 or not isinstance(points[0], dict):
        raise PlaybackConfigError("recorded sent command must contain one point")
    point = points[0]
    try:
        action = tuple(float(value) for values in (
            point["left_delta_xyz"],
            point["left_delta_rotation6d"],
            point["right_delta_xyz"],
            point["right_delta_rotation6d"],
            point["left_hand_targets"],
            point["right_hand_targets"],
        ) for value in values)
    except (KeyError, TypeError, ValueError) as exc:
        raise PlaybackConfigError(f"recorded sent command is malformed: {exc}") from exc
    if len(action) != 30:
        raise PlaybackConfigError("recorded sent command is not a 30-D action")
    # Reuse the production command validator before any ROS publisher exists.
    BimanualCommand.from_policy_vectors(
        [action],
        session_id="offline-validation",
        source=CommandSource.REPLAY,
        sequence=1,
        ttl_s=0.2,
        frame_id="world",
        deadman=True,
        valid_mask=ValidMask(valid_mask),
        issued_monotonic_ns=1,
    )
    return int(command.get("sequence", 0)), valid_mask, action


def extract_replay_commands(
    records: Iterable[DeviceIORecord], replay: ReplaySpec
) -> list[RecordedCommand]:
    commands: list[RecordedCommand] = []
    for record in records:
        if record.topic.rstrip("/") != "/control/sent_command":
            continue
        if not record.valid or not record.timing_valid:
            raise PlaybackConfigError(
                "hardware replay rejects invalid or unmapped sent commands"
            )
        original_sequence, valid_mask, action = _command_action(record.payload)
        if not replay.include_hands:
            valid_mask &= ~0xC
        if valid_mask == 0:
            raise PlaybackConfigError(
                "replay.include_hands=false removed every valid target"
            )
        commands.append(
            RecordedCommand(
                timestamp_ns=record.timestamp_ns,
                original_sequence=original_sequence,
                valid_mask=valid_mask,
                action=action,
            )
        )
    if not commands:
        raise PlaybackConfigError("episode contains no replayable sent commands")
    maximum_gap_ns = int(replay.max_inter_command_gap_s * 1e9)
    for previous, current in zip(commands, commands[1:]):
        gap = current.timestamp_ns - previous.timestamp_ns
        if gap <= 0:
            raise PlaybackConfigError("replay command timestamps are not strictly increasing")
        if gap > maximum_gap_ns:
            raise PlaybackConfigError(
                f"replay command gap {gap / 1e9:.6f}s exceeds configured maximum"
            )
    return commands


def validate_replay_timing(
    commands: list[RecordedCommand], replay: ReplaySpec
) -> None:
    """Ensure slowed scheduling cannot outlive the preceding command TTL."""

    if len(commands) < 2:
        return
    largest_gap_s = max(
        (current.timestamp_ns - previous.timestamp_ns) / 1e9
        for previous, current in zip(commands, commands[1:])
    )
    scaled_gap_s = largest_gap_s / replay.speed
    if scaled_gap_s >= replay.ttl_s * 0.8:
        raise PlaybackConfigError(
            "replay speed/TTL would make commands stale between samples; "
            "increase replay.ttl_s or replay.speed"
        )


def validate_replay_home_origin(
    records: Iterable[DeviceIORecord],
    commands: list[RecordedCommand],
    *,
    left_home: Iterable[float],
    right_home: Iterable[float],
    tolerance_rad: float,
) -> None:
    """Prove the first executed delta was recorded from the configured Home."""

    if not commands:
        raise PlaybackConfigError("cannot validate Home origin without commands")
    targets = {
        "left": tuple(float(value) for value in left_home),
        "right": tuple(float(value) for value in right_home),
    }
    if any(len(value) != 7 for value in targets.values()):
        raise PlaybackConfigError("configured Home must contain seven joints per arm")
    first_command_time = commands[0].timestamp_ns
    measured: dict[str, tuple[float, ...]] = {}
    measured_at: dict[str, int] = {}
    for record in records:
        if record.timestamp_ns > first_command_time:
            break
        if not record.valid or not record.timing_valid or not isinstance(record.payload, dict):
            continue
        normalized = record.topic.rstrip("/")
        for side in ("left", "right"):
            if normalized == f"/robot/{side}_arm/state":
                try:
                    q = tuple(float(value) for value in record.payload["q"])
                except (KeyError, TypeError, ValueError):
                    continue
                if len(q) == 7 and all(math.isfinite(value) for value in q):
                    measured[side] = q
                    measured_at[side] = record.timestamp_ns
    for side in ("left", "right"):
        if side not in measured:
            raise PlaybackConfigError(
                f"no valid {side} arm state exists before the first replay command"
            )
        age_ns = first_command_time - measured_at[side]
        if age_ns > 100_000_000:
            raise PlaybackConfigError(
                f"recorded {side} arm state is stale before the first replay command"
            )
        error = max(
            abs(actual - expected)
            for actual, expected in zip(measured[side], targets[side])
        )
        if error > tolerance_rad:
            raise PlaybackConfigError(
                f"recorded {side} start is not configured Home "
                f"(max joint error {error:.6f} rad > {tolerance_rad:.6f} rad)"
            )
