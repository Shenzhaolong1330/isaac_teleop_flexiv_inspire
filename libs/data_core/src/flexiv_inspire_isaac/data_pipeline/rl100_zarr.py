"""Atomic RL-100 Zarr writer for the dual-arm RGB diffusion-policy profile."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

import numcodecs
import numpy as np
import yaml
import zarr

from policy_contracts import get_profile, joint_minimal_state, native_action30_to_policy24

from .export_spec import ActionView
from .lerobot_v3 import _decode_image, _field_vector, _validated_action
from .profile_export import InvalidPolicyFrame, _required_row_value


SCHEMA_ID = "flexiv_rl100_dp_rgb_v1"
PROFILE_ID = "joint_proprio_cartesian_v1"
IMAGE_SOURCES = {
    "rgb_head": "observation.images.head",
    "rgb_left_wrist": "observation.images.left_wrist",
    "rgb_right_wrist": "observation.images.right_wrist",
}


class RL100ZarrSpecError(ValueError):
    pass


@dataclass(frozen=True)
class RL100ZarrSpec:
    profile: str = PROFILE_ID
    timeline_source: str = "control/sent_command"
    fps: float = 30.0
    resample_timeline: bool = False
    action: ActionView = ActionView()
    camera_alignment: str = "causal"
    channels: Mapping[str, str] | None = None
    gap_threshold_s: float = 0.05
    min_frames: int = 9
    stitch_gaps: bool = False
    max_period_error_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.channels is None:
            object.__setattr__(self, "channels", {})


@dataclass(frozen=True)
class RL100SourceEpisode:
    rows: Sequence[Mapping[str, Any]]
    source_manifest: str
    task: str = ""


@dataclass(frozen=True)
class RL100ZarrExportResult:
    output_root: str
    schema_id: str
    profile: str
    source_episodes: int
    episodes_written: int
    frames_written: int
    frames_dropped_invalid_action: int
    frames_dropped_invalid_image: int
    frames_dropped_invalid_observation: int
    frames_dropped_short_segment: int
    timeline_fps: float


def load_rl100_zarr_spec(path: str | Path) -> RL100ZarrSpec:
    source = Path(path).expanduser().resolve(strict=True)
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise RL100ZarrSpecError("conversion config must be a mapping")
    raw = document.get("rl100_zarr_export")
    if not isinstance(raw, Mapping) or int(raw.get("schema_version", 0)) != 1:
        raise RL100ZarrSpecError(
            "conversion config must define rl100_zarr_export.schema_version: 1"
        )

    timeline = raw.get("timeline", {})
    action = raw.get("action", {})
    segments = raw.get("segments", {})
    channels = raw.get("channels", {})
    if not all(isinstance(item, Mapping) for item in (timeline, action, segments, channels)):
        raise RL100ZarrSpecError(
            "timeline, action, channels and segments must be mappings"
        )

    profile = str(raw.get("profile", "")).strip()
    source_name = str(timeline.get("source", "")).lstrip("/")
    fps = float(timeline.get("fps", 0.0))
    resample = timeline.get("resample", False)
    action_view = str(action.get("view", ""))
    camera_alignment = str(raw.get("camera_alignment", ""))
    gap_threshold_s = float(segments.get("gap_threshold_s", 0.0))
    min_frames = int(segments.get("min_frames", 0))
    stitch_gaps = segments.get("stitch_gaps", False)
    max_period_error_fraction = float(
        timeline.get("max_period_error_fraction", 0.25)
    )

    if profile != PROFILE_ID:
        raise RL100ZarrSpecError(
            f"RL-100 MVP requires profile: {PROFILE_ID}"
        )
    get_profile(profile)
    if source_name != "control/sent_command":
        raise RL100ZarrSpecError(
            "RL-100 MVP timeline.source must be control/sent_command"
        )
    if fps != 30.0 or resample is not False:
        raise RL100ZarrSpecError(
            "RL-100 MVP requires a native 30 Hz command timeline without resampling"
        )
    if action_view != "sent_command":
        raise RL100ZarrSpecError("RL-100 MVP requires action.view: sent_command")
    if camera_alignment != "causal":
        raise RL100ZarrSpecError("RL-100 MVP requires camera_alignment: causal")
    if not 0.05 <= gap_threshold_s <= 60.0:
        raise RL100ZarrSpecError("segments.gap_threshold_s must be in [0.05,60]")
    if min_frames < 2:
        raise RL100ZarrSpecError("segments.min_frames must be at least 2")
    if not isinstance(stitch_gaps, bool):
        raise RL100ZarrSpecError("segments.stitch_gaps must be a bool")
    if not 0.0 < max_period_error_fraction <= 0.5:
        raise RL100ZarrSpecError(
            "timeline.max_period_error_fraction must be in (0,0.5]"
        )
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in channels.items()):
        raise RL100ZarrSpecError("channels must map strings to strings")

    return RL100ZarrSpec(
        profile=profile,
        timeline_source=source_name,
        fps=fps,
        resample_timeline=resample,
        action=ActionView(action_view),
        camera_alignment=camera_alignment,
        channels={key.lstrip("/"): value.lstrip("/") for key, value in channels.items()},
        gap_threshold_s=gap_threshold_s,
        min_frames=min_frames,
        stitch_gaps=stitch_gaps,
        max_period_error_fraction=max_period_error_fraction,
    )


def aligned_row_to_rl100_frame(row: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Convert one aligned native row to the exact 26D/24D/three-RGB contract."""

    try:
        arm_q = np.concatenate(
            [
                _field_vector(
                    _required_row_value(
                        row,
                        f"observation.{side}_arm.state",
                        category="observation",
                    ),
                    "q",
                    7,
                )
                for side in ("left", "right")
            ]
        )
        hand_angle = np.concatenate(
            [
                _field_vector(
                    _required_row_value(
                        row,
                        f"observation.{side}_hand.state",
                        category="observation",
                    ),
                    "angle",
                    6,
                )
                for side in ("left", "right")
            ]
        )
        state = joint_minimal_state(arm_q, hand_angle).astype(np.float32)
    except InvalidPolicyFrame:
        raise
    except ValueError as exc:
        raise InvalidPolicyFrame("observation", str(exc)) from exc

    try:
        native_action = _validated_action(
            _required_row_value(row, "action", category="action"), ActionView()
        )
        action = native_action30_to_policy24(native_action).astype(np.float32)
    except InvalidPolicyFrame:
        raise
    except ValueError as exc:
        raise InvalidPolicyFrame("action", str(exc)) from exc

    images: dict[str, np.ndarray] = {}
    for target, source in IMAGE_SOURCES.items():
        try:
            image = _decode_image(
                _required_row_value(row, source, category="image")
            )
        except InvalidPolicyFrame:
            raise
        except ValueError as exc:
            raise InvalidPolicyFrame("image", f"{source}: {exc}") from exc
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise InvalidPolicyFrame("image", f"{source}: expected HWC uint8 RGB")
        images[target] = np.ascontiguousarray(image.transpose(2, 0, 1))

    if state.shape != (26,) or action.shape != (24,):
        raise InvalidPolicyFrame("observation", "policy mapping returned wrong dimensions")
    return {"state": state, "action": action, **images}


class _StreamingZarrWriter:
    def __init__(
        self,
        root: zarr.Group,
        *,
        fps: float,
        source_manifests: Sequence[str],
        stitch_gaps: bool,
    ) -> None:
        self.root = root
        self.data = root.create_group("data")
        self.meta = root.create_group("meta")
        self.arrays: dict[str, zarr.Array] = {}
        self.image_shapes: dict[str, tuple[int, int, int]] | None = None
        self.episode_ends: list[int] = []
        self.episode_sources: list[int] = []
        self.count = 0
        profile = get_profile(PROFILE_ID)
        root.attrs.update(
            {
                "schema_id": SCHEMA_ID,
                "schema_version": 1,
                "profile": PROFILE_ID,
                "fps": float(fps),
                "image_layout": "CHW",
                "image_dtype": "uint8",
                "state_dimension": 26,
                "action_dimension": 24,
                "state_names": list(profile.state_names),
                "action_names": list(profile.action_names),
                "state_semantics": profile.state_semantics,
                "action_semantics": profile.action_semantics,
                "timeline_source": "control/sent_command",
                "timestamp_semantics": (
                    "pedal_gaps_removed" if stitch_gaps else "source_monotonic"
                ),
                "observation_alignment": "latest_causal",
                "reward_available": False,
                "offline_rl_ready": False,
                "source_manifests": list(source_manifests),
            }
        )

    def _initialize(self, frame: Mapping[str, np.ndarray]) -> None:
        image_shapes = {
            key: tuple(int(item) for item in frame[key].shape)
            for key in IMAGE_SOURCES
        }
        compressor = numcodecs.Blosc(
            cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE
        )
        self.arrays["state"] = self.data.empty(
            "state", shape=(0, 26), chunks=(1024, 26), dtype="f4", compressor=compressor
        )
        self.arrays["action"] = self.data.empty(
            "action", shape=(0, 24), chunks=(1024, 24), dtype="f4", compressor=compressor
        )
        for key, shape in image_shapes.items():
            self.arrays[key] = self.data.empty(
                key,
                shape=(0, *shape),
                chunks=(1, *shape),
                dtype="u1",
                compressor=compressor,
            )
        for key, dtype in (
            ("timestamp_ns", "i8"),
            ("source_timestamp_ns", "i8"),
            ("source_episode_index", "i4"),
            ("capture_segment", "i4"),
        ):
            self.arrays[key] = self.data.empty(
                key, shape=(0,), chunks=(4096,), dtype=dtype, compressor=compressor
            )
        self.image_shapes = image_shapes

    def validate_shapes(self, frame: Mapping[str, np.ndarray]) -> None:
        if self.image_shapes is None:
            return
        actual = {key: tuple(frame[key].shape) for key in IMAGE_SOURCES}
        if actual != self.image_shapes:
            raise InvalidPolicyFrame(
                "image", f"camera image shape changed: {actual} != {self.image_shapes}"
            )

    def append(
        self,
        records: Sequence[tuple[dict[str, np.ndarray], int, int, int, int]],
    ) -> None:
        if not records:
            return
        if not self.arrays:
            self._initialize(records[0][0])
        for frame, _, _, _, _ in records:
            self.validate_shapes(frame)
        start = self.count
        end = start + len(records)
        for array in self.arrays.values():
            array.resize((end, *array.shape[1:]))
        self.arrays["state"][start:end] = np.stack([item[0]["state"] for item in records])
        self.arrays["action"][start:end] = np.stack([item[0]["action"] for item in records])
        for key in IMAGE_SOURCES:
            self.arrays[key][start:end] = np.stack([item[0][key] for item in records])
        self.arrays["timestamp_ns"][start:end] = [item[1] for item in records]
        self.arrays["source_timestamp_ns"][start:end] = [item[2] for item in records]
        self.arrays["source_episode_index"][start:end] = [item[3] for item in records]
        self.arrays["capture_segment"][start:end] = [item[4] for item in records]
        self.count = end

    def finish_episode(self, source_index: int) -> None:
        if self.episode_ends and self.episode_ends[-1] == self.count:
            raise RuntimeError("cannot write an empty RL-100 episode")
        self.episode_ends.append(self.count)
        self.episode_sources.append(int(source_index))

    def finalize(self) -> None:
        if not self.episode_ends or self.episode_ends[-1] != self.count:
            raise RuntimeError("RL-100 Zarr has no complete episodes")
        self.meta.array(
            "episode_ends",
            np.asarray(self.episode_ends, dtype=np.int64),
            chunks=(max(1, len(self.episode_ends)),),
            compressor=None,
        )
        self.meta.array(
            "episode_source_index",
            np.asarray(self.episode_sources, dtype=np.int32),
            chunks=(max(1, len(self.episode_sources)),),
            compressor=None,
        )


def export_rl100_zarr(
    episodes: Sequence[RL100SourceEpisode],
    *,
    output_root: str | Path,
    spec: RL100ZarrSpec,
) -> RL100ZarrExportResult:
    if not episodes:
        raise ValueError("at least one source episode is required")
    destination = Path(output_root).expanduser().resolve()
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError(f"refusing to overwrite existing dataset: {destination}")
        destination.rmdir()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.incomplete-{uuid.uuid4().hex}")

    dropped = {"action": 0, "image": 0, "observation": 0, "short": 0}
    writer: _StreamingZarrWriter | None = None
    gap_ns = int(spec.gap_threshold_s * 1e9)
    expected_period_ns = 1e9 / spec.fps
    try:
        root = zarr.group(store=zarr.DirectoryStore(str(staging)), overwrite=True)
        writer = _StreamingZarrWriter(
            root,
            fps=spec.fps,
            source_manifests=[item.source_manifest for item in episodes],
            stitch_gaps=spec.stitch_gaps,
        )
        pending: list[tuple[dict[str, np.ndarray], int, int, int, int]] = []
        active = False
        active_source = -1
        previous_timestamp: int | None = None
        previous_source_timestamp: int | None = None
        previous_segment: tuple[int, int] | None = None
        run_periods_ns: list[int] = []
        timestamp_offset_ns = 0
        gaps_stitched = 0
        gap_duration_removed_ns = 0

        def finish_run() -> None:
            nonlocal pending, active, active_source, previous_timestamp
            nonlocal previous_source_timestamp, previous_segment, run_periods_ns
            nonlocal timestamp_offset_ns
            if active:
                median_period_ns = float(np.median(run_periods_ns))
                relative_error = abs(median_period_ns - expected_period_ns) / expected_period_ns
                if relative_error > spec.max_period_error_fraction:
                    actual_hz = 1e9 / median_period_ns
                    raise RL100ZarrSpecError(
                        "native sent_command cadence is incompatible with the "
                        f"30 Hz dataset contract: median={actual_hz:.3f} Hz"
                    )
                writer.append(pending)
                writer.finish_episode(active_source)
            else:
                dropped["short"] += len(pending)
            pending = []
            active = False
            active_source = -1
            previous_timestamp = None
            previous_source_timestamp = None
            previous_segment = None
            run_periods_ns = []
            timestamp_offset_ns = 0

        for source_index, episode in enumerate(episodes):
            finish_run()
            for row in episode.rows:
                source_timestamp = int(row["timestamp_ns"])
                segment = int(row.get("observation.capture_segment", 0))
                segment_key = (source_index, segment)
                boundary = (
                    previous_segment is not None
                    and (
                        segment_key != previous_segment
                        or source_timestamp - int(previous_source_timestamp) > gap_ns
                    )
                )
                try:
                    frame = aligned_row_to_rl100_frame(row)
                    writer.validate_shapes(frame)
                except InvalidPolicyFrame as exc:
                    category = exc.category if exc.category in dropped else "observation"
                    dropped[category] += 1
                    if not spec.stitch_gaps:
                        finish_run()
                    continue

                timestamp = source_timestamp - timestamp_offset_ns
                if boundary:
                    if spec.stitch_gaps:
                        expected_timestamp = int(previous_timestamp) + round(expected_period_ns)
                        removed_ns = timestamp - expected_timestamp
                        timestamp_offset_ns += removed_ns
                        timestamp = expected_timestamp
                        gaps_stitched += 1
                        gap_duration_removed_ns += removed_ns
                    else:
                        finish_run()
                        timestamp = source_timestamp

                if previous_timestamp is not None:
                    period_ns = timestamp - previous_timestamp
                    if period_ns <= 0:
                        raise RL100ZarrSpecError(
                            "sent_command timestamps must be strictly increasing"
                        )
                    run_periods_ns.append(period_ns)
                pending.append(
                    (frame, timestamp, source_timestamp, source_index, segment)
                )
                active_source = source_index
                previous_timestamp = timestamp
                previous_source_timestamp = source_timestamp
                previous_segment = segment_key
                if len(pending) >= spec.min_frames:
                    active = True
                if active and len(pending) >= 64:
                    writer.append(pending)
                    pending = []
        finish_run()
        writer.finalize()

        result = RL100ZarrExportResult(
            output_root=str(destination),
            schema_id=SCHEMA_ID,
            profile=spec.profile,
            source_episodes=len(episodes),
            episodes_written=len(writer.episode_ends),
            frames_written=writer.count,
            frames_dropped_invalid_action=dropped["action"],
            frames_dropped_invalid_image=dropped["image"],
            frames_dropped_invalid_observation=dropped["observation"],
            frames_dropped_short_segment=dropped["short"],
            timeline_fps=spec.fps,
        )
        (staging / "export_validation.json").write_text(
            json.dumps(
                {
                    **asdict(result),
                    "source_manifests": [item.source_manifest for item in episodes],
                    "task_descriptions": [item.task for item in episodes],
                    "timeline_source": spec.timeline_source,
                    "timeline_resampled": spec.resample_timeline,
                    "camera_alignment": spec.camera_alignment,
                    "segment_gap_threshold_s": spec.gap_threshold_s,
                    "stitch_gaps": spec.stitch_gaps,
                    "gaps_stitched": gaps_stitched,
                    "gap_duration_removed_s": gap_duration_removed_ns / 1e9,
                    "minimum_segment_frames": spec.min_frames,
                    "maximum_period_error_fraction": spec.max_period_error_fraction,
                    "reward_available": False,
                    "offline_rl_ready": False,
                    "offline_rl_blocker": "reward/success labels are not present in teleoperation MCAP",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        staging.replace(destination)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
