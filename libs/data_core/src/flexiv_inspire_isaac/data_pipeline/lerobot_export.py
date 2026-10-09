"""Deterministic LeRobot v3 offline-view construction.

This module produces aligned rows and validity masks. The environment-specific
writer then feeds those rows to ``lerobot==0.6.0``. MCAP remains immutable.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .export_spec import ActionView
from .alignment import (
    AlignedValue,
    Pose,
    TimedSample,
    interpolate_pose,
)


@dataclass(frozen=True)
class ExportTolerance:
    image_ns: int = 50_000_000
    force_ns: int = 20_000_000
    tactile_ns: int = 50_000_000
    state_ns: int = 20_000_000
    # Inspire state is nominally 15 Hz and the real capture rate can be
    # 12--14 Hz.  A 20 ms arm-state tolerance therefore rejects most valid hand
    # samples.  Keep hand causal and bounded, but allow up to two nominal
    # periods, matching the online policy client's hand age limit.
    hand_ns: int = 150_000_000
    pose_bracket_ns: int = 20_000_000
    action_ns: int = 40_000_000


def _pose_at(
    samples: Sequence[TimedSample[Pose]],
    times: Sequence[int],
    timestamp_ns: int,
    tolerance_ns: int,
) -> AlignedValue:
    right_index = bisect_left(times, timestamp_ns)
    if right_index == 0 or right_index == len(samples):
        return AlignedValue(None, None, None, False, "pose-not-bracketed")
    return interpolate_pose(
        samples[right_index - 1],
        samples[right_index],
        timestamp_ns,
        tolerance_ns,
    )


class EpisodeAligner:
    """Align asynchronous streams to camera frames or a uniform target grid."""

    def __init__(
        self,
        streams: Mapping[str, Sequence[TimedSample]],
        *,
        tolerance: ExportTolerance = ExportTolerance(),
        timeline_source: str = "camera/head/jpeg",
        timeline_hz: float | None = None,
        high_rate_arm_samples_per_frame: int = 0,
        action: ActionView = ActionView(),
        depth_cameras: Sequence[str] = (),
        segment_gap_threshold_s: float = 0.25,
    ) -> None:
        self.streams = streams
        self.tolerance = tolerance
        self.timeline_source = timeline_source.lstrip("/")
        if timeline_hz is not None and not 0.0 < timeline_hz <= 1000.0:
            raise ValueError("timeline_hz must be in (0,1000]")
        self.timeline_hz = timeline_hz
        if not 0 <= high_rate_arm_samples_per_frame <= 128:
            raise ValueError("high_rate_arm_samples_per_frame must be in [0,128]")
        self.high_rate_arm_samples_per_frame = high_rate_arm_samples_per_frame
        self.action = action
        if action.name != "sent_command":
            raise ValueError("absolute action export unsupported: historical views used measured state, not control targets; use sent_command or command-supervision-v2 evidence")
        from .action_contract import BOUNDARIES
        self._boundaries = sorted({
            int(sample.value.get("effective_time_ns", sample.alignment_time_ns))
            for topic in ("episode/events", "maintenance/events")
            for sample in streams.get(topic, ())
            if sample.alignment_time_ns is not None and isinstance(sample.value, Mapping)
            and (sample.value.get("event_type") in BOUNDARIES or sample.value.get("discontinuity"))
            and not (sample.value.get("event_type") in {"pedal_pressed", "pedal_released"} and sample.value.get("recording_gate") is False)
        })
        self.depth_cameras = tuple(depth_cameras)
        if any(
            camera not in {"head", "left_wrist", "right_wrist"}
            for camera in self.depth_cameras
        ):
            raise ValueError("unsupported depth camera")
        if not 0.05 <= segment_gap_threshold_s <= 60.0:
            raise ValueError("segment_gap_threshold_s must be in [0.05,60]")
        self.segment_gap_threshold_ns = int(segment_gap_threshold_s * 1e9)
        self._mapped_streams: dict[str, tuple[TimedSample, ...]] = {}
        self._stream_times: dict[str, tuple[int, ...]] = {}
        for name, samples in streams.items():
            mapped = tuple(
                sample
                for sample in samples
                if sample.alignment_time_ns is not None
            )
            times = tuple(int(sample.alignment_time_ns) for sample in mapped)
            if any(left > right for left, right in zip(times, times[1:])):
                raise ValueError(
                    f"stream {name} must be sorted by alignment_time_ns"
                )
            self._mapped_streams[name] = mapped
            self._stream_times[name] = times

    def rows(self) -> list[dict[str, Any]]:
        reference = self.streams.get(self.timeline_source, ())
        mapped_reference = [
            sample for sample in reference if sample.alignment_time_ns is not None
        ]
        if self.timeline_hz is None:
            timeline = [
                (int(sample.alignment_time_ns), sample)
                for sample in mapped_reference
            ]
        elif mapped_reference:
            start_ns = int(mapped_reference[0].alignment_time_ns)
            end_ns = int(mapped_reference[-1].alignment_time_ns)
            period_ns = int(round(1e9 / self.timeline_hz))
            timeline = []
            for timestamp in range(start_ns, end_ns + 1, period_ns):
                head = self._causal(
                    self.timeline_source, timestamp, self.tolerance.image_ns
                )
                timeline.append((timestamp, head))
        else:
            timeline = []
        output: list[dict[str, Any]] = []
        previous_timestamp: int | None = None
        capture_segment = 0
        prior_event_segment = None
        frame_in_segment = 0
        for timestamp, head in timeline:
            source_gap_ns = (
                0 if previous_timestamp is None else timestamp - previous_timestamp
            )
            event_segment = bisect_right(self._boundaries, timestamp)
            self._segment_lower_ns = self._boundaries[event_segment - 1] if event_segment else 0
            segment_start = previous_timestamp is None or (
                event_segment != prior_event_segment if self._boundaries else source_gap_ns > self.segment_gap_threshold_ns
            )
            prior_event_segment = event_segment
            if segment_start and previous_timestamp is not None:
                capture_segment += 1
                frame_in_segment = 0
            if isinstance(head, TimedSample):
                head_value = AlignedValue(
                    head.value if head.valid else None,
                    timestamp,
                    0,
                    head.valid,
                    head.invalid_reason,
                )
            else:
                head_value = head
            row: dict[str, Any] = {
                "timestamp_ns": timestamp,
                "action.contract": "legacy-camera-sampled-v1",
                "segment.boundary_basis": "explicit-event" if self._boundaries else "legacy-gap-inference-uncertain",
                "observation.source_timestamp_ns": timestamp,
                "observation.source_gap_s": source_gap_ns / 1e9,
                "observation.capture_segment": capture_segment,
                "observation.frame_in_segment": frame_in_segment,
                "observation.capture_segment_start": segment_start,
            }
            self._put(row, "observation.images.head", head_value)
            for camera in ("left_wrist", "right_wrist"):
                self._put_nearest(
                    row,
                    f"observation.images.{camera}",
                    f"camera/{camera}/jpeg",
                    timestamp,
                    self.tolerance.image_ns,
                    allow_future=False,
                )
            for camera in self.depth_cameras:
                self._put_nearest(
                    row,
                    f"observation.depth.{camera}",
                    f"camera/{camera}/depth_z16",
                    timestamp,
                    self.tolerance.image_ns,
                    allow_future=False,
                )
            for side in ("left", "right"):
                pose_stream = f"robot/{side}_arm/tcp_pose"
                raw_pose = self._causal(pose_stream, timestamp, self.tolerance.state_ns)
                if raw_pose.valid and raw_pose.value is not None:
                    from .alignment import InterpolatedPose, quaternion_xyzw_to_matrix, matrix_to_rot6d
                    raw = raw_pose.value
                    pose = AlignedValue(InterpolatedPose(raw, matrix_to_rot6d(quaternion_xyzw_to_matrix(raw.quaternion_xyzw)), 0, 0),
                                        raw_pose.source_time_ns, raw_pose.age_ns, True)
                else:
                    pose = raw_pose
                self._put(row, f"observation.{side}_arm.pose", pose)
                arm_state_stream = f"robot/{side}_arm/state"
                if not self.streams.get(arm_state_stream):
                    arm_state_stream = f"robot/{side}_arm/joint_state"
                self._put_nearest(
                    row,
                    f"observation.{side}_arm.state",
                    arm_state_stream,
                    timestamp,
                    self.tolerance.state_ns,
                )
                for field in ("tcp_twist", "raw_ft", "tcp_wrench"):
                    stream = f"robot/{side}_arm/{field}"
                    if not self.streams.get(stream):
                        stream = arm_state_stream
                    self._put_nearest(
                        row,
                        f"observation.{side}_arm.{field}",
                        stream,
                        timestamp,
                        self.tolerance.force_ns
                        if field in {"raw_ft", "tcp_wrench"}
                        else self.tolerance.state_ns,
                    )
                self._put_nearest(
                    row,
                    f"observation.{side}_hand.state",
                    f"robot/{side}_hand/state",
                    timestamp,
                    self.tolerance.hand_ns,
                )
                self._put_nearest(
                    row,
                    f"observation.{side}_hand.tactile",
                    f"robot/{side}_hand/tactile_raw",
                    timestamp,
                    self.tolerance.tactile_ns,
                )
                history, history_valid, history_age_ns = self._arm_history_at(
                    side, timestamp
                )
                row[f"observation.{side}_arm.high_rate"] = history
                row[f"observation.{side}_arm.high_rate.valid"] = history_valid
                row[f"observation.{side}_arm.high_rate.age_ns"] = history_age_ns
            self._put(row, "action", self._action_at(timestamp))
            from .action_contract import modality_valid
            for side in ("left", "right"):
                for unit in ("arm", "hand"):
                    key = f"observation.{side}_{unit}.state"
                    if not modality_valid(f"robot/{side}_{unit}/state",
                            {"valid": row.get(key + ".valid"), "payload": row.get(key)},
                            time_ns=timestamp, max_age_ns=self.tolerance.hand_ns if unit == "hand" else self.tolerance.state_ns):
                        row[key + ".valid"] = False
                        row[key + ".invalid_reason"] = "required-state-fields-invalid"
                key = f"observation.{side}_hand.tactile"
                if not modality_valid(f"robot/{side}_hand/tactile_raw", {"valid": row.get(key + ".valid"), "payload": row.get(key)}):
                    row[key] = None
                    row[key + ".valid"] = False
                for field in ("tcp_twist", "raw_ft", "tcp_wrench"):
                    key = f"observation.{side}_arm.{field}"
                    value = row.get(key)
                    if isinstance(value, Mapping):
                        value = value.get(field, value.get("values"))
                    try:
                        vector = np.asarray(value, dtype=float)
                        valid = vector.shape == (6,) and bool(np.all(np.isfinite(vector)))
                    except (TypeError, ValueError):
                        valid = False
                    if not valid:
                        row[key] = None
                        row[key + ".valid"] = False
                        row[key + ".invalid_reason"] = "optional-field-missing-or-invalid"
            required = ["action", *[f"observation.images.{camera}" for camera in ("head", "left_wrist", "right_wrist")],
                        *[f"observation.{side}_{unit}.state" for side in ("left", "right") for unit in ("arm", "hand")],
                        *[f"observation.{side}_arm.pose" for side in ("left", "right")]]
            row["training.rejection_reasons"] = [key + ":" + row.get(key + ".invalid_reason", "invalid")
                                                  for key in required if not row.get(key + ".valid", False)]
            row["training.valid"] = not row["training.rejection_reasons"]
            output.append(row)
            previous_timestamp = timestamp
            frame_in_segment += 1
        return output

    def _arm_history_at(
        self, side: str, timestamp_ns: int
    ) -> tuple[list[Any | None], list[bool], list[int | None]]:
        count = self.high_rate_arm_samples_per_frame
        if count == 0:
            return [], [], []
        name = f"robot/{side}_arm/state"
        samples = self._mapped_streams.get(name, ())
        times = self._stream_times.get(name, ())
        end = bisect_right(times, timestamp_ns)
        start = max(0, end - count)
        selected = list(samples[start:end])
        selected_times = list(times[start:end])
        padding = count - len(selected)
        values: list[Any | None] = [None] * padding
        valid = [False] * padding
        ages: list[int | None] = [None] * padding
        # At 300/15 Hz the oldest expected sample is about 63 ms old. A
        # 150 ms cutoff prevents a disconnected arm's stale history from
        # being presented as current while retaining normal timing jitter.
        for sample, sample_time in zip(selected, selected_times):
            age = timestamp_ns - sample_time
            sample_valid = bool(sample.valid) and sample_time >= getattr(self, "_segment_lower_ns", 0) and 0 <= age <= 150_000_000
            values.append(sample.value if sample_valid else None)
            valid.append(sample_valid)
            ages.append(age if age >= 0 else None)
        return values, valid, ages

    def _action_at(self, timestamp_ns: int) -> AlignedValue:
        if self.action.name == "sent_command":
            # Exact RDK-acknowledged command: default behavioural-cloning label.
            sample = self._causal("control/sent_command", timestamp_ns, self.tolerance.action_ns)
            if isinstance(sample.value, dict):
                metadata = dict(zip(sample.value.get("metadata_keys", []), sample.value.get("metadata_values", [])))
                if metadata.get("world_state") and metadata["world_state"] != "FOLLOW":
                    return AlignedValue(None, sample.source_time_ns, sample.age_ns, False, "world-maintenance-not-follow")
            return sample
        raise ValueError("absolute command targets unavailable")

    def _put_nearest(
        self,
        row: dict[str, Any],
        output_name: str,
        stream_name: str,
        timestamp_ns: int,
        tolerance_ns: int,
        *,
        allow_future: bool = False,
    ) -> None:
        aligned = (
            self._nearest(stream_name, timestamp_ns, tolerance_ns)
            if allow_future
            else self._causal(stream_name, timestamp_ns, tolerance_ns)
        )
        self._put(row, output_name, aligned)

    def _nearest(
        self, stream_name: str, timestamp_ns: int, tolerance_ns: int
    ) -> AlignedValue:
        """Match camera observations to the closest frame on either side.

        Cameras are independent producers.  A wrist frame can legitimately
        arrive a few milliseconds after the head frame that defines the
        training timeline, so requiring a causal image wrongly selects the
        previous (one-frame-old) wrist image.
        """

        samples = self._mapped_streams.get(stream_name, ())
        times = self._stream_times.get(stream_name, ())
        if not samples:
            return AlignedValue(None, None, None, False, "timing-unmapped")
        right = bisect_left(times, timestamp_ns)
        indices = [
            index for index in (right - 1, right) if 0 <= index < len(samples)
        ]
        if not indices:
            return AlignedValue(None, None, None, False, "no-nearby-sample")
        # Prefer the earlier sample on an exact tie, while still accepting an
        # almost simultaneous frame produced just after the head camera.
        index = min(
            indices,
            key=lambda item: (
                abs(times[item] - timestamp_ns),
                times[item] > timestamp_ns,
            ),
        )
        sample = samples[index]
        sample_time = times[index]
        offset = timestamp_ns - sample_time
        if not sample.valid:
            return AlignedValue(
                None,
                sample_time,
                offset,
                False,
                sample.invalid_reason or "invalid",
            )
        if abs(offset) > tolerance_ns:
            return AlignedValue(
                None, sample_time, offset, False, "outside-tolerance"
            )
        return AlignedValue(sample.value, sample_time, offset, True)

    def _causal(
        self, stream_name: str, timestamp_ns: int, tolerance_ns: int
    ) -> AlignedValue:
        samples = self._mapped_streams.get(stream_name, ())
        times = self._stream_times.get(stream_name, ())
        if not samples:
            return AlignedValue(None, None, None, False, "timing-unmapped")
        index = bisect_right(times, timestamp_ns) - 1
        if index < 0:
            return AlignedValue(None, None, None, False, "no-causal-sample")
        sample = samples[index]
        sample_time = times[index]
        age = timestamp_ns - sample_time
        if not sample.valid:
            return AlignedValue(
                None,
                sample_time,
                age,
                False,
                sample.invalid_reason or "invalid",
            )
        if sample_time < getattr(self, "_segment_lower_ns", 0):
            return AlignedValue(None, sample_time, age, False, "cross-segment")
        if age > tolerance_ns:
            return AlignedValue(
                None, sample_time, age, False, "outside-tolerance"
            )
        return AlignedValue(sample.value, sample_time, age, True)

    @staticmethod
    def _put(row: dict[str, Any], name: str, aligned: AlignedValue) -> None:
        row[name] = aligned.value if aligned.valid else None
        row[f"{name}.valid"] = aligned.valid
        row[f"{name}.age_ns"] = aligned.age_ns
        row[f"{name}.source_time_ns"] = aligned.source_time_ns
        row[f"{name}.invalid_reason"] = aligned.reason
