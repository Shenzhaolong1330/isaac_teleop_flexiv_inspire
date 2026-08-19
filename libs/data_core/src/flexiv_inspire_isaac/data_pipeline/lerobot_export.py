"""Deterministic LeRobot v3 offline-view construction.

This module produces aligned rows and validity masks. The environment-specific
writer then feeds those rows to ``lerobot==0.6.0``. MCAP remains immutable.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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
        allow_future_camera_matches: bool = True,
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
        self.depth_cameras = tuple(depth_cameras)
        self.allow_future_camera_matches = bool(allow_future_camera_matches)
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
        frame_in_segment = 0
        for timestamp, head in timeline:
            source_gap_ns = (
                0 if previous_timestamp is None else timestamp - previous_timestamp
            )
            segment_start = previous_timestamp is None or (
                source_gap_ns > self.segment_gap_threshold_ns
            )
            if segment_start and previous_timestamp is not None:
                capture_segment += 1
                frame_in_segment = 0
            if self.timeline_source != "camera/head/jpeg":
                head_value = self._causal(
                    "camera/head/jpeg", timestamp, self.tolerance.image_ns
                )
            elif isinstance(head, TimedSample):
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
                    allow_future=self.allow_future_camera_matches,
                )
            for camera in self.depth_cameras:
                self._put_nearest(
                    row,
                    f"observation.depth.{camera}",
                    f"camera/{camera}/depth_z16",
                    timestamp,
                    self.tolerance.image_ns,
                    allow_future=self.allow_future_camera_matches,
                )
            for side in ("left", "right"):
                pose_stream = f"robot/{side}_arm/tcp_pose"
                pose = _pose_at(
                    self._mapped_streams.get(pose_stream, ()),
                    self._stream_times.get(pose_stream, ()),
                    timestamp,
                    self.tolerance.pose_bracket_ns,
                )
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
            sample_valid = bool(sample.valid) and 0 <= age <= 150_000_000
            values.append(sample.value if sample_valid else None)
            valid.append(sample_valid)
            ages.append(age if age >= 0 else None)
        return values, valid, ages

    def _action_at(self, timestamp_ns: int) -> AlignedValue:
        if self.action.name == "sent_command":
            # Exact RDK-acknowledged command: default behavioural-cloning label.
            return self._causal(
                "control/sent_command", timestamp_ns, self.tolerance.action_ns
            )
        if self.action.name == "absolute_joint_position":
            values: list[float] = []
            ages: list[int] = []
            for side in ("left", "right"):
                value = self._causal(
                    f"robot/{side}_arm/state",
                    timestamp_ns,
                    self.tolerance.state_ns,
                )
                if not value.valid or not isinstance(value.value, Mapping):
                    return AlignedValue(None, None, value.age_ns, False, f"absolute-joint-{side}:{value.reason}")
                try:
                    joint = [float(item) for item in value.value["q"]]
                except (KeyError, TypeError, ValueError):
                    return AlignedValue(None, None, value.age_ns, False, f"absolute-joint-{side}:q-missing")
                if len(joint) != 7:
                    return AlignedValue(None, None, value.age_ns, False, f"absolute-joint-{side}:q-not-7d")
                values.extend(joint)
                ages.append(int(value.age_ns or 0))
            return AlignedValue(tuple(values), timestamp_ns, max(ages, default=0), True)
        # Absolute Cartesian pose uses the same valid SO(3)-interpolated poses
        # as observations: xyz plus Rotation-6D for left then right.
        values: list[float] = []
        ages: list[int] = []
        for side in ("left", "right"):
            pose_stream = f"robot/{side}_arm/tcp_pose"
            pose = _pose_at(
                self._mapped_streams.get(pose_stream, ()),
                self._stream_times.get(pose_stream, ()),
                timestamp_ns,
                self.tolerance.pose_bracket_ns,
            )
            if not pose.valid or pose.value is None:
                return AlignedValue(None, None, pose.age_ns, False, f"absolute-cartesian-{side}:{pose.reason}")
            values.extend((*pose.value.pose.xyz, *pose.value.rotation6d))
            ages.append(int(pose.age_ns or 0))
        return AlignedValue(tuple(values), timestamp_ns, max(ages, default=0), True)

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
        row[f"{name}.invalid_reason"] = aligned.reason
