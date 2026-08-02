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
    """Align all streams to head RGB source timestamps at 30 Hz."""

    def __init__(
        self,
        streams: Mapping[str, Sequence[TimedSample]],
        *,
        tolerance: ExportTolerance = ExportTolerance(),
        timeline_source: str = "camera/head/jpeg",
        action: ActionView = ActionView(),
    ) -> None:
        self.streams = streams
        self.tolerance = tolerance
        self.timeline_source = timeline_source.lstrip("/")
        self.action = action
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
        output: list[dict[str, Any]] = []
        for head in reference:
            timestamp = head.alignment_time_ns
            if timestamp is None:
                continue
            row: dict[str, Any] = {
                "timestamp_ns": timestamp,
                "observation.images.head": head.value if head.valid else None,
                "observation.images.head.valid": head.valid,
                "observation.images.head.age_ns": 0,
            }
            for camera in ("left_wrist", "right_wrist"):
                self._put_nearest(
                    row,
                    f"observation.images.{camera}",
                    f"camera/{camera}/jpeg",
                    timestamp,
                    self.tolerance.image_ns,
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
                    self.tolerance.state_ns,
                )
                self._put_nearest(
                    row,
                    f"observation.{side}_hand.tactile",
                    f"robot/{side}_hand/tactile_raw",
                    timestamp,
                    self.tolerance.tactile_ns,
                )
            self._put(row, "action", self._action_at(timestamp))
            output.append(row)
        return output

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
    ) -> None:
        aligned = self._causal(stream_name, timestamp_ns, tolerance_ns)
        self._put(row, output_name, aligned)

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
