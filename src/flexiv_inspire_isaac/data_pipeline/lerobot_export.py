"""Deterministic LeRobot v3 offline-view construction.

This module produces aligned rows and validity masks. The environment-specific
writer then feeds those rows to ``lerobot==0.6.0``. MCAP remains immutable.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .alignment import (
    AlignedValue,
    Pose,
    TimedSample,
    causal_nearest,
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
    timestamp_ns: int,
    tolerance_ns: int,
) -> AlignedValue:
    mapped_samples = [
        sample for sample in samples if sample.alignment_time_ns is not None
    ]
    times = [int(sample.alignment_time_ns) for sample in mapped_samples]
    right_index = bisect_left(times, timestamp_ns)
    if right_index == 0 or right_index == len(mapped_samples):
        return AlignedValue(None, None, None, False, "pose-not-bracketed")
    return interpolate_pose(
        mapped_samples[right_index - 1],
        mapped_samples[right_index],
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
    ) -> None:
        self.streams = streams
        self.tolerance = tolerance

    def rows(self) -> list[dict[str, Any]]:
        reference = self.streams.get("camera/head/jpeg", ())
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
                pose = _pose_at(
                    self.streams.get(f"robot/{side}_arm/tcp_pose", ()),
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
            # Training labels are the exact commands acknowledged by the RDK,
            # never merely requested or bridge-safe commands.
            self._put_nearest(
                row,
                "action",
                "control/sent_command",
                timestamp,
                self.tolerance.action_ns,
            )
            output.append(row)
        return output

    def _put_nearest(
        self,
        row: dict[str, Any],
        output_name: str,
        stream_name: str,
        timestamp_ns: int,
        tolerance_ns: int,
    ) -> None:
        aligned = causal_nearest(
            self.streams.get(stream_name, ()), timestamp_ns, tolerance_ns
        )
        self._put(row, output_name, aligned)

    @staticmethod
    def _put(row: dict[str, Any], name: str, aligned: AlignedValue) -> None:
        row[name] = aligned.value if aligned.valid else None
        row[f"{name}.valid"] = aligned.valid
        row[f"{name}.age_ns"] = aligned.age_ns
        row[f"{name}.invalid_reason"] = aligned.reason
