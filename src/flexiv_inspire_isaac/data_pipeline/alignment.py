"""Timestamp alignment primitives for the reproducible LeRobot view."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Generic, Sequence, TypeVar

import numpy as np


T = TypeVar("T")


@dataclass(frozen=True)
class TimedSample(Generic[T]):
    value: T
    source_time_ns: int
    host_receive_time_ns: int
    sequence: int
    valid: bool = True
    invalid_reason: str = ""
    source_clock_domain: str = "host_monotonic"
    host_clock_domain: str = "host_monotonic"
    mapped_host_time_ns: int | None = None
    require_explicit_mapping: bool = False

    @property
    def alignment_time_ns(self) -> int | None:
        if self.mapped_host_time_ns is not None:
            return self.mapped_host_time_ns
        if not self.require_explicit_mapping and self.source_clock_domain == self.host_clock_domain:
            return self.source_time_ns
        return None

    @property
    def timing_valid(self) -> bool:
        return self.alignment_time_ns is not None



@dataclass(frozen=True)
class AlignedValue(Generic[T]):
    value: T | None
    source_time_ns: int | None
    age_ns: int | None
    valid: bool
    reason: str = ""


def causal_nearest(
    samples: Sequence[TimedSample[T]],
    reference_time_ns: int,
    tolerance_ns: int,
) -> AlignedValue[T]:
    """Return the newest valid sample no later than the reference time."""

    if tolerance_ns < 0:
        raise ValueError("tolerance_ns must be non-negative")
    mapped_samples = [
        (sample.alignment_time_ns, sample)
        for sample in samples
        if sample.alignment_time_ns is not None
    ]
    if not mapped_samples:
        return AlignedValue(None, None, None, False, "timing-unmapped")
    times = [int(item[0]) for item in mapped_samples]
    if any(a > b for a, b in zip(times, times[1:])):
        raise ValueError("samples must be sorted by alignment_time_ns")
    index = bisect_right(times, reference_time_ns) - 1
    if index < 0:
        return AlignedValue(None, None, None, False, "no-causal-sample")
    sample_time, sample = mapped_samples[index]
    age = reference_time_ns - int(sample_time)
    if not sample.valid:
        return AlignedValue(
            None, int(sample_time), age, False, sample.invalid_reason or "invalid"
        )
    if age > tolerance_ns:
        return AlignedValue(None, int(sample_time), age, False, "outside-tolerance")
    return AlignedValue(sample.value, int(sample_time), age, True)


def normalize_quaternion_xyzw(q: Sequence[float]) -> np.ndarray:
    value = np.asarray(q, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(value))
    if norm < 1e-10:
        raise ValueError("quaternion norm is too small")
    return value / norm


def slerp_xyzw(
    q0: Sequence[float], q1: Sequence[float], alpha: float
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    qa = normalize_quaternion_xyzw(q0)
    qb = normalize_quaternion_xyzw(q1)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalize_quaternion_xyzw(qa + alpha * (qb - qa))
    angle = math.acos(dot)
    denominator = math.sin(angle)
    return (
        math.sin((1.0 - alpha) * angle) / denominator * qa
        + math.sin(alpha * angle) / denominator * qb
    )


def quaternion_xyzw_to_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_rot6d(rotation: Sequence[Sequence[float]]) -> tuple[float, ...]:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6):
        raise ValueError("rotation is not orthogonal")
    if abs(float(np.linalg.det(matrix)) - 1.0) > 1e-6:
        raise ValueError("rotation determinant is not +1")
    return tuple(float(value) for value in matrix[:, :2].T.reshape(-1))


@dataclass(frozen=True)
class Pose:
    xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class InterpolatedPose:
    pose: Pose
    rotation6d: tuple[float, ...]
    left_sequence: int
    right_sequence: int


def interpolate_pose(
    left: TimedSample[Pose],
    right: TimedSample[Pose],
    reference_time_ns: int,
    max_bracket_ns: int,
) -> AlignedValue[InterpolatedPose]:
    if not left.valid or not right.valid:
        return AlignedValue(None, None, None, False, "invalid-pose-bracket")
    left_time = left.alignment_time_ns
    right_time = right.alignment_time_ns
    if left_time is None or right_time is None:
        return AlignedValue(None, None, None, False, "unmapped-pose-bracket")
    if not left_time <= reference_time_ns <= right_time:
        return AlignedValue(None, None, None, False, "reference-not-bracketed")
    span = right_time - left_time
    if span <= 0 or span > max_bracket_ns:
        return AlignedValue(None, None, span, False, "pose-bracket-too-wide")
    alpha = (reference_time_ns - left_time) / span
    xyz = (1.0 - alpha) * np.asarray(left.value.xyz) + alpha * np.asarray(
        right.value.xyz
    )
    quaternion = slerp_xyzw(
        left.value.quaternion_xyzw, right.value.quaternion_xyzw, alpha
    )
    # Rotation-6D is derived after SO(3) interpolation.  It is never linearly
    # interpolated in six-dimensional representation space.
    rot6d = matrix_to_rot6d(quaternion_xyzw_to_matrix(quaternion))
    result = InterpolatedPose(
        pose=Pose(
            tuple(float(v) for v in xyz),
            tuple(float(v) for v in quaternion),
        ),
        rotation6d=rot6d,
        left_sequence=left.sequence,
        right_sequence=right.sequence,
    )
    return AlignedValue(result, reference_time_ns, 0, True)
