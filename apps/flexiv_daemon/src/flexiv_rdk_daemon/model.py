"""Typed RDK observations and statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import numpy as np


def finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain exactly {size} finite values")
    return array


@dataclass(frozen=True)
class ArmSample:
    side: str
    connected: bool
    robot_time_ns: int
    robot_time_sec: int
    robot_time_nsec: int
    clock_domain: str
    host_receive_monotonic_ns: int
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    tau_des: np.ndarray
    tau_ext: np.ndarray
    tau_interact: np.ndarray
    tcp_pose_rdk: np.ndarray
    tcp_velocity: np.ndarray
    raw_ft: np.ndarray
    external_wrench: np.ndarray
    temperature: np.ndarray
    connection_generation: int
    host_receive_unix_ns: int = 0
    fault: str = ""

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        for name in ("q", "dq", "tau", "tau_des", "tau_ext", "tau_interact"):
            object.__setattr__(self, name, finite_vector(getattr(self, name), 7, name))
        object.__setattr__(self, "tcp_pose_rdk", finite_vector(self.tcp_pose_rdk, 7, "tcp_pose_rdk"))
        object.__setattr__(self, "tcp_velocity", finite_vector(self.tcp_velocity, 6, "tcp_velocity"))
        object.__setattr__(self, "raw_ft", finite_vector(self.raw_ft, 6, "raw_ft"))
        object.__setattr__(self, "external_wrench", finite_vector(self.external_wrench, 6, "external_wrench"))
        temperature = np.asarray(self.temperature, dtype=np.float64).reshape(-1)
        if temperature.size == 0 or not np.all(np.isfinite(temperature)):
            raise ValueError("temperature must be a non-empty finite vector")
        object.__setattr__(self, "temperature", temperature)
        if self.connection_generation < 0:
            raise ValueError("connection_generation cannot be negative")

    def to_wire(self) -> dict[str, object]:
        return {
            "side": self.side,
            "connected": self.connected,
            "robot_time_ns": str(self.robot_time_ns),
            "robot_time_sec": str(self.robot_time_sec),
            "robot_time_nsec": self.robot_time_nsec,
            "clock_domain": self.clock_domain,
            "host_receive_monotonic_ns": str(self.host_receive_monotonic_ns),
            "host_receive_unix_ns": str(self.host_receive_unix_ns),
            "q": self.q.tolist(),
            "dq": self.dq.tolist(),
            "tau": self.tau.tolist(),
            "tau_des": self.tau_des.tolist(),
            "tau_ext": self.tau_ext.tolist(),
            "tau_interact": self.tau_interact.tolist(),
            "tcp_pose_rdk_xyz_wxyz": self.tcp_pose_rdk.tolist(),
            "tcp_velocity": self.tcp_velocity.tolist(),
            "raw_ft": self.raw_ft.tolist(),
            "external_wrench": self.external_wrench.tolist(),
            "temperature": self.temperature.tolist(),
            "connection_generation": str(self.connection_generation),
            "fault": self.fault,
        }


@dataclass(frozen=True)
class DualArmSample:
    left: ArmSample
    right: ArmSample

    def __post_init__(self) -> None:
        if self.left.side != "left" or self.right.side != "right":
            raise ValueError("DualArmSample ordering must be left then right")
        if self.left.connection_generation != self.right.connection_generation:
            raise ValueError("arm samples belong to different connection generations")

    def to_wire(self) -> dict[str, object]:
        return {"left": self.left.to_wire(), "right": self.right.to_wire()}


@dataclass(frozen=True)
class VectorStatistics:
    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    peak_absolute: tuple[float, ...]

    def to_wire(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ObservationWindowStatistics:
    side: str
    sample_count: int
    duration_s: float
    raw_ft: VectorStatistics
    external_wrench: VectorStatistics
    max_dq_norm: float
    max_tcp_velocity_norm: float
    stable: bool
    rejection_reason: str = ""

    def to_wire(self) -> dict[str, object]:
        result = asdict(self)
        return result


def _vector_statistics(matrix: np.ndarray) -> VectorStatistics:
    return VectorStatistics(
        mean=tuple(float(v) for v in np.mean(matrix, axis=0)),
        standard_deviation=tuple(float(v) for v in np.std(matrix, axis=0)),
        peak_absolute=tuple(float(v) for v in np.max(np.abs(matrix), axis=0)),
    )


def compute_window_statistics(
    samples: Iterable[ArmSample],
    *,
    duration_s: float,
    max_joint_velocity_norm: float,
    max_tcp_velocity_norm: float,
    max_wrench_std_force_n: float,
    max_wrench_std_torque_nm: float,
) -> ObservationWindowStatistics:
    values = list(samples)
    if not values:
        raise ValueError("statistics window has no samples")
    side = values[0].side
    generation = values[0].connection_generation
    if any(sample.side != side for sample in values):
        raise ValueError("statistics window mixes arm sides")
    if any(sample.connection_generation != generation for sample in values):
        raise ValueError("RDK reconnected during statistics window")
    if any(not sample.connected or sample.fault for sample in values):
        raise ValueError("arm disconnected or faulted during statistics window")

    raw = np.stack([sample.raw_ft for sample in values])
    external = np.stack([sample.external_wrench for sample in values])
    dq_max = max(float(np.linalg.norm(sample.dq)) for sample in values)
    tcp_max = max(float(np.linalg.norm(sample.tcp_velocity)) for sample in values)
    raw_stats = _vector_statistics(raw)
    external_stats = _vector_statistics(external)
    reasons: list[str] = []
    if dq_max > max_joint_velocity_norm:
        reasons.append(f"dq_norm={dq_max:.6g}")
    if tcp_max > max_tcp_velocity_norm:
        reasons.append(f"tcp_velocity_norm={tcp_max:.6g}")
    for label, stats in (("raw_ft", raw_stats), ("external_wrench", external_stats)):
        if np.linalg.norm(stats.standard_deviation[:3]) > max_wrench_std_force_n:
            reasons.append(f"{label}_force_std")
        if np.linalg.norm(stats.standard_deviation[3:]) > max_wrench_std_torque_nm:
            reasons.append(f"{label}_torque_std")
    return ObservationWindowStatistics(
        side=side,
        sample_count=len(values),
        duration_s=float(duration_s),
        raw_ft=raw_stats,
        external_wrench=external_stats,
        max_dq_norm=dq_max,
        max_tcp_velocity_norm=tcp_max,
        stable=not reasons,
        rejection_reason=",".join(reasons),
    )


def residual_within_limits(
    stats: ObservationWindowStatistics,
    *,
    max_force_n: float,
    max_torque_nm: float,
) -> tuple[bool, str]:
    reasons: list[str] = []
    for label, vector in (
        ("raw_ft", stats.raw_ft.mean),
        ("external_wrench", stats.external_wrench.mean),
    ):
        if math.sqrt(sum(v * v for v in vector[:3])) > max_force_n:
            reasons.append(f"{label}_force_residual")
        if math.sqrt(sum(v * v for v in vector[3:])) > max_torque_nm:
            reasons.append(f"{label}_torque_residual")
    return not reasons, ",".join(reasons)
