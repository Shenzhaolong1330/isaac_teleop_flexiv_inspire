"""Pure fail-closed SE(3) clutch mapper for Quest wrist poses."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from isaac_teleop_core.rotation6d import (
    IDENTITY_ROT6D,
    geodesic_distance_rad,
    matrix_to_rotvec,
    matrix_to_rotation6d,
    quaternion_xyzw_to_matrix,
    rotvec_to_matrix,
)


@dataclass(frozen=True)
class TrackedPose:
    position: np.ndarray
    quaternion_xyzw: np.ndarray

    @classmethod
    def make(cls, position, quaternion_xyzw) -> "TrackedPose":
        position_array = np.asarray(position, dtype=np.float64).reshape(-1)
        if position_array.shape != (3,) or not np.all(np.isfinite(position_array)):
            raise ValueError("tracked position must contain three finite values")
        # Conversion performs strict finite/norm validation while preserving the
        # Quest quaternion semantics until the SE(3) mapping boundary.
        quaternion_array = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(-1)
        quaternion_xyzw_to_matrix(quaternion_array)
        return cls(position_array.copy(), quaternion_array.copy())

    @property
    def rotation(self) -> np.ndarray:
        return quaternion_xyzw_to_matrix(self.quaternion_xyzw)


@dataclass(frozen=True)
class TrackingSample:
    left: TrackedPose
    right: TrackedPose
    sequence: int
    source_time_ns: int
    receive_monotonic_ns: int
    frame_id: str


@dataclass(frozen=True)
class TeleopDelta:
    left_xyz: np.ndarray
    left_rotation6d: np.ndarray
    right_xyz: np.ndarray
    right_rotation6d: np.ndarray
    active: bool
    hold_latched: bool
    reason: str

    @classmethod
    def identity(cls, *, active: bool, hold: bool = False, reason: str = ""):
        return cls(
            np.zeros(3),
            IDENTITY_ROT6D.copy(),
            np.zeros(3),
            IDENTITY_ROT6D.copy(),
            active,
            hold,
            reason,
        )


class QuestSE3Mapper:
    def __init__(
        self,
        *,
        world_frame: str = "world",
        axis_rotation: np.ndarray | None = None,
        translation_gain: float = 1.0,
        rotation_gain: float = 1.0,
        max_age_s: float = 0.12,
        max_translation_jump_m: float = 0.12,
        max_rotation_jump_rad: float = 0.6,
        max_translation_step_m: float = 0.02,
        max_rotation_step_rad: float = 0.10,
    ) -> None:
        self.world_frame = world_frame
        self.axis = (
            np.eye(3)
            if axis_rotation is None
            else np.asarray(axis_rotation, dtype=np.float64)
        )
        if self.axis.shape != (3, 3) or not np.allclose(
            self.axis.T @ self.axis, np.eye(3), atol=1e-7
        ) or not np.isclose(np.linalg.det(self.axis), 1.0, atol=1e-7):
            raise ValueError("axis_rotation must be a proper rotation")
        if not 0.0 < translation_gain <= 3.0:
            raise ValueError("translation_gain must be in (0,3]")
        if not 0.0 < rotation_gain <= 3.0:
            raise ValueError("rotation_gain must be in (0,3]")
        self.translation_gain = translation_gain
        self.rotation_gain = rotation_gain
        self.max_age_ns = int(max_age_s * 1e9)
        self.max_translation_jump_m = max_translation_jump_m
        self.max_rotation_jump_rad = max_rotation_jump_rad
        self.max_translation_step_m = max_translation_step_m
        self.max_rotation_step_rad = max_rotation_step_rad
        self._last: TrackingSample | None = None
        self._deadman = False
        self._hold = False

    def update(
        self,
        sample: TrackingSample | None,
        *,
        deadman: bool,
        now_ns: int | None = None,
    ) -> TeleopDelta:
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        if not deadman:
            self._deadman = False
            self._last = sample
            self._hold = False
            return TeleopDelta.identity(active=False, reason="clutch_released")
        if self._hold:
            return TeleopDelta.identity(
                active=False, hold=True, reason="tracking_hold_latched"
            )
        reason = self._invalid_reason(sample, now)
        if reason:
            self._hold = True
            return TeleopDelta.identity(active=False, hold=True, reason=reason)
        assert sample is not None
        if not self._deadman or self._last is None:
            self._deadman = True
            self._last = sample
            return TeleopDelta.identity(active=True, reason="anchor_captured")
        if (
            sample.sequence < self._last.sequence
            or (
                sample.source_time_ns
                and self._last.source_time_ns
                and sample.source_time_ns < self._last.source_time_ns
            )
        ):
            self._hold = True
            return TeleopDelta.identity(
                active=False, hold=True, reason="tracking_time_or_sequence_reversed"
            )
        if (
            sample.sequence == self._last.sequence
            or (
                sample.source_time_ns
                and self._last.source_time_ns
                and sample.source_time_ns == self._last.source_time_ns
            )
        ):
            # The 60 Hz timer can run before a new 60 Hz PoseArray arrives.
            # A still-fresh duplicate is an active identity command, not a fault.
            return TeleopDelta.identity(active=True, reason="tracking_repeat")
        values: dict[str, tuple[np.ndarray, np.ndarray, bool]] = {}
        for side in ("left", "right"):
            current = getattr(sample, side)
            previous = getattr(self._last, side)
            raw_translation = current.position - previous.position
            raw_rotation = current.rotation @ previous.rotation.T
            translation_jump = float(np.linalg.norm(raw_translation))
            rotation_jump = geodesic_distance_rad(raw_rotation, np.eye(3))
            if translation_jump > self.max_translation_jump_m:
                self._hold = True
                return TeleopDelta.identity(
                    active=False,
                    hold=True,
                    reason=f"{side}_tracking_translation_jump",
                )
            if rotation_jump > self.max_rotation_jump_rad:
                self._hold = True
                return TeleopDelta.identity(
                    active=False, hold=True, reason=f"{side}_tracking_rotation_jump"
                )
            mapped_translation = (
                self.translation_gain * self.axis @ raw_translation
            )
            mapped_rotation = self.axis @ raw_rotation @ self.axis.T
            translation_step = float(np.linalg.norm(mapped_translation))
            limited = False
            if translation_step > self.max_translation_step_m:
                mapped_translation *= self.max_translation_step_m / translation_step
                limited = True
            rotation_vector = self.rotation_gain * matrix_to_rotvec(
                mapped_rotation
            )
            mapped_rotation = rotvec_to_matrix(rotation_vector)
            rotation_step = float(np.linalg.norm(rotation_vector))
            if rotation_step > self.max_rotation_step_rad:
                mapped_rotation = rotvec_to_matrix(
                    rotation_vector * (self.max_rotation_step_rad / rotation_step)
                )
                limited = True
            values[side] = (
                mapped_translation,
                matrix_to_rotation6d(mapped_rotation),
                limited,
            )
        self._last = sample
        return TeleopDelta(
            values["left"][0],
            values["left"][1],
            values["right"][0],
            values["right"][1],
            active=True,
            hold_latched=False,
            reason=(
                "mapped_step_limited"
                if values["left"][2] or values["right"][2]
                else "mapped"
            ),
        )

    def _invalid_reason(
        self, sample: TrackingSample | None, now_ns: int
    ) -> str:
        if sample is None:
            return "tracking_absent"
        if sample.frame_id.strip().lstrip("/") != self.world_frame.strip().lstrip("/"):
            return "tracking_frame_mismatch"
        if now_ns - sample.receive_monotonic_ns > self.max_age_ns:
            return "tracking_stale"
        if sample.sequence < 0:
            return "tracking_sequence_invalid"
        return ""
