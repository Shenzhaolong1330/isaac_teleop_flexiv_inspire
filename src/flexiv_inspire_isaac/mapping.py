"""Safe absolute-pose to existing Flexiv delta-action mapping.

The existing robot wrapper accumulates each Cartesian delta into an internal
command target.  This mapper therefore anchors the absolute XR poses on clutch,
maintains its own commanded relative target, and only advances that target after
an acknowledgement from the hardware gateway.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .config import MappingConfig, SafetyConfig, SideMapping
from .math3d import (
    clip_norm,
    conjugate_rotation_by_matrix,
    normalize_quaternion_xyzw,
    quat_inverse_xyzw,
    quat_multiply_xyzw,
    quat_to_rotvec,
    relative_quaternion_xyzw,
    rotvec_to_quat,
    scaled_quaternion_xyzw,
)

SIDES = ("left", "right")
AXES = ("x", "y", "z", "rx", "ry", "rz")


class BridgeState(str, Enum):
    READY = "ready"
    ARMING = "arming"
    WAIT_ARM_ACK = "wait_arm_ack"
    ACTIVE = "active"
    HOLD_LATCHED = "hold_latched"


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    orientation_xyzw: np.ndarray

    @classmethod
    def from_values(cls, position: Any, orientation_xyzw: Any) -> "Pose":
        pos = np.asarray(position, dtype=float).reshape(-1)
        if pos.shape != (3,) or not np.all(np.isfinite(pos)):
            raise ValueError("position must contain three finite values")
        return cls(
            position=pos.copy(),
            orientation_xyzw=normalize_quaternion_xyzw(orientation_xyzw),
        )


@dataclass(frozen=True)
class PosePairSample:
    left: Pose
    right: Pose
    sequence: int
    receive_monotonic_ns: int
    source_stamp_ns: int
    frame_id: str
    valid: bool


@dataclass(frozen=True)
class BridgeDecision:
    kind: str
    state: BridgeState
    epoch: int
    proposal_id: int | None = None
    requested_action: dict[str, float | bool] | None = None
    applied_action: dict[str, float | bool] | None = None
    reason: str = ""


@dataclass
class _SideRuntime:
    anchor: Pose
    commanded_position: np.ndarray
    commanded_orientation_xyzw: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray


@dataclass(frozen=True)
class _PendingProposal:
    proposal_id: int
    epoch: int
    created_ns: int
    left_position_step: np.ndarray
    left_rotation_step: np.ndarray
    right_position_step: np.ndarray
    right_rotation_step: np.ndarray
    left_linear_velocity: np.ndarray
    left_angular_velocity: np.ndarray
    right_linear_velocity: np.ndarray
    right_angular_velocity: np.ndarray


def action_from_vectors(
    left_position: Any,
    left_rotation: Any,
    right_position: Any,
    right_rotation: Any,
    *,
    teleop_enabled: bool = True,
) -> dict[str, float | bool]:
    action: dict[str, float | bool] = {}
    for side, position, rotation in (
        ("left", left_position, left_rotation),
        ("right", right_position, right_rotation),
    ):
        values = np.concatenate(
            [
                np.asarray(position, dtype=float).reshape(3),
                np.asarray(rotation, dtype=float).reshape(3),
            ]
        )
        for axis, value in zip(AXES, values, strict=True):
            action[f"{side}_delta_ee_pose.{axis}"] = float(value)
    action["teleop_enable_pressed"] = bool(teleop_enabled)
    return action


def action_vectors(
    action: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vectors: list[np.ndarray] = []
    for side in SIDES:
        values = np.array(
            [float(action[f"{side}_delta_ee_pose.{axis}"]) for axis in AXES],
            dtype=float,
        )
        vectors.extend((values[:3], values[3:]))
    return vectors[0], vectors[1], vectors[2], vectors[3]


class DualArmAbsoluteMapper:
    """Stateful, acknowledgement-aware dual-arm mapper."""

    def __init__(self, mapping: MappingConfig, safety: SafetyConfig):
        self.mapping = mapping
        self.safety = safety
        self.state = BridgeState.READY
        self.deadman_pressed = False
        self.epoch = 0
        self._next_proposal_id = 1
        self._sample: PosePairSample | None = None
        self._last_ingested_sample: PosePairSample | None = None
        self._valid_streak = 0
        self._arm_after_sequence = -1
        self._arm_request_ns: int | None = None
        self._last_control_ns: int | None = None
        self._anchor_frame = ""
        self._runtime: dict[str, _SideRuntime] = {}
        self._pending: _PendingProposal | None = None
        self._hold_pending_reason: str | None = None
        self.last_reason = "startup"

    @property
    def pending_proposal_id(self) -> int | None:
        return None if self._pending is None else self._pending.proposal_id

    def ingest_sample(self, sample: PosePairSample) -> None:
        if sample.sequence < 0:
            raise ValueError("sample sequence must be non-negative")
        if sample.receive_monotonic_ns <= 0:
            raise ValueError("sample receive_monotonic_ns must be positive")

        previous = self._last_ingested_sample
        if previous is not None:
            if sample.sequence <= previous.sequence:
                return
            if (
                sample.source_stamp_ns > 0
                and previous.source_stamp_ns > 0
                and sample.source_stamp_ns < previous.source_stamp_ns
                and self.state
                in {
                    BridgeState.ARMING,
                    BridgeState.WAIT_ARM_ACK,
                    BridgeState.ACTIVE,
                }
            ):
                self._latch_hold("ROS/source time moved backwards")
            if (
                sample.frame_id
                and previous.frame_id
                and sample.frame_id != previous.frame_id
                and self.state
                in {BridgeState.WAIT_ARM_ACK, BridgeState.ACTIVE}
            ):
                self._latch_hold(
                    f"tracking frame changed: {previous.frame_id!r} -> "
                    f"{sample.frame_id!r}"
                )
            if (
                sample.valid
                and previous.valid
                and self.state
                in {
                    BridgeState.ARMING,
                    BridgeState.WAIT_ARM_ACK,
                    BridgeState.ACTIVE,
                }
            ):
                jump_reason = self._tracking_jump_reason(previous, sample)
                if jump_reason:
                    self._latch_hold(jump_reason)

        self._sample = sample
        self._last_ingested_sample = sample
        if sample.valid:
            self._valid_streak += 1
        else:
            self._valid_streak = 0

    def set_deadman(self, pressed: bool, now_ns: int) -> None:
        pressed = bool(pressed)
        if pressed == self.deadman_pressed:
            return
        was_engaged = self.state in {
            BridgeState.ARMING,
            BridgeState.WAIT_ARM_ACK,
            BridgeState.ACTIVE,
            BridgeState.HOLD_LATCHED,
        }
        self.deadman_pressed = pressed
        if not pressed:
            if was_engaged:
                self._hold_pending_reason = "deadman released"
            self._clear_motion_state()
            self.state = BridgeState.READY
            self.last_reason = "deadman released"
            return

        if self.state == BridgeState.HOLD_LATCHED:
            return
        self.state = BridgeState.ARMING
        self._arm_after_sequence = (
            self._sample.sequence if self._sample is not None else -1
        )
        self._valid_streak = 0
        self._arm_request_ns = None
        self._last_control_ns = int(now_ns)
        self.last_reason = "waiting for a fresh post-deadman tracking sample"

    def force_hold(self, reason: str) -> None:
        self._latch_hold(str(reason))

    def tick(self, now_ns: int) -> BridgeDecision | None:
        now_ns = int(now_ns)
        if self._hold_pending_reason is not None:
            reason = self._hold_pending_reason
            self._hold_pending_reason = None
            return BridgeDecision(
                kind="hold",
                state=self.state,
                epoch=self.epoch,
                reason=reason,
            )

        if self.state == BridgeState.READY:
            return None
        if self.state == BridgeState.HOLD_LATCHED:
            return None
        if self.state == BridgeState.WAIT_ARM_ACK:
            if (
                self._arm_request_ns is not None
                and (now_ns - self._arm_request_ns) * 1e-9
                > self.safety.ack_timeout_s
            ):
                self._latch_hold("gateway arm acknowledgement timed out")
                return self.tick(now_ns)
            return None
        if self._pending is not None:
            if (
                (now_ns - self._pending.created_ns) * 1e-9
                > self.safety.ack_timeout_s
            ):
                self._latch_hold("gateway action acknowledgement timed out")
                return self.tick(now_ns)
            return None

        sample_reason = self._sample_invalid_reason(now_ns)
        if sample_reason:
            if self.state == BridgeState.ARMING:
                self.last_reason = sample_reason
                return None
            self._latch_hold(sample_reason)
            return self.tick(now_ns)

        if self.state == BridgeState.ARMING:
            assert self._sample is not None
            if self._sample.sequence <= self._arm_after_sequence:
                return None
            if self._valid_streak < self.safety.warmup_samples:
                self.last_reason = (
                    f"tracking warmup {self._valid_streak}/"
                    f"{self.safety.warmup_samples}"
                )
                return None
            self.epoch += 1
            self._anchor_frame = self._sample.frame_id
            self._runtime = {
                "left": self._new_side_runtime(self._sample.left),
                "right": self._new_side_runtime(self._sample.right),
            }
            self._arm_request_ns = now_ns
            self.state = BridgeState.WAIT_ARM_ACK
            self.last_reason = "waiting for gateway to pin the current robot pose"
            return BridgeDecision(
                kind="arm",
                state=self.state,
                epoch=self.epoch,
                reason="fresh tracking anchor captured; first action is zero",
            )

        if self.state != BridgeState.ACTIVE:
            return None
        if self._last_control_ns is None:
            self._last_control_ns = now_ns
            return None
        dt = (now_ns - self._last_control_ns) * 1e-9
        if dt <= 0:
            self._latch_hold("control monotonic clock did not advance")
            return self.tick(now_ns)
        if dt > self.safety.max_control_dt_s:
            self._latch_hold(
                f"control loop overrun: {dt:.4f}s > "
                f"{self.safety.max_control_dt_s:.4f}s"
            )
            return self.tick(now_ns)

        desired: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        errors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        steps: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        assert self._sample is not None
        for side in SIDES:
            side_map = self.mapping.left if side == "left" else self.mapping.right
            pose = self._sample.left if side == "left" else self._sample.right
            runtime = self._runtime[side]
            desired_position, desired_orientation = self._desired_relative_pose(
                pose, runtime.anchor, side_map
            )
            desired[side] = (desired_position, desired_orientation)
            anchor_translation = float(np.linalg.norm(desired_position))
            anchor_rotation = float(np.linalg.norm(quat_to_rotvec(desired_orientation)))
            if anchor_translation > self.safety.max_anchor_translation_m:
                self._latch_hold(
                    f"{side} anchor translation exceeded: {anchor_translation:.4f}m"
                )
                return self.tick(now_ns)
            if anchor_rotation > self.safety.max_anchor_rotation_rad:
                self._latch_hold(
                    f"{side} anchor rotation exceeded: {anchor_rotation:.4f}rad"
                )
                return self.tick(now_ns)

            position_error = desired_position - runtime.commanded_position
            orientation_error = quat_to_rotvec(
                quat_multiply_xyzw(
                    desired_orientation,
                    quat_inverse_xyzw(runtime.commanded_orientation_xyzw),
                )
            )
            errors[side] = (position_error, orientation_error)
            if (
                float(np.linalg.norm(position_error))
                > self.safety.max_tracking_lag_translation_m
            ):
                self._latch_hold(f"{side} commanded target translation lag too large")
                return self.tick(now_ns)
            if (
                float(np.linalg.norm(orientation_error))
                > self.safety.max_tracking_lag_rotation_rad
            ):
                self._latch_hold(f"{side} commanded target rotation lag too large")
                return self.tick(now_ns)

            position_step, linear_velocity = self._limited_step(
                position_error,
                runtime.linear_velocity,
                dt,
                self.safety.max_linear_speed_m_s,
                self.safety.max_linear_accel_m_s2,
                self.safety.max_translation_step_m,
            )
            rotation_step, angular_velocity = self._limited_step(
                orientation_error,
                runtime.angular_velocity,
                dt,
                self.safety.max_angular_speed_rad_s,
                self.safety.max_angular_accel_rad_s2,
                self.safety.max_rotation_step_rad,
            )
            steps[side] = (
                position_step,
                rotation_step,
                linear_velocity,
                angular_velocity,
            )

        requested_action = action_from_vectors(
            errors["left"][0],
            errors["left"][1],
            errors["right"][0],
            errors["right"][1],
        )
        applied_action = action_from_vectors(
            steps["left"][0],
            steps["left"][1],
            steps["right"][0],
            steps["right"][1],
        )
        moving = any(
            float(np.linalg.norm(value)) > 1e-10
            for side in SIDES
            for value in steps[side][:2]
        )
        if not moving:
            for side in SIDES:
                self._runtime[side].linear_velocity[:] = 0.0
                self._runtime[side].angular_velocity[:] = 0.0
            self._last_control_ns = now_ns
            return BridgeDecision(
                kind="idle",
                state=self.state,
                epoch=self.epoch,
                requested_action=requested_action,
                applied_action=applied_action,
                reason="absolute target reached",
            )

        proposal_id = self._next_proposal_id
        self._next_proposal_id += 1
        self._pending = _PendingProposal(
            proposal_id=proposal_id,
            epoch=self.epoch,
            created_ns=now_ns,
            left_position_step=steps["left"][0],
            left_rotation_step=steps["left"][1],
            right_position_step=steps["right"][0],
            right_rotation_step=steps["right"][1],
            left_linear_velocity=steps["left"][2],
            left_angular_velocity=steps["left"][3],
            right_linear_velocity=steps["right"][2],
            right_angular_velocity=steps["right"][3],
        )
        return BridgeDecision(
            kind="action",
            state=self.state,
            epoch=self.epoch,
            proposal_id=proposal_id,
            requested_action=requested_action,
            applied_action=applied_action,
            reason="bounded step proposed; target advances only after ACK",
        )

    def acknowledge_arm(
        self, *, epoch: int, success: bool, now_ns: int, reason: str = ""
    ) -> None:
        if self.state != BridgeState.WAIT_ARM_ACK or int(epoch) != self.epoch:
            return
        if not success:
            self._latch_hold(reason or "gateway rejected arm")
            return
        self.state = BridgeState.ACTIVE
        self._arm_request_ns = None
        self._last_control_ns = int(now_ns)
        self.last_reason = "active"

    def acknowledge_action(
        self,
        *,
        epoch: int,
        proposal_id: int,
        success: bool,
        now_ns: int,
        reason: str = "",
    ) -> None:
        pending = self._pending
        if (
            pending is None
            or int(epoch) != pending.epoch
            or int(proposal_id) != pending.proposal_id
        ):
            return
        if not success:
            self._pending = None
            self._latch_hold(reason or "gateway rejected action")
            return
        self._commit_side(
            "left",
            pending.left_position_step,
            pending.left_rotation_step,
            pending.left_linear_velocity,
            pending.left_angular_velocity,
        )
        self._commit_side(
            "right",
            pending.right_position_step,
            pending.right_rotation_step,
            pending.right_linear_velocity,
            pending.right_angular_velocity,
        )
        self._pending = None
        self._last_control_ns = int(now_ns)
        self.last_reason = "active"

    def _commit_side(
        self,
        side: str,
        position_step: np.ndarray,
        rotation_step: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> None:
        runtime = self._runtime[side]
        runtime.commanded_position = runtime.commanded_position + position_step
        runtime.commanded_orientation_xyzw = quat_multiply_xyzw(
            rotvec_to_quat(rotation_step), runtime.commanded_orientation_xyzw
        )
        runtime.linear_velocity = linear_velocity.copy()
        runtime.angular_velocity = angular_velocity.copy()

    def _clear_motion_state(self) -> None:
        self._runtime.clear()
        self._pending = None
        self._arm_request_ns = None
        self._last_control_ns = None
        self._anchor_frame = ""
        self._valid_streak = 0

    def _latch_hold(self, reason: str) -> None:
        if self.state != BridgeState.HOLD_LATCHED:
            self._hold_pending_reason = str(reason)
        self._pending = None
        self._arm_request_ns = None
        self.state = BridgeState.HOLD_LATCHED
        self.last_reason = str(reason)

    def _sample_invalid_reason(self, now_ns: int) -> str | None:
        if self._sample is None:
            return "no XR pose sample"
        age_s = (int(now_ns) - self._sample.receive_monotonic_ns) * 1e-9
        if age_s < 0:
            return "XR receive monotonic timestamp is in the future"
        if age_s > self.safety.max_input_age_s:
            return (
                f"XR pose stale: {age_s:.4f}s > "
                f"{self.safety.max_input_age_s:.4f}s"
            )
        if not self._sample.valid:
            return "XR pose validity gate is false"
        if (
            self._anchor_frame
            and self._sample.frame_id
            and self._sample.frame_id != self._anchor_frame
        ):
            return "XR frame differs from armed frame"
        return None

    def _tracking_jump_reason(
        self, previous: PosePairSample, current: PosePairSample
    ) -> str | None:
        for side in SIDES:
            before = previous.left if side == "left" else previous.right
            after = current.left if side == "left" else current.right
            translation = float(np.linalg.norm(after.position - before.position))
            rotation = float(
                np.linalg.norm(
                    quat_to_rotvec(
                        relative_quaternion_xyzw(
                            after.orientation_xyzw, before.orientation_xyzw
                        )
                    )
                )
            )
            if translation > self.safety.tracking_jump_translation_m:
                return f"{side} XR translation jump: {translation:.4f}m"
            if rotation > self.safety.tracking_jump_rotation_rad:
                return f"{side} XR rotation jump: {rotation:.4f}rad"
        return None

    @staticmethod
    def _new_side_runtime(anchor: Pose) -> _SideRuntime:
        return _SideRuntime(
            anchor=anchor,
            commanded_position=np.zeros(3, dtype=float),
            commanded_orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            linear_velocity=np.zeros(3, dtype=float),
            angular_velocity=np.zeros(3, dtype=float),
        )

    @staticmethod
    def _desired_relative_pose(
        current: Pose, anchor: Pose, side_map: SideMapping
    ) -> tuple[np.ndarray, np.ndarray]:
        axis = np.asarray(side_map.axis_rotation, dtype=float)
        position = (
            side_map.translation_gain * axis @ (current.position - anchor.position)
        )
        relative = relative_quaternion_xyzw(
            current.orientation_xyzw, anchor.orientation_xyzw
        )
        mapped = conjugate_rotation_by_matrix(relative, axis)
        orientation = scaled_quaternion_xyzw(mapped, side_map.rotation_gain)
        return position, orientation

    @staticmethod
    def _limited_step(
        error: np.ndarray,
        previous_velocity: np.ndarray,
        dt: float,
        max_speed: float,
        max_acceleration: float,
        hard_step_limit: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        error = np.asarray(error, dtype=float)
        if float(np.linalg.norm(error)) < 1e-10:
            return np.zeros_like(error), np.zeros_like(error)
        target_velocity = clip_norm(error / dt, max_speed)
        velocity_change = clip_norm(
            target_velocity - previous_velocity, max_acceleration * dt
        )
        velocity = previous_velocity + velocity_change
        step = clip_norm(velocity * dt, hard_step_limit)
        if float(np.dot(step, error)) <= 0.0:
            return np.zeros_like(error), np.zeros_like(error)
        if float(np.linalg.norm(step)) > float(np.linalg.norm(error)):
            step = error.copy()
            velocity = step / dt
        return step, velocity

