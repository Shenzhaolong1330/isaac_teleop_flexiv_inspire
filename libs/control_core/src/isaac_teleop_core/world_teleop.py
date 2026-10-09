"""Versioned fixed-space teleop, calibration and transactional target manager.

All poses are xyz + xyzw, metres, in the named world. No hardware imports.
The bridge is the sole execution owner. A candidate is never an acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import uuid

import numpy as np
import yaml

from .rotation6d import (
    quaternion_xyzw_to_matrix,
    matrix_to_quaternion_xyzw,
    matrix_to_rotvec,
    rotvec_to_matrix,
    matrix_to_rotation6d,
)
from .command import CommandPoint

SIDES = ("left", "right")


def vector(value, n):
    a = np.asarray(value, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(f"expected {n} finite values")
    return a


@dataclass(frozen=True)
class Pose:
    p: np.ndarray
    r: np.ndarray

    @classmethod
    def read(cls, value):
        a = vector(value, 7)
        return cls(a[:3].copy(), quaternion_xyzw_to_matrix(a[3:]))

    def wire(self):
        return [*self.p.tolist(), *matrix_to_quaternion_xyzw(self.r).tolist()]

    def compose(self, other):
        return Pose(self.p + self.r @ other.p, self.r @ other.r)

    def inverse(self):
        return Pose(-self.r.T @ self.p, self.r.T)


def error(a, b):
    return float(np.linalg.norm(a.p - b.p)), float(
        np.linalg.norm(matrix_to_rotvec(a.r @ b.r.T))
    )


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def installation(raw):
    """Validate explicitly measured/manual parameters; a static capture cannot solve them."""
    if raw.get("schema_version") != 1 or raw.get("mode") != "world_absolute":
        raise ValueError("world_absolute installation schema_version must be 1")
    for name in (
        "input_backend",
        "tracking_origin_id",
        "input_session_id",
        "tool_identity",
        "world_frame",
    ):
        if not isinstance(raw.get(name), str) or not raw[name].strip():
            raise ValueError(f"{name} must be explicit")
    if raw["world_frame"] != "world":
        raise ValueError("stage A requires robot world frame named world")
    if not isinstance(raw.get("origin_reported"), bool):
        raise ValueError("origin_reported capability must be explicit")
    axis = np.asarray(raw["axis_rotation"], dtype=float)
    if (
        axis.shape != (3, 3)
        or not np.isfinite(axis).all()
        or not np.allclose(axis.T @ axis, np.eye(3), atol=1e-7)
        or not np.isclose(np.linalg.det(axis), 1)
    ):
        raise ValueError("axis_rotation must be SO(3)")
    scale = float(raw["translation_scale"])
    if not math.isfinite(scale) or not 0 < scale <= 3:
        raise ValueError("translation_scale outside (0,3]")
    for name in ("world_from_base", "controller_from_reference", "tcp_from_reference"):
        if set(raw[name]) != set(SIDES):
            raise ValueError(f"{name} requires both sides")
        for side in SIDES:
            Pose.read(raw[name][side])
    for name in (
        "age_s",
        "pair_skew_s",
        "stable_position_m",
        "stable_rotation_rad",
        "relative_position_m",
        "relative_rotation_rad",
    ):
        if (
            not math.isfinite(float(raw["calibration_limits"][name]))
            or raw["calibration_limits"][name] <= 0
        ):
            raise ValueError(f"invalid calibration_limits.{name}")
    return raw


def robust_pose(values, p_limit, r_limit):
    poses = [Pose.read(v) for v in values]
    p = np.median([v.p for v in poses], axis=0)
    # Chordal SO(3) estimate, quaternion sign invariant; no Euler averaging.
    u, _, vt = np.linalg.svd(sum(v.r for v in poses))
    r = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    result = Pose(p, r)
    if any(
        error(v, result)[0] > p_limit or error(v, result)[1] > r_limit for v in poses
    ):
        raise ValueError("reference capture is unstable")
    return result


def calibrate(raw, samples):
    installation(raw)
    limits = raw["calibration_limits"]
    if len(samples) < 10:
        raise ValueError("at least 10 paired reference samples required")
    human, robot = {}, {}
    previous = -1
    previous_source = -1
    for row in samples:
        t = int(row["capture_ns"])
        if t <= previous:
            raise ValueError("capture clock repeated or reversed")
        previous = t
        if "human_source_ns" in row:
            source_stamp = int(row["human_source_ns"])
            if source_stamp <= max(0, previous_source):
                raise ValueError("reference source clock frozen or reversed")
            previous_source = source_stamp
        if row["identity"] != identity(raw):
            raise ValueError("tracking identity changed during capture")
        times = [int(row[k][s + "_time_ns"]) for k in ("human", "robot") for s in SIDES]
        if any(t - v < 0 or t - v > limits["age_s"] * 1e9 for v in times):
            raise ValueError("stale/future reference")
        if max(times) - min(times) > limits["pair_skew_s"] * 1e9:
            raise ValueError("reference samples are not paired")
        for side in SIDES:
            # Validate every rotation, including samples later rejected as outliers.
            Pose.read(row["human"][side])
            Pose.read(row["robot"][side])
    if samples[-1]["capture_ns"] - samples[0]["capture_ns"] < 200_000_000:
        raise ValueError("reference duration must be at least 0.2 s")
    for side in SIDES:
        human[side] = robust_pose(
            [v["human"][side] for v in samples],
            limits["stable_position_m"],
            limits["stable_rotation_rad"],
        ).wire()
        robot[side] = robust_pose(
            [v["robot"][side] for v in samples],
            limits["stable_position_m"],
            limits["stable_rotation_rad"],
        ).wire()
    result = {
        "schema_version": 1,
        "installation_sha256": digest(raw),
        "identity": identity(raw),
        "mapping_epoch": str(uuid.uuid4()),
        "human_reference": human,
        "robot_reference": robot,
        "capture_count": len(samples),
        "duration_ns": samples[-1]["capture_ns"] - samples[0]["capture_ns"],
    }
    # One common similarity translation. Two independent zero points are disallowed.
    mapper = FixedMapping(raw, result, check_relative=False)
    offsets = [
        mapper.robot_ref[s].p - mapper.scale * mapper.axis @ mapper.human_ref[s].p
        for s in SIDES
    ]
    if np.linalg.norm(offsets[0] - offsets[1]) > limits["relative_position_m"]:
        raise ValueError(
            "left/right reference geometry inconsistent with common similarity transform"
        )
    result["world_translation_m"] = np.mean(offsets, axis=0).tolist()
    mapper = FixedMapping(raw, result)
    return result


def identity(raw):
    return {
        k: raw[k]
        for k in (
            "input_backend",
            "tracking_origin_id",
            "input_session_id",
            "tool_identity",
        )
    }


class FixedMapping:
    def __init__(self, raw, calibration, *, check_relative=True):
        self.raw = installation(raw)
        self.calibration = calibration
        if calibration.get("schema_version") != 1 or not calibration.get(
            "mapping_epoch"
        ):
            raise ValueError("invalid calibration version/epoch")
        if calibration["installation_sha256"] != digest(raw) or calibration[
            "identity"
        ] != identity(raw):
            raise ValueError("calibration installation/identity invalidated")
        self.axis = np.asarray(raw["axis_rotation"], dtype=float)
        self.scale = float(raw["translation_scale"])
        self.human_ref, self.robot_ref = {}, {}
        for side in SIDES:
            self.human_ref[side] = Pose.read(
                calibration["human_reference"][side]
            ).compose(Pose.read(raw["controller_from_reference"][side]))
            self.robot_ref[side] = Pose.read(
                calibration["robot_reference"][side]
            ).compose(Pose.read(raw["tcp_from_reference"][side]))
        self.translation = vector(calibration.get("world_translation_m", [0, 0, 0]), 3)
        self.orientation_offset = {
            s: self.human_ref[s].r.T @ self.axis.T @ self.robot_ref[s].r for s in SIDES
        }
        if check_relative:
            limits = raw["calibration_limits"]
            for s in SIDES:
                if (
                    np.linalg.norm(
                        self.robot_ref[s].p
                        - (
                            self.translation
                            + self.scale * self.axis @ self.human_ref[s].p
                        )
                    )
                    > limits["relative_position_m"]
                ):
                    raise ValueError("inconsistent common-space reference")
            if (
                np.linalg.norm(
                    matrix_to_rotvec(
                        self.orientation_offset["left"]
                        @ self.orientation_offset["right"].T
                    )
                )
                > limits["relative_rotation_rad"]
            ):
                raise ValueError("tool reference orientation offsets inconsistent")

    @classmethod
    def load(cls, config_path, calibration_path):
        return cls(
            yaml.safe_load(Path(config_path).read_text()),
            yaml.safe_load(Path(calibration_path).read_text()),
        )

    def targets(self, tracking):
        if tracking["identity"] != identity(self.raw):
            raise ValueError(
                "tracking identity changed: explicit recalibration required"
            )
        result = {}
        for s in SIDES:
            h = Pose.read(tracking[s]).compose(
                Pose.read(self.raw["controller_from_reference"][s])
            )
            ref = Pose(
                self.translation + self.scale * self.axis @ h.p,
                self.axis @ h.r @ self.orientation_offset[s],
            )
            result[s] = ref.compose(
                Pose.read(self.raw["tcp_from_reference"][s]).inverse()
            )
        return result


@dataclass(frozen=True)
class Limits:
    linear_speed: float = 0.05
    angular_speed: float = 0.2
    linear_acceleration: float = 0.2
    angular_acceleration: float = 0.5
    position_lead: float = 0.02
    rotation_lead: float = 0.08
    max_dt: float = 0.1
    tracking_age: float = 0.12
    feedback_age: float = 0.1
    tracking_jump: float = 0.1
    tracking_rotation_jump: float = 0.5
    align_position: float = 0.005
    align_rotation: float = 0.03
    human_motion_position: float = 0.01
    human_motion_rotation: float = 0.05
    target_speed: float = 0.02
    target_angular_speed: float = 0.1
    dwell: float = 0.5
    timeout: float = 20.0
    blocked_timeout: float = 2.0
    finger_takeover_tolerance: float = 10.0
    finger_lead_units: float = 10.0
    jerk: str = "unsupported"

    def __post_init__(self):
        for key, value in vars(self).items():
            if key == "jerk":
                if value != "unsupported":
                    raise ValueError("jerk limiting is not supported")
            elif not math.isfinite(value) or value <= 0:
                raise ValueError(f"positive finite {key} required")


class ApprovedRegion:
    """Site-approved convex TCP region; NOT a whole-arm collision checker.

    Approval identifies the mapping epoch and tool. Site must validate every
    approach in this region with the actual robot, links and environment.
    """

    def __init__(self, raw=None):
        self.raw = raw or {}
        if self.raw:
            for s in SIDES:
                lo, hi = (
                    vector(self.raw[s]["lower_m"], 3),
                    vector(self.raw[s]["upper_m"], 3),
                )
                if np.any(lo >= hi):
                    raise ValueError("invalid approved region")

    def approve(self, start, end, mapping):
        if (
            not self.raw.get("approved_by")
            or self.raw.get("mapping_epoch") != mapping.calibration["mapping_epoch"]
            or self.raw.get("tool_identity") != mapping.raw["tool_identity"]
        ):
            return False
        for s in SIDES:
            for pose in (start[s], end[s]):
                if np.any(pose.p < self.raw[s]["lower_m"]) or np.any(
                    pose.p > self.raw[s]["upper_m"]
                ):
                    return False
        return True


def bounded(v, maximum):
    length = np.linalg.norm(v)
    return v * min(1.0, maximum / max(length, 1e-15))


class WorldController:
    """Bridge-owned desired/limited/accepted/measured and non-resuming state.

    Single in-flight proposal; caller serializes propose/send/ack and epoch changes.
    Long dt, source anomalies, sustained leading and moved alignment snapshots
    latch HOLD. Explicit start recaptures execution, never mapping.
    """

    def __init__(self, mapping, limits=None, path=None):
        self.mapping = mapping
        self.limits = limits or Limits()
        self.path = path or ApprovedRegion()
        self.state, self.reason = "PREVIEW", "session_not_started"
        self.execution_epoch = ""
        self.accepted, self.limited, self.desired, self.measured = {}, {}, {}, {}
        self.measured_q = {}
        self.velocity = {s: (np.zeros(3), np.zeros(3)) for s in SIDES}
        self.pending = None
        self.last_ns = None
        self.last_tracking = None
        self.fingers = False
        self.started_ns = 0
        self.dwell_ns = None
        self.blocked_ns = None
        self.generation = None
        self._source_stamp = None
        self._source_seen_ns = None
        self._mapping_invalid = False

    @property
    def authority(self):
        return self.state in {"ALIGNING", "FOLLOW"}

    def hold(self, reason, *, invalid=False):
        self._mapping_invalid |= invalid
        invalid = self._mapping_invalid
        self.state, self.reason = ("INVALID" if invalid else "HOLD"), reason
        self.pending = None
        self.fingers = False
        self.velocity = {s: (np.zeros(3), np.zeros(3)) for s in SIDES}

    def home(self):
        self.hold("home_exclusive")
        self.state = "HOMING"
        self.accepted = {}
        self.execution_epoch = ""

    def home_complete(self, success):
        self.hold("home_complete_explicit_start_required" if success else "home_failed")

    def start(self, tracking, measured, generation, now_ns):
        if self.state in {"HOMING", "INVALID"} or self._mapping_invalid:
            raise ValueError("Home/invalid mapping requires recovery or recalibration")
        self._validate(tracking, measured, now_ns)
        desired = self.mapping.targets(tracking)
        actual = {s: Pose.read(measured[s]) for s in SIDES}
        if not self.path.approve(actual, desired, self.mapping):
            self.hold("approach_path_unapproved")
            raise ValueError(self.reason)
        self.execution_epoch = str(uuid.uuid4())
        self.generation = generation
        self.accepted = actual.copy()
        self.limited = actual.copy()
        self.desired, self.measured = desired, actual
        self.measured_q = measured.get("q", {})
        self.snapshot = desired.copy()
        self.last_tracking = tracking.copy()
        self.last_ns = now_ns
        self.started_ns = now_ns
        self.dwell_ns = self.blocked_ns = None
        self.pending = None
        self.velocity = {s: (np.zeros(3), np.zeros(3)) for s in SIDES}
        self.state, self.reason = "ALIGNING", "captured_fixed_target"
        self.fingers = False

    def _validate(self, tracking, measured, now):
        if tracking["identity"] != identity(self.mapping.raw):
            self.hold("tracking_identity_changed", invalid=True)
            raise ValueError(self.reason)
        for t, limit in (
            (tracking["receive_ns"], self.limits.tracking_age),
            (measured["receive_ns"], self.limits.feedback_age),
        ):
            if not 0 <= now - int(t) <= limit * 1e9:
                raise ValueError("tracking_or_feedback_stale_future")
        if tracking["frame_id"] != self.mapping.raw["world_frame"]:
            raise ValueError("tracking_frame_mismatch")
        stamp = int(tracking["source_ns"])
        if stamp <= 0:
            raise ValueError("tracking_source_time_missing")
        if stamp != self._source_stamp:
            self._source_stamp, self._source_seen_ns = stamp, now
        elif now - self._source_seen_ns > self.limits.tracking_age * 1e9:
            raise ValueError("tracking_source_clock_frozen")
        for s in SIDES:
            Pose.read(tracking[s])
            Pose.read(measured[s])
        if self.last_tracking is not None:
            old = self.last_tracking
            if (
                int(tracking["receive_ns"]) - int(old["receive_ns"])
                > self.limits.tracking_age * 1e9
            ):
                self.hold("input_stream_gap_requires_recalibration", invalid=True)
                raise ValueError(self.reason)
            if (
                tracking["sequence"] < old["sequence"]
                or tracking["source_ns"] < old["source_ns"]
            ):
                self.hold("input_restart_or_clock_reversed", invalid=True)
                raise ValueError(self.reason)
            if tracking["sequence"] != old["sequence"]:
                for s in SIDES:
                    p, r = error(Pose.read(tracking[s]), Pose.read(old[s]))
                    if (
                        p > self.limits.tracking_jump
                        or r > self.limits.tracking_rotation_jump
                    ):
                        self.hold("detectable_tracking_relocation", invalid=True)
                        raise ValueError(self.reason)
            elif any(
                error(Pose.read(tracking[s]), Pose.read(old[s])) != (0.0, 0.0)
                for s in SIDES
            ):
                # Numerically equal rotations can have tiny log errors; compare wire directly.
                if any(not np.allclose(tracking[s], old[s], atol=1e-12) for s in SIDES):
                    raise ValueError("duplicate_sequence_changed_pose")

    def propose(self, tracking, measured, generation, now_ns):
        if not self.authority:
            raise ValueError(self.reason)
        if self.pending is not None:
            raise ValueError("proposal already in flight")
        try:
            self._validate(tracking, measured, now_ns)
            if generation != self.generation:
                raise ValueError("execution_generation_changed")
            dt = (now_ns - self.last_ns) / 1e9
            if dt <= 0 or dt > self.limits.max_dt:
                raise ValueError("control_clock_or_long_gap")
            self.measured = {s: Pose.read(measured[s]) for s in SIDES}
            self.measured_q = measured.get("q", {})
            latest = self.mapping.targets(tracking)
            changing = False
            for s in SIDES:
                p, r = error(latest[s], self.desired[s])
                changing |= (
                    p / dt > self.limits.target_speed
                    or r / dt > self.limits.target_angular_speed
                )
            self.desired = latest
            if self.state == "ALIGNING":
                if (now_ns - self.started_ns) / 1e9 > self.limits.timeout:
                    raise ValueError("align_timeout")
                if any(
                    error(latest[s], self.snapshot[s])[0]
                    > self.limits.human_motion_position
                    or error(latest[s], self.snapshot[s])[1]
                    > self.limits.human_motion_rotation
                    for s in SIDES
                ):
                    raise ValueError("human_moved_during_alignment")
                target = self.snapshot
            else:
                target = latest
            if not self.path.approve(self.accepted, target, self.mapping):
                raise ValueError("path_or_workspace_rejected")
            lead = any(
                error(self.accepted[s], self.measured[s])[0]
                >= self.limits.position_lead
                or error(self.accepted[s], self.measured[s])[1]
                >= self.limits.rotation_lead
                for s in SIDES
            )
            if lead:
                self.blocked_ns = self.blocked_ns or now_ns
                if (now_ns - self.blocked_ns) / 1e9 >= self.limits.blocked_timeout:
                    raise ValueError("sustained_tracking_obstruction")
                # No integration during obstruction. Restart velocity from zero.
                self.velocity = {s: (np.zeros(3), np.zeros(3)) for s in SIDES}
                candidates = self.accepted.copy()
                velocities = self.velocity.copy()
            else:
                candidate_blocked = False
                candidates, velocities = {}, {}
                for s in SIDES:
                    current = self.accepted[s]
                    dp = target[s].p - current.p
                    dr = matrix_to_rotvec(target[s].r @ current.r.T)
                    vp, vr = self.velocity[s]
                    vp = vp + bounded(
                        bounded(4.0 * dp, self.limits.linear_speed) - vp,
                        self.limits.linear_acceleration * dt,
                    )
                    vr = vr + bounded(
                        bounded(4.0 * dr, self.limits.angular_speed) - vr,
                        self.limits.angular_acceleration * dt,
                    )
                    # Damped approach; acceleration bounds apply to integrated
                    # velocity, including endpoint deceleration (no endpoint snap).
                    step_p, step_r = vp * dt, vr * dt
                    candidate = Pose(
                        current.p + step_p, rotvec_to_matrix(step_r) @ current.r
                    )
                    if (
                        error(candidate, self.measured[s])[0]
                        > self.limits.position_lead
                        or error(candidate, self.measured[s])[1]
                        > self.limits.rotation_lead
                    ):
                        candidate = current
                        candidate_blocked = True
                        vp, vr = np.zeros(3), np.zeros(3)
                    candidates[s], velocities[s] = candidate, (vp, vr)
                if candidate_blocked:
                    self.blocked_ns = self.blocked_ns or now_ns
                    if now_ns - self.blocked_ns >= self.limits.blocked_timeout * 1e9:
                        raise ValueError("sustained_tracking_obstruction")
                else:
                    self.blocked_ns = None
            converged = all(
                error(self.measured[s], latest[s])[0] <= self.limits.align_position
                and error(self.measured[s], latest[s])[1] <= self.limits.align_rotation
                and error(self.accepted[s], latest[s])[0] <= self.limits.align_position
                and error(self.accepted[s], latest[s])[1] <= self.limits.align_rotation
                for s in SIDES
            )
            if not self.path.approve(self.accepted, candidates, self.mapping):
                raise ValueError("limited_path_or_workspace_rejected")
            if self.state == "ALIGNING" and converged and not changing:
                self.dwell_ns = self.dwell_ns or now_ns
                if now_ns - self.dwell_ns >= self.limits.dwell * 1e9:
                    self.state, self.reason = (
                        "FOLLOW",
                        "bilateral_latest_target_converged",
                    )
            else:
                self.dwell_ns = None
            self.last_ns, self.last_tracking = now_ns, tracking.copy()
            self.limited = candidates
            self.pending = (candidates, velocities)
            kwargs = {}
            for s in SIDES:
                kwargs[s + "_delta_xyz"] = candidates[s].p - self.accepted[s].p
                kwargs[s + "_delta_rotation6d"] = matrix_to_rotation6d(
                    candidates[s].r @ self.accepted[s].r.T
                )
                kwargs[s + "_hand_targets"] = np.zeros(6)
            return CommandPoint(execute_after_s=0.0, **kwargs)
        except Exception as exc:
            if self.state != "INVALID":
                self.hold(str(exc))
            raise

    def acknowledge(self, accepted_sides, *, complete, proposal=None):
        # A watchdog can revoke authority during a blocking SDK call. Its
        # returned evidence still commits the exact proposal that was sent.
        proposal = self.pending if proposal is None else proposal
        if proposal is None:
            raise ValueError("no in-flight proposal")
        candidates, velocities = proposal
        for s in accepted_sides:
            self.accepted[s] = candidates[s]
            self.velocity[s] = (
                velocities[s] if self.authority else (np.zeros(3), np.zeros(3))
            )
        self.pending = None
        if not complete or set(accepted_sides) != set(SIDES):
            self.hold("partial_or_failed_execution_rebuild_required")

    def audit(self):
        return {
            "schema": "world-teleop-v1",
            "state": self.state,
            "reason": self.reason,
            "mapping_epoch": self.mapping.calibration["mapping_epoch"],
            "execution_epoch": self.execution_epoch,
            "fingers_taken_over": self.fingers,
            "measured_q": self.measured_q,
            **{
                k: {s: v.wire() for s, v in getattr(self, k).items()}
                for k in ("desired", "limited", "accepted", "measured")
            },
        }
