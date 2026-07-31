"""Fail-closed request dispatcher for the isolated RDK daemon."""

from __future__ import annotations

import json
import hmac
import secrets
import threading
import time
from typing import Any, Callable
import uuid

import numpy as np

from .ft_zero import (
    CONFIRMATION_TOKEN,
    FTZeroManager,
    MaintenanceInterlock,
    ZeroFTRequest,
)
from .ipc import peer_has_local_tty
from isaac_teleop_core.deviceio import record_envelope


class DaemonInterlock(MaintenanceInterlock):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.maintenance_active = False
        self.ready_generation: int | None = None

    def acquire(self, session_id: str) -> None:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("maintenance or command transaction already active")
        self.maintenance_active = True
        self.ready_generation = None

    def release(self, *, success: bool, connection_generation: int | None) -> None:
        self.ready_generation = connection_generation if success else None
        self.maintenance_active = False
        self._lock.release()


class HandObservationCache:
    def __init__(self, max_age_s: float = 0.2) -> None:
        self._max_age_ns = int(max_age_s * 1e9)
        self._positions: tuple[np.ndarray, np.ndarray] | None = None
        self._received_ns = 0
        self._source_sequence = 0
        self._source_time_ns = 0
        self._source_clock_domain = ""
        self._update_count = 0
        self._monitor_reference: tuple[np.ndarray, np.ndarray] | None = None
        self._monitor_max_delta = np.zeros(2, dtype=np.float64)
        self._lock = threading.Lock()

    def update(
        self,
        left: object,
        right: object,
        *,
        source_sequence: int = 0,
        source_time_ns: int = 0,
        source_clock_domain: str = "",
    ) -> None:
        left_array = np.asarray(left, dtype=np.float64).reshape(-1)
        right_array = np.asarray(right, dtype=np.float64).reshape(-1)
        if left_array.shape != (6,) or right_array.shape != (6,):
            raise ValueError("hand observation must contain two 6-vectors")
        if not np.all(np.isfinite(left_array)) or not np.all(np.isfinite(right_array)):
            raise ValueError("hand observation contains NaN or Inf")
        with self._lock:
            self._positions = left_array.copy(), right_array.copy()
            self._received_ns = time.monotonic_ns()
            self._source_sequence = int(source_sequence)
            self._source_time_ns = int(source_time_ns)
            self._source_clock_domain = str(source_clock_domain)
            self._update_count += 1
            if self._monitor_reference is not None:
                deltas = np.array(
                    [
                        np.max(np.abs(left_array - self._monitor_reference[0])),
                        np.max(np.abs(right_array - self._monitor_reference[1])),
                    ]
                )
                self._monitor_max_delta = np.maximum(self._monitor_max_delta, deltas)

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if self._positions is None:
                raise RuntimeError("no hand observation is available")
            if time.monotonic_ns() - self._received_ns > self._max_age_ns:
                raise RuntimeError("hand observation is stale")
            return self._positions[0].copy(), self._positions[1].copy()

    def begin_monitor(self, left: object, right: object) -> None:
        left_array = np.asarray(left, dtype=np.float64).reshape(6)
        right_array = np.asarray(right, dtype=np.float64).reshape(6)
        with self._lock:
            self._monitor_reference = (left_array.copy(), right_array.copy())
            self._monitor_max_delta = np.zeros(2, dtype=np.float64)

    def monitor_snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "active": self._monitor_reference is not None,
                "left_max_delta": float(self._monitor_max_delta[0]),
                "right_max_delta": float(self._monitor_max_delta[1]),
                "last_source_sequence": self._source_sequence,
                "last_source_time_ns": self._source_time_ns,
                "source_clock_domain": self._source_clock_domain,
                "last_host_receive_monotonic_ns": self._received_ns,
                "update_count": self._update_count,
            }

    def end_monitor(self) -> dict[str, object]:
        with self._lock:
            snapshot = {
                "active": self._monitor_reference is not None,
                "left_max_delta": float(self._monitor_max_delta[0]),
                "right_max_delta": float(self._monitor_max_delta[1]),
                "last_source_sequence": self._source_sequence,
                "last_source_time_ns": self._source_time_ns,
                "source_clock_domain": self._source_clock_domain,
                "last_host_receive_monotonic_ns": self._received_ns,
                "update_count": self._update_count,
            }
            self._monitor_reference = None
            return snapshot


class _LocalAuthorization:
    """Short-lived single-use token bound to explicit fields and a local TTY."""

    def __init__(self, lifetime_s: float = 30.0) -> None:
        self._lifetime_ns = int(lifetime_s * 1e9)
        self._tokens: dict[str, tuple[int, tuple[str, ...]]] = {}
        self._lock = threading.Lock()

    def mint(
        self,
        *,
        pid: int,
        confirmation: str,
        expected_confirmation: str,
        binding: tuple[str, ...],
    ) -> tuple[str, int]:
        if not peer_has_local_tty(pid):
            raise PermissionError("authorization requires a same-host local TTY")
        if confirmation != expected_confirmation:
            raise PermissionError("local confirmation token is incorrect")
        if any(not field for field in binding):
            raise ValueError("authorization binding fields cannot be empty")
        token = secrets.token_urlsafe(32)
        expires = time.monotonic_ns() + self._lifetime_ns
        with self._lock:
            self._purge_locked(time.monotonic_ns())
            self._tokens[token] = (expires, binding)
        return token, expires

    def consume(self, token: str, *, binding: tuple[str, ...]) -> None:
        now = time.monotonic_ns()
        with self._lock:
            self._purge_locked(now)
            matching_key = next(
                (candidate for candidate in self._tokens if hmac.compare_digest(candidate, token)),
                None,
            )
            if matching_key is None:
                raise PermissionError("authorization is missing, expired or consumed")
            expires, expected = self._tokens.pop(matching_key)
        if now > expires:
            raise PermissionError("authorization expired")
        if binding != expected:
            raise PermissionError("authorization binding does not match")

    def _purge_locked(self, now: int) -> None:
        for token, record in tuple(self._tokens.items()):
            if now > record[0]:
                del self._tokens[token]


class LocalZeroAuthorization:
    def __init__(self, lifetime_s: float = 30.0) -> None:
        self._inner = _LocalAuthorization(lifetime_s)

    def mint(
        self,
        *,
        pid: int,
        session_id: str,
        confirmation: str,
        tool_payload_config_hash: str,
    ) -> tuple[str, int]:
        if len(tool_payload_config_hash) < 8:
            raise ValueError("tool/payload hash is missing or too short")
        return self._inner.mint(
            pid=pid,
            confirmation=confirmation,
            expected_confirmation=CONFIRMATION_TOKEN,
            binding=(session_id, tool_payload_config_hash),
        )

    def consume(self, token: str, *, session_id: str, tool_payload_config_hash: str) -> None:
        self._inner.consume(token, binding=(session_id, tool_payload_config_hash))


class LocalControlAuthorization:
    CONFIRMATION = "FLEXIV-CONTROL-ARM"

    def __init__(self, lifetime_s: float = 30.0) -> None:
        self._inner = _LocalAuthorization(lifetime_s)

    def mint(
        self,
        *,
        pid: int,
        session_id: str,
        source: str,
        confirmation: str,
    ) -> tuple[str, int]:
        if source not in {"teleop", "policy", "replay"}:
            raise ValueError("unknown command source")
        return self._inner.mint(
            pid=pid,
            confirmation=confirmation,
            expected_confirmation=self.CONFIRMATION,
            binding=(session_id, source),
        )

    def consume(self, token: str, *, session_id: str, source: str) -> None:
        self._inner.consume(token, binding=(session_id, source))


class RDKRequestDispatcher:
    """Last-line RDK safety supervisor.

    The ROS controller remains the primary arbiter. This process independently
    enforces a single source/session lease, local arm token, F/T-zero generation,
    command TTL, monotonic sequence and an in-process watchdog.
    """

    def __init__(
        self,
        backend: Any,
        ft_zero: FTZeroManager,
        hands: HandObservationCache,
        interlock: DaemonInterlock,
        *,
        observe_provider: Callable[[], Any] | None = None,
        zero_authorizations: LocalZeroAuthorization | None = None,
        control_authorizations: LocalControlAuthorization | None = None,
        deviceio_emitter: Any | None = None,
        watchdog_period_s: float = 0.005,
    ) -> None:
        self._backend = backend
        self._ft_zero = ft_zero
        self._hands = hands
        self._interlock = interlock
        self._observe_provider = observe_provider or backend.observe_both
        self._zero_authorizations = zero_authorizations or LocalZeroAuthorization()
        self._control_authorizations = control_authorizations or LocalControlAuthorization()
        self._deviceio_emitter = deviceio_emitter
        self._deviceio_sequence = {"left": 0, "right": 0}
        self._last_command_sequence: dict[str, int] = {}
        self._active = False
        self._active_source: str | None = None
        self._active_session: str | None = None
        self._owner_pid: int | None = None
        self._command_deadline_ns = 0
        self._hold_latched = False
        self._hold_reason = ""
        self._lock = threading.RLock()
        self._watchdog_period_s = watchdog_period_s
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self.daemon_instance_id = uuid.uuid4().hex

    @property
    def hold_latched(self) -> bool:
        with self._lock:
            return self._hold_latched

    @property
    def active_source(self) -> str | None:
        with self._lock:
            return self._active_source

    def start_watchdog(self) -> None:
        with self._lock:
            if self._watchdog_thread is not None:
                return
            thread = threading.Thread(
                target=self._watchdog_loop,
                name="rdk-command-watchdog",
                daemon=True,
            )
            self._watchdog_thread = thread
            thread.start()

    def close(self) -> None:
        self.shutdown_hold()

    def shutdown_hold(self, reason: str = "daemon_shutdown") -> None:
        """Synchronously hold an active session before RDK disconnect."""

        with self._lock:
            if self._active:
                self._latch_hold_locked(reason, force_hardware_hold=True)
            else:
                self._hold_latched = True
                self._hold_reason = reason
                self._command_deadline_ns = 0
        self._watchdog_stop.set()
        thread = self._watchdog_thread
        if thread is not None:
            thread.join(timeout=1.0)

    def peer_disconnected(self, credentials: tuple[int, int, int]) -> None:
        pid = credentials[0]
        with self._lock:
            if self._active and self._owner_pid == pid:
                self._latch_hold_locked("command_peer_disconnected")

    def __call__(
        self,
        kind: str,
        sequence: int,
        payload: dict[str, Any],
        credentials: tuple[int, int, int],
    ) -> tuple[str, dict[str, Any]]:
        pid, _, _ = credentials

        if kind == "hello":
            return self._ack(sequence, True, "hello")
        if kind == "observe":
            observation = self._observe_provider().to_wire()
            self._emit_deviceio(observation)
            observation["daemon_instance_id"] = self.daemon_instance_id
            return "dual_arm_state", observation
        if kind == "hand_observation":
            self._hands.update(
                payload.get("left"),
                payload.get("right"),
                source_sequence=int(payload.get("source_sequence", "0")),
                source_time_ns=int(payload.get("source_time_ns", "0")),
                source_clock_domain=str(payload.get("source_clock_domain", "")),
            )
            return self._ack(sequence, True, "hand_observation_updated")
        if kind == "authorize_zero_ft":
            token, expires = self._zero_authorizations.mint(
                pid=pid,
                session_id=str(payload.get("session_id", "")),
                confirmation=str(payload.get("operator_confirmation", "")),
                tool_payload_config_hash=str(payload.get("tool_payload_config_hash", "")),
            )
            return "authorize_zero_ft_result", {
                "authorized": True,
                "one_time_token": token,
                "expires_monotonic_ns": str(expires),
                "reason": "local_tty_confirmed",
            }
        if kind == "authorize_control":
            return self._authorize_control(pid, payload)
        if kind == "zero_ft":
            return self._zero_ft_request(payload)
        if kind == "cartesian_command":
            return self._cartesian_command(payload, owner_pid=pid)
        if kind == "hold":
            return self._hold(sequence, str(payload.get("reason", "hold")))
        raise ValueError(f"packet kind {kind!r} is not valid as a request")

    def _emit_deviceio(self, observation: dict[str, Any]) -> None:
        """Mirror each native RDK sample before protobuf/ROS conversion."""
        emitter = self._deviceio_emitter
        if emitter is None:
            return
        for side in ("left", "right"):
            arm = observation[side]
            self._deviceio_sequence[side] += 1
            sequence = self._deviceio_sequence[side]
            host_ns = int(arm["host_receive_monotonic_ns"])
            source_ns = int(arm["robot_time_ns"])
            common = {
                "producer": "rdk",
                "source_time_ns": source_ns,
                "host_receive_time_ns": host_ns,
                "sequence": sequence,
                "valid": bool(arm.get("connected", False)) and not bool(arm.get("fault", "")),
                "invalid_reason": str(arm.get("fault", "")),
                "source_clock_domain": str(arm.get("clock_domain", "flexiv_controller")),
                "host_clock_domain": "host_monotonic",
                "mapped_host_time_ns": host_ns,
                "timing_valid": True,
            }
            pose = list(arm["tcp_pose_rdk_xyz_wxyz"])
            payloads = {
                f"/robot/{side}_arm/state": arm,
                f"/robot/{side}_arm/tcp_pose": {
                    "xyz": pose[:3],
                    "quaternion_xyzw": [pose[4], pose[5], pose[6], pose[3]],
                },
                f"/robot/{side}_arm/tcp_twist": {"values": list(arm["tcp_velocity"])},
                f"/robot/{side}_arm/raw_ft": {"values": list(arm["raw_ft"])},
                f"/robot/{side}_arm/tcp_wrench": {"values": list(arm["external_wrench"])},
            }
            for topic, payload in payloads.items():
                try:
                    emitter.emit(record_envelope(topic=topic, payload=payload, **common))
                except (BufferError, RuntimeError, ValueError):
                    # Native capture must never delay or fault the RDK observation path.
                    pass

    def _authorize_control(self, pid: int, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        session = str(payload.get("session_id", ""))
        source = str(payload.get("source", ""))
        token, expires = self._control_authorizations.mint(
            pid=pid,
            session_id=session,
            source=source,
            confirmation=str(payload.get("operator_confirmation", "")),
        )
        if bool(payload.get("clear_hold_latched", False)):
            with self._lock:
                if not self._ft_zero.is_zeroed_for(session):
                    raise PermissionError("cannot clear hold before this session is F/T-zeroed")
                self._active = False
                self._active_source = None
                self._active_session = None
                self._owner_pid = None
                self._command_deadline_ns = 0
                self._hold_latched = False
                self._hold_reason = ""
        return "authorize_control_result", {
            "authorized": True,
            "one_time_token": token,
            "expires_monotonic_ns": str(expires),
            "reason": "local_tty_confirmed",
        }

    def _zero_ft_request(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        session_id = str(payload.get("session_id", ""))
        config_hash = str(payload.get("tool_payload_config_hash", ""))
        self._zero_authorizations.consume(
            str(payload.get("local_authorization_token", "")),
            session_id=session_id,
            tool_payload_config_hash=config_hash,
        )
        with self._lock:
            if self._active:
                raise RuntimeError("cannot zero F/T while a command source is active")
            self._hold_latched = True
            self._hold_reason = "maintenance"
        request = ZeroFTRequest(
            session_id=session_id,
            operator_confirmation=CONFIRMATION_TOKEN,
            local_console=True,
            tool_payload_config_hash=config_hash,
            left_hand_position=payload.get("left_hand_position", []),
            right_hand_position=payload.get("right_hand_position", []),
        )
        self._hands.begin_monitor(
            request.left_hand_position, request.right_hand_position
        )
        try:
            result = self._ft_zero.zero(request)
        finally:
            hand_monitor = self._hands.end_monitor()
        result.available_statistics["hand_monitor"] = hand_monitor
        with self._lock:
            # A separate local control-arm authorization is always required after
            # maintenance; successful zero does not activate or arm motion.
            self._hold_reason = "control_rearm_required" if result.success else "ft_zero_failed"
        response: dict[str, Any] = {
            "success": result.success,
            "failure_reason": result.failure_reason,
            "connection_generation": "0"
            if result.connection_generation is None
            else str(result.connection_generation),
            "event_json": result.event_id,
            "daemon_instance_id": self.daemon_instance_id,
            "failure_phase": result.failure_phase,
            "cleanup_json": json.dumps(result.cleanup, sort_keys=True),
            "hand_monitor_json": json.dumps(hand_monitor, sort_keys=True),
            "available_statistics": {
                name: self._statistics_wire(value)
                for name, value in result.available_statistics.items()
                if isinstance(value, dict) and "raw_ft" in value
            },
        }
        for side in ("left", "right"):
            arm = getattr(result, side)
            if arm is not None:
                response[side] = {
                    "side": side,
                    "before": self._statistics_wire(arm.before),
                    "after": self._statistics_wire(arm.after),
                }
        return "zero_ft_result", response

    @staticmethod
    def _statistics_wire(value: Any) -> dict[str, Any]:
        raw = value if isinstance(value, dict) else value.to_wire()
        raw_ft = raw["raw_ft"]
        external = raw["external_wrench"]
        return {
            "samples": str(raw["sample_count"]),
            "mean": list(raw_ft["mean"]),
            "standard_deviation": list(raw_ft["standard_deviation"]),
            "peak_absolute": list(raw_ft["peak_absolute"]),
            "external_mean": list(external["mean"]),
            "external_standard_deviation": list(external["standard_deviation"]),
            "external_peak_absolute": list(external["peak_absolute"]),
            "duration_s": float(raw["duration_s"]),
            "max_dq_norm": float(raw["max_dq_norm"]),
            "max_tcp_velocity_norm": float(raw["max_tcp_velocity_norm"]),
            "stable": bool(raw["stable"]),
            "reason": str(raw.get("rejection_reason", "")),
            "rejection_reason": str(raw.get("rejection_reason", "")),
        }

    def _cartesian_command(
        self,
        payload: dict[str, Any],
        *,
        owner_pid: int,
    ) -> tuple[str, dict[str, Any]]:
        now = time.monotonic_ns()
        session_id = str(payload.get("session_id", ""))
        source = str(payload.get("source", ""))
        source_sequence = self._uint64(payload.get("source_sequence"), "source_sequence")
        expires = self._uint64(payload.get("expires_monotonic_ns"), "expires_monotonic_ns")
        valid_mask = int(payload.get("valid_mask", 0))
        if self._interlock.maintenance_active:
            return self._ack(source_sequence, False, "maintenance_active")
        if not self._ft_zero.is_zeroed_for(session_id):
            return self._ack(source_sequence, False, "ft_not_zeroed_for_session")
        if self._interlock.ready_generation != self._backend.connection_generation:
            return self._ack(source_sequence, False, "rdk_connection_generation_changed")
        if source not in {"teleop", "policy", "replay"}:
            return self._ack(source_sequence, False, "unknown_source")
        if not bool(payload.get("safety_validated", False)):
            return self._ack(source_sequence, False, "safety_not_validated")
        if not bool(payload.get("local_permission", False)):
            return self._ack(source_sequence, False, "local_permission_missing")
        if not bool(payload.get("physical_pedal", False)):
            return self._ack(source_sequence, False, "physical_pedal_released")
        if expires <= now:
            return self._ack(source_sequence, False, "command_expired")
        if expires - now > 1_000_000_000:
            return self._ack(source_sequence, False, "command_ttl_exceeds_one_second")
        if valid_mask == 0 or valid_mask & ~0x3:
            return self._ack(source_sequence, False, "invalid_arm_mask")
        try:
            validated_targets = {
                side: self._validate_target(side, payload.get(side, {}))
                for side, bit in (("left", 0x1), ("right", 0x2))
                if valid_mask & bit
            }
        except Exception:
            with self._lock:
                if self._active:
                    self._latch_hold_locked(
                        "invalid_cartesian_target", force_hardware_hold=True
                    )
            raise
        with self._lock:
            if self._hold_latched:
                return self._ack(source_sequence, False, f"hold_latched:{self._hold_reason}")
            if self._active_source is None:
                self._control_authorizations.consume(
                    str(payload.get("local_arm_token", "")),
                    session_id=session_id,
                    source=source,
                )
                self._active_source = source
                self._active_session = session_id
                self._owner_pid = owner_pid
            elif source != self._active_source or session_id != self._active_session:
                self._latch_hold_locked(
                    "source_or_session_conflict", force_hardware_hold=True
                )
                return self._ack(source_sequence, False, "source_or_session_conflict")
            elif owner_pid != self._owner_pid:
                self._latch_hold_locked(
                    "command_owner_pid_changed", force_hardware_hold=True
                )
                return self._ack(source_sequence, False, "command_owner_pid_changed")
            if source_sequence <= self._last_command_sequence.get(source, -1):
                self._latch_hold_locked(
                    "non_monotonic_source_sequence", force_hardware_hold=True
                )
                return self._ack(source_sequence, False, "non_monotonic_source_sequence")
            self._last_command_sequence[source] = source_sequence
            # Mark active before the first arm write so a partial two-arm send
            # is always followed by a best-effort hold on both arms.
            self._active = True
            self._command_deadline_ns = expires
            try:
                for side in ("left", "right"):
                    if side in validated_targets:
                        self._send_validated_target(side, validated_targets[side])
            except Exception:
                self._latch_hold_locked("send_failure", force_hardware_hold=True)
                raise
        return self._ack(source_sequence, True, "sent")

    @staticmethod
    def _validate_target(
        side: str, target: object
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        if not isinstance(target, dict):
            raise ValueError(f"{side} Cartesian target is missing")
        pose = np.asarray(target.get("tcp_pose_rdk", []), dtype=np.float64).reshape(-1)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError(f"{side} Cartesian pose must be a finite 7-vector")
        quaternion_norm = float(np.linalg.norm(pose[3:]))
        if not np.isfinite(quaternion_norm) or abs(quaternion_norm - 1.0) > 1e-3:
            raise ValueError(f"{side} Cartesian quaternion is not normalized")
        limits = (
            float(target.get("max_linear_velocity", 0.0)),
            float(target.get("max_angular_velocity", 0.0)),
            float(target.get("max_linear_acceleration", 0.0)),
            float(target.get("max_angular_acceleration", 0.0)),
        )
        maxima = (0.10, 0.25, 0.50, 1.0)
        if any(
            not np.isfinite(value) or value <= 0.0 or value > maximum
            for value, maximum in zip(limits, maxima, strict=True)
        ):
            raise ValueError(f"{side} Cartesian safety limit is invalid")
        return pose, limits

    def _send_validated_target(
        self,
        side: str,
        validated: tuple[np.ndarray, tuple[float, float, float, float]],
    ) -> None:
        pose, limits = validated
        self._backend.send_cartesian_target(
            side,
            pose,
            max_linear_velocity=limits[0],
            max_angular_velocity=limits[1],
            max_linear_acceleration=limits[2],
            max_angular_acceleration=limits[3],
            local_authorized=True,
        )

    def _hold(self, sequence: int, reason: str) -> tuple[str, dict[str, Any]]:
        with self._lock:
            # Explicit retries always re-issue both measurement holds.
            errors = self._latch_hold_locked(reason, force_hardware_hold=True)
            if errors:
                return self._ack(sequence, False, f"hold_failed:{';'.join(errors)}")
        return self._ack(sequence, True, f"hold_latched:{reason}")

    def _latch_hold_locked(
        self, reason: str, *, force_hardware_hold: bool = False
    ) -> list[str]:
        errors: list[str] = []
        if self._active or force_hardware_hold:
            for side in ("left", "right"):
                try:
                    self._backend.send_hold_from_measurement(side, local_authorized=True)
                except Exception as exc:
                    errors.append(f"{side}:{type(exc).__name__}:{exc}")
        self._active = False
        self._hold_latched = True
        self._hold_reason = reason
        self._command_deadline_ns = 0
        return errors

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(self._watchdog_period_s):
            with self._lock:
                if self._active and time.monotonic_ns() >= self._command_deadline_ns:
                    self._latch_hold_locked("command_ttl_expired")

    @staticmethod
    def _uint64(value: object, name: str) -> int:
        if isinstance(value, str) and value.isdecimal():
            parsed = int(value)
        elif isinstance(value, int) and not isinstance(value, bool):
            parsed = value
        else:
            raise ValueError(f"{name} must be uint64")
        if parsed < 0 or parsed >= 1 << 64:
            raise ValueError(f"{name} is outside uint64")
        return parsed

    @staticmethod
    def _ack(sequence: int, accepted: bool, reason: str) -> tuple[str, dict[str, Any]]:
        return "command_ack", {
            "source_sequence": str(sequence),
            "accepted": accepted,
            "reason": reason,
            "applied_monotonic_ns": str(time.monotonic_ns()) if accepted else "0",
        }
