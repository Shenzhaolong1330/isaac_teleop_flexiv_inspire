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


class LocalHomeAuthorization:
    """One-shot bootstrap token for a process-bound Home session lease."""

    CONFIRMATION = "FLEXIV-HOME-MOVE"

    def __init__(self, lifetime_s: float = 30.0) -> None:
        self._inner = _LocalAuthorization(lifetime_s)

    def mint(
        self,
        *,
        pid: int,
        session_id: str,
        confirmation: str,
    ) -> tuple[str, int]:
        return self._inner.mint(
            pid=pid,
            confirmation=confirmation,
            expected_confirmation=self.CONFIRMATION,
            binding=(session_id,),
        )

    def consume(self, token: str, *, session_id: str) -> None:
        self._inner.consume(token, binding=(session_id,))


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
        home_authorizations: LocalHomeAuthorization | None = None,
        deviceio_emitter: Any | None = None,
        cartesian_limits: tuple[float, float, float, float] = (
            0.10,
            0.25,
            0.50,
            1.0,
        ),
        watchdog_period_s: float = 0.005,
    ) -> None:
        self._backend = backend
        self._ft_zero = ft_zero
        self._hands = hands
        self._interlock = interlock
        self._observe_provider = observe_provider or backend.observe_both
        self._zero_authorizations = zero_authorizations or LocalZeroAuthorization()
        self._control_authorizations = control_authorizations or LocalControlAuthorization()
        self._home_authorizations = home_authorizations or LocalHomeAuthorization()
        self._deviceio_emitter = deviceio_emitter
        configured_limits = np.asarray(
            cartesian_limits, dtype=np.float64
        ).reshape(-1)
        if (
            configured_limits.shape != (4,)
            or not np.all(np.isfinite(configured_limits))
            or np.any(configured_limits <= 0.0)
        ):
            raise ValueError(
                "cartesian_limits must contain four positive finite values"
            )
        self._cartesian_limits = tuple(
            float(value) for value in configured_limits
        )
        self._deviceio_sequence = {"left": 0, "right": 0}
        self._last_command_sequence: dict[str, int] = {}
        self._last_home_sequence = -1
        self._active = False
        self._home_active = False
        self._home_started_ns = 0
        self._home_timeout_ns = 0
        self._home_signature: tuple[object, ...] | None = None
        self._home_phase = ""
        self._home_phase_started_ns = 0
        self._home_lift_timeout_ns = 0
        self._home_lift_targets: dict[str, np.ndarray] = {}
        self._home_lift_active: tuple[str, ...] = ()
        self._home_lift_queue: list[str] = []
        self._home_authorized_session: str | None = None
        self._home_authorized_owner_pid: int | None = None
        self._home_authorized_generation: int | None = None
        # The control bridge already samples both arms at high rate. Home uses
        # that fresh sample instead of issuing a second pair of blocking RDK
        # state reads on every 40 ms keepalive.
        self._observation_lock = threading.Lock()
        self._latest_observation: Any | None = None
        self._latest_observation_ns = 0
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
            if self._home_authorized_owner_pid == pid:
                self._clear_home_authorization_locked()

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
            sample = self._observe_provider()
            self._cache_observation(sample)
            observation_wire = sample.to_wire()
            self._emit_deviceio(observation_wire)
            observation_wire["daemon_instance_id"] = self.daemon_instance_id
            return "dual_arm_state", observation_wire
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
        if kind == "authorize_home":
            return self._authorize_home(pid, payload)
        if kind == "zero_ft":
            return self._zero_ft_request(payload)
        if kind == "cartesian_command":
            return self._cartesian_command(payload, owner_pid=pid)
        if kind == "home_command":
            return self._home_command(payload, owner_pid=pid)
        if kind == "hold":
            return self._hold(sequence, str(payload.get("reason", "hold")))
        raise ValueError(f"packet kind {kind!r} is not valid as a request")

    def _cache_observation(self, sample: Any) -> None:
        with self._observation_lock:
            self._latest_observation = sample
            self._latest_observation_ns = time.monotonic_ns()

    def _home_observation(self, max_age_ns: int = 100_000_000) -> Any:
        """Return a recent dual-arm sample without duplicating normal polling."""

        now = time.monotonic_ns()
        with self._observation_lock:
            sample = self._latest_observation
            received_ns = self._latest_observation_ns
        if (
            sample is not None
            and now - received_ns <= max_age_ns
            and sample.left.connection_generation
            == self._backend.connection_generation
            and sample.right.connection_generation
            == self._backend.connection_generation
        ):
            return sample
        # Direct dispatcher tests and startup can reach Home before the first
        # bridge poll. One synchronous read preserves the daemon's independent
        # health check; subsequent keepalives reuse the normal observation feed.
        sample = self._observe_provider()
        self._cache_observation(sample)
        return sample

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
                if self._active:
                    raise RuntimeError("cannot clear hold while motion is active")
                # A latched hold leaves both controllers in IDLE via Robot.Stop.
                # Prepare a fresh measured Cartesian anchor before allowing a
                # new command token to cross the hardware boundary.
                if self._hold_latched:
                    errors: list[str] = []
                    for side in ("left", "right"):
                        try:
                            self._backend.switch_cartesian_mode(
                                side, local_console=True
                            )
                        except Exception as exc:
                            errors.append(
                                f"{side}:mode:{type(exc).__name__}:{exc}"
                            )
                    if not errors:
                        for side in ("left", "right"):
                            try:
                                self._backend.send_hold_from_measurement(
                                    side, local_authorized=True
                                )
                            except Exception as exc:
                                errors.append(
                                    f"{side}:anchor:{type(exc).__name__}:{exc}"
                                )
                    if errors:
                        raise RuntimeError(
                            "cannot prepare Cartesian control after hold: "
                            + ";".join(errors)
                        )
                self._active = False
                self._active_source = None
                self._active_session = None
                self._owner_pid = None
                self._command_deadline_ns = 0
                self._hold_latched = False
                self._hold_reason = ""
                self._last_command_sequence.pop(source, None)
        return "authorize_control_result", {
            "authorized": True,
            "one_time_token": token,
            "expires_monotonic_ns": str(expires),
            "reason": "local_tty_confirmed",
        }

    def _authorize_home(
        self, pid: int, payload: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        clear_hold_latched = bool(payload.get("clear_hold_latched", False))
        recover_robot_faults = bool(
            payload.get("recover_robot_faults", False)
        )
        if recover_robot_faults and not clear_hold_latched:
            raise PermissionError(
                "controller fault recovery requires clear_hold_latched"
            )
        if recover_robot_faults:
            with self._lock:
                if self._active:
                    raise RuntimeError(
                        "cannot recover controller faults while motion is active"
                    )
                if not self._ft_zero.is_zeroed_for(
                    str(payload.get("session_id", ""))
                ):
                    raise PermissionError(
                        "cannot recover controller faults before this session "
                        "is F/T-zeroed"
                    )
            self._recover_robots_for_reset()

        session = str(payload.get("session_id", ""))
        token, expires = self._home_authorizations.mint(
            pid=pid,
            session_id=session,
            confirmation=str(payload.get("operator_confirmation", "")),
        )
        if clear_hold_latched:
            with self._lock:
                if not self._ft_zero.is_zeroed_for(session):
                    raise PermissionError(
                        "cannot clear hold before this session is F/T-zeroed"
                    )
                if self._active:
                    raise RuntimeError("cannot clear hold while motion is active")
                self._active_source = None
                self._active_session = None
                self._owner_pid = None
                self._command_deadline_ns = 0
                self._hold_latched = False
                self._hold_reason = ""
                self._last_home_sequence = -1
        return "authorize_home_result", {
            "authorized": True,
            "one_time_token": token,
            "expires_monotonic_ns": str(expires),
            "reason": "local_tty_confirmed",
        }

    def _recover_robots_for_reset(self, timeout_s: float = 20.0) -> None:
        """Clear Flexiv faults and restore both arms to operational for Reset.

        This intentionally matches the working one-shot reset sequence:
        ``ClearFault`` when faulted, ``Enable``, then wait for
        ``operational()`` before any Home mode switch.  It is reachable only
        from the explicit local Reset Home authorization, never from routine
        teleoperation re-authorization.
        """

        for side in ("left", "right"):
            self._backend.clear_fault(side, local_console=True)
        for side in ("left", "right"):
            if not self._backend.operational(side):
                self._backend.enable(side, local_console=True)

        deadline = time.monotonic() + timeout_s
        pending = ["left", "right"]
        while pending and time.monotonic() < deadline:
            pending = [
                side
                for side in pending
                if not self._backend.operational(side)
            ]
            if pending:
                time.sleep(0.2)
        if pending:
            raise RuntimeError(
                "Flexiv arm did not become operational after ClearFault/Enable: "
                + ", ".join(pending)
            )
        # The bridge may have cached the pre-recovery sample with fault=true.
        # Force Home's health check to read the post-ClearFault controller
        # state instead of rejecting a recovered arm on that stale sample.
        with self._observation_lock:
            self._latest_observation = None
            self._latest_observation_ns = 0

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
            self._clear_home_authorization_locked()
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

    def _validate_target(
        self, side: str, target: object
    ) -> tuple[
        str,
        np.ndarray,
        tuple[float, float, float, float],
        np.ndarray,
        np.ndarray,
    ]:
        if not isinstance(target, dict):
            raise ValueError(f"{side} Cartesian target is missing")
        # Old replay/test clients did not carry this field and historically
        # used impedance. Current control bridges always send it explicitly.
        control_mode = str(
            target.get("control_mode", "impedance")
        ).strip().lower()
        if control_mode not in {"position", "impedance"}:
            raise ValueError(
                f"{side} Cartesian control_mode must be position or impedance"
            )
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
        maxima = self._cartesian_limits
        if any(
            not np.isfinite(value) or value <= 0.0 or value > maximum
            for value, maximum in zip(limits, maxima, strict=True)
        ):
            raise ValueError(f"{side} Cartesian safety limit is invalid")
        stiffness = np.asarray(
            target.get("cartesian_stiffness", []), dtype=np.float64
        ).reshape(-1)
        damping_ratio = np.asarray(
            target.get("cartesian_damping_ratio", []), dtype=np.float64
        ).reshape(-1)
        if stiffness.shape != (6,) or not np.all(np.isfinite(stiffness)):
            raise ValueError(f"{side} Cartesian stiffness must be a finite 6-vector")
        if damping_ratio.shape != (6,) or not np.all(np.isfinite(damping_ratio)):
            raise ValueError(
                f"{side} Cartesian damping ratio must be a finite 6-vector"
            )
        nominal = np.asarray(
            self._backend.nominal_cartesian_stiffness(side), dtype=np.float64
        )
        if nominal.shape != (6,) or np.any(stiffness < 0.0) or np.any(
            stiffness > nominal
        ):
            raise ValueError(
                f"{side} Cartesian stiffness exceeds RobotInfo.K_x_nom"
            )
        if np.any(damping_ratio < 0.3) or np.any(damping_ratio > 0.8):
            raise ValueError(
                f"{side} Cartesian damping ratio must be in [0.3,0.8]"
            )
        return control_mode, pose, limits, stiffness, damping_ratio

    def _send_validated_target(
        self,
        side: str,
        validated: tuple[
            str,
            np.ndarray,
            tuple[float, float, float, float],
            np.ndarray,
            np.ndarray,
        ],
    ) -> None:
        control_mode, pose, limits, stiffness, damping_ratio = validated
        if control_mode == "impedance":
            self._backend.set_cartesian_impedance(
                side,
                stiffness,
                damping_ratio,
                local_authorized=True,
            )
        self._backend.send_cartesian_target(
            side,
            pose,
            max_linear_velocity=limits[0],
            max_angular_velocity=limits[1],
            max_linear_acceleration=limits[2],
            max_angular_acceleration=limits[3],
            local_authorized=True,
        )

    def _home_command(
        self,
        payload: dict[str, Any],
        *,
        owner_pid: int,
    ) -> tuple[str, dict[str, Any]]:
        """Start or keep alive one guarded, configured dual-arm Home motion."""

        now = time.monotonic_ns()
        session_id = str(payload.get("session_id", ""))
        sequence = self._uint64(payload.get("request_sequence"), "request_sequence")
        expires = self._uint64(
            payload.get("expires_monotonic_ns"), "expires_monotonic_ns"
        )
        if self._interlock.maintenance_active:
            return self._home_result(False, False, "maintenance_active")
        if not self._ft_zero.is_zeroed_for(session_id):
            return self._home_result(False, False, "ft_not_zeroed_for_session")
        if self._interlock.ready_generation != self._backend.connection_generation:
            return self._home_result(
                False, False, "rdk_connection_generation_changed"
            )
        for field, reason in (
            ("safety_validated", "safety_not_validated"),
            ("local_permission", "local_permission_missing"),
            ("collision_clear", "collision_not_clear"),
        ):
            if not bool(payload.get(field, False)):
                with self._lock:
                    if self._home_active:
                        self._latch_hold_locked(reason, force_hardware_hold=True)
                return self._home_result(False, False, reason)
        if expires <= now:
            return self._home_result(False, False, "home_keepalive_expired")
        if expires - now > 250_000_000:
            return self._home_result(
                False, False, "home_keepalive_ttl_exceeds_250_ms"
            )
        try:
            targets, limits, lift, signature = self._validate_home(payload)
        except Exception:
            with self._lock:
                if self._home_active:
                    self._latch_hold_locked(
                        "invalid_home_command", force_hardware_hold=True
                    )
            raise
        with self._lock:
            generation = self._backend.connection_generation
            lease_active = (
                session_id == self._home_authorized_session
                and owner_pid == self._home_authorized_owner_pid
                and generation == self._home_authorized_generation
            )
            if not self._home_active and not lease_active:
                self._home_authorizations.consume(
                    str(payload.get("local_authorization_token", "")),
                    session_id=session_id,
                )
            if self._hold_latched:
                if (
                    not self._home_active
                    and bool(payload.get("clear_routine_hold", False))
                    and self._hold_reason == "episode_home_transition"
                ):
                    self._hold_latched = False
                    self._hold_reason = ""
                else:
                    return self._home_result(
                        False, False, f"hold_latched:{self._hold_reason}"
                    )
            if self._active and not self._home_active:
                return self._home_result(False, False, "command_source_active")
            if self._home_active:
                if owner_pid != self._owner_pid or session_id != self._active_session:
                    self._latch_hold_locked(
                        "home_owner_or_session_changed",
                        force_hardware_hold=True,
                    )
                    return self._home_result(
                        False, False, "home_owner_or_session_changed"
                    )
                if signature != self._home_signature:
                    self._latch_hold_locked(
                        "home_configuration_changed",
                        force_hardware_hold=True,
                    )
                    return self._home_result(
                        False, False, "home_configuration_changed"
                    )
                if sequence <= self._last_home_sequence:
                    self._latch_hold_locked(
                        "non_monotonic_home_sequence",
                        force_hardware_hold=True,
                    )
                    return self._home_result(
                        False, False, "non_monotonic_home_sequence"
                    )
            else:
                sample = self._home_observation()
                for side in ("left", "right"):
                    arm = getattr(sample, side)
                    if not arm.connected or arm.fault:
                        raise RuntimeError(f"{side} arm is not healthy for Home")
                    if np.max(np.abs(arm.dq)) > 0.05:
                        raise RuntimeError(f"{side} arm must be stationary before Home")
                    if not self._backend.operational(side):
                        raise RuntimeError(f"{side} arm is not operational")
                if not lease_active:
                    self._home_authorized_session = session_id
                    self._home_authorized_owner_pid = owner_pid
                    self._home_authorized_generation = generation
                self._active = True
                self._home_active = True
                self._active_source = "home"
                self._active_session = session_id
                self._owner_pid = owner_pid
                self._home_timeout_ns = int(limits[3] * 1e9)
                self._home_lift_timeout_ns = int(lift["timeout_s"] * 1e9)
                self._home_signature = signature
                # Mark active before the first arm write so partial mode/command
                # transitions are always stopped by the same hold path.
                try:
                    self._begin_home_motion_locked(
                        sample, targets, limits, lift, now
                    )
                except Exception:
                    self._latch_hold_locked(
                        "home_start_failure", force_hardware_hold=True
                    )
                    raise
                now = time.monotonic_ns()
            self._last_home_sequence = sequence
            if self._home_phase == "lift":
                sample = self._home_observation()
                for side in self._home_lift_active:
                    arm = getattr(sample, side)
                    if not arm.connected or arm.fault:
                        self._latch_hold_locked(
                            "home_lift_arm_unhealthy",
                            force_hardware_hold=True,
                        )
                        return self._home_result(
                            False, False, f"{side}_arm_unhealthy_during_home_lift"
                        )
                if now - self._home_phase_started_ns > self._home_lift_timeout_ns:
                    detail = ",".join(
                        f"{side}(current_z={float(getattr(sample, side).tcp_pose_rdk[2]):.4f},"
                        f"target_z={float(self._home_lift_targets[side][2]):.4f},"
                        f"remaining={max(0.0, float(self._home_lift_targets[side][2]) - float(getattr(sample, side).tcp_pose_rdk[2])):.4f})"
                        for side in self._home_lift_active
                    )
                    self._latch_hold_locked(
                        "home_lift_timeout", force_hardware_hold=True
                    )
                    return self._home_result(
                        False, False, f"home_lift_timeout:{detail}"
                    )
                reached = all(
                    float(getattr(sample, side).tcp_pose_rdk[2])
                    >= float(self._home_lift_targets[side][2])
                    - float(lift["tolerance_m"])
                    for side in self._home_lift_active
                )
                if reached:
                    try:
                        if self._home_lift_queue:
                            next_side = self._home_lift_queue.pop(0)
                            self._start_home_lift_locked(
                                (next_side,), lift, now
                            )
                        else:
                            self._start_joint_home_locked(
                                targets, limits, now
                            )
                    except Exception:
                        self._latch_hold_locked(
                            "home_phase_transition_failure",
                            force_hardware_hold=True,
                        )
                        raise
                if self._home_phase == "lift":
                    remaining = max(
                        max(
                            0.0,
                            float(self._home_lift_targets[side][2])
                            - float(getattr(sample, side).tcp_pose_rdk[2]),
                        )
                        for side in self._home_lift_active
                    )
                    # RDK mode changes and observations above are synchronous
                    # and may legitimately take longer than the request's
                    # 150 ms wire TTL.  Arm the watchdog only after that
                    # accepted work has finished; otherwise it can expire the
                    # instant this lock is released, before the bridge has any
                    # opportunity to send its next keepalive.
                    self._command_deadline_ns = time.monotonic_ns() + 250_000_000
                    return self._home_result(
                        True, False, "home_lift_in_progress", remaining
                    )
            if now - self._home_started_ns > self._home_timeout_ns:
                self._latch_hold_locked("home_timeout", force_hardware_hold=True)
                return self._home_result(False, False, "home_timeout")
            sample = self._home_observation()
            errors = [
                float(np.max(np.abs(getattr(sample, side).q - targets[side])))
                for side in ("left", "right")
            ]
            max_error = max(errors)
            max_speed = max(
                float(np.max(np.abs(getattr(sample, side).dq)))
                for side in ("left", "right")
            )
            if max_error <= limits[2] and max_speed <= 0.02:
                cleanup_errors = self._complete_home_locked()
                if cleanup_errors:
                    return self._home_result(
                        False,
                        False,
                        f"home_completion_hold_failed:{';'.join(cleanup_errors)}",
                        max_error,
                    )
                return self._home_result(
                    True, True, "home_complete", max_error
                )
            # As with the lift phase, start the keepalive grace period after
            # the blocking RDK observation, not before it.
            self._command_deadline_ns = time.monotonic_ns() + 250_000_000
            return self._home_result(True, False, "home_in_progress", max_error)

    def _clear_home_authorization_locked(self) -> None:
        self._home_authorized_session = None
        self._home_authorized_owner_pid = None
        self._home_authorized_generation = None

    def _validate_home(
        self, payload: dict[str, Any]
    ) -> tuple[
        dict[str, np.ndarray],
        tuple[float, float, float, float],
        dict[str, Any],
        tuple[object, ...],
    ]:
        targets: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            target = np.asarray(
                payload.get(f"{side}_joint_positions", []),
                dtype=np.float64,
            ).reshape(-1)
            if target.shape != (7,) or not np.all(np.isfinite(target)):
                raise ValueError(f"{side} Home target must be a finite 7-vector")
            lower, upper = self._backend.joint_position_limits(side)
            if np.any(target < lower) or np.any(target > upper):
                raise ValueError(
                    f"{side} Home target exceeds RobotInfo joint limits"
                )
            targets[side] = target
        max_velocity = float(payload.get("max_velocity_rad_s", 0.0))
        max_acceleration = float(payload.get("max_acceleration_rad_s2", 0.0))
        tolerance = float(payload.get("tolerance_rad", 0.0))
        timeout = float(payload.get("timeout_s", 0.0))
        if not 0.0 < max_velocity <= 0.75:
            raise ValueError("Home max velocity must be in (0,0.75] rad/s")
        if not 0.0 < max_acceleration <= 2.0:
            raise ValueError("Home max acceleration must be in (0,2.0] rad/s^2")
        if not 0.0 < tolerance <= 0.1:
            raise ValueError("Home tolerance must be in (0,0.1] rad")
        if not 1.0 <= timeout <= 60.0:
            raise ValueError("Home timeout must be in [1,60] seconds")
        limits = (max_velocity, max_acceleration, tolerance, timeout)
        lift_enabled_raw = payload.get("lift_enabled", False)
        if not isinstance(lift_enabled_raw, bool):
            raise ValueError("Home lift_enabled must be a bool")
        lift: dict[str, Any] = {
            "enabled": lift_enabled_raw,
            "safe_z": {},
            "motion_limits": (0.0, 0.0, 0.0, 0.0),
            "tolerance_m": 0.0,
            "timeout_s": 0.0,
            "parallel": False,
            "stiffness": np.zeros(6),
            "damping_ratio": np.full(6, 0.7),
        }
        lift_signature: tuple[object, ...] = (False,)
        if lift_enabled_raw:
            safe_z = {
                side: float(payload.get(f"{side}_lift_safe_z_m", np.nan))
                for side in ("left", "right")
            }
            if any(
                not np.isfinite(value) or not -2.0 <= value <= 2.0
                for value in safe_z.values()
            ):
                raise ValueError("Home lift safe Z values must be finite in [-2,2]")
            motion_limits = tuple(
                float(payload.get(name, 0.0))
                for name in (
                    "lift_max_linear_velocity",
                    "lift_max_angular_velocity",
                    "lift_max_linear_acceleration",
                    "lift_max_angular_acceleration",
                )
            )
            if any(
                not np.isfinite(value) or value <= 0.0 or value > maximum
                for value, maximum in zip(
                    motion_limits, self._cartesian_limits, strict=True
                )
            ):
                raise ValueError("Home lift Cartesian safety limit is invalid")
            lift_tolerance = float(payload.get("lift_tolerance_m", 0.0))
            lift_timeout = float(payload.get("lift_timeout_s", 0.0))
            if not 0.0 < lift_tolerance <= 0.05:
                raise ValueError("Home lift tolerance must be in (0,0.05] m")
            if not 1.0 <= lift_timeout <= 30.0:
                raise ValueError("Home lift timeout must be in [1,30] seconds")
            lift_parallel = payload.get("lift_parallel", False)
            if not isinstance(lift_parallel, bool):
                raise ValueError("Home lift_parallel must be a bool")
            stiffness = np.asarray(
                payload.get("lift_cartesian_stiffness", []),
                dtype=np.float64,
            ).reshape(-1)
            damping_ratio = np.asarray(
                payload.get("lift_cartesian_damping_ratio", []),
                dtype=np.float64,
            ).reshape(-1)
            if stiffness.shape != (6,) or not np.all(np.isfinite(stiffness)):
                raise ValueError("Home lift stiffness must be a finite 6-vector")
            if damping_ratio.shape != (6,) or not np.all(np.isfinite(damping_ratio)):
                raise ValueError(
                    "Home lift damping ratio must be a finite 6-vector"
                )
            if np.any(stiffness < 0.0) or any(
                np.any(stiffness > self._backend.nominal_cartesian_stiffness(side))
                for side in ("left", "right")
            ):
                raise ValueError("Home lift stiffness exceeds RobotInfo.K_x_nom")
            if np.any(damping_ratio < 0.3) or np.any(damping_ratio > 0.8):
                raise ValueError("Home lift damping ratio must be in [0.3,0.8]")
            lift = {
                "enabled": True,
                "safe_z": safe_z,
                "motion_limits": motion_limits,
                "tolerance_m": lift_tolerance,
                "timeout_s": lift_timeout,
                "parallel": lift_parallel,
                "stiffness": stiffness,
                "damping_ratio": damping_ratio,
            }
            lift_signature = (
                True,
                safe_z["left"],
                safe_z["right"],
                *motion_limits,
                lift_tolerance,
                lift_timeout,
                lift_parallel,
                tuple(float(value) for value in stiffness),
                tuple(float(value) for value in damping_ratio),
            )
        signature: tuple[object, ...] = (
            tuple(float(value) for value in targets["left"]),
            tuple(float(value) for value in targets["right"]),
            *limits,
            *lift_signature,
        )
        return targets, limits, lift, signature

    def _begin_home_motion_locked(
        self,
        sample: Any,
        targets: dict[str, np.ndarray],
        limits: tuple[float, float, float, float],
        lift: dict[str, Any],
        now: int,
    ) -> None:
        self._home_lift_targets = {}
        self._home_lift_active = ()
        self._home_lift_queue = []
        if lift["enabled"]:
            needing_lift: list[str] = []
            for side in ("left", "right"):
                current = getattr(sample, side).tcp_pose_rdk.copy()
                target = current.copy()
                target[2] = max(
                    float(current[2]), float(lift["safe_z"][side])
                )
                self._home_lift_targets[side] = target
                if target[2] > current[2] + float(lift["tolerance_m"]):
                    needing_lift.append(side)
            if needing_lift:
                if lift["parallel"]:
                    active = tuple(needing_lift)
                else:
                    active = (needing_lift[0],)
                    self._home_lift_queue = needing_lift[1:]
                self._start_home_lift_locked(active, lift, now)
                return
        self._start_joint_home_locked(targets, limits, now)

    def _start_home_lift_locked(
        self,
        sides: tuple[str, ...],
        lift: dict[str, Any],
        now: int,
    ) -> None:
        for side in sides:
            self._backend.switch_cartesian_mode(side, local_console=True)
        for side in sides:
            # Reset lift is rigid Cartesian position tracking. The mode switch
            # already disabled all force-control axes; do not replace the
            # controller's position gains with teleop impedance settings.
            motion_limits = lift["motion_limits"]
            self._backend.send_cartesian_target(
                side,
                self._home_lift_targets[side],
                max_linear_velocity=motion_limits[0],
                max_angular_velocity=motion_limits[1],
                max_linear_acceleration=motion_limits[2],
                max_angular_acceleration=motion_limits[3],
                local_authorized=True,
            )
        self._home_lift_active = sides
        self._home_phase = "lift"
        self._home_phase_started_ns = now

    def _start_joint_home_locked(
        self,
        targets: dict[str, np.ndarray],
        limits: tuple[float, float, float, float],
        now: int,
    ) -> None:
        # RDK expects the first NRT joint target immediately after SwitchMode.
        # Switching both arms first left the first arm in NRT_JOINT_POSITION
        # for roughly one hardware round-trip without a target. On the real
        # station that produced controller event 303010 (no feasible NRT
        # trajectory) before the second arm had even changed mode. This is the
        # same per-arm SwitchMode -> SendJointPosition ordering used by the
        # proven dual_arm_teleop reset implementation.
        for side in ("left", "right"):
            self._backend.switch_joint_position_mode(
                side, local_console=True
            )
            self._backend.send_joint_position(
                side,
                targets[side],
                max_velocity=limits[0],
                max_acceleration=limits[1],
                local_authorized=True,
            )
        self._home_phase = "joint_home"
        self._home_started_ns = now

    def _complete_home_locked(self) -> list[str]:
        errors: list[str] = []
        for side in ("left", "right"):
            try:
                # SwitchMode stops ongoing motion before transitioning. Stop
                # both arms first, then issue either arm's measured hold.
                self._backend.switch_cartesian_mode(
                    side, local_console=True
                )
            except Exception as exc:
                errors.append(f"{side}:stop:{type(exc).__name__}:{exc}")
        for side in ("left", "right"):
            try:
                self._backend.send_hold_from_measurement(
                    side, local_authorized=True
                )
            except Exception as exc:
                errors.append(f"{side}:hold:{type(exc).__name__}:{exc}")
        self._active = False
        self._home_active = False
        self._active_source = None
        self._active_session = None
        self._owner_pid = None
        self._command_deadline_ns = 0
        self._home_started_ns = 0
        self._home_timeout_ns = 0
        self._home_signature = None
        self._home_phase = ""
        self._home_phase_started_ns = 0
        self._home_lift_timeout_ns = 0
        self._home_lift_targets = {}
        self._home_lift_active = ()
        self._home_lift_queue = []
        self._hold_latched = bool(errors)
        self._hold_reason = "home_completion_hold_failed" if errors else ""
        return errors

    @staticmethod
    def _home_result(
        accepted: bool,
        completed: bool,
        reason: str,
        max_error: float = 0.0,
    ) -> tuple[str, dict[str, Any]]:
        return "home_result", {
            "accepted": accepted,
            "completed": completed,
            "reason": reason,
            "max_position_error_rad": float(max_error),
            "applied_monotonic_ns": (
                str(time.monotonic_ns()) if accepted else "0"
            ),
        }

    def _hold(self, sequence: int, reason: str) -> tuple[str, dict[str, Any]]:
        with self._lock:
            routine_reasons = {
                "physical_pedal_released",
                "source_deadman_released",
                "source_heartbeat_stale",
                "command_stale",
                "stop_requested",
                "local_stop",
                "command_ttl_expired",
                "control_authorization_expired",
                "episode_home_transition",
                # The local bridge already stopped motion on the limit. Once
                # live arm observations are healthy again, a pedal re-arm or
                # explicit Home should not remain blocked by the old latch.
                "safety_limit",
            }
            if (
                reason == "episode_home_transition"
                and self._hold_latched
                and self._hold_reason not in routine_reasons
            ):
                return self._ack(
                    sequence,
                    False,
                    f"non_routine_hold_preserved:{self._hold_reason}",
                )
            # Explicit retries always re-issue Stop to both controllers.
            errors = self._latch_hold_locked(reason, force_hardware_hold=True)
            if errors:
                return self._ack(sequence, False, f"hold_failed:{';'.join(errors)}")
        return self._ack(sequence, True, f"hold_latched:{reason}")

    def _latch_hold_locked(
        self, reason: str, *, force_hardware_hold: bool = False
    ) -> list[str]:
        errors: list[str] = []
        was_home = self._home_active
        if self._active or force_hardware_hold:
            for side in ("left", "right"):
                try:
                    if was_home or self._active or force_hardware_hold:
                        self._backend.stop(side, local_console=True)
                except Exception as exc:
                    errors.append(f"{side}:stop:{type(exc).__name__}:{exc}")
        self._active = False
        self._home_active = False
        self._home_started_ns = 0
        self._home_timeout_ns = 0
        self._home_signature = None
        self._home_phase = ""
        self._home_phase_started_ns = 0
        self._home_lift_timeout_ns = 0
        self._home_lift_targets = {}
        self._home_lift_active = ()
        self._home_lift_queue = []
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
