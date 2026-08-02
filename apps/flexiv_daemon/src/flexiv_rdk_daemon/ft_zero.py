"""Protected, auditable two-arm ZeroFTSensor maintenance transaction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hmac
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Callable, Protocol
import uuid

import numpy as np

from .backend import ArmBackend
from .configuration import ToolPayloadIdentity
from .guard import HardwareWriteGuard
from .model import (
    ObservationWindowStatistics,
    compute_window_statistics,
    finite_vector,
    residual_within_limits,
)

CONFIRMATION_TOKEN = "FLEXIV-FT-UNLOADED"


class MaintenanceInterlock(Protocol):
    """Bridge hook that excludes teleop, policy and replay during maintenance."""

    def acquire(self, session_id: str) -> None: ...
    def release(self, *, success: bool, connection_generation: int | None) -> None: ...


@dataclass(frozen=True)
class FTZeroConfig:
    sample_window_s: float = 2.0
    sample_rate_hz: float = 200.0
    min_samples: int = 100
    operational_timeout_s: float = 30.0
    primitive_timeout_s: float = 30.0
    poll_interval_s: float = 0.02
    max_joint_velocity_norm: float = 0.01
    max_tcp_velocity_norm: float = 0.01
    max_wrench_std_force_n: float = 0.5
    max_wrench_std_torque_nm: float = 0.05
    max_residual_force_n: float = 1.0
    max_residual_torque_nm: float = 0.1
    max_pre_external_mean_force_n: float = 3.0
    max_pre_external_mean_torque_nm: float = 0.3
    max_pre_external_peak_force_n: float = 5.0
    max_pre_external_peak_torque_nm: float = 0.5
    max_hand_delta: float = 5.0

    def __post_init__(self) -> None:
        numeric = (
            self.sample_window_s,
            self.sample_rate_hz,
            self.operational_timeout_s,
            self.primitive_timeout_s,
            self.poll_interval_s,
            self.max_joint_velocity_norm,
            self.max_tcp_velocity_norm,
            self.max_wrench_std_force_n,
            self.max_wrench_std_torque_nm,
            self.max_residual_force_n,
            self.max_residual_torque_nm,
            self.max_pre_external_mean_force_n,
            self.max_pre_external_mean_torque_nm,
            self.max_pre_external_peak_force_n,
            self.max_pre_external_peak_torque_nm,
            self.max_hand_delta,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in numeric):
            raise ValueError("F/T zero configuration values must be finite and non-negative")
        if self.sample_rate_hz <= 0.0 or self.min_samples <= 0:
            raise ValueError("sample_rate_hz and min_samples must be positive")


@dataclass(frozen=True)
class ZeroFTRequest:
    session_id: str
    operator_confirmation: str
    local_console: bool
    tool_payload_config_hash: str
    left_hand_position: np.ndarray
    right_hand_position: np.ndarray

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id is required")
        if len(self.tool_payload_config_hash) < 8:
            raise ValueError("tool_payload_config_hash is missing or implausibly short")
        object.__setattr__(
            self,
            "left_hand_position",
            finite_vector(self.left_hand_position, 6, "left_hand_position"),
        )
        object.__setattr__(
            self,
            "right_hand_position",
            finite_vector(self.right_hand_position, 6, "right_hand_position"),
        )


@dataclass(frozen=True)
class ArmZeroResult:
    side: str
    before: ObservationWindowStatistics
    after: ObservationWindowStatistics


@dataclass(frozen=True)
class ZeroFTResult:
    success: bool
    event_id: str
    session_id: str
    connection_generation: int | None
    left: ArmZeroResult | None = None
    right: ArmZeroResult | None = None
    failure_reason: str = ""
    available_statistics: dict[str, object] = field(default_factory=dict)
    failure_phase: str = ""
    cleanup: dict[str, str] = field(default_factory=dict)


class FTZeroFailure(RuntimeError):
    pass


class JsonlEventRecorder:
    """Append-only event sink with restrictive permissions."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not path.exists():
            path.touch(mode=0o600)
        path.chmod(0o600)
        self._lock = threading.Lock()

    def __call__(self, event: dict[str, object]) -> None:
        record = {"event_schema": "flexiv_ft_maintenance/v1", **event}
        line = json.dumps(record, separators=(",", ":"), sort_keys=True, allow_nan=False)
        with self._lock, self._path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())


class FTZeroManager:
    def __init__(
        self,
        backend: ArmBackend,
        *,
        write_guard: HardwareWriteGuard,
        interlock: MaintenanceInterlock,
        read_hand_positions: Callable[[], tuple[np.ndarray, np.ndarray]],
        event_sink: Callable[[dict[str, object]], None],
        config: FTZeroConfig | None = None,
        tool_payload_identity: Callable[[], ToolPayloadIdentity] | None = None,
        hand_monitor_snapshot: Callable[[], dict[str, object]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._backend = backend
        self._guard = write_guard
        self._interlock = interlock
        self._read_hand_positions = read_hand_positions
        self._event_sink = event_sink
        self._config = config or FTZeroConfig()
        self._tool_payload_identity = tool_payload_identity
        self._hand_monitor_snapshot = hand_monitor_snapshot
        self._sleep = sleep
        self._monotonic = monotonic
        self._transaction_lock = threading.Lock()
        self._zeroed_session: str | None = None
        self._zeroed_generation: int | None = None
        self._zeroed_tool_payload_fingerprint: str | None = None

    def is_zeroed_for(self, session_id: str) -> bool:
        generation = self._backend.connection_generation
        if self._zeroed_generation != generation:
            self.invalidate("rdk_connection_generation_changed")
        if (
            self._zeroed_tool_payload_fingerprint is not None
            and self._tool_payload_identity is not None
        ):
            try:
                current = self._tool_payload_identity()
            except Exception as exc:
                self.invalidate(
                    f"tool_payload_configuration_unreadable:{type(exc).__name__}"
                )
            else:
                if current.fingerprint != self._zeroed_tool_payload_fingerprint:
                    self.invalidate("tool_payload_configuration_changed")
        return self._zeroed_session == session_id and self._zeroed_generation == generation

    def invalidate(self, reason: str) -> None:
        if self._zeroed_session is not None or self._zeroed_generation is not None:
            self._event_sink(
                {
                    "event_type": "ft_zero_invalidated",
                    "session_id": self._zeroed_session,
                    "connection_generation": self._zeroed_generation,
                    "reason": reason,
                    "host_monotonic_ns": time.monotonic_ns(),
                }
            )
        self._zeroed_session = None
        self._zeroed_generation = None
        self._zeroed_tool_payload_fingerprint = None

    def zero(self, request: ZeroFTRequest) -> ZeroFTResult:
        """Run the complete left-then-right transaction.

        Any exception becomes a failed result, invalidates the zero status, and
        leaves the control interlock non-ready.
        """

        if not self._transaction_lock.acquire(blocking=False):
            raise FTZeroFailure("another F/T zero transaction is already active")
        event_id = str(uuid.uuid4())
        generation: int | None = None
        left_result: ArmZeroResult | None = None
        right_result: ArmZeroResult | None = None
        interlock_acquired = False
        writes_authorized = False
        phase = "validate_request"
        hand_before: tuple[np.ndarray, np.ndarray] | None = None
        hand_after: tuple[np.ndarray, np.ndarray] | None = None
        available_statistics: dict[str, object] = {}
        cleanup: dict[str, str] = {}
        tool_identity: ToolPayloadIdentity | None = None
        try:
            tool_identity = self._validate_request(request)
            self._guard.require(
                "ZeroFTSensor transaction",
                local_console=request.local_console,
            )
            writes_authorized = True
            generation = self._backend.connection_generation
            self.invalidate("new_ft_zero_transaction")
            self._interlock.acquire(request.session_id)
            interlock_acquired = True
            phase = "hand_preflight"
            hand_before = self._read_hands()
            self._verify_hand_reference(request, hand_before)
            self._emit_start(event_id, request, generation, hand_before)

            # One simultaneous dual-arm window establishes the initial
            # unloaded state. The non-active arm then remains continuously
            # monitored during every primitive poll and post window.
            phase = "dual_arm_initial_pre_window"
            left_before, right_initial = self._sample_dual_window(
                generation, hand_before
            )
            available_statistics["left_before"] = asdict(left_before)
            available_statistics["right_before"] = asdict(right_initial)
            for stats in (left_before, right_initial):
                self._reject_pre_zero_external_contact(stats)
                if not stats.stable:
                    raise FTZeroFailure(
                        f"{stats.side} pre-zero stability check failed: "
                        f"{stats.rejection_reason}"
                    )

            phase = "left_zero_and_post_window"
            left_after = self._zero_one("left", generation, hand_before)
            available_statistics["left_after"] = asdict(left_after)
            left_result = ArmZeroResult("left", left_before, left_after)

            # The right-arm pre-zero statistics must be immediately fresh; the
            # initial window cannot be reused after the complete left operation.
            phase = "right_fresh_pre_window"
            _, right_before = self._sample_dual_window(generation, hand_before)
            available_statistics["right_before"] = asdict(right_before)
            self._reject_pre_zero_external_contact(right_before)
            if not right_before.stable:
                raise FTZeroFailure(
                    "right pre-zero stability check failed: "
                    f"{right_before.rejection_reason}"
                )
            phase = "right_zero_and_post_window"
            right_after = self._zero_one("right", generation, hand_before)
            available_statistics["right_after"] = asdict(right_after)
            right_result = ArmZeroResult("right", right_before, right_after)

            hand_after = self._read_hands()
            self._verify_hands_unchanged(hand_before, hand_after)
            self._verify_hand_monitor_unchanged()
            self._observe_dual_transaction(generation)
            if self._backend.connection_generation != generation:
                raise FTZeroFailure("RDK reconnected during F/T zero transaction")
            if tool_identity is not None:
                current_tool_identity = self._tool_payload_identity()
                if current_tool_identity.fingerprint != tool_identity.fingerprint:
                    raise FTZeroFailure(
                        "local tool/payload configuration changed during F/T zero"
                    )

            self._zeroed_session = request.session_id
            self._zeroed_generation = generation
            self._zeroed_tool_payload_fingerprint = (
                None if tool_identity is None else tool_identity.fingerprint
            )
            self._interlock.release(success=True, connection_generation=generation)
            interlock_acquired = False
            if self._hand_monitor_snapshot is not None:
                available_statistics["hand_monitor"] = self._hand_monitor_snapshot()
            result = ZeroFTResult(
                success=True,
                event_id=event_id,
                session_id=request.session_id,
                connection_generation=generation,
                left=left_result,
                right=right_result,
                available_statistics=available_statistics.copy(),
                cleanup=cleanup.copy(),
            )
            self._event_sink(
                self._result_event(
                    result,
                    request,
                    hand_before,
                    hand_after,
                    available_statistics,
                    "complete",
                    cleanup,
                )
            )
            return result
        except Exception as exc:
            self._zeroed_session = None
            self._zeroed_generation = None
            if writes_authorized:
                for side in ("left", "right"):
                    try:
                        self._backend.switch_idle(side, local_console=True)
                        cleanup[f"{side}_idle"] = "ok"
                    except Exception as cleanup_exc:
                        cleanup[f"{side}_idle"] = (
                            f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                        )
            try:
                hand_after = self._read_hands()
            except Exception as hand_exc:
                cleanup["hand_after"] = f"{type(hand_exc).__name__}: {hand_exc}"
            if interlock_acquired:
                self._interlock.release(success=False, connection_generation=None)
            if self._hand_monitor_snapshot is not None:
                available_statistics["hand_monitor"] = self._hand_monitor_snapshot()
            result = ZeroFTResult(
                success=False,
                event_id=event_id,
                session_id=request.session_id,
                connection_generation=generation,
                left=left_result,
                right=right_result,
                failure_reason=f"{type(exc).__name__}: {exc}",
                available_statistics=available_statistics.copy(),
                failure_phase=phase,
                cleanup=cleanup.copy(),
            )
            self._event_sink(
                self._result_event(
                    result,
                    request,
                    hand_before,
                    hand_after,
                    available_statistics,
                    phase,
                    cleanup,
                )
            )
            return result
        finally:
            self._transaction_lock.release()

    def _validate_request(
        self, request: ZeroFTRequest
    ) -> ToolPayloadIdentity | None:
        if not request.local_console:
            raise FTZeroFailure("remote clients cannot trigger or confirm F/T zero")
        if request.operator_confirmation != CONFIRMATION_TOKEN:
            raise FTZeroFailure("operator confirmation token is incorrect")
        if self._tool_payload_identity is None:
            return None
        identity = self._tool_payload_identity()
        if not hmac.compare_digest(
            request.tool_payload_config_hash, identity.sha256
        ):
            raise FTZeroFailure(
                "tool/payload config hash does not match daemon-owned local config"
            )
        return identity

    def _read_hands(self) -> tuple[np.ndarray, np.ndarray]:
        left, right = self._read_hand_positions()
        return finite_vector(left, 6, "measured_left_hand"), finite_vector(right, 6, "measured_right_hand")

    def _verify_hand_reference(
        self,
        request: ZeroFTRequest,
        measured: tuple[np.ndarray, np.ndarray],
    ) -> None:
        if np.max(np.abs(measured[0] - request.left_hand_position)) > self._config.max_hand_delta:
            raise FTZeroFailure("left hand does not match the confirmed unloaded pose")
        if np.max(np.abs(measured[1] - request.right_hand_position)) > self._config.max_hand_delta:
            raise FTZeroFailure("right hand does not match the confirmed unloaded pose")

    def _verify_hands_unchanged(
        self,
        reference: tuple[np.ndarray, np.ndarray],
        current: tuple[np.ndarray, np.ndarray],
    ) -> None:
        if np.max(np.abs(current[0] - reference[0])) > self._config.max_hand_delta:
            raise FTZeroFailure("left hand moved during F/T zero")
        if np.max(np.abs(current[1] - reference[1])) > self._config.max_hand_delta:
            raise FTZeroFailure("right hand moved during F/T zero")

    def _sample_window(
        self,
        side: str,
        generation: int,
        hand_reference: tuple[np.ndarray, np.ndarray],
    ) -> ObservationWindowStatistics:
        config = self._config
        start = self._monotonic()
        deadline = start + config.sample_window_s
        samples = []
        period = 1.0 / config.sample_rate_hz
        while self._monotonic() < deadline or len(samples) < config.min_samples:
            dual = self._observe_dual_transaction(generation)
            samples.append(getattr(dual, side))
            self._verify_hands_unchanged(hand_reference, self._read_hands())
            self._verify_hand_monitor_unchanged()
            if period > 0.0:
                self._sleep(period)
        self._verify_hands_unchanged(hand_reference, self._read_hands())
        self._verify_hand_monitor_unchanged()
        return compute_window_statistics(
            samples,
            duration_s=max(0.0, self._monotonic() - start),
            max_joint_velocity_norm=config.max_joint_velocity_norm,
            max_tcp_velocity_norm=config.max_tcp_velocity_norm,
            max_wrench_std_force_n=config.max_wrench_std_force_n,
            max_wrench_std_torque_nm=config.max_wrench_std_torque_nm,
        )

    def _sample_dual_window(
        self,
        generation: int,
        hand_reference: tuple[np.ndarray, np.ndarray],
    ) -> tuple[ObservationWindowStatistics, ObservationWindowStatistics]:
        config = self._config
        start = self._monotonic()
        deadline = start + config.sample_window_s
        samples = {"left": [], "right": []}
        period = 1.0 / config.sample_rate_hz
        while (
            self._monotonic() < deadline
            or len(samples["left"]) < config.min_samples
        ):
            dual = self._observe_dual_transaction(generation)
            samples["left"].append(dual.left)
            samples["right"].append(dual.right)
            self._verify_hands_unchanged(hand_reference, self._read_hands())
            self._verify_hand_monitor_unchanged()
            if period > 0.0:
                self._sleep(period)
        duration = max(0.0, self._monotonic() - start)
        return tuple(
            compute_window_statistics(
                samples[side],
                duration_s=duration,
                max_joint_velocity_norm=config.max_joint_velocity_norm,
                max_tcp_velocity_norm=config.max_tcp_velocity_norm,
                max_wrench_std_force_n=config.max_wrench_std_force_n,
                max_wrench_std_torque_nm=config.max_wrench_std_torque_nm,
            )
            for side in ("left", "right")
        )

    def _observe_dual_transaction(self, generation: int):
        if self._backend.connection_generation != generation:
            raise FTZeroFailure("RDK reconnected during F/T zero transaction")
        try:
            dual = self._backend.observe_both()
        except Exception as exc:
            if self._backend.connection_generation != generation:
                raise FTZeroFailure(
                    "RDK reconnected during F/T zero transaction"
                ) from exc
            raise
        for sample in (dual.left, dual.right):
            if sample.connection_generation != generation:
                raise FTZeroFailure("RDK reconnected during F/T zero transaction")
            if not sample.connected or sample.fault:
                raise FTZeroFailure(
                    f"{sample.side} disconnected or faulted during F/T zero"
                )
            dq_norm = float(np.linalg.norm(sample.dq))
            tcp_norm = float(np.linalg.norm(sample.tcp_velocity))
            if (
                dq_norm > self._config.max_joint_velocity_norm
                or tcp_norm > self._config.max_tcp_velocity_norm
            ):
                raise FTZeroFailure(
                    f"{sample.side} stability/motion check failed during "
                    f"F/T zero: dq={dq_norm:.6g},tcp={tcp_norm:.6g}"
                )
            external = sample.external_wrench
            if (
                np.linalg.norm(external[:3])
                > self._config.max_pre_external_peak_force_n
                or np.linalg.norm(external[3:])
                > self._config.max_pre_external_peak_torque_nm
            ):
                raise FTZeroFailure(
                    f"{sample.side} external contact detected during F/T zero"
                )
        return dual

    def _verify_hand_monitor_unchanged(self) -> None:
        if self._hand_monitor_snapshot is None:
            return
        snapshot = self._hand_monitor_snapshot()
        for side in ("left", "right"):
            delta = float(snapshot.get(f"{side}_max_delta", 0.0))
            if not math.isfinite(delta) or delta > self._config.max_hand_delta:
                raise FTZeroFailure(
                    f"{side} hand moved during F/T zero (latched delta={delta})"
                )

    def _reject_pre_zero_external_contact(
        self, stats: ObservationWindowStatistics
    ) -> None:
        external = stats.external_wrench
        mean_force = float(np.linalg.norm(external.mean[:3]))
        mean_torque = float(np.linalg.norm(external.mean[3:]))
        peak_force = float(np.linalg.norm(external.peak_absolute[:3]))
        peak_torque = float(np.linalg.norm(external.peak_absolute[3:]))
        limits = self._config
        reasons: list[str] = []
        if mean_force > limits.max_pre_external_mean_force_n:
            reasons.append(f"mean_force={mean_force:.6g}N")
        if mean_torque > limits.max_pre_external_mean_torque_nm:
            reasons.append(f"mean_torque={mean_torque:.6g}Nm")
        if peak_force > limits.max_pre_external_peak_force_n:
            reasons.append(f"peak_force={peak_force:.6g}N")
        if peak_torque > limits.max_pre_external_peak_torque_nm:
            reasons.append(f"peak_torque={peak_torque:.6g}Nm")
        if reasons:
            raise FTZeroFailure(
                f"{stats.side} pre-zero external contact detected: "
                + ",".join(reasons)
            )

    def _zero_one(
        self,
        side: str,
        generation: int,
        hand_reference: tuple[np.ndarray, np.ndarray],
    ) -> ObservationWindowStatistics:
        self._observe_dual_transaction(generation)
        self._verify_hand_monitor_unchanged()
        self._backend.enable(side, local_console=True)
        self._observe_dual_transaction(generation)
        operational_deadline = self._monotonic() + self._config.operational_timeout_s
        while not self._backend.operational(side):
            self._observe_dual_transaction(generation)
            self._verify_hands_unchanged(hand_reference, self._read_hands())
            self._verify_hand_monitor_unchanged()
            self._check_deadline(operational_deadline, f"{side} operational timeout", generation)
            self._sleep(self._config.poll_interval_s)
        self._observe_dual_transaction(generation)
        self._backend.switch_primitive_mode(side, local_console=True)
        self._observe_dual_transaction(generation)
        self._backend.execute_zero_ft(side, local_console=True)
        primitive_deadline = self._monotonic() + self._config.primitive_timeout_s
        while True:
            self._observe_dual_transaction(generation)
            self._verify_hands_unchanged(hand_reference, self._read_hands())
            self._verify_hand_monitor_unchanged()
            state = self._backend.primitive_state(side)
            if self._primitive_failed(state):
                raise FTZeroFailure(f"{side} ZeroFTSensor primitive failed: {dict(state)}")
            if self._primitive_terminated(state):
                break
            self._check_deadline(primitive_deadline, f"{side} ZeroFTSensor timeout", generation)
            self._sleep(self._config.poll_interval_s)
        after = self._sample_window(side, generation, hand_reference)
        if not after.stable:
            raise FTZeroFailure(f"{side} post-zero stability check failed: {after.rejection_reason}")
        residual_ok, residual_reason = residual_within_limits(
            after,
            max_force_n=self._config.max_residual_force_n,
            max_torque_nm=self._config.max_residual_torque_nm,
        )
        if not residual_ok:
            raise FTZeroFailure(f"{side} post-zero residual check failed: {residual_reason}")
        self._backend.switch_idle(side, local_console=True)
        self._backend.switch_cartesian_mode(side, local_console=True)
        self._backend.rebase_from_measurement(side)
        self._observe_dual_transaction(generation)
        self._verify_hand_monitor_unchanged()
        return after

    def _check_deadline(self, deadline: float, reason: str, generation: int) -> None:
        if self._backend.connection_generation != generation:
            raise FTZeroFailure("RDK reconnected during F/T zero")
        if self._monotonic() >= deadline:
            raise FTZeroFailure(reason)

    @staticmethod
    def _primitive_terminated(state: object) -> bool:
        if not isinstance(state, dict):
            return False
        value = state.get("terminated", False)
        return value is True or value == 1 or (isinstance(value, str) and value.lower() in {"true", "1"})

    @staticmethod
    def _primitive_failed(state: object) -> bool:
        if not isinstance(state, dict):
            return True
        failed = state.get("failed", False)
        error = state.get("error", "")
        return failed is True or failed == 1 or bool(error)

    def _emit_start(
        self,
        event_id: str,
        request: ZeroFTRequest,
        generation: int,
        hands: tuple[np.ndarray, np.ndarray],
    ) -> None:
        self._event_sink(
            {
                "event_type": "ft_zero_started",
                "event_id": event_id,
                "session_id": request.session_id,
                "connection_generation": generation,
                "tool_payload_config_hash": request.tool_payload_config_hash,
                "operator_confirmation": True,
                "left_hand_position": hands[0].tolist(),
                "right_hand_position": hands[1].tolist(),
                "host_monotonic_ns": time.monotonic_ns(),
            }
        )

    @staticmethod
    def _result_event(
        result: ZeroFTResult,
        request: ZeroFTRequest,
        hands_before: tuple[np.ndarray, np.ndarray] | None,
        hands_after: tuple[np.ndarray, np.ndarray] | None,
        available_statistics: dict[str, object],
        phase: str,
        cleanup: dict[str, str],
    ) -> dict[str, object]:
        def arm(value: ArmZeroResult | None) -> object:
            return None if value is None else asdict(value)

        return {
            "event_type": "ft_zero_completed" if result.success else "ft_zero_failed",
            "event_id": result.event_id,
            "session_id": result.session_id,
            "success": result.success,
            "failure_reason": result.failure_reason,
            "connection_generation": result.connection_generation,
            "tool_payload_config_hash": request.tool_payload_config_hash,
            "left": arm(result.left),
            "right": arm(result.right),
            "available_statistics": available_statistics,
            "phase": phase,
            "cleanup": cleanup,
            "left_hand_before": None if hands_before is None else hands_before[0].tolist(),
            "right_hand_before": None if hands_before is None else hands_before[1].tolist(),
            "left_hand_after": None if hands_after is None else hands_after[0].tolist(),
            "right_hand_after": None if hands_after is None else hands_after[1].tolist(),
            "host_monotonic_ns": time.monotonic_ns(),
        }
