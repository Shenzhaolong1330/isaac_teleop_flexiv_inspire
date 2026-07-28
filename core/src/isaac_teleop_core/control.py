"""Fail-closed source arbitration and control state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
import time
from typing import Callable

from .command import BimanualCommand, CommandSource, ValidMask, hold_command


class ControlState(str, Enum):
    DISABLED = "DISABLED"
    MAINTENANCE = "MAINTENANCE"
    FT_ZEROED = "FT_ZEROED"
    READY = "READY"
    TELEOP_ARMED = "TELEOP_ARMED"
    POLICY_ARMED = "POLICY_ARMED"
    REPLAY_ARMED = "REPLAY_ARMED"
    ACTIVE = "ACTIVE"
    HOLD_LATCHED = "HOLD_LATCHED"
    FAULT = "FAULT"


class HoldReason(str, Enum):
    NONE = "none"
    LOCAL_PERMISSION = "local_permission_missing"
    FT_NOT_ZEROED = "ft_not_zeroed"
    PEDAL_RELEASED = "physical_pedal_released"
    DEADMAN_RELEASED = "source_deadman_released"
    HEARTBEAT_STALE = "source_heartbeat_stale"
    COMMAND_STALE = "command_stale"
    SOURCE_MISMATCH = "source_mismatch"
    SESSION_MISMATCH = "session_mismatch"
    SEQUENCE = "non_monotonic_sequence"
    ARM_OFFLINE = "arm_offline"
    HAND_OFFLINE = "hand_offline"
    LIMIT = "safety_limit"
    COLLISION = "collision"
    HARDWARE_FAULT = "hardware_fault"
    STOP = "stop_requested"
    INVALID_COMMAND = "invalid_command"


@dataclass(frozen=True)
class GateInputs:
    local_permission: bool = False
    physical_pedal: bool = False
    arms_online: bool = False
    hands_online: bool = False
    limits_ok: bool = False
    collision_clear: bool = False
    hardware_fault: bool = False


@dataclass(frozen=True)
class ControlSnapshot:
    state: ControlState
    session_id: str | None
    active_source: CommandSource | None
    hold_reason: HoldReason
    ft_zero_generation: int | None
    last_requested: BimanualCommand | None
    last_safe: BimanualCommand | None
    last_sent: BimanualCommand | None


class TransitionError(RuntimeError):
    pass


class ControlArbiter:
    """Thread-safe, non-resuming arbiter for teleop/policy/replay.

    A valid command is exposed as `last_safe`; this class never talks to
    hardware. The hardware bridge must mark a command as sent only after its own
    final checks and successful IPC acknowledgement.
    """

    def __init__(
        self,
        *,
        heartbeat_timeouts_s: dict[CommandSource, float] | None = None,
        policy_period_s: float | None = None,
        command_validator: Callable[[BimanualCommand], None] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._state = ControlState.DISABLED
        self._session_id: str | None = None
        self._active_source: CommandSource | None = None
        self._hold_reason = HoldReason.NONE
        self._ft_zero_generation: int | None = None
        self._gates = GateInputs()
        self._heartbeats: dict[CommandSource, int] = {}
        self._last_sequence: dict[CommandSource, int] = {}
        self._last_requested: BimanualCommand | None = None
        self._last_safe: BimanualCommand | None = None
        self._last_sent: BimanualCommand | None = None
        self._deadman_release_seen = False
        self._validator = command_validator
        configured = heartbeat_timeouts_s or {
            CommandSource.TELEOP: 0.100,
            CommandSource.POLICY: 0.200,
            CommandSource.REPLAY: 0.200,
        }
        self._heartbeat_timeouts_ns = {
            source: int(timeout * 1e9) for source, timeout in configured.items()
        }
        if policy_period_s is not None:
            self.set_policy_period(policy_period_s)

    def set_policy_period(self, period_s: float) -> None:
        if period_s <= 0.0:
            raise ValueError("policy period must be positive")
        with self._lock:
            self._heartbeat_timeouts_ns[CommandSource.POLICY] = int(
                max(0.200, 3.0 * period_s) * 1e9
            )

    @property
    def snapshot(self) -> ControlSnapshot:
        with self._lock:
            return ControlSnapshot(
                state=self._state,
                session_id=self._session_id,
                active_source=self._active_source,
                hold_reason=self._hold_reason,
                ft_zero_generation=self._ft_zero_generation,
                last_requested=self._last_requested,
                last_safe=self._last_safe,
                last_sent=self._last_sent,
            )

    def begin_hardware_session(self, session_id: str) -> None:
        """Invalidate any previous F/T zero and enter maintenance."""

        if not session_id:
            raise ValueError("session_id is required")
        with self._lock:
            self._session_id = session_id
            self._active_source = None
            self._ft_zero_generation = None
            self._heartbeats.clear()
            self._last_sequence.clear()
            self._clear_commands()
            self._hold_reason = HoldReason.NONE
            self._deadman_release_seen = False
            self._state = ControlState.MAINTENANCE

    def on_rdk_reconnect(self) -> None:
        with self._lock:
            self._ft_zero_generation = None
            self._active_source = None
            self._clear_commands()
            self._hold_reason = HoldReason.FT_NOT_ZEROED
            self._state = ControlState.MAINTENANCE

    def mark_ft_zeroed(self, *, session_id: str, connection_generation: int) -> None:
        with self._lock:
            if self._state is not ControlState.MAINTENANCE:
                raise TransitionError("F/T zero result is accepted only in MAINTENANCE")
            if session_id != self._session_id:
                raise TransitionError("F/T zero session does not match current hardware session")
            if connection_generation < 0:
                raise ValueError("connection_generation cannot be negative")
            self._ft_zero_generation = connection_generation
            self._hold_reason = HoldReason.NONE
            self._state = ControlState.FT_ZEROED

    def declare_ready(self, *, connection_generation: int) -> None:
        with self._lock:
            if self._state is not ControlState.FT_ZEROED:
                raise TransitionError("READY requires FT_ZEROED")
            if self._ft_zero_generation != connection_generation:
                self._state = ControlState.MAINTENANCE
                self._ft_zero_generation = None
                raise TransitionError("RDK connection changed after F/T zero")
            self._state = ControlState.READY

    def update_gates(self, gates: GateInputs, *, now_monotonic_ns: int | None = None) -> None:
        now = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        with self._lock:
            self._gates = gates
            if self._state is ControlState.ACTIVE:
                reason = self._gate_failure(now, require_command=True)
                if reason is not HoldReason.NONE:
                    self._latch(reason, now)

    def heartbeat(self, source: CommandSource, *, now_monotonic_ns: int | None = None) -> None:
        now = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        with self._lock:
            self._heartbeats[CommandSource(source)] = now

    def arm(self, source: CommandSource) -> None:
        source = CommandSource(source)
        with self._lock:
            if self._state is not ControlState.READY:
                raise TransitionError("a command source can only arm from READY")
            if not self._gates.local_permission:
                raise TransitionError("local permission is required before arming")
            if self._active_source is not None:
                raise TransitionError("another command source already owns control")
            self._active_source = source
            self._hold_reason = HoldReason.NONE
            self._state = {
                CommandSource.TELEOP: ControlState.TELEOP_ARMED,
                CommandSource.POLICY: ControlState.POLICY_ARMED,
                CommandSource.REPLAY: ControlState.REPLAY_ARMED,
            }[source]

    def disarm(self) -> None:
        with self._lock:
            if self._state is ControlState.ACTIVE:
                self._latch(HoldReason.STOP, time.monotonic_ns())
                return
            if self._state in {
                ControlState.TELEOP_ARMED,
                ControlState.POLICY_ARMED,
                ControlState.REPLAY_ARMED,
            }:
                self._active_source = None
                self._clear_commands()
                self._state = ControlState.READY

    def submit(
        self,
        command: BimanualCommand,
        *,
        now_monotonic_ns: int | None = None,
    ) -> BimanualCommand:
        """Validate and approve a command, or latch hold on an active-chain fault."""

        now = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        with self._lock:
            self._last_requested = command
            try:
                self._validate_envelope(command, now)
                if self._validator is not None:
                    self._validator(command)
            except Exception:
                if self._is_armed_or_active():
                    self._latch(HoldReason.INVALID_COMMAND, now)
                raise
            if not command.deadman:
                if self._is_armed_or_active():
                    self._latch(HoldReason.DEADMAN_RELEASED, now)
                    self._deadman_release_seen = True
                raise TransitionError("deadman is not asserted")
            reason = self._gate_failure(now, require_command=False)
            if reason is not HoldReason.NONE:
                if self._is_armed_or_active():
                    self._latch(reason, now)
                raise TransitionError(f"control gate rejected command: {reason.value}")
            self._last_safe = command
            self._state = ControlState.ACTIVE
            return command

    def reject_invalid_command(
        self,
        source: CommandSource,
        *,
        now_monotonic_ns: int | None = None,
    ) -> None:
        """Latch a malformed ROS/gRPC packet that could not be constructed."""

        now = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        with self._lock:
            if source == self._active_source and self._is_armed_or_active():
                self._latch(HoldReason.INVALID_COMMAND, now)

    def tick(self, *, now_monotonic_ns: int | None = None) -> ControlSnapshot:
        now = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        with self._lock:
            if self._state is ControlState.ACTIVE:
                reason = self._gate_failure(now, require_command=True)
                if reason is not HoldReason.NONE:
                    self._latch(reason, now)
            return self.snapshot

    def mark_sent(self, command: BimanualCommand) -> None:
        with self._lock:
            if self._state is not ControlState.ACTIVE:
                raise TransitionError("cannot mark a command sent outside ACTIVE")
            if command is not self._last_safe:
                raise TransitionError("only the currently approved safe command may be sent")
            self._last_sent = command

    def stop(self) -> None:
        with self._lock:
            if self._state in {
                ControlState.ACTIVE,
                ControlState.TELEOP_ARMED,
                ControlState.POLICY_ARMED,
                ControlState.REPLAY_ARMED,
            }:
                self._latch(HoldReason.STOP, time.monotonic_ns())

    def fault(self, reason: HoldReason = HoldReason.HARDWARE_FAULT) -> None:
        with self._lock:
            self._clear_commands()
            self._active_source = None
            self._hold_reason = reason
            self._state = ControlState.FAULT

    def observe_deadman_released(self, source: CommandSource) -> None:
        with self._lock:
            if source == self._active_source:
                self._deadman_release_seen = True

    def clear_hold(self, *, local_acknowledged: bool) -> None:
        """Clear a latch only after local acknowledgement and deadman release."""

        with self._lock:
            if self._state is not ControlState.HOLD_LATCHED:
                raise TransitionError("no HOLD_LATCHED state to clear")
            if not local_acknowledged:
                raise TransitionError("local acknowledgement is required")
            if not self._deadman_release_seen:
                raise TransitionError("source deadman must be released before rearming")
            if self._ft_zero_generation is None:
                self._state = ControlState.MAINTENANCE
            else:
                self._state = ControlState.READY
            self._active_source = None
            self._hold_reason = HoldReason.NONE
            self._deadman_release_seen = False
            self._clear_commands()

    def _validate_envelope(self, command: BimanualCommand, now: int) -> None:
        if self._state not in {
            ControlState.TELEOP_ARMED,
            ControlState.POLICY_ARMED,
            ControlState.REPLAY_ARMED,
            ControlState.ACTIVE,
        }:
            raise TransitionError("command source is not armed")
        if command.session_id != self._session_id:
            raise TransitionError(HoldReason.SESSION_MISMATCH.value)
        if command.source != self._active_source:
            raise TransitionError(HoldReason.SOURCE_MISMATCH.value)
        previous_sequence = self._last_sequence.get(command.source, -1)
        if command.sequence <= previous_sequence:
            raise TransitionError(HoldReason.SEQUENCE.value)
        if not command.is_fresh(now):
            raise TransitionError(HoldReason.COMMAND_STALE.value)
        if command.valid_mask == ValidMask.NONE:
            raise TransitionError("hold command cannot activate control")
        self._last_sequence[command.source] = command.sequence

    def _gate_failure(self, now: int, *, require_command: bool) -> HoldReason:
        gates = self._gates
        if gates.hardware_fault:
            return HoldReason.HARDWARE_FAULT
        if self._ft_zero_generation is None:
            return HoldReason.FT_NOT_ZEROED
        if not gates.local_permission:
            return HoldReason.LOCAL_PERMISSION
        if not gates.physical_pedal:
            return HoldReason.PEDAL_RELEASED
        if not gates.arms_online:
            return HoldReason.ARM_OFFLINE
        if not gates.hands_online:
            return HoldReason.HAND_OFFLINE
        if not gates.limits_ok:
            return HoldReason.LIMIT
        if not gates.collision_clear:
            return HoldReason.COLLISION
        if self._active_source is None:
            return HoldReason.SOURCE_MISMATCH
        heartbeat = self._heartbeats.get(self._active_source)
        timeout = self._heartbeat_timeouts_ns[self._active_source]
        if heartbeat is None or now - heartbeat > timeout:
            return HoldReason.HEARTBEAT_STALE
        if require_command:
            if self._last_safe is None or not self._last_safe.is_fresh(now):
                return HoldReason.COMMAND_STALE
            if not self._last_safe.deadman:
                return HoldReason.DEADMAN_RELEASED
        return HoldReason.NONE

    def _latch(self, reason: HoldReason, now: int) -> None:
        source = self._active_source or CommandSource.TELEOP
        session = self._session_id or "no-session"
        sequence = max(self._last_sequence.get(source, 0) + 1, 1)
        explicit_hold = hold_command(
            session_id=session,
            source=source,
            sequence=sequence,
            now_monotonic_ns=now,
            reason=reason.value,
        )
        self._last_safe = explicit_hold
        self._last_sent = None
        self._hold_reason = reason
        self._deadman_release_seen = False
        self._state = ControlState.HOLD_LATCHED

    def _is_armed_or_active(self) -> bool:
        return self._state in {
            ControlState.TELEOP_ARMED,
            ControlState.POLICY_ARMED,
            ControlState.REPLAY_ARMED,
            ControlState.ACTIVE,
        }

    def _clear_commands(self) -> None:
        self._last_requested = None
        self._last_safe = None
        self._last_sent = None
