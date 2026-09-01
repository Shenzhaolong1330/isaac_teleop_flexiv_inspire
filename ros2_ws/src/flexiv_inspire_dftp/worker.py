"""One deterministic I/O worker per DFTP hand."""

from __future__ import annotations

import threading
import time
from typing import Callable, Protocol

from .models import Acquisition, HandCommand, HandState, TactileFrame, TactileSurface
from .profiles import HandProfile, hand_profile
from .protocol import (
    HAND_STATE_BLOCK_START,
    HAND_STATE_BLOCK_WORDS,
    Register,
    TACTILE_LAYOUT,
    TOTAL_TAXELS,
    decode_hand_state_block,
    decode_tactile,
)


class Reader(Protocol):
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def read_raw(self, byte_address: int, word_count: int) -> bytes: ...


class CommandSink(Protocol):
    def apply(self, command: HandCommand) -> None: ...


class LatestOnlyMailbox:
    """A single-slot mailbox; a newer sequence atomically replaces the old one."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: HandCommand | None = None
        self._last_taken_sequence = -1

    def put(self, command: HandCommand) -> bool:
        with self._lock:
            newest = max(
                self._last_taken_sequence,
                -1 if self._pending is None else self._pending.sequence,
            )
            if command.sequence <= newest:
                return False
            self._pending = command
            return True

    def take(self) -> HandCommand | None:
        with self._lock:
            command, self._pending = self._pending, None
            if command is not None:
                self._last_taken_sequence = command.sequence
            return command

    def discard(self) -> None:
        """Discard queued work and remember its sequence."""
        with self._lock:
            if self._pending is not None:
                self._last_taken_sequence = self._pending.sequence
            self._pending = None

    @property
    def pending_sequence(self) -> int | None:
        with self._lock:
            return None if self._pending is None else self._pending.sequence


class DftpProtocolReader:
    """Read-only protocol decoder.  It does not expose a write method."""

    def __init__(
        self,
        transport: Reader,
        side: str,
        clock_ns: Callable[[], int],
        *,
        profile: HandProfile | None = None,
    ) -> None:
        self.transport = transport
        self.side = side
        self.clock_ns = clock_ns
        self.profile = profile or hand_profile("rh56dftp_2", side=side)
        self.tactile_layout = self.profile.tactile_layout
        self.tactile_taxels = self.profile.tactile_taxel_count
        self.state_sequence = 0
        self.tactile_sequence = 0

    def connect(self) -> None:
        self.transport.connect()

    def close(self) -> None:
        self.transport.close()

    def read_state(self) -> HandState:
        frame_start = self.clock_ns()
        payload = self.transport.read_raw(
            HAND_STATE_BLOCK_START, HAND_STATE_BLOCK_WORDS
        )
        frame_end = self.clock_ns()
        values = decode_hand_state_block(payload)
        # Every field is sampled atomically by the same Modbus transaction.
        field_times = {
            name: (frame_start, frame_end)
            for name in (
                "position",
                "angle",
                "force",
                "current",
                "error",
                "status",
                "temperature",
            )
        }
        self.state_sequence += 1
        return HandState(
            side=self.side,
            actuator_position=values["position"],
            actuator_angle=self.profile.canonicalize_angles(
                values["position"], values["angle"]
            ),
            actual_force_g=values["force"],
            current_ma=values["current"],
            temperature_c=values["temperature"],
            error_code=values["error"],
            status_code=values["status"],
            acquisition=Acquisition(
                # DFTP exposes no device clock; midpoint is the least biased
                # host-clock estimate and field_times preserve the uncertainty.
                source_time_ns=(frame_start + frame_end) // 2,
                host_receive_time_ns=frame_end,
                sequence=self.state_sequence,
            ),
            field_times_ns=field_times,
        )

    def read_surface(self, spec) -> TactileSurface:
        start = self.clock_ns()
        payload = self.transport.read_raw(spec.start_address, spec.taxels)
        end = self.clock_ns()
        return TactileSurface(
            name=spec.name,
            rows=spec.rows,
            cols=spec.cols,
            values=decode_tactile(payload, spec),
            acquisition_start_ns=start,
            acquisition_end_ns=end,
        )


class DftpCommandSink:
    """Thin motion sink, constructed only with a command-capable transport."""

    def __init__(self, transport) -> None:
        if not hasattr(transport, "write_i16"):
            raise TypeError("command sink requires a command-capable transport")
        self.transport = transport

    def apply(self, command: HandCommand) -> None:
        # Set conservative force limits before commanding an angle.
        self.transport.write_i16(Register.FORCE_LIMIT, command.force_limits)
        self.transport.write_i16(Register.ANGLE_TARGET, command.angles)


class DftpHandWorker:
    """Own deterministic state/command I/O for one hand.

    Production supplies a second read-only connection for tactile acquisition,
    so its 17 transactions cannot block the 200 Hz state/command path.  The
    single-reader fallback is retained for focused tests and compatibility.
    """

    def __init__(
        self,
        side: str,
        reader: DftpProtocolReader,
        *,
        tactile_reader: DftpProtocolReader | None = None,
        command_sink: CommandSink | None = None,
        state_hz: float = 50.0,
        tactile_hz: float = 30.0,
        safe_force_limits: tuple[int, ...] = (600, 600, 600, 600, 500, 500),
        clock_ns: Callable[[], int] = time.monotonic_ns,
        observation_clock_ns: Callable[[], int] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        on_state: Callable[[HandState], None] | None = None,
        on_tactile: Callable[[TactileFrame], None] | None = None,
        on_fault: Callable[[str], None] | None = None,
    ) -> None:
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        if not 1.0 <= state_hz <= 200.0:
            raise ValueError("state_hz must be in 1..200")
        if not 1.0 <= tactile_hz <= 40.0:
            raise ValueError("tactile_hz must be in 1..40")
        if len(safe_force_limits) != 6 or any(
            value < 0 or value > 3000 for value in safe_force_limits
        ):
            raise ValueError("safe_force_limits must be six values in 0..3000")
        self.side = side
        self.reader = reader
        self.tactile_reader = tactile_reader
        self.command_sink = command_sink
        self.state_period_ns = round(1e9 / state_hz)
        self.tactile_period_ns = round(1e9 / tactile_hz)
        self.safe_force_limits = tuple(safe_force_limits)
        self.clock_ns = clock_ns
        # Scheduling and watchdog deadlines stay monotonic. Observations use
        # one explicit clock shared with DftpProtocolReader, so frame bounds
        # cannot mix monotonic time with wall-clock surface timestamps.
        self.observation_clock_ns = observation_clock_ns or getattr(
            tactile_reader or reader, "clock_ns", clock_ns
        )
        self.sleeper = sleeper
        self.on_state = on_state or (lambda _state: None)
        self.on_tactile = on_tactile or (lambda _frame: None)
        self.on_fault = on_fault or (lambda _message: None)
        self.mailbox = LatestOnlyMailbox()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tactile_thread: threading.Thread | None = None
        self._latest_state: HandState | None = None
        self._last_applied: HandCommand | None = None
        self._timeout_hold_sent = False
        self._tactile_sequence = 0
        tactile_source = tactile_reader or reader
        self.tactile_layout = tuple(
            getattr(tactile_source, "tactile_layout", TACTILE_LAYOUT)
        )
        self.tactile_taxels = int(
            getattr(tactile_source, "tactile_taxels", TOTAL_TAXELS)
        )

    @property
    def read_only(self) -> bool:
        return self.command_sink is None

    def submit(self, command: HandCommand) -> bool:
        if self.read_only:
            return False
        return self.mailbox.put(command)

    def request_hold(self, sequence: int, reason: str) -> bool:
        """Queue a measured-position hold on the owning worker."""
        if self.read_only or self._latest_state is None:
            return False
        now_ns = self.clock_ns()
        return self.mailbox.put(
            HandCommand(
                sequence=sequence,
                angles=self._latest_state.actuator_angle,
                force_limits=self.safe_force_limits,
                deadline_ns=now_ns + 1_000_000_000,
                source=f"local-safety-hold:{reason}",
            )
        )

    def start(self) -> None:
        if self._thread is not None or self._tactile_thread is not None:
            raise RuntimeError("worker already started")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run, name=f"dftp-{self.side}-worker", daemon=True
        )
        self._thread.start()
        if self.tactile_reader is not None:
            self._tactile_thread = threading.Thread(
                target=self.run_tactile,
                name=f"dftp-{self.side}-tactile-worker",
                daemon=True,
            )
            self._tactile_thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        tactile_thread, self._tactile_thread = self._tactile_thread, None
        deadline = time.monotonic() + timeout_s
        alive = []
        for name, selected in (
            ("state", thread),
            ("tactile", tactile_thread),
        ):
            if selected is None:
                continue
            selected.join(max(0.0, deadline - time.monotonic()))
            if selected.is_alive():
                alive.append(name)
        if alive:
            raise TimeoutError(
                f"{self.side} DFTP {'/'.join(alive)} worker did not stop"
            )

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self.reader.connect()
                self.mailbox.discard()
                self._latest_state = None
                self._last_applied = None
                self._timeout_hold_sent = False
                self._run_connected()
            except Exception as exc:
                self.on_fault(f"{self.side}: Modbus connection lost: {exc}")
            finally:
                self.reader.close()
            if not self._stop.is_set():
                self.sleeper(0.5)

    def _run_connected(self) -> None:
        next_state = self.clock_ns()
        next_tactile = next_state
        while not self._stop.is_set():
            now = self.clock_ns()
            self._service_command(now)
            if now >= next_state:
                self._read_state_safely()
                next_state = max(next_state + self.state_period_ns, now)
            if self.tactile_reader is None and now >= next_tactile:
                self._read_tactile_frame_safely()
                next_tactile = max(next_tactile + self.tactile_period_ns, now)
            next_deadline = (
                next_state
                if self.tactile_reader is not None
                else min(next_state, next_tactile)
            )
            remaining_s = max(0.0, (next_deadline - self.clock_ns()) / 1e9)
            self.sleeper(min(remaining_s, 0.002))

    def run_tactile(self) -> None:
        """Run low-rate tactile acquisition on its independent connection."""

        assert self.tactile_reader is not None
        while not self._stop.is_set():
            try:
                self.tactile_reader.connect()
                self._run_tactile_connected()
            except Exception as exc:
                self.on_fault(
                    f"{self.side}: tactile Modbus connection lost: {exc}"
                )
            finally:
                self.tactile_reader.close()
            if not self._stop.is_set():
                self.sleeper(0.5)

    def _run_tactile_connected(self) -> None:
        next_tactile = self.clock_ns()
        while not self._stop.is_set():
            now = self.clock_ns()
            if now >= next_tactile:
                self._read_tactile_frame_safely()
                next_tactile = max(next_tactile + self.tactile_period_ns, now)
            remaining_s = max(
                0.0, (next_tactile - self.clock_ns()) / 1e9
            )
            self.sleeper(min(remaining_s, 0.002))

    def _read_state_safely(self) -> None:
        try:
            state = self.reader.read_state()
            if not state.acquisition.valid:
                self.on_state(state)
                raise RuntimeError(state.acquisition.invalid_reason)

            self._latest_state = state
            self.on_state(state)
        except Exception as exc:
            self.on_fault(f"{self.side}: state read failed: {exc}")
            raise

    def _read_tactile_frame_safely(self) -> None:
        start = self.observation_clock_ns()
        surfaces: list[TactileSurface] = []
        reader = self.tactile_reader or self.reader
        try:
            for spec in self.tactile_layout:
                # In compatibility mode state, tactile and commands share one
                # connection, so retain command preemption between surfaces.
                if self.tactile_reader is None:
                    self._service_command(self.clock_ns())
                surfaces.append(reader.read_surface(spec))
                if self.tactile_reader is None:
                    self._service_command(self.clock_ns())
            end = self.observation_clock_ns()
            self._tactile_sequence += 1
            frame = TactileFrame(
                side=self.side,
                sequence=self._tactile_sequence,
                acquisition_start_ns=start,
                acquisition_end_ns=end,
                surfaces=tuple(surfaces),
            )
            if frame.taxel_count != self.tactile_taxels:
                raise RuntimeError(
                    "tactile frame taxel count is not "
                    f"{self.tactile_taxels}"
                )
            self.on_tactile(frame)
        except Exception as exc:
            self.on_fault(f"{self.side}: tactile read failed: {exc}")
            raise

    def _service_command(self, now_ns: int) -> None:
        if self.command_sink is None:
            return
        command = self.mailbox.take()
        if command is not None:
            if command.deadline_ns <= now_ns:
                self.on_fault(
                    f"{self.side}: rejected expired hand command {command.sequence}"
                )
                self._send_timeout_hold(now_ns, reason="expired-before-execution")
            else:
                self.command_sink.apply(command)
                self._last_applied = command
                self._timeout_hold_sent = False

        if (
            self._last_applied is not None
            and now_ns >= self._last_applied.deadline_ns
            and not self._timeout_hold_sent
        ):
            self._send_timeout_hold(now_ns, reason="command-watchdog")

    def _send_timeout_hold(self, now_ns: int, reason: str) -> None:
        if self.command_sink is None or self._latest_state is None:
            self.on_fault(f"{self.side}: cannot hold without a measured hand state")
            self._timeout_hold_sent = True
            return
        hold = HandCommand(
            sequence=(
                0 if self._last_applied is None else self._last_applied.sequence + 1
            ),
            angles=self._latest_state.actuator_angle,
            force_limits=self.safe_force_limits,
            deadline_ns=now_ns + 1_000_000_000,
            source=f"watchdog-hold:{reason}",
        )
        self.command_sink.apply(hold)
        self._timeout_hold_sent = True
