from __future__ import annotations

from dataclasses import replace

import numpy as np

from flexiv_rdk_daemon.ft_zero import (
    CONFIRMATION_TOKEN,
    FTZeroConfig,
    FTZeroManager,
    ZeroFTRequest,
)
from flexiv_rdk_daemon.guard import HardwareWriteGuard
from flexiv_rdk_daemon.mock_backend import MockBackend


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += max(duration, 0.001)


class Interlock:
    def __init__(self) -> None:
        self.acquired = False
        self.released: list[tuple[bool, int | None]] = []

    def acquire(self, session_id: str) -> None:
        assert not self.acquired
        self.acquired = True

    def release(self, *, success: bool, connection_generation: int | None) -> None:
        self.released.append((success, connection_generation))
        self.acquired = False


def config() -> FTZeroConfig:
    return FTZeroConfig(
        sample_window_s=0.0,
        sample_rate_hz=1000.0,
        min_samples=3,
        operational_timeout_s=0.01,
        primitive_timeout_s=0.01,
        poll_interval_s=0.002,
        max_joint_velocity_norm=0.01,
        max_tcp_velocity_norm=0.01,
        max_wrench_std_force_n=0.5,
        max_wrench_std_torque_nm=0.05,
        max_residual_force_n=1.0,
        max_residual_torque_nm=0.1,
        max_hand_delta=0.1,
    )


def request() -> ZeroFTRequest:
    return ZeroFTRequest(
        session_id="session",
        operator_confirmation=CONFIRMATION_TOKEN,
        local_console=True,
        tool_payload_config_hash="deadbeefcafebabe",
        left_hand_position=np.zeros(6),
        right_hand_position=np.zeros(6),
    )


def manager(backend: MockBackend, hands=lambda: (np.zeros(6), np.zeros(6))):
    clock = Clock()
    interlock = Interlock()
    events: list[dict[str, object]] = []
    value = FTZeroManager(
        backend,
        write_guard=HardwareWriteGuard(test_backend=True),
        interlock=interlock,
        read_hand_positions=hands,
        event_sink=events.append,
        config=config(),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    return value, interlock, events, clock


def test_success_is_left_then_right_and_records_statistics() -> None:
    backend = MockBackend()
    value, interlock, events, _ = manager(backend)
    result = value.zero(request())
    assert result.success
    assert value.is_zeroed_for("session")
    assert interlock.released == [(True, 1)]
    assert backend.events.index(("left", "zero_ft")) < backend.events.index(("right", "zero_ft"))
    assert result.left.before.sample_count == 3
    assert result.right.after.sample_count == 3
    assert events[0]["event_type"] == "ft_zero_started"
    assert events[-1]["event_type"] == "ft_zero_completed"


def test_operational_timeout_fails_and_never_becomes_ready() -> None:
    backend = MockBackend()
    backend.is_operational["left"] = False
    value, interlock, events, _ = manager(backend)
    result = value.zero(request())
    assert not result.success
    assert "operational timeout" in result.failure_reason
    assert not value.is_zeroed_for("session")
    assert interlock.released[-1] == (False, None)
    assert events[-1]["event_type"] == "ft_zero_failed"


def test_primitive_failure_and_disconnect_are_fail_closed() -> None:
    backend = MockBackend()
    backend.primitive_sequences["left"] = [{"failed": True, "error": "mock"}]
    value, _, _, _ = manager(backend)
    assert not value.zero(request()).success

    class DisconnectBackend(MockBackend):
        def observe(self, side: str):
            sample = super().observe(side)
            self.reconnect()
            return sample

    disconnected = DisconnectBackend()
    value2, _, _, _ = manager(disconnected)
    result = value2.zero(request())
    assert not result.success
    assert "reconnect" in result.failure_reason.lower()


def test_joint_or_tcp_motion_rejects_preflight() -> None:
    backend = MockBackend()
    backend.samples["left"] = replace(backend.samples["left"], dq=np.ones(7))
    value, _, _, _ = manager(backend)
    result = value.zero(request())
    assert not result.success
    assert "stability" in result.failure_reason


def test_hand_motion_rejects_transaction() -> None:
    calls = 0

    def hands():
        nonlocal calls
        calls += 1
        delta = 0.0 if calls < 3 else 1.0
        return np.full(6, delta), np.zeros(6)

    backend = MockBackend()
    value, _, _, _ = manager(backend, hands)
    result = value.zero(request())
    assert not result.success
    assert "hand moved" in result.failure_reason


def test_residual_rejects_and_reconnect_invalidates_old_zero() -> None:
    class ResidualBackend(MockBackend):
        def execute_zero_ft(self, side: str, *, local_console: bool) -> None:
            super().execute_zero_ft(side, local_console=local_console)
            self.samples[side] = replace(
                self.samples[side],
                raw_ft=np.array([10.0, 0, 0, 0, 0, 0]),
            )

    residual = ResidualBackend()
    value, _, _, _ = manager(residual)
    result = value.zero(request())
    assert not result.success
    assert "residual" in result.failure_reason

    normal = MockBackend()
    value2, _, _, _ = manager(normal)
    assert value2.zero(request()).success
    normal.reconnect()
    assert not value2.is_zeroed_for("session")


def test_remote_or_wrong_confirmation_is_rejected_without_enable() -> None:
    backend = MockBackend()
    value, _, _, _ = manager(backend)
    remote = replace(request(), local_console=False)
    assert not value.zero(remote).success
    wrong = replace(request(), operator_confirmation="WRONG")
    assert not value.zero(wrong).success
    assert not any(event[1] == "enable" for event in backend.events)
