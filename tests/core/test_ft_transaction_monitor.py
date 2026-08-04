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
from flexiv_rdk_daemon.model import DualArmSample


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.now += max(duration, 0.001)


class Interlock:
    def acquire(self, session_id: str) -> None:
        pass

    def release(self, *, success: bool, connection_generation: int | None) -> None:
        pass


def request() -> ZeroFTRequest:
    return ZeroFTRequest(
        session_id="session",
        operator_confirmation=CONFIRMATION_TOKEN,
        local_console=True,
        tool_payload_config_hash="deadbeef",
        left_hand_position=np.zeros(6),
        right_hand_position=np.zeros(6),
    )


def make_manager(backend, *, hand_monitor=None) -> FTZeroManager:
    clock = Clock()
    return FTZeroManager(
        backend,
        write_guard=HardwareWriteGuard(test_backend=True),
        interlock=Interlock(),
        read_hand_positions=lambda: (np.zeros(6), np.zeros(6)),
        event_sink=lambda event: None,
        config=FTZeroConfig(
            sample_window_s=0.0,
            sample_rate_hz=1000.0,
            min_samples=3,
            operational_timeout_s=0.02,
            primitive_timeout_s=0.02,
            enable_settle_timeout_s=0.01,
            enable_settle_window_s=0.003,
            poll_interval_s=0.001,
            max_hand_delta=0.1,
        ),
        hand_monitor_snapshot=hand_monitor,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )


def test_latched_hand_motion_is_rejected_even_after_return_to_reference() -> None:
    # The sampled hand reader has returned to zero, but HandObservationCache's
    # transaction latch reports the transient excursion.
    monitor = lambda: {
        "left_max_delta": 1.0,
        "right_max_delta": 0.0,
    }
    result = make_manager(MockBackend(), hand_monitor=monitor).zero(request())
    assert not result.success
    assert "left hand moved" in result.failure_reason
    assert not any(event[1] == "enable" for event in result.cleanup)


def test_opposite_arm_motion_that_never_settles_is_rejected() -> None:
    class PersistentRightMotion(MockBackend):
        def __init__(self):
            super().__init__()
            self.left_enabled = False
            self.is_operational["left"] = False

        def enable(self, side: str, *, local_console: bool) -> None:
            super().enable(side, local_console=local_console)
            if side == "left":
                self.left_enabled = True
                self.is_operational[side] = True

        def observe_both(self):
            left = self.observe("left")
            right = self.observe("right")
            if self.left_enabled:
                right = replace(right, dq=np.ones(7))
            return DualArmSample(left, right)

    backend = PersistentRightMotion()
    result = make_manager(backend).zero(request())
    assert not result.success
    assert "did not settle after enabling left" in result.failure_reason
    assert ("right", "zero_ft") not in backend.events


def test_opposite_arm_transient_contact_is_rejected_during_left_zero() -> None:
    class TransientRightContact(MockBackend):
        def __init__(self):
            super().__init__()
            self.observations = 0

        def observe_both(self):
            self.observations += 1
            left = self.observe("left")
            right = self.observe("right")
            if self.observations == 5:
                right = replace(
                    right,
                    external_wrench=np.array([10.0, 0, 0, 0, 0, 0]),
                )
            return DualArmSample(left, right)

    result = make_manager(TransientRightContact()).zero(request())
    assert not result.success
    assert "right external contact" in result.failure_reason
