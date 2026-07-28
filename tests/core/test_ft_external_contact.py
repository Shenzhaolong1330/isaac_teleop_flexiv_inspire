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
    value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, duration):
        self.value += max(duration, 0.001)


class Interlock:
    def acquire(self, session_id):
        pass

    def release(self, **kwargs):
        pass


def run(backend):
    clock = Clock()
    manager = FTZeroManager(
        backend,
        write_guard=HardwareWriteGuard(test_backend=True),
        interlock=Interlock(),
        read_hand_positions=lambda: (np.zeros(6), np.zeros(6)),
        event_sink=lambda event: None,
        config=FTZeroConfig(
            sample_window_s=0.0,
            sample_rate_hz=1000.0,
            min_samples=3,
            max_pre_external_mean_force_n=3.0,
            max_pre_external_mean_torque_nm=0.3,
            max_pre_external_peak_force_n=5.0,
            max_pre_external_peak_torque_nm=0.5,
        ),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    return manager.zero(
        ZeroFTRequest(
            session_id="session",
            operator_confirmation=CONFIRMATION_TOKEN,
            local_console=True,
            tool_payload_config_hash="deadbeef",
            left_hand_position=np.zeros(6),
            right_hand_position=np.zeros(6),
        )
    )


def test_constant_external_contact_is_rejected_before_enable():
    backend = MockBackend()
    backend.samples["left"] = replace(
        backend.samples["left"], external_wrench=np.array([4.0, 0, 0, 0, 0, 0])
    )
    result = run(backend)
    assert not result.success
    assert "external contact" in result.failure_reason
    assert ("left", "enable") not in backend.events


def test_pulse_external_contact_is_rejected_before_enable():
    class PulseBackend(MockBackend):
        count = 0

        def observe(self, side):
            sample = super().observe(side)
            if side == "left":
                self.count += 1
                if self.count == 2:
                    return replace(
                        sample,
                        external_wrench=np.array([6.0, 0, 0, 0, 0, 0]),
                    )
            return sample

    backend = PulseBackend()
    result = run(backend)
    assert not result.success
    assert "external contact" in result.failure_reason
    assert ("left", "enable") not in backend.events
