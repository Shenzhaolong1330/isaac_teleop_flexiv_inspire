from __future__ import annotations

import numpy as np
import pytest

from flexiv_rdk_daemon.backend import FlexivRDKBackend
from flexiv_rdk_daemon.ft_zero import (
    CONFIRMATION_TOKEN,
    FTZeroConfig,
    FTZeroManager,
    ZeroFTRequest,
)
from flexiv_rdk_daemon.guard import HardwareWriteGuard
from flexiv_rdk_daemon.mock_backend import MockBackend


def test_rdk_19_tuple_timestamp_and_legacy_integer_normalization() -> None:
    nanoseconds, seconds, subsecond = FlexivRDKBackend._robot_time(
        (1_785_275_099, 182_913_000)
    )
    assert nanoseconds == 1_785_275_099_182_913_000
    assert seconds == 1_785_275_099
    assert subsecond == 182_913_000
    assert FlexivRDKBackend._robot_time(nanoseconds) == (
        nanoseconds,
        seconds,
        subsecond,
    )
    with pytest.raises(RuntimeError):
        FlexivRDKBackend._robot_time((1, 1_000_000_000))


class Interlock:
    def __init__(self) -> None:
        self.result = None

    def acquire(self, session_id: str) -> None:
        pass

    def release(self, *, success: bool, connection_generation: int | None) -> None:
        self.result = success, connection_generation


def test_ft_failure_best_effort_idles_and_keeps_available_statistics() -> None:
    backend = MockBackend()
    backend.primitive_sequences["left"] = [{"failed": True, "error": "mock"}]
    interlock = Interlock()
    events: list[dict[str, object]] = []
    tick = 0.0

    def monotonic() -> float:
        nonlocal tick
        tick += 0.001
        return tick

    manager = FTZeroManager(
        backend,
        write_guard=HardwareWriteGuard(test_backend=True),
        interlock=interlock,
        read_hand_positions=lambda: (np.zeros(6), np.zeros(6)),
        event_sink=events.append,
        config=FTZeroConfig(
            sample_window_s=0.0,
            sample_rate_hz=1000.0,
            min_samples=2,
            operational_timeout_s=0.01,
            primitive_timeout_s=0.01,
            enable_settle_timeout_s=0.01,
            enable_settle_window_s=0.002,
            poll_interval_s=0.0,
        ),
        sleep=lambda duration: None,
        monotonic=monotonic,
    )
    result = manager.zero(
        ZeroFTRequest(
            session_id="session",
            operator_confirmation=CONFIRMATION_TOKEN,
            local_console=True,
            tool_payload_config_hash="deadbeef",
            left_hand_position=np.zeros(6),
            right_hand_position=np.zeros(6),
        )
    )
    assert not result.success
    assert interlock.result == (False, None)
    assert ("left", "idle") in backend.events
    assert ("right", "idle") in backend.events
    failure = events[-1]
    assert failure["phase"] == "left_zero_and_post_window"
    assert set(failure["available_statistics"]) == {
        "left_before",
        "right_before",
    }
    assert failure["cleanup"] == {"left_idle": "ok", "right_idle": "ok"}
