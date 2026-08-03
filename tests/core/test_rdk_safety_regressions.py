from __future__ import annotations

import time

import pytest

from flexiv_rdk_daemon.mock_backend import MockBackend
from flexiv_rdk_daemon.server import (
    DaemonInterlock,
    HandObservationCache,
    RDKRequestDispatcher,
)


class FakeFT:
    def is_zeroed_for(self, session_id: str) -> bool:
        return session_id == "session"


class TokenAuthority:
    def consume(self, token: str, *, session_id: str, source: str) -> None:
        if token != "token":
            raise PermissionError("bad token")


def command(sequence: int = 1) -> dict:
    target = {
        "tcp_pose_rdk": [0, 0, 0, 1, 0, 0, 0],
        "max_linear_velocity": 0.01,
        "max_angular_velocity": 0.05,
        "max_linear_acceleration": 0.05,
        "max_angular_acceleration": 0.1,
        "cartesian_stiffness": [1200.0, 1200.0, 1200.0, 80.0, 80.0, 80.0],
        "cartesian_damping_ratio": [0.7] * 6,
    }
    return {
        "session_id": "session",
        "source": "teleop",
        "source_sequence": str(sequence),
        "expires_monotonic_ns": str(time.monotonic_ns() + 100_000_000),
        "left": dict(target),
        "right": dict(target),
        "valid_mask": 3,
        "safety_validated": True,
        "local_permission": True,
        "physical_pedal": True,
        "local_arm_token": "token",
    }


def dispatcher(backend: MockBackend) -> RDKRequestDispatcher:
    interlock = DaemonInterlock()
    interlock.ready_generation = backend.connection_generation
    hands = HandObservationCache()
    value = RDKRequestDispatcher(
        backend,
        FakeFT(),
        hands,
        interlock,
        control_authorizations=TokenAuthority(),
        watchdog_period_s=0.001,
    )
    return value


def test_both_targets_are_validated_before_either_arm_is_sent() -> None:
    backend = MockBackend()
    value = dispatcher(backend)
    payload = command()
    payload["right"]["tcp_pose_rdk"] = [0, 0, 0]
    with pytest.raises(ValueError):
        value("cartesian_command", 1, payload, (111, 0, 0))
    assert ("left", "send_cartesian") not in backend.events
    assert ("right", "send_cartesian") not in backend.events


def test_partial_dual_arm_send_failure_holds_both_arms() -> None:
    class FailRight(MockBackend):
        def send_cartesian_target(self, side, pose_rdk, **kwargs):
            if side == "right":
                raise RuntimeError("right send failed")
            return super().send_cartesian_target(side, pose_rdk, **kwargs)

    backend = FailRight()
    value = dispatcher(backend)
    with pytest.raises(RuntimeError, match="right send failed"):
        value("cartesian_command", 1, command(), (222, 0, 0))
    assert ("left", "send_cartesian") in backend.events
    assert ("left", "stop") in backend.events
    assert ("right", "stop") in backend.events
    assert value.hold_latched


def test_active_lease_is_bound_to_owner_pid() -> None:
    backend = MockBackend()
    value = dispatcher(backend)
    _, accepted = value("cartesian_command", 1, command(1), (333, 0, 0))
    assert accepted["accepted"]
    _, rejected = value("cartesian_command", 1, command(2), (334, 0, 0))
    assert not rejected["accepted"]
    assert rejected["reason"] == "command_owner_pid_changed"
    assert value.hold_latched
    assert ("left", "stop") in backend.events
    assert ("right", "stop") in backend.events


def test_shutdown_synchronously_holds_both_active_arms() -> None:
    backend = MockBackend()
    value = dispatcher(backend)
    value("cartesian_command", 1, command(), (444, 0, 0))
    value.shutdown_hold()
    assert ("left", "stop") in backend.events
    assert ("right", "stop") in backend.events
    assert value.hold_latched


def test_explicit_hold_retry_reissues_hardware_stop() -> None:
    class FailFirstLeftStop(MockBackend):
        def __init__(self):
            super().__init__()
            self.failed = False

        def stop(self, side: str, *, local_console: bool):
            if side == "left" and not self.failed:
                self.failed = True
                raise RuntimeError("transient stop failure")
            return super().stop(side, local_console=local_console)

    backend = FailFirstLeftStop()
    value = dispatcher(backend)
    value("cartesian_command", 1, command(), (555, 0, 0))
    _, first = value("hold", 2, {"reason": "test"}, (555, 0, 0))
    assert not first["accepted"]
    _, second = value("hold", 3, {"reason": "test"}, (555, 0, 0))
    assert second["accepted"]
    assert backend.events.count(("left", "stop")) == 1
    assert backend.events.count(("right", "stop")) == 2
