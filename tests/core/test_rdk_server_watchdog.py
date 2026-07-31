from __future__ import annotations

import time

import numpy as np
import pytest

import flexiv_rdk_daemon.server as server_module
from flexiv_rdk_daemon.mock_backend import MockBackend
from flexiv_rdk_daemon.server import (
    DaemonInterlock,
    HandObservationCache,
    LocalControlAuthorization,
    LocalHomeAuthorization,
    LocalZeroAuthorization,
    RDKRequestDispatcher,
)


class FakeFT:
    def __init__(self) -> None:
        self.valid = True

    def is_zeroed_for(self, session_id: str) -> bool:
        return self.valid and session_id == "session"


class TokenAuthority:
    def __init__(self) -> None:
        self.consumed: list[tuple[str, str, str]] = []

    def mint(self, **kwargs):
        return "token", time.monotonic_ns() + 1_000_000_000

    def consume(self, token: str, *, session_id: str, source: str) -> None:
        if token != "token":
            raise PermissionError("bad token")
        self.consumed.append((token, session_id, source))


class HomeTokenAuthority:
    def __init__(self) -> None:
        self.consumed = 0

    def consume(self, token: str, *, session_id: str) -> None:
        if token != "home-token" or session_id != "session":
            raise PermissionError("bad Home token")
        self.consumed += 1


def dispatcher():
    backend = MockBackend()
    interlock = DaemonInterlock()
    interlock.ready_generation = 1
    hands = HandObservationCache()
    hands.update(np.zeros(6), np.zeros(6))
    value = RDKRequestDispatcher(
        backend,
        FakeFT(),
        hands,
        interlock,
        control_authorizations=TokenAuthority(),
        home_authorizations=HomeTokenAuthority(),
        watchdog_period_s=0.001,
    )
    return value, backend


def command(source: str, sequence: int, ttl_s: float = 0.05):
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
        "source": source,
        "source_sequence": str(sequence),
        "expires_monotonic_ns": str(time.monotonic_ns() + int(ttl_s * 1e9)),
        "left": target,
        "right": target,
        "valid_mask": 3,
        "safety_validated": True,
        "local_permission": True,
        "physical_pedal": True,
        "local_arm_token": "token",
    }


def home_command(
    sequence: int,
    *,
    target: float = 0.0,
    physical_pedal: bool = True,
) -> dict:
    return {
        "session_id": "session",
        "request_sequence": str(sequence),
        "expires_monotonic_ns": str(time.monotonic_ns() + 100_000_000),
        "left_joint_positions": [target] * 7,
        "right_joint_positions": [target] * 7,
        "max_velocity_rad_s": 0.5,
        "max_acceleration_rad_s2": 1.0,
        "tolerance_rad": 0.01,
        "timeout_s": 5.0,
        "safety_validated": True,
        "local_permission": True,
        "physical_pedal": physical_pedal,
        "collision_clear": True,
        "local_authorization_token": "home-token",
    }


def test_watchdog_holds_on_ttl_expiry_and_latches() -> None:
    value, backend = dispatcher()
    value.start_watchdog()
    kind, payload = value("cartesian_command", 1, command("teleop", 1, 0.01), (111, 0, 0))
    assert payload["accepted"]
    deadline = time.monotonic() + 0.5
    while not value.hold_latched and time.monotonic() < deadline:
        time.sleep(0.002)
    value.close()
    assert value.hold_latched
    assert ("left", "send_hold") in backend.events
    _, rejected = value("cartesian_command", 2, command("teleop", 2), (111, 0, 0))
    assert not rejected["accepted"]
    assert "hold_latched" in rejected["reason"]


def test_owner_disconnect_holds_immediately() -> None:
    value, backend = dispatcher()
    value("cartesian_command", 1, command("teleop", 1), (222, 0, 0))
    value.peer_disconnected((222, 0, 0))
    assert value.hold_latched
    assert ("right", "send_hold") in backend.events


def test_different_source_conflict_holds_both_arms() -> None:
    value, backend = dispatcher()
    value("cartesian_command", 1, command("teleop", 1), (333, 0, 0))
    _, response = value("cartesian_command", 2, command("policy", 1), (333, 0, 0))
    assert not response["accepted"]
    assert value.hold_latched
    assert value.active_source == "teleop"


def test_remote_process_cannot_mint_zero_or_control_authorization(monkeypatch) -> None:
    monkeypatch.setattr(server_module, "peer_has_local_tty", lambda pid: False)
    with pytest.raises(PermissionError):
        LocalZeroAuthorization().mint(
            pid=1,
            session_id="session",
            confirmation="FLEXIV-FT-UNLOADED",
            tool_payload_config_hash="deadbeef",
        )
    with pytest.raises(PermissionError):
        LocalControlAuthorization().mint(
            pid=1,
            session_id="session",
            source="teleop",
            confirmation="FLEXIV-CONTROL-ARM",
        )


def test_local_authorization_is_single_use(monkeypatch) -> None:
    monkeypatch.setattr(server_module, "peer_has_local_tty", lambda pid: True)
    authority = LocalControlAuthorization()
    token, _ = authority.mint(
        pid=10,
        session_id="session",
        source="teleop",
        confirmation="FLEXIV-CONTROL-ARM",
    )
    authority.consume(token, session_id="session", source="teleop")
    with pytest.raises(PermissionError):
        authority.consume(token, session_id="session", source="teleop")


def test_home_authorization_requires_separate_confirmation(monkeypatch) -> None:
    monkeypatch.setattr(server_module, "peer_has_local_tty", lambda pid: True)
    authority = LocalHomeAuthorization()
    with pytest.raises(PermissionError):
        authority.mint(
            pid=10,
            session_id="session",
            confirmation="FLEXIV-CONTROL-ARM",
        )
    token, _ = authority.mint(
        pid=10,
        session_id="session",
        confirmation="FLEXIV-HOME-MOVE",
    )
    authority.consume(token, session_id="session")


def test_home_sends_joint_target_then_returns_to_cartesian_hold() -> None:
    value, backend = dispatcher()
    kind, response = value(
        "home_command", 1, home_command(1), (777, 0, 0)
    )
    assert kind == "home_result"
    assert response["accepted"] and response["completed"]
    for side in ("left", "right"):
        assert (side, "joint_position_mode") in backend.events
        assert (side, "send_joint_position") in backend.events
        assert (side, "cartesian_mode") in backend.events
        assert (side, "send_hold") in backend.events
    assert not value.hold_latched


def test_home_does_not_require_pedal_and_reuses_process_session_lease() -> None:
    value, _ = dispatcher()
    first = home_command(1, physical_pedal=False)
    _, first_response = value("home_command", 1, first, (777, 0, 0))
    second = home_command(2, physical_pedal=False)
    second["local_authorization_token"] = ""
    _, second_response = value("home_command", 2, second, (777, 0, 0))

    assert first_response["accepted"] and first_response["completed"]
    assert second_response["accepted"] and second_response["completed"]


def test_home_session_lease_is_revoked_when_bridge_disconnects() -> None:
    value, _ = dispatcher()
    value("home_command", 1, home_command(1), (780, 0, 0))
    value.peer_disconnected((780, 0, 0))
    retry = home_command(2)
    retry["local_authorization_token"] = ""

    with pytest.raises(PermissionError, match="bad Home token"):
        value("home_command", 2, retry, (780, 0, 0))


def test_home_watchdog_stops_joint_motion_without_keepalive() -> None:
    value, backend = dispatcher()
    value.start_watchdog()
    _, response = value(
        "home_command", 1, home_command(1, target=0.2), (778, 0, 0)
    )
    assert response["accepted"] and not response["completed"]
    deadline = time.monotonic() + 0.5
    while not value.hold_latched and time.monotonic() < deadline:
        time.sleep(0.002)
    value.close()
    assert value.hold_latched
    assert ("left", "cartesian_mode") in backend.events
    assert ("right", "cartesian_mode") in backend.events


def test_cartesian_command_applies_impedance_before_motion() -> None:
    value, backend = dispatcher()
    _, response = value(
        "cartesian_command", 1, command("teleop", 1), (779, 0, 0)
    )
    assert response["accepted"]
    for side in ("left", "right"):
        impedance = backend.events.index((side, "set_cartesian_impedance"))
        motion = backend.events.index((side, "send_cartesian"))
        assert impedance < motion


class CaptureEmitter:
    def __init__(self) -> None:
        self.records = []

    def emit(self, record, *, critical=False) -> None:
        assert not critical
        self.records.append(record)


def test_observe_emits_pre_dds_arm_deviceio_records() -> None:
    backend = MockBackend()
    interlock = DaemonInterlock()
    hands = HandObservationCache()
    hands.update(np.zeros(6), np.zeros(6))
    capture = CaptureEmitter()
    value = RDKRequestDispatcher(
        backend,
        FakeFT(),
        hands,
        interlock,
        deviceio_emitter=capture,
    )
    kind, payload = value("observe", 1, {}, (123, 0, 0))
    value.close()
    assert kind == "dual_arm_state"
    assert payload["left"]["tcp_pose_rdk_xyz_wxyz"][3] == 1.0
    by_topic = {record["topic"]: record for record in capture.records}
    assert len(by_topic) == 10
    assert by_topic["/robot/left_arm/state"]["producer"] == "rdk"
    assert by_topic["/robot/right_arm/tcp_pose"]["payload"]["quaternion_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert by_topic["/robot/left_arm/raw_ft"]["timing_valid"] is True
