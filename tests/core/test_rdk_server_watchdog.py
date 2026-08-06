from __future__ import annotations

from dataclasses import replace
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

    def mint(self, **kwargs):
        return "home-token", time.monotonic_ns() + 1_000_000_000

    def consume(self, token: str, *, session_id: str) -> None:
        if token != "home-token" or session_id != "session":
            raise PermissionError("bad Home token")
        self.consumed += 1


def dispatcher(
    *, cartesian_limits: tuple[float, float, float, float] = (
        0.10,
        0.25,
        0.50,
        1.0,
    ),
):
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
        cartesian_limits=cartesian_limits,
        watchdog_period_s=0.001,
    )
    return value, backend


def command(source: str, sequence: int, ttl_s: float = 0.05):
    target = {
        "tcp_pose_rdk": [0, 0, 0, 1, 0, 0, 0],
        "control_mode": "impedance",
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


def enable_home_lift(payload: dict) -> dict:
    payload.update(
        {
            "lift_enabled": True,
            "left_lift_target_x_m": 0.925,
            "left_lift_target_y_m": 0.345,
            "left_lift_safe_z_m": -0.38,
            "right_lift_target_x_m": 0.952,
            "right_lift_target_y_m": -0.152,
            "right_lift_safe_z_m": -0.41,
            "lift_max_linear_velocity": 0.10,
            "lift_max_angular_velocity": 0.20,
            "lift_max_linear_acceleration": 0.40,
            "lift_max_angular_acceleration": 0.80,
            "lift_tolerance_m": 0.005,
            "lift_timeout_s": 12.0,
            "lift_parallel": False,
            "lift_cartesian_stiffness": [3000.0, 3000.0, 3000.0, 200.0, 200.0, 200.0],
            "lift_cartesian_damping_ratio": [0.7] * 6,
        }
    )
    return payload


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
    assert ("left", "stop") in backend.events
    _, rejected = value("cartesian_command", 2, command("teleop", 2), (111, 0, 0))
    assert not rejected["accepted"]
    assert "hold_latched" in rejected["reason"]


def test_owner_disconnect_holds_immediately() -> None:
    value, backend = dispatcher()
    value("cartesian_command", 1, command("teleop", 1), (222, 0, 0))
    value.peer_disconnected((222, 0, 0))
    assert value.hold_latched
    assert ("right", "stop") in backend.events


def test_clear_routine_hold_reanchors_both_stopped_arms_in_cartesian_mode() -> None:
    value, backend = dispatcher()
    value("cartesian_command", 1, command("teleop", 1), (223, 0, 0))
    value("hold", 2, {"reason": "physical_pedal_released"}, (223, 0, 0))
    assert value.hold_latched

    kind, response = value(
        "authorize_control",
        3,
        {
            "session_id": "session",
            "source": "teleop",
            "operator_confirmation": "FLEXIV-CONTROL-ARM",
            "clear_hold_latched": True,
        },
        (223, 0, 0),
    )

    assert kind == "authorize_control_result"
    assert response["authorized"]
    assert not value.hold_latched
    for side in ("left", "right"):
        stop = backend.events.index((side, "stop"))
        mode = backend.events.index((side, "cartesian_mode"))
        anchor = backend.events.index((side, "send_hold"))
        assert stop < mode < anchor


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


def test_policy_control_authorization_does_not_require_a_tty(monkeypatch) -> None:
    monkeypatch.setattr(server_module, "peer_has_local_tty", lambda pid: False)
    authority = LocalControlAuthorization()
    token, expires = authority.mint(
        pid=1,
        session_id="session",
        source="policy",
        confirmation="FLEXIV-CONTROL-ARM",
    )

    assert token
    assert expires > time.monotonic_ns()
    authority.consume(token, session_id="session", source="policy")


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


def test_reset_home_authorization_clears_fault_enables_and_waits_operational() -> None:
    value, backend = dispatcher()
    backend.is_faulted["left"] = True
    backend.is_operational["left"] = False
    value("observe", 0, {}, (10, 0, 0))
    assert value._latest_observation is not None

    original_enable = backend.enable

    def enable(side: str, *, local_console: bool) -> None:
        original_enable(side, local_console=local_console)
        backend.is_operational[side] = True

    backend.enable = enable
    kind, response = value(
        "authorize_home",
        1,
        {
            "session_id": "session",
            "operator_confirmation": "FLEXIV-HOME-MOVE",
            "clear_hold_latched": True,
            "recover_robot_faults": True,
        },
        (10, 0, 0),
    )

    assert kind == "authorize_home_result"
    assert response["authorized"]
    assert ("left", "clear_fault") in backend.events
    assert ("left", "enable") in backend.events
    assert backend.is_operational["left"]
    assert not backend.is_faulted["left"]
    assert value._latest_observation is None


def test_ordinary_home_authorization_does_not_clear_controller_fault() -> None:
    value, backend = dispatcher()
    backend.is_faulted["left"] = True

    kind, response = value(
        "authorize_home",
        1,
        {
            "session_id": "session",
            "operator_confirmation": "FLEXIV-HOME-MOVE",
            "clear_hold_latched": True,
        },
        (10, 0, 0),
    )

    assert kind == "authorize_home_result"
    assert response["authorized"]
    assert ("left", "clear_fault") not in backend.events


def test_home_sends_joint_target_then_returns_to_cartesian_hold() -> None:
    value, backend = dispatcher()
    for side in ("left", "right"):
        backend.samples[side] = replace(
            backend.samples[side], q=np.full(7, 0.2)
        )
    kind, response = value(
        "home_command", 1, home_command(1), (777, 0, 0)
    )
    assert kind == "home_result"
    assert response["accepted"] and not response["completed"]
    for side in ("left", "right"):
        assert (side, "joint_position_mode") in backend.events
        assert (side, "send_joint_position") in backend.events
        backend.samples[side] = replace(
            backend.samples[side], q=np.zeros(7)
        )
    value("observe", 2, {}, (777, 0, 0))
    follow_up = home_command(2)
    follow_up["local_authorization_token"] = ""
    _, completed = value("home_command", 3, follow_up, (777, 0, 0))
    assert completed["accepted"] and completed["completed"]
    for side in ("left", "right"):
        assert (side, "cartesian_mode") in backend.events
        assert (side, "send_hold") in backend.events
    assert not value.hold_latched


def test_home_sends_each_target_immediately_after_its_mode_switch() -> None:
    value, backend = dispatcher()
    for side in ("left", "right"):
        backend.samples[side] = replace(
            backend.samples[side], q=np.full(7, 0.2)
        )

    _, response = value(
        "home_command", 1, home_command(1), (777, 0, 0)
    )

    assert response["accepted"] and not response["completed"]
    left_mode = backend.events.index(("left", "joint_position_mode"))
    left_target = backend.events.index(("left", "send_joint_position"))
    right_mode = backend.events.index(("right", "joint_position_mode"))
    right_target = backend.events.index(("right", "send_joint_position"))
    assert left_mode < left_target < right_mode < right_target


def test_home_lifts_only_the_arm_outside_joint_home() -> None:
    value, backend = dispatcher()
    # Left is already at the requested joint Home even though its TCP is below
    # the safe lift Z. Right is away from Home and below its safe lift Z.
    for side in ("left", "right"):
        pose = backend.samples[side].tcp_pose_rdk.copy()
        pose[2] = -0.60
        q = np.zeros(7) if side == "left" else np.full(7, 0.2)
        backend.samples[side] = replace(
            backend.samples[side], q=q, tcp_pose_rdk=pose
        )
    request = enable_home_lift(home_command(1))

    _, response = value("home_command", 1, request, (789, 0, 0))

    assert response["accepted"] and not response["completed"]
    assert ("left", "send_cartesian") not in backend.events
    assert ("right", "send_cartesian") in backend.events
    assert "left" not in backend.cartesian_targets
    np.testing.assert_allclose(
        backend.cartesian_targets["right"],
        [0.952, -0.152, -0.41, 1.0, 0.0, 0.0, 0.0],
    )

    right_pose = backend.samples["right"].tcp_pose_rdk.copy()
    right_pose[:3] = [0.952, -0.152, -0.41]
    backend.samples["right"] = replace(
        backend.samples["right"], tcp_pose_rdk=right_pose
    )
    value("observe", 2, {}, (789, 0, 0))
    follow_up = enable_home_lift(home_command(2))
    follow_up["local_authorization_token"] = ""
    _, joint_home = value("home_command", 3, follow_up, (789, 0, 0))

    assert joint_home["accepted"] and not joint_home["completed"]
    assert ("left", "send_joint_position") not in backend.events
    assert ("right", "send_joint_position") in backend.events


def test_home_pre_aligns_each_tcp_xyz_before_joint_home() -> None:
    value, backend = dispatcher()
    for side in ("left", "right"):
        pose = backend.samples[side].tcp_pose_rdk.copy()
        pose[2] = -0.60
        backend.samples[side] = replace(
            backend.samples[side], q=np.full(7, 0.2), tcp_pose_rdk=pose
        )
    request = enable_home_lift(home_command(1))

    _, first = value("home_command", 1, request, (790, 0, 0))
    assert first["accepted"] and not first["completed"]
    assert ("left", "send_cartesian") in backend.events
    assert ("left", "disable_force_control_axes") in backend.events
    assert ("left", "set_cartesian_impedance") not in backend.events
    np.testing.assert_allclose(
        backend.cartesian_targets["left"],
        [0.925, 0.345, -0.38, 1.0, 0.0, 0.0, 0.0],
    )
    assert ("left", "send_joint_position") not in backend.events
    assert ("right", "send_cartesian") not in backend.events

    left_pose = backend.samples["left"].tcp_pose_rdk.copy()
    left_pose[:3] = [0.925, 0.345, -0.38]
    backend.samples["left"] = replace(
        backend.samples["left"], q=np.full(7, 0.2), tcp_pose_rdk=left_pose
    )
    value("observe", 2, {}, (790, 0, 0))
    request = enable_home_lift(home_command(2))
    request["local_authorization_token"] = ""
    _, second = value("home_command", 3, request, (790, 0, 0))
    assert second["accepted"] and not second["completed"]
    assert ("right", "send_cartesian") in backend.events
    assert ("left", "send_joint_position") not in backend.events

    right_pose = backend.samples["right"].tcp_pose_rdk.copy()
    right_pose[:3] = [0.952, -0.152, -0.41]
    backend.samples["right"] = replace(
        backend.samples["right"], tcp_pose_rdk=right_pose
    )
    value("observe", 4, {}, (790, 0, 0))
    request = enable_home_lift(home_command(3))
    request["local_authorization_token"] = ""
    _, third = value("home_command", 5, request, (790, 0, 0))

    assert third["accepted"] and not third["completed"]
    for side in ("left", "right"):
        lift_write = backend.events.index((side, "send_cartesian"))
        joint_write = backend.events.index((side, "send_joint_position"))
        assert lift_write < joint_write
        backend.samples[side] = replace(
            backend.samples[side], q=np.zeros(7)
        )
    value("observe", 6, {}, (790, 0, 0))
    request = enable_home_lift(home_command(4))
    request["local_authorization_token"] = ""
    _, completed = value("home_command", 7, request, (790, 0, 0))
    assert completed["accepted"] and completed["completed"]


def test_home_lift_never_descends_an_arm_already_above_safe_z() -> None:
    value, backend = dispatcher()
    for side in ("left", "right"):
        backend.samples[side] = replace(
            backend.samples[side], q=np.full(7, 0.2)
        )
    request = enable_home_lift(home_command(1))
    request["left_lift_safe_z_m"] = -0.20
    request["right_lift_safe_z_m"] = -0.20

    _, response = value("home_command", 1, request, (791, 0, 0))

    # Mock TCP Z is 0.0, already above -0.20. XY still aligns, but the target Z
    # remains at 0.0 and therefore never commands a descent.
    assert response["accepted"] and not response["completed"]
    assert backend.cartesian_targets["left"][2] == pytest.approx(0.0)


def test_home_lift_timeout_reports_arm_height_and_remaining_distance() -> None:
    value, backend = dispatcher()
    left_pose = backend.samples["left"].tcp_pose_rdk.copy()
    left_pose[2] = -0.60
    backend.samples["left"] = replace(
        backend.samples["left"], q=np.full(7, 0.2), tcp_pose_rdk=left_pose
    )
    request = enable_home_lift(home_command(1))

    _, first = value("home_command", 1, request, (792, 0, 0))
    assert first["accepted"] and not first["completed"]
    value._home_phase_started_ns = time.monotonic_ns() - 13_000_000_000
    retry = enable_home_lift(home_command(2))
    retry["local_authorization_token"] = ""

    _, timed_out = value("home_command", 2, retry, (792, 0, 0))

    assert not timed_out["accepted"]
    assert timed_out["reason"].startswith("home_lift_timeout:left(")
    assert "current_xyz=[0.0, 0.0, -0.6]" in timed_out["reason"]
    assert "target_xyz=[0.925, 0.345, -0.38]" in timed_out["reason"]
    assert "max_remaining=0.9250" in timed_out["reason"]


def test_home_reuses_fresh_bridge_observation_instead_of_polling_rdk_again() -> None:
    backend = MockBackend()
    interlock = DaemonInterlock()
    interlock.ready_generation = 1
    hands = HandObservationCache()
    hands.update(np.zeros(6), np.zeros(6))
    observation_count = 0

    def observe_both():
        nonlocal observation_count
        observation_count += 1
        return backend.observe_both()

    value = RDKRequestDispatcher(
        backend,
        FakeFT(),
        hands,
        interlock,
        observe_provider=observe_both,
        home_authorizations=HomeTokenAuthority(),
    )
    value("observe", 1, {}, (777, 0, 0))

    _, response = value(
        "home_command", 2, home_command(1, target=0.2), (777, 0, 0)
    )
    value.close()

    assert response["accepted"] and not response["completed"]
    assert observation_count == 1


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
    assert ("left", "stop") in backend.events
    assert ("right", "stop") in backend.events


def test_home_watchdog_deadline_starts_after_blocking_observation() -> None:
    """Slow RDK reads must not consume the bridge's entire keepalive grace."""

    backend = MockBackend()
    interlock = DaemonInterlock()
    interlock.ready_generation = 1
    hands = HandObservationCache()
    hands.update(np.zeros(6), np.zeros(6))

    def slow_observe_both():
        time.sleep(0.18)
        return backend.observe_both()

    value = RDKRequestDispatcher(
        backend,
        FakeFT(),
        hands,
        interlock,
        observe_provider=slow_observe_both,
        home_authorizations=HomeTokenAuthority(),
        watchdog_period_s=0.001,
    )
    value.start_watchdog()

    _, first = value(
        "home_command", 1, home_command(1, target=0.2), (783, 0, 0)
    )
    assert first["accepted"] and not first["completed"]
    assert not value.hold_latched

    # Let the cached observation age out, then make the next accepted
    # keepalive spend longer in RDK observation than its former absolute
    # deadline allowed.  The watchdog thread waits on the dispatcher lock and
    # must see a fresh server-side grace period when the call returns.
    time.sleep(0.11)
    keepalive = home_command(2, target=0.2)
    keepalive["expires_monotonic_ns"] = str(
        time.monotonic_ns() + 240_000_000
    )
    keepalive["local_authorization_token"] = ""
    _, second = value("home_command", 2, keepalive, (783, 0, 0))
    assert second["accepted"] and not second["completed"]
    time.sleep(0.02)
    assert not value.hold_latched

    value.close()


def test_home_clears_only_episode_transition_hold() -> None:
    value, _ = dispatcher()
    _, active = value(
        "cartesian_command", 1, command("teleop", 1), (781, 0, 0)
    )
    assert active["accepted"]
    _, held = value(
        "hold", 2, {"reason": "episode_home_transition"}, (781, 0, 0)
    )
    assert held["accepted"] and value.hold_latched
    request = home_command(1)
    request["clear_routine_hold"] = True

    _, response = value("home_command", 3, request, (781, 0, 0))

    assert response["accepted"] and response["completed"]
    assert not value.hold_latched


def test_episode_transition_never_overwrites_or_clears_severe_hold() -> None:
    value, _ = dispatcher()
    _, severe = value(
        "hold", 1, {"reason": "hardware_fault"}, (782, 0, 0)
    )
    assert severe["accepted"]
    _, transition = value(
        "hold", 2, {"reason": "episode_home_transition"}, (782, 0, 0)
    )
    assert not transition["accepted"]
    assert "hardware_fault" in transition["reason"]
    request = home_command(1)
    request["clear_routine_hold"] = True

    _, response = value("home_command", 3, request, (782, 0, 0))

    assert not response["accepted"]
    assert response["reason"] == "hold_latched:hardware_fault"


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


def test_cartesian_position_command_does_not_apply_impedance() -> None:
    value, backend = dispatcher()
    payload = command("teleop", 1)
    payload["left"]["control_mode"] = "position"
    payload["right"]["control_mode"] = "position"

    _, response = value("cartesian_command", 1, payload, (779, 0, 0))

    assert response["accepted"]
    for side in ("left", "right"):
        assert (side, "set_cartesian_impedance") not in backend.events
        assert (side, "send_cartesian") in backend.events


def test_cartesian_command_accepts_site_limits_below_daemon_ceiling() -> None:
    value, _ = dispatcher(cartesian_limits=(0.35, 1.0, 1.0, 2.0))
    payload = command("teleop", 1)
    for side in ("left", "right"):
        payload[side]["max_linear_velocity"] = 0.20
        payload[side]["max_angular_velocity"] = 0.60
        payload[side]["max_linear_acceleration"] = 1.0
        payload[side]["max_angular_acceleration"] = 2.0

    _, response = value(
        "cartesian_command", 1, payload, (779, 0, 0)
    )

    assert response["accepted"]


def test_cartesian_command_rejects_limit_above_daemon_ceiling() -> None:
    value, _ = dispatcher(cartesian_limits=(0.35, 1.0, 1.0, 2.0))
    payload = command("teleop", 1)
    payload["left"]["max_linear_velocity"] = 0.36

    with pytest.raises(ValueError, match="left Cartesian safety limit"):
        value("cartesian_command", 1, payload, (779, 0, 0))


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
