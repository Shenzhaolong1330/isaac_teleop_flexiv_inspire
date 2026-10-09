"""Stage B offline gates: real mapping/control/daemon adapters, synthetic external I/O."""

from __future__ import annotations
import ast
import copy
import importlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS, ModuleType
import time

import numpy as np
import pytest
import yaml
from isaac_teleop_core.world_teleop import (
    Pose,
    FixedMapping,
    WorldController,
    Limits,
    ApprovedRegion,
    calibrate,
    identity,
)
from isaac_teleop_core.rotation6d import rotvec_to_matrix, matrix_to_quaternion_xyzw
from isaac_teleop_core.command import BimanualCommand, CommandSource, ValidMask
from isaac_teleop_core.control import (
    ControlArbiter,
    GateInputs,
    ControlState,
    TransitionError,
)
from flexiv_rdk_daemon.nullspace import NullspaceAdapter, validate_nullspace

ROOT = Path(__file__).resolve().parents[1]


def raw():
    return yaml.safe_load((ROOT / "config/world_absolute.example.yaml").read_text())


def pose(x=0, y=0, z=0, r=None):
    return [x, y, z, *matrix_to_quaternion_xyzw(np.eye(3) if r is None else r)]


def samples(cfg=None):
    cfg = cfg or raw()
    return [
        {
            "capture_ns": 1_000_000_000 + i * 30_000_000,
            "identity": identity(cfg),
            "human": {
                "left": pose(-0.1),
                "right": pose(0.1),
                "left_time_ns": 1_000_000_000 + i * 30_000_000,
                "right_time_ns": 1_000_000_000 + i * 30_000_000,
            },
            "robot": {
                "left": pose(-0.1),
                "right": pose(0.1),
                "left_time_ns": 1_000_000_000 + i * 30_000_000,
                "right_time_ns": 1_000_000_000 + i * 30_000_000,
            },
        }
        for i in range(12)
    ]


def mapping(cfg=None):
    cfg = cfg or raw()
    return FixedMapping(cfg, calibrate(cfg, samples(cfg)))


def tracking(m, now, index=0, x=0, r=None):
    return {
        "identity": identity(m.raw),
        "receive_ns": now,
        "source_ns": now,
        "sequence": index,
        "frame_id": "world",
        "left": pose(-0.1 + x, r=r),
        "right": pose(0.1 + x, r=r),
    }


def measured(m, now):
    return {"receive_ns": now, **m.calibration["robot_reference"]}


def controller(m=None, **kw):
    m = m or mapping()
    approval = {
        "approved_by": "synthetic-test-only",
        "mapping_epoch": m.calibration["mapping_epoch"],
        "tool_identity": m.raw["tool_identity"],
        **{s: {"lower_m": [-2] * 3, "upper_m": [2] * 3} for s in ("left", "right")},
    }
    return WorldController(m, Limits(**kw), ApprovedRegion(approval))


def test_mapping_noncommuting_sign_shared_tool_and_closed_loop():
    cfg = raw()
    cfg["translation_scale"] = 2
    cfg["axis_rotation"] = rotvec_to_matrix([0, 0, np.pi / 2]).tolist()
    cfg["controller_from_reference"] = {
        s: pose(0.02, r=rotvec_to_matrix([0.3, 0, 0])) for s in ("left", "right")
    }
    cfg["tcp_from_reference"] = {
        s: pose(0, 0.03, r=rotvec_to_matrix([0, 0.4, 0])) for s in ("left", "right")
    }
    rows = samples(cfg)
    axis = np.asarray(cfg["axis_rotation"])
    for row in rows:
        for s in ("left", "right"):
            h = Pose.read(row["human"][s]).compose(
                Pose.read(cfg["controller_from_reference"][s])
            )
            ref = Pose(2 * axis @ h.p + np.array([0.5, 0.6, 0.7]), axis @ h.r)
            row["robot"][s] = ref.compose(
                Pose.read(cfg["tcp_from_reference"][s]).inverse()
            ).wire()
    m = FixedMapping(cfg, calibrate(cfg, rows))
    h = tracking(
        m,
        2_000_000_000,
        1,
        x=0.05,
        r=rotvec_to_matrix([0, 0, 0.7]) @ rotvec_to_matrix([0.4, 0, 0]),
    )
    a = m.targets(h)
    href = Pose.read(h["left"]).compose(
        Pose.read(cfg["controller_from_reference"]["left"])
    )
    ref = a["left"].compose(Pose.read(cfg["tcp_from_reference"]["left"]))
    np.testing.assert_allclose(ref.p, np.array([0.5, 0.6, 0.7]) + 2 * axis @ href.p)
    np.testing.assert_allclose(ref.r, axis @ href.r, atol=1e-12)
    h["left"][3:] = [-v for v in h["left"][3:]]
    np.testing.assert_allclose(m.targets(h)["left"].r, a["left"].r)
    # relative geometry maps in one common space, even with tool offsets.
    np.testing.assert_allclose(
        a["right"].p - a["left"].p, 2 * axis @ np.array([0.2, 0, 0]), atol=1e-12
    )
    for x in (0, 0.05, 0.02, 0):
        m.targets(tracking(m, 2_000_000_000, x=x))
    np.testing.assert_allclose(
        m.targets(tracking(m, 2_000_000_000))["left"].wire(),
        rows[0]["robot"]["left"],
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "change",
    [
        "stale",
        "skew",
        "unstable",
        "rotation",
        "identity",
        "clock",
        "relative",
        "version",
    ],
)
def test_calibration_refuses_bad_references(change):
    cfg = raw()
    rows = samples(cfg)
    if change == "stale":
        rows[0]["human"]["left_time_ns"] -= 200_000_000
    if change == "skew":
        rows[0]["human"]["left_time_ns"] -= 30_000_000
    if change == "unstable":
        rows[2]["human"]["left"][0] += 0.1
    if change == "rotation":
        rows[2]["human"]["left"][3:] = [0] * 4
    if change == "identity":
        rows[2]["identity"]["input_session_id"] = "restart"
    if change == "clock":
        rows[2]["capture_ns"] = rows[1]["capture_ns"]
    if change == "relative":
        rows[2]["robot"]["right"][0] += 0.2
    if change == "version":
        cfg["schema_version"] = 2
    with pytest.raises(ValueError):
        calibrate(cfg, rows)


def test_calibration_missing_assets_or_changed_installation(tmp_path):
    m = mapping()
    (tmp_path / "i.yaml").write_text(yaml.safe_dump(m.raw))
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(m.calibration))
    FixedMapping.load(tmp_path / "i.yaml", tmp_path / "c.yaml")
    with pytest.raises(FileNotFoundError):
        FixedMapping.load(tmp_path / "i.yaml", tmp_path / "absent")
    cfg = copy.deepcopy(m.raw)
    cfg["translation_scale"] = 1.2
    with pytest.raises(ValueError):
        FixedMapping(cfg, m.calibration)


@pytest.mark.parametrize("hz", [30, 60, 90])
def test_time_speed_acceleration_and_no_unaccepted_commit(hz):
    c = controller(position_lead=1, tracking_jump=1)
    now = 2_000_000_000
    c.start(tracking(c.mapping, now), measured(c.mapping, now), 1, now)
    c.state = "FOLLOW"
    previous = 0
    velocity = 0
    for i in range(1, hz + 1):
        t = now + round(i * 1e9 / hz)
        actual = {"receive_ns": t, **{s: p.wire() for s, p in c.accepted.items()}}
        point = c.propose(tracking(c.mapping, t, i, x=0.5), actual, 1, t)
        step = np.linalg.norm(point.left_delta_xyz)
        v = step * hz
        assert v <= c.limits.linear_speed + 1e-7
        assert abs(v - velocity) * hz <= c.limits.linear_acceleration + 1e-5
        assert c.accepted["left"].p[0] == previous - 0.1
        c.acknowledge(["left", "right"], complete=True)
        previous = c.accepted["left"].p[0] + 0.1
        velocity = v
    assert 0.04 < previous < 0.05


@pytest.mark.parametrize(
    "fault", ["gap", "clock", "reverse", "identity", "jump", "generation", "stale"]
)
def test_anomalies_fail_closed_without_target_jump(fault):
    c = controller()
    now = 2_000_000_000
    c.start(tracking(c.mapping, now, 1), measured(c.mapping, now), 1, now)
    t = now + 20_000_000
    h = tracking(c.mapping, t, 2)
    q = measured(c.mapping, t)
    gen = 1
    if fault == "gap":
        t += 200_000_000
        h["receive_ns"] = t
        q["receive_ns"] = t
    if fault == "clock":
        t = now
        h["receive_ns"] = t
        q["receive_ns"] = t
    if fault == "reverse":
        h["sequence"] = 0
    if fault == "identity":
        h["identity"]["tracking_origin_id"] = "recenter"
    if fault == "jump":
        h["left"][0] += 0.3
    if fault == "generation":
        gen = 2
    if fault == "stale":
        h["receive_ns"] -= 200_000_000
    before = c.accepted["left"].wire()
    with pytest.raises(ValueError):
        c.propose(h, q, gen, t)
    assert c.state in {"HOLD", "INVALID"}
    assert c.accepted["left"].wire() == before


def test_home_mapping_epoch_bilateral_latest_dwell_moving_path_timeout_partial():
    c = controller(dwell=0.05, timeout=0.5)
    now = 2_000_000_000
    epoch = c.mapping.calibration["mapping_epoch"]
    c.start(tracking(c.mapping, now), measured(c.mapping, now), 1, now)
    old = c.execution_epoch
    c.home()
    assert not c.authority
    c.home_complete(True)
    h = tracking(c.mapping, now + 20_000_000, 1, x=0.02)
    q = measured(c.mapping, now + 20_000_000)
    c.start(h, q, 1, now + 20_000_000)
    assert c.execution_epoch != old and c.mapping.calibration["mapping_epoch"] == epoch
    for i in range(2, 8):
        t = now + i * 20_000_000
        q = {"receive_ns": t, **{s: p.wire() for s, p in c.accepted.items()}}
        # one side still far -> no FOLLOW even with left at latest target
        q["left"] = c.snapshot["left"].wire()
        c.propose(tracking(c.mapping, t, i, x=0.02), q, 1, t)
        c.acknowledge(["left", "right"], complete=True)
        assert c.state == "ALIGNING"
    with pytest.raises(ValueError, match="human_moved"):
        t = now + 160_000_000
        c.propose(tracking(c.mapping, t, 8, x=0.04), {**q, "receive_ns": t}, 1, t)
    c = controller()
    c.path = ApprovedRegion()
    with pytest.raises(ValueError, match="unapproved"):
        c.start(tracking(c.mapping, now), measured(c.mapping, now), 1, now)
    c = controller(timeout=0.01)
    c.start(tracking(c.mapping, now), measured(c.mapping, now), 1, now)
    with pytest.raises(ValueError, match="timeout"):
        c.propose(
            tracking(c.mapping, now + 20_000_000, 1),
            measured(c.mapping, now + 20_000_000),
            1,
            now + 20_000_000,
        )
    c = controller()
    c.start(tracking(c.mapping, now), measured(c.mapping, now), 1, now)
    c.state = "FOLLOW"
    c.propose(
        tracking(c.mapping, now + 20_000_000, 1, x=0.01),
        measured(c.mapping, now + 20_000_000),
        1,
        now + 20_000_000,
    )
    right = c.accepted["right"].wire()
    c.acknowledge(["left"], complete=False)
    assert (
        c.state == "HOLD"
        and c.accepted["right"].wire() == right
        and c.accepted["left"].p[0] > -0.1
    )


def test_blocked_does_not_store_velocity_and_release_is_bounded():
    c = controller(position_lead=0.001, blocked_timeout=0.15)
    now = 2_000_000_000
    c.start(tracking(c.mapping, now), measured(c.mapping, now), 1, now)
    c.state = "FOLLOW"
    for i in range(1, 10):
        t = now + i * 20_000_000
        c.propose(tracking(c.mapping, t, i, x=0.05), measured(c.mapping, t), 1, t)
        c.acknowledge(["left", "right"], complete=True)
        assert c.accepted["left"].p[0] + 0.1 <= 0.001 + 1e-8
    t += 20_000_000
    feedback = {"receive_ns": t, **{s: p.wire() for s, p in c.accepted.items()}}
    p = c.propose(tracking(c.mapping, t, 10, x=0.05), feedback, 1, t)
    assert (
        np.linalg.norm(p.left_delta_xyz)
        <= c.limits.linear_acceleration * 0.02**2 + 1e-8
    )


@pytest.mark.parametrize("source", list(CommandSource))
def test_core_authority_no_pedal_is_only_teleop(source):
    a = ControlArbiter()
    a.begin_hardware_session("s")
    a.mark_ft_zeroed(session_id="s", connection_generation=1)
    a.declare_ready(connection_generation=1)
    a.update_gates(
        GateInputs(
            local_permission=True,
            teleop_session_authority=True,
            arms_online=True,
            hands_online=True,
            limits_ok=True,
            collision_clear=True,
        ),
        now_monotonic_ns=1_000_000_000,
    )
    a.arm(source)
    a.heartbeat(source, now_monotonic_ns=1_000_000_000)
    vector = [0] * 30
    vector[3:9] = [1, 0, 0, 0, 1, 0]
    vector[12:18] = [1, 0, 0, 0, 1, 0]
    command = BimanualCommand.from_policy_vectors(
        [vector],
        session_id="s",
        source=source,
        sequence=1,
        ttl_s=0.1,
        frame_id="world",
        deadman=True,
        issued_monotonic_ns=1_000_000_000,
    )
    if source is CommandSource.TELEOP:
        a.submit(command, now_monotonic_ns=1_000_000_000)
        assert a.snapshot.state is ControlState.ACTIVE
        a.tick(now_monotonic_ns=1_200_000_000)
        assert a.snapshot.state is ControlState.HOLD_LATCHED
    else:
        with pytest.raises(TransitionError):
            a.submit(command, now_monotonic_ns=1_000_000_000)


def ns_config():
    return {
        "schema_version": 1,
        **{
            s: {
                "enabled": True,
                "reference_strategy": "takeover_measured",
                "failure_policy": "hold",
                "rdk_version": "1.9.x",
                "weights": [0, 0, 0.5],
            }
            for s in ("left", "right")
        },
    }


def test_nullspace_adapter_fixed_reference_reentry_capabilities_failure_disabled():
    n = NullspaceAdapter(ns_config())
    calls = []
    robot = NS(
        SetNullSpacePosture=lambda q: calls.append(q),
        SetNullSpaceObjectives=lambda *w: calls.append(w),
    )
    n.mode_entry("left", robot, np.zeros(7), np.full(7, -2), np.full(7, 2), "1.9.0")
    n.mode_entry("left", robot, np.ones(7), np.full(7, -2), np.full(7, 2), "1.9.0")
    assert calls[0] == calls[2] == [0] * 7
    with pytest.raises(RuntimeError, match="unsupported"):
        n.mode_entry("right", NS(), np.zeros(7), np.full(7, -2), np.full(7, 2), "1.9.0")
    assert n.results["right"]["result"] == "failed"

    def fail(q):
        raise RuntimeError("SDK fail")

    with pytest.raises(RuntimeError):
        n.mode_entry(
            "left",
            NS(SetNullSpacePosture=fail, SetNullSpaceObjectives=lambda *w: None),
            np.zeros(7),
            np.full(7, -2),
            np.full(7, 2),
            "1.9.0",
        )
    NullspaceAdapter().mode_entry(
        "left", NS(), np.zeros(7), np.full(7, -2), np.full(7, 2), "old"
    )
    cfg = ns_config()
    cfg["left"]["weights"] = [0, 0, 0]
    with pytest.raises(ValueError):
        validate_nullspace(cfg)


@pytest.mark.parametrize(
    "world_enabled,source,partial,expected",
    [
        (True, "teleop", False, True),
        (False, "teleop", False, False),
        (True, "policy", False, False),
        (True, "replay", False, False),
        (True, "teleop", True, False),
    ],
)
def test_actual_daemon_pedal_authority_partial_typed_protocol(
    world_enabled, source, partial, expected
):
    from flexiv_rdk_daemon.mock_backend import MockBackend
    from flexiv_rdk_daemon.server import (
        DaemonInterlock,
        HandObservationCache,
        RDKRequestDispatcher,
    )
    from flexiv_rdk_daemon.typed_ipc import TypedEnvelopeCodec

    class Backend(MockBackend):
        def send_cartesian_target(self, side, pose_rdk, **kwargs):
            if partial and side == "right":
                raise RuntimeError("synthetic right failure")
            super().send_cartesian_target(side, pose_rdk, **kwargs)

    b = Backend()
    interlock = DaemonInterlock()
    interlock.ready_generation = b.connection_generation
    d = RDKRequestDispatcher(
        b,
        NS(is_zeroed_for=lambda s: True),
        HandObservationCache(),
        interlock,
        control_authorizations=NS(consume=lambda *a, **kw: None),
        world_absolute_enabled=world_enabled,
    )
    target = {
        "control_mode": "impedance",
        "tcp_pose_rdk": [0, 0, 0, 1, 0, 0, 0],
        "max_linear_velocity": 0.01,
        "max_angular_velocity": 0.05,
        "max_linear_acceleration": 0.05,
        "max_angular_acceleration": 0.1,
        "cartesian_stiffness": [1200] * 3 + [80] * 3,
        "cartesian_damping_ratio": [0.7] * 6,
    }
    payload = {
        "session_id": "s",
        "source": source,
        "source_sequence": "1",
        "expires_monotonic_ns": str(time.monotonic_ns() + 100_000_000),
        "valid_mask": 3,
        "safety_validated": True,
        "local_permission": True,
        "physical_pedal": False,
        "world_session_authority": True,
        "mapping_epoch": "synthetic-map",
        "left": target,
        "right": target,
    }
    codec = TypedEnvelopeCodec.load()
    _, _, decoded = codec.decode(codec.encode("cartesian_command", 1, payload))
    assert decoded["world_session_authority"] and not decoded.get(
        "physical_pedal", False
    )
    try:
        kind, result = d("cartesian_command", 1, decoded, (222, 0, 0))
        assert result["accepted"] == expected
        if partial:
            assert result["sdk_returned_sides"] == ["left"]
        _, _, ack = codec.decode(codec.encode(kind, 2, result))
        assert ack.get("accepted", False) == expected
    finally:
        d.close()


def load_bridge(monkeypatch):
    std = ModuleType("std_msgs")
    msg = ModuleType("std_msgs.msg")
    msg.String = lambda: NS(data="")
    std.msg = msg
    monkeypatch.setitem(sys.modules, "std_msgs", std)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", msg)
    return importlib.import_module("flexiv_inspire_control.world_bridge").WorldBridge


def test_actual_world_bridge_point_input_preview_fingers_and_true_pedal(monkeypatch):
    WorldBridge = load_bridge(monkeypatch)
    b = WorldBridge()
    b._world = controller()
    m = b._world.mapping
    now = 2_000_000_000
    b._home_inflight = False
    monkeypatch.setattr(time, "monotonic_ns", lambda: now + 20_000_000)
    b._daemon_instance_id = "d"
    b._connection_generation = 1
    feedback = measured(m, now + 20_000_000)
    b._world_feedback = lambda: feedback
    b._world.start(tracking(m, now), measured(m, now), ("d", 1), now)
    b._arbiter = NS(snapshot=NS(state=ControlState.TELEOP_ARMED))
    b._physical_pedal = False
    b._world_event = lambda: None
    h = tracking(m, now + 20_000_000, 1)
    msg = NS(
        metadata_keys=["world_tracking", "world_command_enabled"],
        metadata_values=[json.dumps(h), "true"],
        valid_mask=12,
        deadman=False,
    )
    out = b._world_receive(msg)
    assert out.deadman and out.valid_mask == 3 and not b._physical_pedal
    cmd = NS(metadata={"world_tracking": json.dumps(h)})
    from isaac_teleop_core.command import CommandPoint

    p = b._world_point(cmd, CommandPoint.identity())
    assert np.linalg.norm(p.left_delta_xyz) == 0
    b._world.acknowledge(["left", "right"], complete=True)
    b._world.home()
    out = b._world_receive(msg)
    assert not out.deadman and out.valid_mask == 0


def test_actual_rdk_backend_mode_entry_restores_nullspace_once_not_each_send():
    from flexiv_rdk_daemon.backend import FlexivRDKBackend, RobotSpec

    calls = []
    b = FlexivRDKBackend(
        (RobotSpec("left", "l"), RobotSpec("right", "r")),
        write_guard=NS(require=lambda *a, **k: None),
        nullspace_config=ns_config(),
    )
    b._rdk_version = "1.9.0"
    b._rdk = NS(Mode=NS(NRT_CARTESIAN_MOTION_FORCE=1))
    robot = NS(
        SwitchMode=lambda mode: calls.append("mode"),
        SetForceControlAxis=lambda v: None,
        SetNullSpacePosture=lambda q: calls.append(("q", q)),
        SetNullSpaceObjectives=lambda *w: calls.append(("w", w)),
        SendCartesianMotionForce=lambda *a, **kw: calls.append("send"),
    )
    b._robots["left"] = robot
    b._metadata["left"] = NS(q_min=[-2] * 7, q_max=[2] * 7)
    q = np.ones(7) * 0.2
    b.observe = lambda s: NS(q=q)
    b.switch_cartesian_mode("left", local_console=True)
    for _ in range(20):
        b.send_cartesian_target(
            "left",
            np.array([0, 0, 0, 1, 0, 0, 0]),
            max_linear_velocity=0.01,
            max_angular_velocity=0.05,
            max_linear_acceleration=0.1,
            max_angular_acceleration=0.2,
            local_authorized=True,
        )
    assert sum(isinstance(v, tuple) and v[0] == "q" for v in calls) == 1
    q[:] = 0.5
    b.switch_cartesian_mode("left", local_console=True)
    assert [v[1] for v in calls if isinstance(v, tuple) and v[0] == "q"] == [
        [0.2] * 7,
        [0.2] * 7,
    ]
    b.begin_nullspace_takeover()
    b.switch_cartesian_mode("left", local_console=True)
    assert b.nullspace_results["left"]["q_ref"] == [0.5] * 7


def test_follow_alignment_switch_continuous_and_duplicates_dwell():
    c = controller(dwell=0.05)
    now = 2_000_000_000
    c.start(tracking(c.mapping, now), measured(c.mapping, now), 1, now)
    before = c.accepted["left"].wire()
    # timer updates reuse the same source snapshot, fresh receive time unchanged
    h = tracking(c.mapping, now)
    for i in range(1, 6):
        t = now + i * 20_000_000
        c.propose(h, measured(c.mapping, t), 1, t)
        c.acknowledge(["left", "right"], complete=True)
    assert c.state == "FOLLOW" and c.accepted["left"].wire() == before
    assert c.execution_epoch and c.mapping.calibration["mapping_epoch"]


def test_blocked_latches_before_unreachable_error_can_accumulate():
    c = controller(position_lead=0.001, blocked_timeout=0.1)
    now = 2_000_000_000
    c.start(tracking(c.mapping, now), measured(c.mapping, now), 1, now)
    c.state = "FOLLOW"
    with pytest.raises(ValueError, match="obstruction"):
        for i in range(1, 50):
            t = now + i * 20_000_000
            c.propose(tracking(c.mapping, t, i, x=0.05), measured(c.mapping, t), 1, t)
            c.acknowledge(["left", "right"], complete=True)
    assert c.state == "HOLD" and c.accepted["left"].p[0] <= -0.099


def test_cli_reference_binding_save_validate_preview(tmp_path, monkeypatch, capsys):
    from isaac_teleop_core.world_cli import main

    cfg = raw()
    rows = samples(cfg)
    new = "live-source-boot"
    for row in rows:
        row["identity"]["input_session_id"] = new
    i = tmp_path / "manual.yaml"
    i.write_text(yaml.safe_dump(cfg))
    r = tmp_path / "rows.json"
    r.write_text(json.dumps(rows))
    dest = tmp_path / "assets"
    monkeypatch.setattr(
        sys, "argv", ["world_cli", "calibrate", str(i), str(r), str(dest)]
    )
    main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "world_cli",
            "validate",
            str(dest / "installation.yaml"),
            str(dest / "calibration.yaml"),
        ],
    )
    main()
    m = FixedMapping.load(dest / "installation.yaml", dest / "calibration.yaml")
    assert m.raw["input_session_id"] == new
    t = tmp_path / "tracking.json"
    t.write_text(json.dumps([tracking(m, 2_000_000_000, x=0.01)]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "world_cli",
            "preview",
            str(dest / "installation.yaml"),
            str(dest / "calibration.yaml"),
            str(t),
        ],
    )
    main()
    assert '"valid": true' in capsys.readouterr().out


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("legacy_cache", ["bogus", "missing"])
def test_actual_bridge_sender_world_targets_and_partial_receipts(
    monkeypatch, partial, legacy_cache
):
    WorldBridge = load_bridge(monkeypatch)
    from isaac_teleop_core.scheduling import PointSchedule
    from isaac_teleop_core.evidence import receipt, emit_receipt, command_id
    from isaac_teleop_core.control import HoldReason
    from isaac_teleop_core.rotation6d import ros_pose_to_rdk_pose
    from flexiv_inspire_control.frames import BaseTransform
    from flexiv_inspire_control.conversion import cartesian_target_from_point
    from flexiv_rdk_daemon.mock_backend import MockBackend
    from flexiv_rdk_daemon.server import (
        DaemonInterlock,
        HandObservationCache,
        RDKRequestDispatcher,
    )
    from flexiv_rdk_daemon.typed_ipc import TypedEnvelopeCodec

    tree = ast.parse(
        (
            ROOT / "ros2_ws/src/flexiv_inspire_control/flexiv_inspire_control/node.py"
        ).read_text()
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ControlBridge"
    )
    names = {
        "_execute_chunk",
        "_targets_for_point",
        "_dispatch_chunk",
        "_is_expired_control_authorization",
        "_routine_daemon_hold_reason",
    }
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    # Remove class-only decorators before binding on our dependency-isolated bridge.
    for n in methods:
        n.decorator_list = []
    ns = {
        "time": time,
        "PointSchedule": PointSchedule,
        "ControlState": ControlState,
        "HoldReason": HoldReason,
        "CommandSource": CommandSource,
        "ValidMask": ValidMask,
        "command_id": command_id,
        "receipt": receipt,
        "emit_receipt": emit_receipt,
        "BimanualCommandMsg": object,
        "cartesian_target_from_point": cartesian_target_from_point,
        "matrix_to_quaternion_xyzw": matrix_to_quaternion_xyzw,
        "ros_pose_to_rdk_pose": ros_pose_to_rdk_pose,
        "np": np,
    }
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), "production-bridge", "exec"),
        ns,
    )

    class Bridge(WorldBridge):
        pass

    for name in names:
        setattr(
            Bridge,
            name,
            staticmethod(ns[name])
            if name.startswith("_is_") or name == "_routine_daemon_hold_reason"
            else ns[name],
        )
    b = Bridge()
    b._world = controller()
    b._world_event = lambda: None
    b._world_lock = threading.RLock()
    b._world_send_lock = threading.RLock()
    b._state_lock = threading.RLock()
    b._chunk_lock = threading.Lock()
    b._hardware_command_lock = threading.Lock()
    b._chunk_generation = 1
    b._daemon_instance_id = "d"
    b._connection_generation = 1
    b._home_inflight = False
    b._physical_pedal = False
    b._local_permission = True
    now = time.monotonic_ns()
    h0 = tracking(b._world.mapping, now - 20_000_000, 1)
    actual = measured(b._world.mapping, now - 20_000_000)
    b._world.start(h0, actual, ("d", 1), now - 20_000_000)
    b._world.state = "FOLLOW"
    b._world_feedback = lambda: measured(b._world.mapping, time.monotonic_ns())
    b._safe_pose_rdk = {
        s: np.array([99, 99, 99, 1, 0, 0, 0]) for s in ("left", "right")
    }  # deliberately inconsistent legacy cache
    if legacy_cache == "missing":
        b._safe_pose_rdk = {}
    b._previous_output_quaternion = {s: None for s in ("left", "right")}
    b._world_from_base = {
        s: BaseTransform(np.zeros(3), np.array([0, 0, 0, 1])) for s in ("left", "right")
    }
    params = {
        "cartesian_control_mode": "impedance",
        "cartesian_impedance_stiffness": [1200] * 3 + [80] * 3,
        "cartesian_damping_ratio": [0.7] * 6,
        "max_translation_step_m": 0.01,
        "max_rotation_step_rad": 0.1,
        "software_safety_limits_enabled": False,
        "max_linear_velocity_m_s": 0.05,
        "max_angular_velocity_rad_s": 0.2,
        "max_linear_acceleration_m_s2": 0.2,
        "max_angular_acceleration_rad_s2": 0.5,
    }
    b.get_parameter = lambda k: NS(value=params[k])
    b.get_clock = lambda: NS(now=lambda: NS(to_msg=lambda: NS(sec=0, nanosec=0)))
    sent = []
    b._safe_pub = NS(publish=lambda m: None)
    b._sent_pub = NS(publish=sent.append)
    b._emit_deviceio = lambda *a, **k: True
    b._publish_trace = lambda *a: None
    b._publish_control_state = lambda: None
    b._send_hold_once = lambda *a: None
    b._arbiter = NS(
        snapshot=NS(state=ControlState.ACTIVE),
        tick=lambda **kw: NS(state=ControlState.ACTIVE),
        mark_sent=lambda c: None,
        require_reauthorization=lambda *a, **kw: None,
        reject_invalid_command=lambda *a, **kw: None,
    )
    b._update_gates = lambda now: None
    b._effective_motion_pedal = lambda: True
    b._pending_arm_token = "synthetic-token"
    b._pending_arm_token_expiry_ns = now + 1_000_000_000
    b._rdk_control_lease_active = False
    b._single_point_message = lambda m, *a, **kw: copy.deepcopy(m)

    class Backend(MockBackend):
        def send_cartesian_target(self, side, p, **kw):
            if partial and side == "right":
                raise RuntimeError("synthetic partial")
            super().send_cartesian_target(side, p, **kw)

    backend = Backend()
    interlock = DaemonInterlock()
    interlock.ready_generation = backend.connection_generation
    d = RDKRequestDispatcher(
        backend,
        NS(is_zeroed_for=lambda s: True),
        HandObservationCache(),
        interlock,
        control_authorizations=NS(consume=lambda *a, **k: None),
        world_absolute_enabled=True,
    )
    codec = TypedEnvelopeCodec.load()

    def request(kind, payload):
        _, _, decoded = codec.decode(codec.encode(kind, 1, payload))
        kind, result = d(kind, 1, decoded, (222, 0, 0))
        return kind, result

    b._ipc_command = NS(request=request)
    command = BimanualCommand.from_policy_vectors(
        [
            np.r_[
                np.zeros(3),
                [1, 0, 0, 0, 1, 0],
                np.zeros(3),
                [1, 0, 0, 0, 1, 0],
                np.zeros(12),
            ]
        ],
        session_id="s",
        source=CommandSource.TELEOP,
        sequence=1,
        ttl_s=0.1,
        frame_id="world",
        deadman=True,
        valid_mask=ValidMask.LEFT_ARM | ValidMask.RIGHT_ARM,
        issued_monotonic_ns=now,
    )
    from dataclasses import replace

    command = replace(
        command,
        metadata={
            "world_tracking": json.dumps(tracking(b._world.mapping, now, 2, x=0.01))
        },
    )
    msg = NS(trajectory=[NS()], metadata_keys=[], metadata_values=[])
    try:
        b._dispatch_chunk(1, command, msg, now)
        assert b._world.accepted["left"].p[0] > -0.1, b._world.reason
        assert (
            b._world.accepted["left"].p[0] < 0
        )  # doesn't integrate desired-measured into bogus 99 cache
        if partial:
            assert (
                b._world.state == "HOLD"
                and b._world.accepted["right"].p[0] == 0.1
                and not sent
            )
        else:
            assert len(sent) == 1 and sent[0].trajectory[0].left_delta_xyz[0] > 0
    finally:
        d.close()


def test_stage_b_production_data_closure(tmp_path):
    from flexiv_inspire_isaac.data_pipeline.stage_b_smoke import run

    result = run(tmp_path / "evidence")
    assert result["status"] == "PASS" and result["continuous_segments"] == 2
    assert result["valid_follow_commands"] < result["raw_commands"]
    assert result["max_limited_linear_acceleration_m_s2"] <= 0.2 + 1e-8
    from flexiv_inspire_isaac.data_pipeline.provenance import (
        verify_archive,
        resolve_asset,
    )

    directory = Path(result["manifest"]).parent
    verify_archive(directory)
    assert "world_absolute" in resolve_asset(directory, "effective_config").read_text()


def test_actual_world_event_native_clock_producer_and_gate_failure(monkeypatch):
    WorldBridge = load_bridge(monkeypatch)
    b = WorldBridge()
    b._world = controller()
    b._physical_pedal = False
    now = time.monotonic_ns()
    b._world.start(
        tracking(b._world.mapping, now), measured(b._world.mapping, now), 1, now
    )
    records = []
    b._world_state_pub = NS(publish=lambda m: None)
    b._deviceio = NS(emit=lambda doc, **kw: records.append(doc))
    b._world_event()
    assert len(records) == 2 and records[0]["producer"] == "control"
    assert (
        records[1]["payload"]["world"]["state"] == "ALIGNING"
        and records[1]["timing_valid"]
    )
    assert records[1]["source_clock_domain"] == "host_monotonic"
    stopped = []
    b._arbiter = NS(stop=lambda: stopped.append(True))
    b._send_hold_once = lambda r: stopped.append(r)
    b.get_logger = lambda: NS(error=lambda m: None)

    def fail(*a, **kw):
        raise BufferError("synthetic audit full")

    b._deviceio = NS(emit=fail)
    b._world_event()
    assert b._world.state == "HOLD" and stopped


def test_world_tool_payload_identity_change_invalidates_actual_bridge(
    tmp_path, monkeypatch
):
    import hashlib

    WorldBridge = load_bridge(monkeypatch)
    b = WorldBridge()
    m = mapping()
    tool = tmp_path / "tool.yaml"
    tool.write_text("tool: original\n")
    cfg = copy.deepcopy(m.raw)
    cfg["tool_identity"] = hashlib.sha256(
        json.dumps(
            {"tool": "original"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    m = mapping(cfg)
    b._world = controller(m)
    b._world_tool_path = tool
    b._world_tool_stat = None
    b._check_world_tool(m)
    tool.write_text("tool: changed-tool\n")
    with pytest.raises(ValueError, match="recalibrate"):
        b._check_world_tool(m)
    assert b._world.state == "INVALID"


@pytest.mark.parametrize(
    "operation", ["toggle", "rerecord", "pause_recording", "resume_recording"]
)
@pytest.mark.parametrize("control_state", ["ACTIVE", "HOLD_LATCHED", "FAULT"])
def test_actual_episode_operations_independent_of_motion_and_home(
    operation, control_state
):
    path = ROOT / "apps/session_manager/src/flexiv_inspire_isaac/episode_control.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "EpisodeController"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_on_control"
    )
    ns = {"String": object}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
    calls = []
    b = NS(
        _control_state_name=control_state,
        get_parameter=lambda k: NS(
            value="world_absolute" if k == "teleop_mapping_mode" else 100
        ),
        _completed_episodes=0,
        _episode_index=1,
        _attempt=1,
        _stop=lambda **kw: calls.append(("save", kw)) or True,
        _start=lambda: calls.append("start"),
        _pause=lambda: calls.append("pause"),
        _resume=lambda: calls.append("resume"),
        _begin_home=lambda a: pytest.fail("save/record operations cannot Home"),
        _fail=lambda r: pytest.fail(r),
    )
    ns["_on_control"](b, NS(data=operation))
    assert calls
    if operation == "toggle":
        assert b._completed_episodes == 1 and b._episode_index == 2
    if operation == "rerecord":
        assert b._completed_episodes == 0 and b._attempt == 2


def test_actual_read_only_capture_uses_source_clock_qos_and_paired_feedback(
    tmp_path, monkeypatch
):
    from isaac_teleop_core.world_cli import capture

    subs = {}
    stamp = [1_000_000_000]
    sensor_qos = object()
    qos_used = {}

    class Node:
        def __init__(self, name):
            pass

        def create_subscription(self, kind, topic, callback, qos):
            subs[topic] = callback
            qos_used[topic] = qos

        def destroy_node(self):
            pass

    rclpy = ModuleType("rclpy")
    rclpy.init = lambda: None
    rclpy.shutdown = lambda: None
    node_mod = ModuleType("rclpy.node")
    node_mod.Node = Node
    qos_mod = ModuleType("rclpy.qos")
    qos_mod.qos_profile_sensor_data = sensor_qos
    msg_mod = ModuleType("flexiv_inspire_interfaces.msg")
    msg_mod.BimanualCommand = object
    msg_mod.ArmState = object
    interfaces = ModuleType("flexiv_inspire_interfaces")
    interfaces.msg = msg_mod
    for name, value in (
        ("rclpy", rclpy),
        ("rclpy.node", node_mod),
        ("rclpy.qos", qos_mod),
        ("flexiv_inspire_interfaces", interfaces),
        ("flexiv_inspire_interfaces.msg", msg_mod),
    ):
        monkeypatch.setitem(sys.modules, name, value)
    m = mapping()

    def spin_once(node, timeout_sec):
        stamp[0] += 20_000_000
        t = stamp[0]
        h = tracking(m, t, t)
        subs["/teleop/command"](
            NS(metadata_keys=["world_tracking"], metadata_values=[json.dumps(h)])
        )
        for side in ("left", "right"):
            p = NS(
                position=NS(x=-0.1 if side == "left" else 0.1, y=0, z=0),
                orientation=NS(x=0, y=0, z=0, w=1),
            )
            acq = NS(
                valid=True,
                timing_valid=True,
                host_clock_domain="host_monotonic",
                mapped_host_time=NS(sec=t // 1_000_000_000, nanosec=t % 1_000_000_000),
            )
            subs[f"/robot/{side}_arm/state"](
                NS(tcp_pose=p, connected=True, fault="", acquisition=acq)
            )

    rclpy.spin_once = spin_once
    monkeypatch.setattr(time, "monotonic_ns", lambda: stamp[0])
    monkeypatch.setattr(time, "monotonic", lambda: stamp[0] / 1e9)
    output = tmp_path / "capture.json"
    capture(output, 0.4)
    rows = json.loads(output.read_text())
    calibrate(m.raw, rows)
    assert len(rows) >= 10 and qos_used["/robot/left_arm/state"] is sensor_qos
    assert rows[-1]["robot"]["left_time_ns"] == rows[-1]["capture_ns"]


def test_invalid_mapping_survives_home_and_ordinary_holds():
    c = controller()
    c.hold("recenter", invalid=True)
    c.hold("stop")
    assert c.state == "INVALID"
    c.home()
    c.home_complete(True)
    assert c.state == "INVALID"
    now = 2_000_000_000
    with pytest.raises(ValueError, match="recalibration"):
        c.start(tracking(c.mapping, now), measured(c.mapping, now), 1, now)


def test_limited_candidate_checks_approved_region_with_velocity_inertia():
    c = controller()
    now = 2_000_000_000
    c.start(tracking(c.mapping, now), measured(c.mapping, now), 1, now)
    c.state = "FOLLOW"
    for side in ("left", "right"):
        boundary = float(c.accepted[side].p[0])
        c.path.raw[side]["upper_m"][0] = boundary
        c.velocity[side] = (np.array([0.04, 0, 0]), np.zeros(3))
    t = now + 20_000_000
    with pytest.raises(ValueError, match="limited_path"):
        c.propose(tracking(c.mapping, t, 1), measured(c.mapping, t), 1, t)
    assert not c.authority and c.pending is None


def test_calibration_capture_source_clock_must_advance():
    cfg = raw()
    rows = samples(cfg)
    for row in rows:
        row["human_source_ns"] = 123
    with pytest.raises(ValueError, match="source clock"):
        calibrate(cfg, rows)


def test_world_config_renders_real_entrypoints_and_requires_capture_archive(tmp_path):
    from flexiv_inspire_isaac.system_config import (
        load_system_config,
        render_runtime_configs,
    )
    from flexiv_inspire_isaac.cli import _episode_command
    from flexiv_inspire_isaac.data_pipeline.provenance import (
        archive_episode,
        verify_archive,
    )

    config = load_system_config(ROOT / "config/site.yaml")
    paths = {}
    for key in ("world_installation", "world_calibration", "world_execution_config"):
        path = tmp_path / (key + ".yaml")
        path.write_text("schema_version: 1\n")
        paths[key] = str(path)
    Path(paths["world_calibration"]).write_text(
        "schema_version: 1\nreference_samples: missing_capture.json\n"
    )
    config.document["teleop"].update(mapping_mode="world_absolute", **paths)
    rendered = render_runtime_configs(config, tmp_path / "rendered")
    for name in ("control_bridge.yaml", "teleop.yaml"):
        params = yaml.safe_load(rendered[name].read_text())["/**"]["ros__parameters"]
        assert params["teleop_mapping_mode"] == "world_absolute"
        assert params["world_installation"] == paths["world_installation"]
    assert "teleop_mapping_mode:=world_absolute" in _episode_command(config, rendered)
    episode = tmp_path / "episode"
    episode.mkdir()
    archived = archive_episode(episode, config={"teleop": paths})
    (episode / "manifest.json").write_text(json.dumps({"provenance": archived}))
    assert not archived["complete"]
    with pytest.raises(ValueError, match="required archive asset missing"):
        verify_archive(episode)


def production_methods(names, namespace):
    path = ROOT / "ros2_ws/src/flexiv_inspire_control/flexiv_inspire_control/node.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ControlBridge"
    )
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    for n in methods:
        n.decorator_list = []
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace


@pytest.mark.parametrize(
    "source", [CommandSource.TELEOP, CommandSource.POLICY, CommandSource.REPLAY]
)
def test_actual_bridge_gates_world_cannot_bypass_limits_or_collision_policy_unchanged(
    source,
):
    ns = production_methods(
        {"_update_gates", "_effective_motion_pedal"},
        {
            "CommandSource": CommandSource,
            "GateInputs": GateInputs,
        },
    )
    now = 2_000_000_000
    seen = []
    b = NS(
        _world=controller(),
        _physical_pedal=False,
        _pending_arm_source=source,
        _arbiter=NS(
            snapshot=NS(active_source=source),
            update_gates=lambda g, **kw: seen.append(g),
        ),
        _hand_connected={s: True for s in ("left", "right")},
        _last_hand_observation_ns={s: now for s in ("left", "right")},
        _last_arm_observation_ns={s: now for s in ("left", "right")},
        _latest_wire={s: {"connected": True, "fault": ""} for s in ("left", "right")},
        _local_permission=True,
        _limits_ok=False,
        _collision_clear=False,
        _state_lock=threading.RLock(),
        _try_arm_pending_authorization=lambda t: None,
        _world_motion_authority=lambda: True,
    )
    params = {
        "software_safety_limits_enabled": False,
        "policy_pedal_required": False,
        "hand_observation_timeout_ms": 150,
        "require_hands_for_arm_control": True,
    }
    b.get_parameter = lambda k: NS(value=params[k])
    b._effective_motion_pedal = lambda: ns["_effective_motion_pedal"](b)
    ns["_update_gates"](b, now)
    gates = seen[-1]
    if source is CommandSource.TELEOP:
        assert gates.teleop_session_authority and not gates.physical_pedal
        assert not gates.limits_ok and not gates.collision_clear
    else:
        assert not gates.teleop_session_authority
        assert gates.limits_ok and gates.collision_clear
        assert gates.physical_pedal == (source is CommandSource.POLICY)


def test_actual_world_feedback_uses_mapped_source_and_measured_q(monkeypatch):
    from flexiv_inspire_control.frames import BaseTransform

    WorldBridge = load_bridge(monkeypatch)
    b = WorldBridge()
    b._world = controller()
    b._state_lock = threading.RLock()
    b._last_arm_mapped_ns = {"left": 100, "right": 90}
    b._last_arm_observation_ns = {"left": 999, "right": 999}
    b._arm_safety_ok = {s: True for s in ("left", "right")}
    b._latest_wire = {
        s: {
            "connected": True,
            "fault": "",
            "q": [0.2] * 7,
            "tcp_pose_rdk_xyz_wxyz": [0, 0, 0, 1, 0, 0, 0],
        }
        for s in ("left", "right")
    }
    b._world_from_base = {
        s: BaseTransform(np.zeros(3), np.array([0, 0, 0, 1])) for s in ("left", "right")
    }
    feedback = b._world_feedback()
    assert feedback["receive_ns"] == 90 and feedback["q"]["left"] == [0.2] * 7
    b._arm_safety_ok["right"] = False
    with pytest.raises(ValueError, match="unsafe"):
        b._world_feedback()


def test_actual_home_request_serializes_sends_and_ready_home_clears_epoch(monkeypatch):
    ns = production_methods(
        {"_begin_home_request", "_begin_home_request_locked"},
        {
            "time": time,
            "ControlState": ControlState,
            "threading": threading,
        },
    )
    c = controller()
    now = time.monotonic_ns()
    c.start(tracking(c.mapping, now), measured(c.mapping, now), 1, now)
    c.hold("ready")
    events = []
    thread_args = []
    # Simulate thread launch, not the production transition.
    monkeypatch.setattr(
        threading, "Thread", lambda **kw: NS(start=lambda: thread_args.append(kw))
    )
    b = NS(
        _world=c,
        _world_send_lock=threading.RLock(),
        _world_lock=threading.RLock(),
        _state_lock=threading.RLock(),
        _home_inflight=False,
        _home_authorization_lease_active=True,
        _pending_home_recovery=False,
        _arbiter=NS(snapshot=NS(state=ControlState.READY)),
        _update_gates=lambda t: None,
        _home_gate_failure=lambda *a, **kw: "",
        _world_event=lambda: events.append(c.state),
        _emit_home_event=lambda *a, **kw: None,
        _run_home=lambda *a: None,
        _publish_home_status=lambda *a, **kw: None,
    )
    b._begin_home_request_locked = lambda r: ns["_begin_home_request_locked"](b, r)
    ns["_begin_home_request"](b, "home-id")
    assert events == ["HOMING"] and b._home_inflight
    assert c.accepted == {} and c.execution_epoch == "" and len(thread_args) == 1


def test_actual_daemon_local_takeover_prepares_enabled_nullspace_even_without_hold():
    from flexiv_rdk_daemon.mock_backend import MockBackend
    from flexiv_rdk_daemon.server import (
        DaemonInterlock,
        HandObservationCache,
        RDKRequestDispatcher,
    )

    calls = []
    backend = MockBackend()
    backend.begin_nullspace_takeover = lambda: True
    backend.switch_cartesian_mode = lambda side, **kw: calls.append(("mode", side))
    backend.send_hold_from_measurement = lambda side, **kw: calls.append(
        ("anchor", side)
    )
    d = RDKRequestDispatcher(
        backend,
        NS(is_zeroed_for=lambda s: True),
        HandObservationCache(),
        DaemonInterlock(),
        control_authorizations=NS(mint=lambda **kw: ("token", 123)),
    )
    try:
        kind, result = d._authorize_control(
            42, {"session_id": "s", "source": "teleop", "clear_hold_latched": True}
        )
        assert kind == "authorize_control_result" and result["authorized"]
        assert calls == [
            ("mode", "left"),
            ("mode", "right"),
            ("anchor", "left"),
            ("anchor", "right"),
        ]
    finally:
        d.close()


def test_sdk_return_after_watchdog_hold_commits_sent_proposal_without_authority():
    c = controller()
    now = 2_000_000_000
    c.start(tracking(c.mapping, now, x=0.005), measured(c.mapping, now), 1, now)
    t = now + 20_000_000
    c.propose(tracking(c.mapping, t, 1, x=0.005), measured(c.mapping, t), 1, t)
    proposal = c.pending
    before = c.accepted["left"].wire()
    c.hold("watchdog_during_sdk")
    c.acknowledge(["left"], complete=False, proposal=proposal)
    assert c.accepted["left"].wire() != before
    assert c.accepted["left"].wire() == proposal[0]["left"].wire()
    assert not c.authority and np.linalg.norm(c.velocity["left"][0]) == 0


def test_actual_world_operation_start_requires_token_and_fingers_takeover_is_bounded(
    monkeypatch,
):
    from isaac_teleop_core.command import CommandPoint

    WorldBridge = load_bridge(monkeypatch)
    b = WorldBridge()
    b._world = controller()
    m = b._world.mapping
    clock = [2_000_000_000]
    monkeypatch.setattr(time, "monotonic_ns", lambda: clock[0])
    b._session_id = "s"
    b._world_send_lock = threading.RLock()
    b._world_lock = threading.RLock()
    b._home_inflight = False
    b._local_permission = True
    b._rdk_control_lease_active = False
    b._pending_arm_token = None
    b._pending_arm_token_expiry_ns = clock[0] + 100_000_000
    b._daemon_instance_id, b._connection_generation = "d", 1
    b._world_tracking = tracking(m, clock[0])
    b._world_feedback = lambda: measured(m, clock[0])
    b._world_event = lambda: None
    errors = []
    b.get_logger = lambda: NS(error=errors.append)
    b._update_gates = lambda t: None
    b._try_arm_pending_authorization = lambda t: None
    b._arbiter = NS(
        snapshot=NS(state=ControlState.READY),
        observe_deadman_released=lambda s: None,
        stop=lambda: None,
    )
    b._send_hold_once = lambda r: None

    def op(action, **kw):
        b._world_operation(
            NS(
                data=json.dumps(
                    {
                        "session_id": "s",
                        "confirm": "WORLD-TELEOP-LOCAL",
                        "operation": action,
                        **kw,
                    }
                )
            )
        )

    op("start")
    assert not b._world.authority and "authorization" in errors[-1]
    b._pending_arm_token = "fresh-local-token"
    op("start")
    assert b._world.state == "ALIGNING" and b._world.execution_epoch
    op("fingers", targets={s: [300] * 6 for s in ("left", "right")})
    assert not b._world.fingers and "FOLLOW" in errors[-1]
    b._world.state = "FOLLOW"  # The convergence algorithm is tested independently.
    b._hand_angles = {s: np.array([300] * 6) for s in ("left", "right")}
    b._last_hand_observation_ns = {s: clock[0] for s in ("left", "right")}
    op("fingers", targets={s: [100] * 6 for s in ("left", "right")})
    assert not b._world.fingers and "match" in errors[-1]
    op("fingers", targets={s: [300] * 6 for s in ("left", "right")})
    assert b._world.fingers
    clock[0] += 20_000_000
    point = CommandPoint.identity()
    from dataclasses import replace

    point = replace(
        point,
        left_hand_targets=np.array([1000] * 6),
        right_hand_targets=np.array([0] * 6),
    )
    cmd = NS(metadata={"world_tracking": json.dumps(tracking(m, clock[0], 1))})
    limited = b._world_point(cmd, point)
    np.testing.assert_allclose(limited.left_hand_targets, [310] * 6)
    np.testing.assert_allclose(limited.right_hand_targets, [290] * 6)
    b._world.acknowledge(["left", "right"], complete=True)
    op("stop")
    assert b._world.state == "HOLD" and not b._world.fingers
    op("invalidate")
    assert b._world.state == "INVALID"
    op("start")
    assert not b._world.authority and "recalibration" in errors[-1]
