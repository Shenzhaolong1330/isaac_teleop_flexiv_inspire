from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import yaml

from flexiv_rdk_daemon.backend import FlexivRDKBackend, RobotSpec
from flexiv_rdk_daemon.configuration import (
    load_daemon_configuration,
    read_tool_payload_identity,
)
from flexiv_rdk_daemon.guard import (
    HardwareWriteGuard,
    HardwareWriteRejected,
)


def tool_document():
    arm = {
        "tool": {
            "name": "test",
            "serial": "tool-test-001",
            "mounting_revision": "rev-a",
        },
        "payload": {
            "mass_kg": 1.0,
            "center_of_mass_m": [0.0, 0.0, 0.1],
            "inertia_kg_m2": [0.1] * 6,
            "tcp_location_xyz_wxyz": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        },
        "locally_verified": True,
    }
    return {
        "schema_version": 1,
        "arms": {"left": arm, "right": copy.deepcopy(arm)},
    }


def test_site_daemon_configuration_exposes_cartesian_command_ceiling():
    path = (
        Path(__file__).parents[2]
        / "apps/flexiv_daemon/config/robots.yaml"
    )

    config = load_daemon_configuration(path)

    assert config.cartesian_limits == (0.35, 1.0, 1.0, 2.0)
    assert config.ft_zero["enable_settle_timeout_s"] == 10.0
    assert config.ft_zero["enable_settle_window_s"] == 0.5
    assert config.ft_zero["external_contact_check_enabled"] is False


def test_tool_payload_sha_is_canonical_but_touch_changes_fingerprint(tmp_path):
    path = tmp_path / "tool.yaml"
    path.write_text(yaml.safe_dump(tool_document()), encoding="utf-8")
    first = read_tool_payload_identity(path)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = read_tool_payload_identity(path)
    assert first.sha256 == second.sha256
    assert first.fingerprint != second.fingerprint


def test_physical_identity_is_optional_traceability_metadata(tmp_path):
    document = tool_document()
    document["arms"]["left"]["tool"]["serial"] = ""
    document["arms"]["left"]["tool"]["mounting_revision"] = "UNVERIFIED"
    path = tmp_path / "tool.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    identity = read_tool_payload_identity(path)
    assert identity.sha256


def test_hardware_write_guard_requires_exact_file_content_and_mode(tmp_path):
    permit = tmp_path / "permit"
    permit.write_text("", encoding="utf-8")
    permit.chmod(0o600)
    guard = HardwareWriteGuard(
        cli_enabled=True,
        environment={
            HardwareWriteGuard.ENVIRONMENT_KEY:
            HardwareWriteGuard.ENVIRONMENT_VALUE
        },
        permit_file=permit,
    )
    with pytest.raises(HardwareWriteRejected, match="content"):
        guard.require("test", local_console=True)
    permit.write_text(HardwareWriteGuard.PERMIT_FILE_VALUE + "\n", encoding="utf-8")
    guard.require("test", local_console=True)
    permit.chmod(0o644)
    with pytest.raises(HardwareWriteRejected, match="0600"):
        guard.require("test", local_console=True)


class _Info:
    serial_number = ""
    model_name = "Rizon4s"
    software_version = "v3.11"
    has_ft_sensor = True
    license_type = "RDK-Professional+TDK-Standard"
    q_min = [-2.0] * 7
    q_max = [2.0] * 7
    dq_max = [1.0] * 7
    K_x_nom = [10000.0, 10000.0, 10000.0, 2500.0, 2500.0, 2500.0]


class _Robot:
    def __init__(self, serial):
        self.serial = serial
        self.calls = []
        self.faulted = False

    def info(self):
        value = _Info()
        value.serial_number = self.serial
        return value

    def SwitchMode(self, mode):
        self.calls.append(("SwitchMode", mode))

    def Stop(self):
        self.calls.append(("Stop",))

    def fault(self):
        return self.faulted

    def ClearFault(self):
        self.calls.append(("ClearFault",))
        self.faulted = False
        return True

    def Enable(self):
        self.calls.append(("Enable",))

    def SetCartesianImpedance(self, stiffness, damping_ratio):
        self.calls.append(
            ("SetCartesianImpedance", stiffness, damping_ratio)
        )

    def SendJointPosition(self, positions, velocities, max_vel, max_acc):
        self.calls.append(
            ("SendJointPosition", positions, velocities, max_vel, max_acc)
        )


class _ToolParams:
    mass = 1.0
    CoM = [0.0, 0.0, 0.1]
    inertia = [0.1] * 6
    tcp_location = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


class _Tool:
    def __init__(self, robot):
        self.robot = robot

    def name(self):
        return "test"

    def params(self):
        return _ToolParams()


def test_connect_runs_read_only_robot_info_compatibility(monkeypatch):
    module = types.SimpleNamespace(
        __version__="1.9.0",
        Robot=_Robot,
        Tool=_Tool,
        Mode=types.SimpleNamespace(
            NRT_CARTESIAN_MOTION_FORCE="cartesian",
            NRT_JOINT_POSITION="joint",
        ),
    )
    monkeypatch.setitem(sys.modules, "flexivrdk", module)
    backend = FlexivRDKBackend(
        (
            RobotSpec("left", "Rizon4s-063326"),
            RobotSpec("right", "Rizon4s-063806"),
        ),
        write_guard=HardwareWriteGuard(test_backend=True),
    )
    backend.connect()
    report = backend.compatibility_metadata
    assert report["left"]["software_version"] == "v3.11"
    assert report["right"]["has_ft_sensor"] is True
    assert report["left"]["active_tool_name"] == "test"
    assert report["right"]["q_min"] == (-2.0,) * 7
    backend.verify_active_tool_payload(
        json.dumps(tool_document(), sort_keys=True, separators=(",", ":"))
    )
    backend.switch_cartesian_mode("left", local_console=True)
    backend.set_cartesian_impedance(
        "left",
        np.asarray([1200.0, 1200.0, 1200.0, 80.0, 80.0, 80.0]),
        np.asarray([0.7] * 6),
        local_authorized=True,
    )
    backend.switch_joint_position_mode("left", local_console=True)
    backend.send_joint_position(
        "left",
        np.zeros(7),
        max_velocity=0.5,
        max_acceleration=1.0,
        local_authorized=True,
    )
    backend.stop("left", local_console=True)
    calls = backend._robots["left"].calls
    assert ("SwitchMode", "cartesian") in calls
    assert any(call[0] == "SetCartesianImpedance" for call in calls)
    assert ("SwitchMode", "joint") in calls
    assert ("Stop",) in calls
    joint_call = next(call for call in calls if call[0] == "SendJointPosition")
    assert joint_call[2] == [0.0] * 7
    assert joint_call[3] == [0.5] * 7
    assert joint_call[4] == [1.0] * 7


def test_reset_clear_fault_and_enable_use_rdk_recovery_order(monkeypatch):
    module = types.SimpleNamespace(
        __version__="1.9.0",
        Robot=_Robot,
        Tool=_Tool,
    )
    monkeypatch.setitem(sys.modules, "flexivrdk", module)
    backend = FlexivRDKBackend(
        (RobotSpec("left", "left"), RobotSpec("right", "right")),
        write_guard=HardwareWriteGuard(test_backend=True),
    )
    backend.connect()
    robot = backend._robots["right"]
    robot.faulted = True

    assert backend.clear_fault("right", local_console=True) is True
    backend.enable("right", local_console=True)

    assert robot.calls[-2:] == [("ClearFault",), ("Enable",)]


def test_connect_rejects_configured_serial_mismatch(monkeypatch):
    class WrongRobot(_Robot):
        def info(self):
            value = super().info()
            value.serial_number = "wrong"
            return value

    monkeypatch.setitem(
        sys.modules,
        "flexivrdk",
        types.SimpleNamespace(__version__="1.9.0", Robot=WrongRobot, Tool=_Tool),
    )
    backend = FlexivRDKBackend(
        (RobotSpec("left", "left"), RobotSpec("right", "right")),
        write_guard=HardwareWriteGuard(test_backend=True),
    )
    with pytest.raises(RuntimeError, match="serial mismatch"):
        backend.connect()


def test_connect_rejects_active_tool_mismatch(monkeypatch):
    module = types.SimpleNamespace(__version__="1.9.0", Robot=_Robot, Tool=_Tool)
    monkeypatch.setitem(sys.modules, "flexivrdk", module)
    backend = FlexivRDKBackend(
        (
            RobotSpec("left", "left", expected_active_tool="different-tool"),
            RobotSpec("right", "right", expected_active_tool="different-tool"),
        ),
        write_guard=HardwareWriteGuard(test_backend=True),
    )
    with pytest.raises(RuntimeError, match="active tool mismatch"):
        backend.connect()


def test_local_payload_must_match_live_controller(monkeypatch):
    module = types.SimpleNamespace(__version__="1.9.0", Robot=_Robot, Tool=_Tool)
    monkeypatch.setitem(sys.modules, "flexivrdk", module)
    backend = FlexivRDKBackend(
        (
            RobotSpec("left", "left", expected_active_tool="test"),
            RobotSpec("right", "right", expected_active_tool="test"),
        ),
        write_guard=HardwareWriteGuard(test_backend=True),
    )
    backend.connect()
    document = tool_document()
    document["arms"]["right"]["payload"]["mass_kg"] = 2.0
    with pytest.raises(RuntimeError, match="right active tool mass_kg differs"):
        backend.verify_active_tool_payload(
            json.dumps(document, sort_keys=True, separators=(",", ":"))
        )
