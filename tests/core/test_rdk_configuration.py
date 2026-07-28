from __future__ import annotations

import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import yaml

from flexiv_rdk_daemon.backend import FlexivRDKBackend, RobotSpec
from flexiv_rdk_daemon.configuration import read_tool_payload_identity
from flexiv_rdk_daemon.guard import (
    HardwareWriteGuard,
    HardwareWriteRejected,
)


def tool_document():
    arm = {
        "tool": {"name": "test"},
        "payload": {
            "mass_kg": 1.0,
            "center_of_mass_m": [0.0, 0.0, 0.1],
            "inertia_kg_m2": [0.1] * 6,
        },
        "locally_verified": True,
    }
    return {"schema_version": 1, "arms": {"left": arm, "right": arm}}


def test_tool_payload_sha_is_canonical_but_touch_changes_fingerprint(tmp_path):
    path = tmp_path / "tool.yaml"
    path.write_text(yaml.safe_dump(tool_document()), encoding="utf-8")
    first = read_tool_payload_identity(path)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = read_tool_payload_identity(path)
    assert first.sha256 == second.sha256
    assert first.fingerprint != second.fingerprint


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


class _Robot:
    def __init__(self, serial):
        self.serial = serial

    def info(self):
        value = _Info()
        value.serial_number = self.serial
        return value


def test_connect_runs_read_only_robot_info_compatibility(monkeypatch):
    module = types.SimpleNamespace(__version__="1.9.0", Robot=_Robot)
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


def test_connect_rejects_configured_serial_mismatch(monkeypatch):
    class WrongRobot(_Robot):
        def info(self):
            value = super().info()
            value.serial_number = "wrong"
            return value

    monkeypatch.setitem(
        sys.modules,
        "flexivrdk",
        types.SimpleNamespace(__version__="1.9.0", Robot=WrongRobot),
    )
    backend = FlexivRDKBackend(
        (RobotSpec("left", "left"), RobotSpec("right", "right")),
        write_guard=HardwareWriteGuard(test_backend=True),
    )
    with pytest.raises(RuntimeError, match="serial mismatch"):
        backend.connect()
