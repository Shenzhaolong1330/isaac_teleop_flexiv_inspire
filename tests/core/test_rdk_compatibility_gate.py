from __future__ import annotations

import sys
import types

import pytest

from flexiv_rdk_daemon.backend import FlexivRDKBackend, RobotSpec
from flexiv_rdk_daemon.guard import HardwareWriteGuard


class Info:
    serial_number = ""
    model_name = "Rizon4s"
    software_version = "v3.11.2"
    has_ft_sensor = True
    license_type = "RDK-Professional+TDK-Standard"
    q_min = [-2.0] * 7
    q_max = [2.0] * 7
    dq_max = [1.0] * 7


class Robot:
    software = "v3.11.2"
    licenses = "RDK-Professional+TDK-Standard"

    def __init__(self, serial):
        self.serial = serial

    def info(self):
        value = Info()
        value.serial_number = self.serial
        value.software_version = self.software
        value.license_type = self.licenses
        return value


class ToolParams:
    mass = 1.0
    CoM = [0.0, 0.0, 0.1]
    inertia = [0.1] * 6
    tcp_location = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


class Tool:
    def __init__(self, robot):
        self.robot = robot

    def name(self):
        return "test"

    def params(self):
        return ToolParams()


def backend() -> FlexivRDKBackend:
    return FlexivRDKBackend(
        (
            RobotSpec("left", "left"),
            RobotSpec("right", "right"),
        ),
        write_guard=HardwareWriteGuard(test_backend=True),
    )


def test_controller_software_prefix_is_fail_closed(monkeypatch) -> None:
    class WrongSoftware(Robot):
        software = "v3.12.0"

    monkeypatch.setitem(
        sys.modules,
        "flexivrdk",
        types.SimpleNamespace(
            __version__="1.9.0", Robot=WrongSoftware, Tool=Tool
        ),
    )
    with pytest.raises(RuntimeError, match="software mismatch"):
        backend().connect()


def test_required_rdk_professional_license_is_fail_closed(monkeypatch) -> None:
    class MissingLicense(Robot):
        licenses = "RDK-Standard"

    monkeypatch.setitem(
        sys.modules,
        "flexivrdk",
        types.SimpleNamespace(
            __version__="1.9.0", Robot=MissingLicense, Tool=Tool
        ),
    )
    with pytest.raises(RuntimeError, match="required license"):
        backend().connect()


def test_rdk_version_must_be_reported_and_locked_to_1_9(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "flexivrdk",
        types.SimpleNamespace(__version__="", Robot=Robot),
    )
    with pytest.raises(RuntimeError, match="RDK 1.9"):
        backend().connect()
