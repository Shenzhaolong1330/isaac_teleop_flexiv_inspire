from __future__ import annotations

from pathlib import Path
import signal
import stat
from types import ModuleType, SimpleNamespace
import sys

import pytest

from flexiv_inspire_isaac import cli
from flexiv_inspire_control.zero_ft_local import (
    _ft_zero_mode,
    _skip_hand_preview,
    _wait_for_ready_before_home,
)


class _InteractiveInput:
    @staticmethod
    def isatty() -> bool:
        return True


class _Config:
    def __init__(self, runtime_root: Path, tool_config: Path) -> None:
        self.document = {
            "session": {"runtime_root": str(runtime_root)},
            "flexiv": {
                "tool_payload_config": str(tool_config),
                "home": {"timeout_s": 23.0},
            },
        }
        self.root = runtime_root.parent
        self.sha256 = "test-config-sha256"

    @staticmethod
    def resolve(raw: str) -> Path:
        return Path(raw)


def test_reset_composes_zero_ft_then_home_without_operator_tokens(
    tmp_path, monkeypatch
):
    tool_config = tmp_path / "tool_payload.yaml"
    tool_config.write_text("schema_version: 1\n", encoding="utf-8")
    calls: list[list[str]] = []
    module = ModuleType("flexiv_inspire_control.zero_ft_local")

    def fake_main(argv: list[str]) -> int:
        calls.append(argv)
        return 17

    module.main = fake_main
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(cli.sys, "stdin", _InteractiveInput())
    monkeypatch.setattr(
        cli, "_ensure_rdk_daemon", lambda config: tmp_path / "rdk.sock"
    )
    monkeypatch.setattr(cli, "_ensure_reset_ros_services", lambda config: None)

    result = cli._run_reset(
        _Config(tmp_path, tool_config),
        SimpleNamespace(preview_seconds=1.5),
    )

    assert result == 17
    assert len(calls) == 1
    argv = calls[0]
    assert argv[argv.index("--rdk-socket") + 1] == str(tmp_path / "rdk.sock")
    assert argv[argv.index("--tool-payload-config") + 1] == str(tool_config)
    assert "--confirm-ft-unloaded" in argv
    assert "--skip-preview-if-ft-zeroed" in argv
    assert "--home-after-zero" in argv
    assert "--cycle-hands-after-home" in argv
    assert "--clear-home-hold-latched" in argv


def test_reset_parser_needs_no_confirmation_argument():
    args = cli._parser().parse_args(["reset"])

    assert args.operation == "reset"
    assert args.preview_seconds == 2.0


def test_reset_reuses_valid_ft_zero_when_session_is_already_ready():
    assert _ft_zero_mode("MAINTENANCE", False) == "execute"
    assert _ft_zero_mode("READY", True) == "reuse"
    assert _ft_zero_mode("TELEOP_ARMED", True) == "reuse"
    assert _ft_zero_mode("POLICY_ARMED", True) == "reuse"
    assert _ft_zero_mode("REPLAY_ARMED", True) == "reuse"
    with pytest.raises(RuntimeError, match="valid session F/T zero"):
        _ft_zero_mode("READY", False)
    with pytest.raises(RuntimeError, match="valid session F/T zero"):
        _ft_zero_mode("TELEOP_ARMED", False)
    with pytest.raises(RuntimeError, match="READY/ARMED"):
        _ft_zero_mode("ACTIVE", True)


def test_reused_armed_zero_does_not_wait_for_ready_before_home_request():
    assert _wait_for_ready_before_home("execute") is True
    assert _wait_for_ready_before_home("reuse") is False
    with pytest.raises(ValueError, match="unsupported"):
        _wait_for_ready_before_home("unavailable")


def test_hand_preview_is_skipped_only_for_confirmed_ft_zero_reuse():
    assert _skip_hand_preview(
        "reuse", requested=True, execution_confirmed=True
    )
    assert not _skip_hand_preview(
        "execute", requested=True, execution_confirmed=True
    )
    assert not _skip_hand_preview(
        "reuse", requested=False, execution_confirmed=True
    )
    assert not _skip_hand_preview(
        "reuse", requested=True, execution_confirmed=False
    )
    with pytest.raises(ValueError, match="unsupported"):
        _skip_hand_preview(
            "unavailable", requested=True, execution_confirmed=True
        )


def test_reset_reuses_running_ros_services_without_restart_or_fixed_wait(
    tmp_path, monkeypatch
):
    tool_config = tmp_path / "tool_payload.yaml"
    tool_config.write_text("schema_version: 1\n", encoding="utf-8")
    config = _Config(tmp_path, tool_config)
    (tmp_path / "launcher").mkdir()
    running = iter((True, True))
    monkeypatch.setattr(cli, "_process_running", lambda *markers: next(running))
    monkeypatch.setattr(
        cli,
        "render_runtime_configs",
        lambda config, target: {"dftp.yaml": tmp_path / "dftp.yaml"},
    )
    monkeypatch.setattr(
        cli,
        "_restart_dftp_processes",
        lambda: pytest.fail("a running DFTP process must not be restarted"),
    )
    monkeypatch.setattr(
        cli,
        "_start",
        lambda commands, directory, log_prefix: (
            pytest.fail("no process should be started") if commands else []
        ),
    )
    monkeypatch.setattr(
        cli,
        "_verify_process_startup",
        lambda *args, **kwargs: pytest.fail(
            "reused services must not incur a startup wait"
        ),
    )

    cli._ensure_reset_ros_services(config)

    state = (tmp_path / "launcher" / "reset-services.json").read_text(
        encoding="utf-8"
    )
    assert '"control_bridge": true' in state
    assert '"dftp": true' in state


def test_launcher_creates_and_reuses_internal_hardware_permit(tmp_path):
    permit = tmp_path / "runtime" / "session.write-permit"

    cli._ensure_write_permit(permit)
    cli._ensure_write_permit(permit)

    assert permit.read_text(encoding="utf-8").strip() == (
        "FLEXIV-RDK-WRITES-ENABLED"
    )
    assert stat.S_IMODE(permit.stat().st_mode) == 0o600


def test_launcher_does_not_chmod_existing_permit_parent(tmp_path):
    parent = tmp_path / "shared-runtime"
    parent.mkdir(mode=0o755)
    before = stat.S_IMODE(parent.stat().st_mode)

    cli._ensure_write_permit(parent / "session.write-permit")

    assert stat.S_IMODE(parent.stat().st_mode) == before


def test_reset_restarts_existing_dftp_process(monkeypatch):
    process_sets = iter(([101, 202], []))
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(cli, "_matching_process_ids", lambda *markers: next(process_sets))
    monkeypatch.setattr(cli.os, "kill", lambda pid, selected: signals.append((pid, selected)))
    monkeypatch.setattr(cli.Path, "exists", lambda path: False)

    cli._restart_dftp_processes()

    assert signals == [(101, signal.SIGTERM), (202, signal.SIGTERM)]


def test_changed_rdk_config_restarts_managed_daemon(tmp_path, monkeypatch):
    socket_path = tmp_path / "rdk.sock"
    state_path = tmp_path / "rdk-daemon.json"
    state_path.write_text(
        '{"pid": 303, "config_sha256": "old"}\n', encoding="utf-8"
    )
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        cli.Path,
        "read_bytes",
        lambda path: f"flexiv-rdk-daemon --socket {socket_path}".encode(),
    )
    monkeypatch.setattr(cli.os, "killpg", lambda pid, selected: signals.append((pid, selected)))
    monkeypatch.setattr(cli, "_rdk_socket_live", lambda path: False)

    restarted = cli._restart_managed_rdk_if_config_changed(
        state_path,
        rdk_socket=socket_path,
        config_sha256="new",
    )

    assert restarted is True
    assert signals == [(303, signal.SIGTERM)]


def test_teleop_command_mode_follows_site_config(tmp_path):
    config = SimpleNamespace(
        document={
            "session": {"id": "test-session"},
            "teleop": {"control_enabled": True},
        }
    )

    command = cli._teleop_input_command(
        config, {"teleop.yaml": tmp_path / "teleop.yaml"}
    )

    assert "command_enabled:=true" in command


def test_record_keeps_collection_processes_out_of_background_launcher():
    assert cli._is_collection_process_command(
        [sys.executable, "-m", "flexiv_inspire_isaac.episode_control"]
    )
    assert cli._is_collection_process_command(
        ["flexiv-inspire-pedal-router", "--ros-args"]
    )
    assert not cli._is_collection_process_command(
        ["flexiv-inspire-camera-node", "--ros-args"]
    )


def test_foreground_collection_ctrl_c_finalizes_controller_and_stops_pedal(
    tmp_path, monkeypatch
):
    class CollectionConfig:
        path = tmp_path / "site.yaml"
        sha256 = "test"
        document = {
            "session": {"runtime_root": str(tmp_path)},
            "recording": {
                "episode_count": 2,
                "dataset_name": "test_dataset",
                "output_root": "sessions",
                "task_description": "test task",
            },
        }

        @staticmethod
        def resolve(raw: str) -> Path:
            return tmp_path / raw

    class FakeProcess:
        def __init__(self, name: str, pid: int) -> None:
            self.name = name
            self.pid = pid
            self.alive = True
            self.signals = []
            self.terminated = False
            self._foreground_wait = True

        def poll(self):
            return None if self.alive else 0

        def wait(self, timeout=None):
            if self.name == "controller" and timeout is None and self._foreground_wait:
                self._foreground_wait = False
                raise KeyboardInterrupt
            self.alive = False
            return 0

        def send_signal(self, selected):
            self.signals.append(selected)

        def terminate(self):
            self.terminated = True
            self.alive = False

        def kill(self):
            self.alive = False

    pedal = FakeProcess("pedal", 501)
    controller = FakeProcess("controller", 502)
    pending = iter((pedal, controller))
    written = []
    removed = []
    monkeypatch.setattr(cli, "_collection_preflight", lambda *_args: None)
    monkeypatch.setattr(cli, "_episode_command", lambda *_args: ["controller"])
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *_args, **_kwargs: next(pending))
    monkeypatch.setattr(
        cli, "_write_state", lambda _directory, _config, processes: written.append(processes)
    )
    monkeypatch.setattr(
        cli, "_remove_state_pids", lambda _directory, pids: removed.append(pids)
    )

    result = cli._run_collection(
        CollectionConfig(), {"pedal.yaml": tmp_path / "pedal.yaml"}
    )

    assert result == 130
    assert controller.signals == [signal.SIGINT]
    assert pedal.terminated is True
    assert written == [[pedal, controller]]
    assert removed == [{501, 502}]


def test_orphaned_managed_process_discovery_is_scoped_to_project(
    tmp_path, monkeypatch
):
    config = _Config(tmp_path / "runtime", tmp_path / "tool.yaml")
    project_process = 401
    other_process = 402
    observed_markers = []

    def matching(*markers):
        observed_markers.extend(markers)
        return [project_process, other_process, cli.os.getpid()]

    monkeypatch.setattr(cli, "_matching_process_ids", matching)

    def command_line(path: Path) -> bytes:
        pid = int(path.parts[2])
        if pid == project_process:
            return (
                f"{config.root}/envs/ros-py312/bin/"
                "flexiv-inspire-control-bridge"
            ).encode()
        return b"/other/workspace/flexiv-inspire-control-bridge"

    monkeypatch.setattr(cli.Path, "read_bytes", command_line)

    assert cli._matching_managed_process_ids(config) == [project_process]
    assert "flexiv-inspire-camera-node" in observed_markers
    assert "flexiv-inspire-episode-controller" in observed_markers
    assert "flexiv-inspire-xr-bridge" in observed_markers
