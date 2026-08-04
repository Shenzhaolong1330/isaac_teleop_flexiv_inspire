from __future__ import annotations

from pathlib import Path
import json
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
    assert "--skip-hand-preview" in argv
    assert argv[argv.index("--max-hand-delta") + 1] == "150"
    assert "--skip-preview-if-ft-zeroed" in argv
    assert "--home-after-zero" in argv
    assert "--cycle-hands-after-home" in argv
    assert "--clear-home-hold-latched" in argv


def test_reset_recovers_fault_by_restarting_bridge_then_retrying(tmp_path, monkeypatch):
    tool_config = tmp_path / "tool_payload.yaml"
    tool_config.write_text("schema_version: 1\n", encoding="utf-8")
    calls = 0
    module = ModuleType("flexiv_inspire_control.zero_ft_local")

    def fake_main(_argv: list[str]) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError(
                "Reset requires MAINTENANCE, or READY with valid zero; got FAULT"
            )
        return 0

    module.main = fake_main
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(cli.sys, "stdin", _InteractiveInput())
    monkeypatch.setattr(
        cli, "_ensure_rdk_daemon", lambda config: tmp_path / "rdk.sock"
    )
    service_starts = []
    monkeypatch.setattr(
        cli, "_ensure_reset_ros_services", lambda config: service_starts.append(config)
    )
    restarts = []
    monkeypatch.setattr(
        cli, "_restart_reset_ros_processes", lambda: restarts.append(True)
    )
    config = _Config(tmp_path, tool_config)

    result = cli._run_reset(
        config,
        SimpleNamespace(preview_seconds=0.0),
    )

    assert result == 0
    assert calls == 2
    assert restarts == [True]
    assert service_starts == [config, config]


def test_reset_restarts_stuck_rdk_stack_then_retries_once(tmp_path, monkeypatch):
    tool_config = tmp_path / "tool_payload.yaml"
    tool_config.write_text("schema_version: 1\n", encoding="utf-8")
    calls = 0
    module = ModuleType("flexiv_inspire_control.zero_ft_local")

    def fake_main(_argv: list[str]) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("timed out")
        return 0

    module.main = fake_main
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(cli.sys, "stdin", _InteractiveInput())
    daemon_starts = []
    monkeypatch.setattr(
        cli,
        "_ensure_rdk_daemon",
        lambda config: daemon_starts.append(config) or tmp_path / "rdk.sock",
    )
    service_starts = []
    monkeypatch.setattr(
        cli, "_ensure_reset_ros_services", lambda config: service_starts.append(config)
    )
    stops = []
    monkeypatch.setattr(
        cli,
        "_stop_managed_services",
        lambda config, require_existing: stops.append((config, require_existing)) or 3,
    )
    config = _Config(tmp_path, tool_config)

    result = cli._run_reset(
        config,
        SimpleNamespace(preview_seconds=0.0),
    )

    assert result == 0
    assert calls == 2
    assert daemon_starts == [config, config]
    assert service_starts == [config, config]
    assert stops == [(config, False)]


def test_reset_recovers_replay_deadman_latch_by_restarting_stack(tmp_path, monkeypatch):
    tool_config = tmp_path / "tool_payload.yaml"
    tool_config.write_text("schema_version: 1\n", encoding="utf-8")
    calls = 0
    module = ModuleType("flexiv_inspire_control.zero_ft_local")

    def fake_main(_argv: list[str]) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("source deadman must be released before rearming")
        return 0

    module.main = fake_main
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(cli.sys, "stdin", _InteractiveInput())
    monkeypatch.setattr(cli, "_ensure_rdk_daemon", lambda config: tmp_path / "rdk.sock")
    monkeypatch.setattr(cli, "_ensure_reset_ros_services", lambda config: None)
    restarts: list[bool] = []
    monkeypatch.setattr(
        cli,
        "_stop_managed_services",
        lambda config, require_existing: restarts.append(require_existing) or 0,
    )

    assert cli._run_reset(_Config(tmp_path, tool_config), SimpleNamespace(preview_seconds=0)) == 0
    assert calls == 2
    assert restarts == [False]


def test_reset_parser_needs_no_confirmation_argument():
    args = cli._parser().parse_args(["reset"])

    assert args.operation == "reset"
    assert args.preview_seconds == 2.0


def test_replay_automatically_prepares_rdk_and_ros_stack(tmp_path, monkeypatch):
    prepared: list[object] = []
    monkeypatch.setattr(
        cli,
        "_run_reset",
        lambda config, args: prepared.append((config, args.preview_seconds)) or 0,
    )
    monkeypatch.setattr(cli, "_start_replay_pedal_router", lambda config: [])
    monkeypatch.setattr(cli, "_publish_replay_deadman_release", lambda: None)
    replay_module = ModuleType("flexiv_inspire_isaac.replay")
    replay_module.main = lambda argv: 23
    monkeypatch.setitem(sys.modules, replay_module.__name__, replay_module)

    tool_config = tmp_path / "tool_payload.yaml"
    tool_config.write_text("schema_version: 1\n", encoding="utf-8")
    config = _Config(tmp_path, tool_config)
    result = cli._main(SimpleNamespace(operation="replay"), config)

    assert result == 23
    assert len(prepared) == 2
    assert all(item[0] is config and item[1] == 0.0 for item in prepared)


def test_manus_startup_reports_valid_bimanual_retargeting(tmp_path):
    log = tmp_path / "teleop.log"
    log.write_text(
        "[INFO] MANUS_HANDS_READY: both gloves are valid\n",
        encoding="utf-8",
    )
    process = SimpleNamespace(poll=lambda: None, returncode=None)

    assert cli._report_manus_status(
        [[sys.executable, "-m", "flexiv_inspire_control.teleop_input_node"]],
        [process],
        [log],
    ) is True


def test_service_group_start_is_atomic_and_closes_parent_logs(
    tmp_path, monkeypatch
):
    class Process:
        pid = 321

        @staticmethod
        def poll():
            return None

    first = Process()
    calls = 0
    observed_logs = []

    def popen(_command, **kwargs):
        nonlocal calls
        calls += 1
        observed_logs.append(kwargs["stdout"])
        if calls == 2:
            raise OSError("missing executable")
        return first

    stopped = []
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(
        cli, "_stop_started_processes", lambda processes: stopped.append(processes)
    )

    with pytest.raises(OSError, match="missing executable"):
        cli._start([["one"], ["two"]], tmp_path, log_prefix="atomic")

    assert stopped == [[first]]
    assert len(observed_logs) == 2
    assert all(log.closed for log in observed_logs)


def test_manus_not_ready_never_stops_record(tmp_path, monkeypatch, capsys):
    log = tmp_path / "teleop.log"
    log.write_text("teleop input started in COMMAND mode\n", encoding="utf-8")
    process = SimpleNamespace(poll=lambda: None, returncode=None)
    monkeypatch.setattr(
        cli,
        "_stop_started_processes",
        lambda _processes: pytest.fail("MANUS readiness must not stop record"),
    )

    ready = cli._report_manus_status(
        [[sys.executable, "-m", "flexiv_inspire_control.teleop_input_node"]],
        [process],
        [log],
    )

    assert ready is False
    assert "record 继续运行" in capsys.readouterr().out


def test_record_failure_still_stops_all_managed_services(tmp_path, monkeypatch):
    config = SimpleNamespace(
        document={"session": {"runtime_root": str(tmp_path / "runtime")}}
    )
    cleanup_calls = []
    monkeypatch.setattr(cli, "load_system_config", lambda path: config)

    def fail_record(args, selected_config):
        assert selected_config is config
        raise RuntimeError("recorder failed")

    monkeypatch.setattr(cli, "_main", fail_record)
    monkeypatch.setattr(
        cli,
        "_stop_managed_services",
        lambda selected_config, require_existing: cleanup_calls.append(
            (selected_config, require_existing)
        )
        or 7,
    )

    with pytest.raises(RuntimeError, match="recorder failed"):
        cli.main(["record"])

    assert cleanup_calls == [(config, False), (config, False)]


def test_managed_service_cleanup_escalates_after_graceful_timeout(
    tmp_path, monkeypatch
):
    runtime = tmp_path / "runtime"
    launcher = runtime / "launcher"
    launcher.mkdir(parents=True)
    managed_pid = 424242
    (launcher / "processes.json").write_text(
        json.dumps({"pids": [managed_pid]}),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        root=tmp_path,
        document={"session": {"runtime_root": str(runtime)}},
    )
    monkeypatch.setattr(
        cli, "_matching_managed_process_ids", lambda selected: [managed_pid]
    )
    times = iter((0.0, 6.0, 6.0, 7.0))
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(times))
    signals = []
    killed = False

    def fake_killpg(pid, requested_signal):
        nonlocal killed
        signals.append((pid, requested_signal))
        if requested_signal == signal.SIGKILL:
            killed = True

    monkeypatch.setattr(cli.os, "killpg", fake_killpg)
    monkeypatch.setattr(cli, "_pid_is_running", lambda pid: not killed)

    assert cli._stop_managed_services(config, require_existing=False) == 1
    assert signals == [
        (managed_pid, signal.SIGTERM),
        (managed_pid, signal.SIGKILL),
    ]


def test_reset_reuses_valid_ft_zero_when_session_is_already_ready():
    assert _ft_zero_mode("MAINTENANCE", False) == "execute"
    assert _ft_zero_mode("READY", True) == "reuse"
    assert _ft_zero_mode("TELEOP_ARMED", True) == "reuse"
    assert _ft_zero_mode("POLICY_ARMED", True) == "reuse"
    assert _ft_zero_mode("REPLAY_ARMED", True) == "reuse"
    assert _ft_zero_mode("HOLD_LATCHED", True) == "reuse"
    with pytest.raises(RuntimeError, match="valid session F/T zero"):
        _ft_zero_mode("READY", False)
    with pytest.raises(RuntimeError, match="valid session F/T zero"):
        _ft_zero_mode("TELEOP_ARMED", False)
    with pytest.raises(RuntimeError, match="READY/ARMED/HOLD_LATCHED"):
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
    (tmp_path / "launcher" / "reset-services.json").write_text(
        '{"config_sha256": "test-config-sha256", '
        '"runtime_sha256": "test-runtime-sha256"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "_reset_ros_runtime_fingerprint",
        lambda _config: "test-runtime-sha256",
    )
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
        "_restart_reset_ros_processes",
        lambda: pytest.fail("matching Reset services must not be restarted"),
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


def test_record_waits_for_rdk_socket_before_collection_preflight(
    tmp_path, monkeypatch
):
    class Process:
        def poll(self):
            return None

    attempts = iter((False, True))
    monkeypatch.setattr(cli, "_rdk_socket_live", lambda _path: next(attempts))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    cli._wait_for_rdk_socket(
        tmp_path / "rdk.sock",
        [Process()],
        [tmp_path / "rdk.log"],
        timeout_s=1.0,
    )


def test_record_waits_for_all_camera_streams(tmp_path, monkeypatch):
    camera_log = tmp_path / "camera.log"
    camera_log.write_text(
        "\n".join(
            f"camera {name} (serial): streaming"
            for name in ("head", "left_wrist", "right_wrist")
        ),
        encoding="utf-8",
    )

    class Process:
        def poll(self):
            return None

    cli._wait_for_camera_streams(
        [["flexiv-inspire-camera-node"]],
        [Process()],
        [camera_log],
        ("head", "left_wrist", "right_wrist"),
        timeout_s=1.0,
    )


def test_record_waits_for_openxr_and_first_decoded_frame(tmp_path):
    receiver_log = tmp_path / "receiver.log"
    receiver_log.write_text(
        "OpenXR session is ready\nSession Initialization Time: 20 ms\n",
        encoding="utf-8",
    )

    class Process:
        def poll(self):
            return None

    assert cli._wait_for_xr_receiver_ready(
        Process(), receiver_log, expected_streams=1, timeout_s=1.0
    ) is True


def test_record_keeps_running_when_quest_handshake_is_late(
    tmp_path, capsys
):
    receiver_log = tmp_path / "receiver.log"
    receiver_log.write_text(
        "No XR headset connected, retrying in 2s\n",
        encoding="utf-8",
    )

    class Process:
        def poll(self):
            return None

    assert cli._wait_for_xr_receiver_ready(
        Process(), receiver_log, expected_streams=1, timeout_s=0.01
    ) is False
    output = capsys.readouterr().out
    assert "record 继续运行" in output
    assert "自动重连" in output


def test_record_keeps_running_when_quest_receiver_exits(tmp_path, capsys):
    receiver_log = tmp_path / "receiver.log"
    receiver_log.write_text("OpenXR unavailable\n", encoding="utf-8")

    class Process:
        def poll(self):
            return 2

    assert cli._wait_for_xr_receiver_ready(
        Process(), receiver_log, expected_streams=1, timeout_s=1.0
    ) is False
    output = capsys.readouterr().out
    assert "record 继续运行" in output
    assert "自动重试" in output


def test_record_waits_for_encoded_rtp_before_starting_receiver(tmp_path):
    bridge_log = tmp_path / "bridge.log"
    bridge_log.write_text(
        "XR_VIDEO_STREAM_READY: head encoder=h264_nvenc sent=1\n",
        encoding="utf-8",
    )

    class Process:
        returncode = None

        def poll(self):
            return None

    assert cli._wait_for_xr_bridge_streams(
        [["flexiv-inspire-xr-bridge"]],
        [Process()],
        [bridge_log],
        ("head",),
        timeout_s=1.0,
    ) is True


def test_record_keeps_running_when_rtp_bridge_has_no_frame(tmp_path, capsys):
    bridge_log = tmp_path / "bridge.log"
    bridge_log.write_text("encoder starting\n", encoding="utf-8")

    class Process:
        def poll(self):
            return None

    assert cli._wait_for_xr_bridge_streams(
        [["flexiv-inspire-xr-bridge"]],
        [Process()],
        [bridge_log],
        ("head",),
        timeout_s=0.01,
    ) is False
    assert "record 和遥操继续" in capsys.readouterr().out


def test_optional_xr_video_reports_encoded_and_displayed_frame(tmp_path):
    bridge_log = tmp_path / "bridge.log"
    receiver_log = tmp_path / "receiver.log"
    bridge_log.write_text(
        "XR_VIDEO_STREAM_READY: head encoder=h264 sent=1\n",
        encoding="utf-8",
    )
    receiver_log.write_text(
        "OpenXR session is ready\nSession Initialization Time: 20 ms\n",
        encoding="utf-8",
    )

    class Process:
        def poll(self):
            return None

    assert cli._report_optional_xr_video_status(
        [
            ["flexiv-inspire-xr-bridge"],
            ["run_isaac_camera_receiver.sh"],
        ],
        [Process(), Process()],
        [bridge_log, receiver_log],
        ("head",),
        bridge_timeout_s=0.1,
        receiver_timeout_s=0.1,
    ) is True


def test_optional_xr_video_does_not_block_without_display(tmp_path, capsys):
    bridge_log = tmp_path / "bridge.log"
    receiver_log = tmp_path / "receiver.log"
    bridge_log.write_text(
        "XR_VIDEO_STREAM_READY: head encoder=h264 sent=1\n",
        encoding="utf-8",
    )
    receiver_log.write_text("waiting for headset\n", encoding="utf-8")

    class Process:
        def poll(self):
            return None

    assert cli._report_optional_xr_video_status(
        [
            ["flexiv-inspire-xr-bridge"],
            ["run_isaac_camera_receiver.sh"],
        ],
        [Process(), Process()],
        [bridge_log, receiver_log],
        ("head",),
        bridge_timeout_s=0.01,
        receiver_timeout_s=0.01,
    ) is False
    assert "record 和遥操继续" in capsys.readouterr().out


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
        cli,
        "_wait_for_collection_processes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )
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


def test_pedal_router_exit_stops_collection(monkeypatch):
    class Process:
        def __init__(self, statuses):
            self._statuses = iter(statuses)

        def poll(self):
            return next(self._statuses)

    controller = Process((None,))
    pedal = Process((7,))
    monkeypatch.setattr(
        cli.time,
        "sleep",
        lambda _seconds: pytest.fail("pedal exit should be detected immediately"),
    )

    with pytest.raises(RuntimeError, match="脚踏输入进程意外退出.*退出码 7"):
        cli._wait_for_collection_processes(controller, pedal)


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
