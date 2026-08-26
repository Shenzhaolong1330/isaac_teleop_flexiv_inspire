import json
import re
import threading
from types import SimpleNamespace
from pathlib import Path

import flexiv_inspire_isaac.episode_control as episode_control
from flexiv_inspire_isaac.episode_control import (
    EpisodeController,
    _next_task_episode_index,
)
from flexiv_inspire_isaac.pedal_router import PedalRouter
from flexiv_inspire_control.foot_pedal import FootPedalMonitor, KEY_DOWN, KEY_SPACE


def test_transient_stream_holds_can_be_rearmed_by_pedal_cycle() -> None:
    assert {
        "command_stale",
        "source_heartbeat_stale",
        "arm_offline",
        "hand_offline",
        "safety_limit",
    } <= episode_control._ROUTINE_CONTROL_HOLD_REASONS


def test_task_episode_index_continues_across_collection_processes(
    tmp_path: Path,
) -> None:
    for name in (
        "pick_place_episode_001_20260806_1000",
        "pick_place_episode_007_20260806_1100",
        "pick_place_episode_007_20260806_1100_01",
        "other_task_episode_099_20260806_1200",
        "episode_000001_20260805_0900",
    ):
        (tmp_path / name).mkdir()

    assert _next_task_episode_index(tmp_path, "pick_place") == 8
    assert _next_task_episode_index(tmp_path, "new_task") == 1


class _Publisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


def test_middle_pedal_accepts_remapped_and_native_key_codes() -> None:
    states: list[bool] = []
    monitor = FootPedalMonitor(
        Path("/dev/input/event-does-not-need-to-exist"),
        states.append,
        enable_key_codes=(KEY_SPACE, KEY_DOWN),
    )

    assert monitor._enable_key_codes == {KEY_SPACE, KEY_DOWN}
    monitor._set_enable(True)
    assert states == [True]


class _Logger:
    def __init__(self) -> None:
        self.warnings = []
        self.infos = []
        self.errors = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def info(self, message: str) -> None:
        self.infos.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)


class _FakeController:
    def __init__(self, manus_calibration: str) -> None:
        self._sequence = 0
        self._episode_index = 3
        self._attempt = 2
        self._current_manifest = None
        self.values = {
            "sessions_root": "/tmp/sessions",
            "dataset_name": "pick_place",
            "task_name": "red_block_pick",
            "task_description": "Pick up the red block.",
            "session_id": "test-session",
            "tool_config": "/tmp/tool.yaml",
            "ft_zero_record": "/tmp/ft-zero.jsonl",
            "camera_config": "/tmp/camera.yaml",
            "camera_head_extrinsics": "",
            "camera_left_wrist_extrinsics": "",
            "camera_right_wrist_extrinsics": "",
            "manus_calibration": manus_calibration,
            "camera_recording_mode": "jpeg",
            "deviceio_mode": "native",
            "deviceio_profile": "training",
            "ros_mcap_enabled": False,
            "record_only_while_pedal_pressed": True,
            "camera_hz": 30.0,
            "action_hz": 30.0,
            "deviceio_socket": "/tmp/deviceio.sock",
            "runtime_dir": "/tmp/runtime",
        }

    def _required(self, name: str) -> str:
        value = self.values[name]
        if not value:
            raise RuntimeError(name)
        return str(value)

    def _state_file(self):
        return EpisodeController._state_file(self)

    def get_parameter(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(value=self.values[name])


def test_episode_command_contains_configured_identity_prompt_and_internal_attempt() -> None:
    fake = _FakeController("")
    command = EpisodeController._command(fake)

    assert not any(item.startswith("manus=") for item in command)
    assert command[command.index("--dataset-name") + 1] == "pick_place"
    assert command[command.index("--storage-subdirectory") + 1] == "raw"
    assert command[command.index("--episode-index") + 1] == "3"
    assert command[command.index("--attempt") + 1] == "2"
    assert command[command.index("--task-name") + 1] == "red_block_pick"
    assert command[command.index("--deviceio-profile") + 1] == "training"
    assert command[command.index("--expected-camera-hz") + 1] == "30.0"
    assert command[command.index("--expected-action-hz") + 1] == "30.0"
    assert "--no-ros-mcap" in command
    assert "--record-only-while-pedal-pressed" in command
    assert command[command.index("--task-description") + 1] == (
        "Pick up the red block."
    )
    episode_name = command[command.index("--episode-directory-name") + 1]
    match = re.fullmatch(
        r"red_block_pick_episode_003_(\d{8}_\d{4})",
        episode_name,
    )
    assert match is not None
    assert command[command.index("--collection-timestamp-local") + 1] == (
        match.group(1)
    )


def test_episode_command_adds_suffix_when_minute_directory_exists(
    tmp_path: Path, monkeypatch
) -> None:
    fake = _FakeController("")
    fake.values["sessions_root"] = str(tmp_path)
    timestamp = "20260802_1349"
    existing = (
        tmp_path
        / "pick_place"
        / "raw"
        / f"red_block_pick_episode_003_{timestamp}"
    )
    existing.mkdir(parents=True)
    monkeypatch.setattr(episode_control, "local_minute_timestamp", lambda: timestamp)

    command = EpisodeController._command(fake)

    assert command[command.index("--episode-directory-name") + 1] == (
        f"red_block_pick_episode_003_{timestamp}_01"
    )
    assert command[command.index("--collection-timestamp-local") + 1] == timestamp


def test_discard_deletes_only_current_episode_directory(tmp_path: Path) -> None:
    fake = _FakeController("")
    fake.values["sessions_root"] = str(tmp_path)
    episode = (
        tmp_path
        / "pick_place"
        / "raw"
        / "red_block_pick_episode_003_20260803_1500"
    )
    episode.mkdir(parents=True)
    (episode / "manifest.json").write_text("{}", encoding="utf-8")
    sibling = (
        tmp_path
        / "pick_place"
        / "raw"
        / "red_block_pick_episode_002_20260803_1459"
    )
    sibling.mkdir()
    fake._current_manifest = episode / "manifest.json"
    logger = _Logger()
    fake.get_logger = lambda: logger

    EpisodeController._delete_discarded_episode(fake)

    assert not episode.exists()
    assert sibling.is_dir()
    assert fake._current_manifest is None
    assert "已删除丢弃的数据" in logger.infos[-1]


def test_manus_calibration_is_recorded_when_configured() -> None:
    fake = _FakeController("/tmp/manus.yaml")
    command = EpisodeController._command(fake)
    assert "manus=/tmp/manus.yaml" in command


def test_camera_extrinsics_are_hashed_into_episode_manifest_inputs() -> None:
    fake = _FakeController("")
    fake.values["camera_head_extrinsics"] = "/tmp/head-extrinsics.yaml"

    command = EpisodeController._command(fake)

    assert "camera_head=/tmp/head-extrinsics.yaml" in command


def test_camera_disconnect_is_prominent_and_recovery_is_reported_once() -> None:
    logger = _Logger()
    fake = SimpleNamespace(
        _camera_status_by_name={},
        get_logger=lambda: logger,
    )
    disconnected = _control(
        json.dumps(
            {
                "camera_name": "right_wrist",
                "serial": "347622074577",
                "connected": False,
                "fault": True,
                "reason": "capture-failed:RuntimeError:No device connected",
            }
        )
    )

    EpisodeController._on_camera_status(fake, "right_wrist", disconnected)
    EpisodeController._on_camera_status(fake, "right_wrist", disconnected)

    assert len(logger.errors) == 1
    assert "数据流中断：右腕相机" in logger.errors[0]
    assert "episode 会自动作废重录" in logger.errors[0]

    EpisodeController._on_camera_status(
        fake,
        "right_wrist",
        _control(
            json.dumps(
                {
                    "camera_name": "right_wrist",
                    "serial": "347622074577",
                    "connected": True,
                    "fault": False,
                    "reason": "streaming",
                }
            )
        ),
    )

    assert len(logger.infos) == 1
    assert "数据流恢复：右腕相机" in logger.infos[0]


class _EpisodeWorkflow:
    def __init__(self, episode_count: int = 2) -> None:
        self._lock = threading.RLock()
        self._pending_home_request_id = None
        self._pending_home_deadline_ns = 0
        self._pending_home_action = None
        self._awaiting_home_authorization = False
        self._home_recovery_attempted = False
        self._control_state_name = "TELEOP_ARMED"
        self._home_request = _Publisher()
        self._episode_index = 1
        self._attempt = 1
        self._completed_episodes = 0
        self._finished = False
        self.values = {
            "home_result_timeout_s": 30.0,
            "session_id": "test-session",
            "episode_count": episode_count,
        }
        self.pauses = 0
        self.resumes = 0
        self.stops = []
        self.starts = 0
        self.authorize_home_calls = 0
        self.authorize_home_clear_flags = []
        self.authorize_control_calls = 0
        self.commit_result = True
        self.statuses = []
        self.faults = []
        self.events = []
        self.logger = _Logger()

    def get_parameter(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(value=self.values[name])

    def get_logger(self):
        return self.logger

    def _required(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _publish(self, value: str) -> None:
        self.statuses.append(value)

    def _publish_progress(self, value: str) -> None:
        self.statuses.append(value)

    def _pause(self) -> None:
        self.pauses += 1

    def _resume(self) -> None:
        self.resumes += 1
        self.events.append("resume")

    def _stop(self, *, rerecord: bool) -> bool:
        self.stops.append(rerecord)
        self.events.append(f"stop:{rerecord}")
        return self.commit_result and not rerecord

    def _start(self) -> None:
        self.starts += 1
        self.events.append("start")

    def _authorize_home(self, *, clear_hold_latched: bool = False) -> None:
        self.authorize_home_calls += 1
        self.authorize_home_clear_flags.append(clear_hold_latched)
        self._awaiting_home_authorization = True

    def _authorize_control(self) -> None:
        self.authorize_control_calls += 1
        self.events.append("authorize")

    def _send_home_request(self) -> None:
        EpisodeController._send_home_request(self)

    def _begin_home(self, action: str) -> None:
        EpisodeController._begin_home(self, action)

    def _fail(self, reason: str) -> None:
        self.faults.append(reason)

    def _recover_from_home_failure(self, reason: str) -> None:
        EpisodeController._recover_from_home_failure(self, reason)



def _control(value: str) -> SimpleNamespace:
    return SimpleNamespace(data=value)


def _authorize_and_complete_home(fake: _EpisodeWorkflow) -> None:
    EpisodeController._on_home_status(
        fake,
        _control(json.dumps({"state": "authorized", "reason": ""})),
    )
    request = json.loads(fake._home_request.messages[-1].data)
    EpisodeController._on_home_status(
        fake,
        _control(json.dumps({
            "request_id": request["request_id"],
            "state": "complete",
            "reason": "",
        })),
    )


def test_right_pedal_pauses_homes_commits_and_starts_next() -> None:
    fake = _EpisodeWorkflow(episode_count=2)

    EpisodeController._on_control(fake, _control("stop"))

    assert fake.pauses == 1
    assert fake.stops == []
    assert fake._pending_home_action == "next"
    _authorize_and_complete_home(fake)
    assert fake.stops == [False]
    assert fake._completed_episodes == 1
    assert fake._episode_index == 2
    assert fake.starts == 1


def test_right_pedal_finishes_at_configured_episode_count() -> None:
    fake = _EpisodeWorkflow(episode_count=1)

    EpisodeController._on_control(fake, _control("stop"))
    _authorize_and_complete_home(fake)

    assert fake.stops == [False]
    assert fake.starts == 0
    assert fake._finished is True
    assert fake.statuses[-1] == "COMPLETE"


def test_right_pedal_empty_action_discards_and_restarts_same_episode() -> None:
    fake = _EpisodeWorkflow(episode_count=1)
    fake.commit_result = False

    EpisodeController._on_control(fake, _control("stop"))
    _authorize_and_complete_home(fake)

    assert fake.stops == [False]
    assert fake._completed_episodes == 0
    assert fake._episode_index == 1
    assert fake._attempt == 2
    assert fake.starts == 1
    assert fake._finished is False
    assert fake.events[-2:] == ["start", "authorize"]
    assert "记录不完整" in fake.logger.warnings[-1]


def test_failed_episode_is_deleted_even_after_recorder_already_exited(
    tmp_path: Path,
) -> None:
    fake = _FakeController("")
    fake.values["sessions_root"] = str(tmp_path)
    fake._lock = threading.RLock()
    fake._process = SimpleNamespace(poll=lambda: 1)
    fake._publish_progress = lambda _state: None
    logger = _Logger()
    fake.get_logger = lambda: logger
    episode = (
        tmp_path
        / "pick_place"
        / "raw"
        / "red_block_pick_episode_003_20260807_1400"
    )
    episode.mkdir(parents=True)
    fake._current_manifest = episode / "manifest.json"
    fake._current_manifest.write_text(
        json.dumps(
            {
                "completed": False,
                "completion_reason": "operator-stop; required streams have no valid samples",
                "streams": {
                    "episode/events": {"samples": 2, "invalid": 0}
                },
            }
        ),
        encoding="utf-8",
    )
    fake._delete_discarded_episode = lambda: EpisodeController._delete_discarded_episode(fake)

    completed = EpisodeController._stop(fake, rerecord=False)

    assert completed is False
    assert not episode.exists()
    assert fake._current_manifest is None
    assert "已自动删除" in logger.warnings[-1]


def test_left_pedal_discards_homes_and_restarts_same_index() -> None:
    fake = _EpisodeWorkflow()

    EpisodeController._on_control(fake, _control("rerecord"))
    _authorize_and_complete_home(fake)

    assert fake.pauses == 1
    assert fake.stops == [True]
    assert fake._episode_index == 1
    assert fake._attempt == 2
    assert fake.starts == 1
    assert fake.events[-3:] == ["stop:True", "start", "authorize"]


def test_episode_input_during_fault_keeps_current_recorder_alive() -> None:
    fake = _EpisodeWorkflow()
    fake._control_state_name = "FAULT"

    EpisodeController._on_control(fake, _control("rerecord"))

    assert fake.pauses == 0
    assert fake.stops == []
    assert fake.starts == 0
    assert fake.faults == []
    assert fake.statuses == ["WAITING_FOR_CONTROL_RECOVERY"]
    assert "recorder remains active" in fake.logger.errors[-1]


def test_quest_a_pauses_homes_and_resumes_same_episode() -> None:
    fake = _EpisodeWorkflow()

    EpisodeController._on_control(fake, _control("home"))
    _authorize_and_complete_home(fake)

    assert fake.pauses == 1
    assert fake.resumes == 1
    assert fake.events[-2:] == ["resume", "authorize"]
    assert fake.stops == []
    assert fake.starts == 0
    assert fake._episode_index == 1


def test_home_failure_never_finalizes_or_starts_episode() -> None:
    fake = _EpisodeWorkflow()
    EpisodeController._on_control(fake, _control("rerecord"))
    EpisodeController._on_home_status(
        fake,
        _control(json.dumps({"state": "authorized", "reason": ""})),
    )
    request = json.loads(fake._home_request.messages[-1].data)

    EpisodeController._on_home_status(
        fake,
        _control(json.dumps({
            "request_id": request["request_id"],
            "state": "rejected",
            "reason": "collision_not_clear",
        })),
    )

    assert fake.stops == [True]
    assert fake.starts == 1
    assert fake._attempt == 2
    assert fake.faults == []
    assert fake._finished is False
    assert fake.statuses[-1] == "RECORDING_HOME_FAILED"


def test_invalid_command_hold_is_cleared_and_home_is_retried() -> None:
    fake = _EpisodeWorkflow()
    EpisodeController._on_control(fake, _control("rerecord"))
    EpisodeController._on_home_status(
        fake,
        _control(json.dumps({"state": "authorized", "reason": ""})),
    )
    first_request = json.loads(fake._home_request.messages[-1].data)

    EpisodeController._on_home_status(
        fake,
        _control(json.dumps({
            "request_id": first_request["request_id"],
            "state": "rejected",
            "reason": "non_routine_hold_preserved:invalid_command",
        })),
    )

    assert fake.authorize_home_clear_flags == [False, True]
    assert fake._pending_home_action == "discard"
    assert fake._finished is False

    _authorize_and_complete_home(fake)
    assert fake.stops == [True]
    assert fake.starts == 1
    assert fake._attempt == 2
    assert fake.faults == []


def test_pedal_router_maps_right_press_to_stop() -> None:
    publisher = _Publisher()
    logger = _Logger()
    fake = SimpleNamespace(
        _publisher=publisher,
        get_logger=lambda: logger,
        get_parameter=lambda name: SimpleNamespace(value={
            "rerecord_key_code": 105,
            "record_toggle_key_code": 106,
        }[name]),
    )

    PedalRouter._on_event(fake, SimpleNamespace(pressed=True, key_code=106))

    assert publisher.messages[-1].data == "stop"
    assert logger.infos == ["右踏板：保存本条并进入下一条"]


def test_middle_pedal_reports_physical_press_until_control_confirms_active() -> None:
    logger = _Logger()
    deadman_states = []
    fake = SimpleNamespace(
        _enabled=False,
        _publish_deadman=lambda: deadman_states.append(fake._enabled),
        get_logger=lambda: logger,
    )

    PedalRouter._on_enable(fake, True)

    assert deadman_states == [True]
    assert logger.infos == [
        "中踏板：已踩下，正在等待 Quest 双腕追踪和控制端确认"
    ]
    assert "机械臂运动已启用" not in logger.infos[-1]


class _ClutchRearm:
    def __init__(self) -> None:
        self._routine_control_rearm_pending = False
        self._routine_control_rearm_pressed_attempted = False
        self._pending_home_action = None
        self._last_announced_control_state = None
        self.calls = []
        self.errors = []
        self.infos = []

    def _authorize_control(self, *, clear_hold_latched: bool = False) -> None:
        self.calls.append(clear_hold_latched)

    def get_logger(self):
        return SimpleNamespace(error=self.errors.append, info=self.infos.append)


def _control_state(
    *, state="HOLD_LATCHED", reason="physical_pedal_released", pedal=False
):
    return SimpleNamespace(
        state_name=state,
        hold_reason=reason,
        physical_pedal=pedal,
        local_permission=True,
        ft_zeroed_for_session=True,
        arms_online=True,
        hands_online=True,
    )


def test_routine_pedal_release_rearms_on_next_press_with_fresh_authorization() -> None:
    fake = _ClutchRearm()

    EpisodeController._on_control_state(fake, _control_state())
    EpisodeController._on_control_state(fake, _control_state())

    assert fake.calls == []
    assert fake._routine_control_rearm_pending is True

    EpisodeController._on_control_state(fake, _control_state(pedal=True))
    EpisodeController._on_control_state(fake, _control_state(pedal=True))

    assert fake.calls == [True]


def test_minor_fault_rearm_is_attempted_once_and_points_to_reset() -> None:
    class FaultingRearm(_ClutchRearm):
        def _authorize_control(self, *, clear_hold_latched: bool = False) -> None:
            self.calls.append(clear_hold_latched)
            raise RuntimeError(
                "right:mode: Robot is not operational: Minor fault occurred"
            )

    fake = FaultingRearm()

    EpisodeController._on_control_state(fake, _control_state(pedal=True))
    EpisodeController._on_control_state(fake, _control_state(pedal=True))

    assert fake.calls == [True]
    assert fake._routine_control_rearm_pending is False
    assert fake._routine_control_rearm_pressed_attempted is True
    assert len(fake.errors) == 1
    assert "robot reset" in fake.errors[0]


def test_expired_control_authorization_rearms_on_next_press() -> None:
    fake = _ClutchRearm()

    EpisodeController._on_control_state(
        fake, _control_state(reason="control_authorization_expired")
    )
    EpisodeController._on_control_state(
        fake, _control_state(reason="control_authorization_expired")
    )

    assert fake.calls == []
    EpisodeController._on_control_state(
        fake,
        _control_state(reason="control_authorization_expired", pedal=True),
    )

    assert fake.calls == [True]


def test_non_routine_hold_is_not_automatically_cleared() -> None:
    fake = _ClutchRearm()

    EpisodeController._on_control_state(
        fake, _control_state(reason="invalid_command")
    )

    assert fake.calls == []


def test_fault_reason_is_printed_for_operator_recovery() -> None:
    fake = _ClutchRearm()

    EpisodeController._on_control_state(
        fake,
        _control_state(state="FAULT", reason="hardware_fault", pedal=False),
    )

    assert fake.calls == []
    assert fake.errors == ["机械臂控制进入 FAULT：hardware_fault"]


def test_fault_is_not_reprinted_on_every_pedal_edge() -> None:
    fake = _ClutchRearm()

    EpisodeController._on_control_state(
        fake,
        _control_state(state="FAULT", reason="hardware_fault", pedal=False),
    )
    EpisodeController._on_control_state(
        fake,
        _control_state(state="FAULT", reason="hardware_fault", pedal=True),
    )
    EpisodeController._on_control_state(
        fake,
        _control_state(state="FAULT", reason="hardware_fault", pedal=False),
    )

    assert fake.errors == ["机械臂控制进入 FAULT：hardware_fault"]


class _StartupGate:
    def __init__(self, *, ready: bool) -> None:
        self._startup_done = False
        self._control_ready_for_recording = ready
        self._control_state_name = "READY" if ready else "MAINTENANCE"
        self._next_ready_wait_log_ns = 0
        self.events = []
        self.statuses = []
        self.logger = _Logger()

    def get_parameter(self, _name: str) -> SimpleNamespace:
        return SimpleNamespace(value=True)

    def get_logger(self):
        return self.logger

    def _publish_progress(self, value: str) -> None:
        self.statuses.append(value)

    def _authorize_control(self) -> None:
        self.events.append("authorize")

    def _start(self) -> None:
        self.events.append("start")

    def _fail(self, reason: str) -> None:
        self.events.append(f"fail:{reason}")


def test_episode_does_not_record_or_authorize_while_in_maintenance() -> None:
    fake = _StartupGate(ready=False)

    EpisodeController._tick(fake)

    assert fake.events == []
    assert fake.statuses == ["WAITING_FOR_READY"]
    assert fake._startup_done is False


def test_episode_starts_deviceio_recorder_then_authorizes_after_ready() -> None:
    fake = _StartupGate(ready=True)

    EpisodeController._tick(fake)

    assert fake.events == ["start", "authorize"]
    assert fake._startup_done is True
