import json
import threading
from types import SimpleNamespace

from flexiv_inspire_isaac.episode_control import EpisodeController
from flexiv_inspire_isaac.pedal_router import PedalRouter


class _Publisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


class _Logger:
    def __init__(self) -> None:
        self.warnings = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class _FakeController:
    def __init__(self, manus_calibration: str) -> None:
        self._sequence = 0
        self._episode_index = 3
        self._attempt = 2
        self._current_manifest = None
        self.values = {
            "sessions_root": "/tmp/sessions",
            "dataset_name": "pick_place",
            "task_description": "Pick up the red block.",
            "session_id": "test-session",
            "tool_config": "/tmp/tool.yaml",
            "ft_zero_record": "/tmp/ft-zero.jsonl",
            "camera_config": "/tmp/camera.yaml",
            "manus_calibration": manus_calibration,
            "camera_recording_mode": "jpeg",
            "deviceio_mode": "native",
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


def test_episode_command_contains_configured_identity_prompt_and_attempt() -> None:
    fake = _FakeController("")
    command = EpisodeController._command(fake)

    assert not any(item.startswith("manus=") for item in command)
    assert command[command.index("--dataset-name") + 1] == "pick_place"
    assert command[command.index("--episode-index") + 1] == "3"
    assert command[command.index("--attempt") + 1] == "2"
    assert command[command.index("--task-description") + 1] == (
        "Pick up the red block."
    )
    assert "episode_000003_attempt_02_" in command[
        command.index("--episode-directory-name") + 1
    ]


def test_manus_calibration_is_recorded_when_configured() -> None:
    fake = _FakeController("/tmp/manus.yaml")
    command = EpisodeController._command(fake)
    assert "manus=/tmp/manus.yaml" in command


class _EpisodeWorkflow:
    def __init__(self, episode_count: int = 2) -> None:
        self._lock = threading.RLock()
        self._pending_home_request_id = None
        self._pending_home_deadline_ns = 0
        self._pending_home_action = None
        self._awaiting_home_authorization = False
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
        self.authorize_control_calls = 0
        self.statuses = []
        self.faults = []
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

    def _stop(self, *, rerecord: bool) -> bool:
        self.stops.append(rerecord)
        return not rerecord

    def _start(self) -> None:
        self.starts += 1

    def _authorize_home(self) -> None:
        self.authorize_home_calls += 1
        self._awaiting_home_authorization = True

    def _authorize_control(self) -> None:
        self.authorize_control_calls += 1

    def _send_home_request(self) -> None:
        EpisodeController._send_home_request(self)

    def _begin_home(self, action: str) -> None:
        EpisodeController._begin_home(self, action)

    def _fail(self, reason: str) -> None:
        self.faults.append(reason)


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


def test_left_pedal_discards_homes_and_restarts_same_index() -> None:
    fake = _EpisodeWorkflow()

    EpisodeController._on_control(fake, _control("rerecord"))
    _authorize_and_complete_home(fake)

    assert fake.pauses == 1
    assert fake.stops == [True]
    assert fake._episode_index == 1
    assert fake._attempt == 2
    assert fake.starts == 1


def test_quest_a_pauses_homes_and_resumes_same_episode() -> None:
    fake = _EpisodeWorkflow()

    EpisodeController._on_control(fake, _control("home"))
    _authorize_and_complete_home(fake)

    assert fake.pauses == 1
    assert fake.resumes == 1
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

    assert fake.stops == []
    assert fake.starts == 0
    assert fake.faults == ["home:collision_not_clear"]


def test_pedal_router_maps_right_press_to_stop() -> None:
    publisher = _Publisher()
    fake = SimpleNamespace(
        _publisher=publisher,
        get_parameter=lambda name: SimpleNamespace(value={
            "rerecord_key_code": 105,
            "record_toggle_key_code": 106,
        }[name]),
    )

    PedalRouter._on_event(fake, SimpleNamespace(pressed=True, key_code=106))

    assert publisher.messages[-1].data == "stop"
