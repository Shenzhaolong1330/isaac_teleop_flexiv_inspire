import json
import threading
from types import SimpleNamespace

from flexiv_inspire_isaac.episode_control import EpisodeController
from flexiv_inspire_isaac.pedal_router import PedalRouter


class _FakeController:
    _sequence = 0

    def __init__(self, manus_calibration: str) -> None:
        self.values = {
            "sessions_root": "/tmp/sessions",
            "session_id": "test-session",
            "tool_config": "/tmp/tool.yaml",
            "ft_zero_record": "/tmp/ft-zero.jsonl",
            "camera_config": "/tmp/camera.yaml",
            "manus_calibration": manus_calibration,
            "camera_recording_mode": "jpeg",
            "deviceio_socket": "/tmp/deviceio.sock",
        }

    def _required(self, name: str) -> str:
        value = self.values[name]
        if not value:
            raise RuntimeError(name)
        return value

    def get_parameter(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(value=self.values[name])


def test_arm_only_recording_does_not_require_manus_calibration() -> None:
    fake = _FakeController("")
    command = EpisodeController._command(fake)
    assert not any(item.startswith("manus=") for item in command)


def test_manus_calibration_is_recorded_when_configured() -> None:
    fake = _FakeController("/tmp/manus.yaml")
    command = EpisodeController._command(fake)
    assert "manus=/tmp/manus.yaml" in command


class _Publisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


class _EpisodeWorkflow:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pending_home_request_id = None
        self._pending_home_deadline_ns = 0
        self._home_request = _Publisher()
        self.stops = []
        self.starts = 0
        self.statuses = []

    def get_parameter(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(value={
            "home_result_timeout_s": 30.0,
            "session_id": "test-session",
        }[name])

    def _required(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _publish(self, value: str) -> None:
        self.statuses.append(value)

    def _stop(self, *, rerecord: bool) -> None:
        self.stops.append(rerecord)
        self._publish("STOPPED")

    def _start(self) -> None:
        self.starts += 1
        self._publish("RECORDING")


def _control(value: str) -> SimpleNamespace:
    return SimpleNamespace(data=value)


def test_right_pedal_is_stop_only_and_cancels_pending_restart() -> None:
    fake = _EpisodeWorkflow()
    fake._pending_home_request_id = "pending"

    EpisodeController._on_control(fake, _control("stop"))

    assert fake.stops == [False]
    assert fake.starts == 0
    assert fake._pending_home_request_id is None


def test_left_pedal_discards_then_waits_for_matching_home() -> None:
    fake = _EpisodeWorkflow()
    fake._request_home_then_start = lambda: EpisodeController._request_home_then_start(fake)

    EpisodeController._on_control(fake, _control("rerecord"))

    assert fake.stops == [True]
    assert fake.starts == 0
    request = json.loads(fake._home_request.messages[-1].data)
    assert request["source"] == "episode_rerecord"
    assert request["session_id"] == "test-session"
    assert fake.statuses[-1] == "WAITING_FOR_HOME"

    EpisodeController._on_home_status(
        fake,
        _control(json.dumps({
            "request_id": "some-other-request",
            "state": "complete",
        })),
    )
    assert fake.starts == 0

    EpisodeController._on_home_status(
        fake,
        _control(json.dumps({
            "request_id": request["request_id"],
            "state": "complete",
            "reason": "",
        })),
    )
    assert fake.starts == 1
    assert fake._pending_home_request_id is None


def test_home_failure_never_starts_new_episode() -> None:
    fake = _EpisodeWorkflow()
    fake._pending_home_request_id = "request"
    fake._pending_home_deadline_ns = 1

    EpisodeController._on_home_status(
        fake,
        _control(json.dumps({
            "request_id": "request",
            "state": "rejected",
            "reason": "collision_not_clear",
        })),
    )

    assert fake.starts == 0
    assert fake.statuses[-1] == "FAULT:home:collision_not_clear"


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
