from types import SimpleNamespace

from flexiv_inspire_isaac.episode_control import EpisodeController


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
