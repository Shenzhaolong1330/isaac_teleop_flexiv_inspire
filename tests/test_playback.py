from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from flexiv_inspire_isaac.data_pipeline.playback import (
    PlaybackConfigError,
    RecordedCommand,
    extract_replay_commands,
    load_playback_config,
    read_deviceio_records,
    resolve_episode,
    validate_replay_home_origin,
    validate_replay_timing,
)
from flexiv_inspire_isaac.data_pipeline.recorder import McapJsonSink, RecordEnvelope
from flexiv_inspire_isaac.replay import _fill_command_message
from isaac_teleop_core.command import ROTATION_ORDER


def _config(tmp_path: Path, *, episode: str = "latest", replay_speed: float = 1.0) -> Path:
    path = tmp_path / "playback.yaml"
    path.write_text(
        f"""
schema_version: 1
site_config: site.yaml
dataset:
  root: episodes
  episode: {episode}
  require_completed: true
visualize:
  sink: save
  save_path: output.rrd
  viewer_port: 9876
  speed: 2.0
  realtime: false
replay:
  enabled: false
  operator_confirmation: ""
  speed: {replay_speed}
  ttl_s: 0.2
  max_inter_command_gap_s: 0.1
  max_schedule_lateness_s: 0.02
  start_delay_s: 2.0
  home_before_start: true
  include_hands: true
""",
        encoding="utf-8",
    )
    (tmp_path / "site.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    return path


def _episode(
    tmp_path: Path,
    name: str,
    *,
    index: int,
    attempt: int = 1,
    completed: bool = True,
    pause_count: int = 1,
) -> Path:
    directory = tmp_path / "episodes" / name
    directory.mkdir(parents=True)
    (directory / "deviceio.mcap").touch()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "episode_uuid": name,
                "episode_index": index,
                "attempt": attempt,
                "completed": completed,
                "completion_reason": "complete" if completed else "aborted",
                "pause_count": pause_count,
                "deviceio_mcap": "deviceio.mcap",
            }
        ),
        encoding="utf-8",
    )
    return directory


def _sent_payload(sequence: int) -> dict:
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    return {
        "schema_version": 1,
        "source": "teleop",
        "sequence": sequence,
        "representation": 1,
        "frame_id": "world",
        "rotation_order": ROTATION_ORDER,
        "valid_mask": 15,
        "deadman": True,
        "trajectory": [
            {
                "left_delta_xyz": [0.001, 0.0, 0.0],
                "left_delta_rotation6d": identity,
                "right_delta_xyz": [0.0, 0.001, 0.0],
                "right_delta_rotation6d": identity,
                "left_hand_targets": [0.0] * 6,
                "right_hand_targets": [0.0] * 6,
            }
        ],
    }


def test_config_and_latest_episode_selection(tmp_path: Path) -> None:
    spec = load_playback_config(_config(tmp_path))
    _episode(tmp_path, "one", index=1)
    selected = _episode(tmp_path, "two", index=2, attempt=2)
    result = resolve_episode(spec)
    assert result.directory == selected
    assert spec.visualize.speed == 2.0
    assert spec.replay.enabled is False


def test_latest_selection_skips_newer_incomplete_attempt(tmp_path: Path) -> None:
    spec = load_playback_config(_config(tmp_path))
    completed = _episode(tmp_path, "complete", index=1)
    _episode(tmp_path, "interrupted", index=2, completed=False)

    result = resolve_episode(spec)

    assert result.directory == completed


def test_hardware_selection_rejects_incomplete_or_mid_episode_home(tmp_path: Path) -> None:
    _episode(tmp_path, "paused", index=1, pause_count=2)
    spec = load_playback_config(_config(tmp_path, episode="paused"))
    with pytest.raises(PlaybackConfigError, match="paused mid-demonstration"):
        resolve_episode(spec, for_hardware=True)

    _episode(tmp_path, "incomplete", index=2, completed=False)
    spec = load_playback_config(_config(tmp_path, episode="incomplete"))
    with pytest.raises(PlaybackConfigError, match="incomplete"):
        resolve_episode(spec, for_hardware=True)


def test_replay_speed_above_recorded_speed_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PlaybackConfigError, match="replay.speed"):
        load_playback_config(_config(tmp_path, replay_speed=1.01))


def test_deviceio_commands_are_validated_and_keep_recorded_timing(tmp_path: Path) -> None:
    path = tmp_path / "commands.mcap"
    sink = McapJsonSink(path)
    try:
        for sequence, timestamp in ((11, 1_000_000_000), (12, 1_020_000_000)):
            sink.write(
                RecordEnvelope(
                    topic="/control/sent_command",
                    source_time_ns=timestamp,
                    host_receive_time_ns=timestamp,
                    mapped_host_time_ns=timestamp,
                    sequence=sequence,
                    valid=True,
                    payload=_sent_payload(sequence),
                )
            )
    finally:
        sink.close()

    spec = load_playback_config(_config(tmp_path))
    records = read_deviceio_records(path)
    commands = extract_replay_commands(records, spec.replay)
    assert [item.original_sequence for item in commands] == [11, 12]
    assert commands[1].timestamp_ns - commands[0].timestamp_ns == 20_000_000
    assert len(commands[0].action) == 30


def test_command_gap_and_unmapped_timing_are_fail_closed(tmp_path: Path) -> None:
    spec = load_playback_config(_config(tmp_path))
    from flexiv_inspire_isaac.data_pipeline.playback import DeviceIORecord

    first = DeviceIORecord(
        "/control/sent_command", 1, True, True, 1, _sent_payload(1), {}
    )
    second = DeviceIORecord(
        "/control/sent_command", 200_000_002, True, True, 2, _sent_payload(2), {}
    )
    with pytest.raises(PlaybackConfigError, match="gap"):
        extract_replay_commands([first, second], spec.replay)
    with pytest.raises(PlaybackConfigError, match="invalid or unmapped"):
        extract_replay_commands(
            [DeviceIORecord(first.topic, 1, True, False, 1, first.payload, {})],
            spec.replay,
        )


def test_slow_replay_must_keep_each_command_inside_ttl(tmp_path: Path) -> None:
    spec = load_playback_config(_config(tmp_path))
    commands = [
        RecordedCommand(0, 1, 3, (0.0,) * 30),
        RecordedCommand(190_000_000, 2, 3, (0.0,) * 30),
    ]
    with pytest.raises(PlaybackConfigError, match="commands stale"):
        validate_replay_timing(commands, spec.replay)


def test_hardware_replay_requires_recorded_home_origin(tmp_path: Path) -> None:
    from flexiv_inspire_isaac.data_pipeline.playback import DeviceIORecord

    command = RecordedCommand(100, 1, 3, (0.0,) * 30)
    records = [
        DeviceIORecord(
            f"/robot/{side}_arm/state",
            90,
            True,
            True,
            index,
            {"q": q},
            {},
        )
        for index, (side, q) in enumerate(
            (("left", [0.0] * 7), ("right", [0.0] * 7)), start=1
        )
    ]
    validate_replay_home_origin(
        records,
        [command],
        left_home=[0.0] * 7,
        right_home=[0.0] * 7,
        tolerance_rad=0.01,
    )
    records[1] = DeviceIORecord(
        "/robot/right_arm/state", 90, True, True, 2, {"q": [0.1] * 7}, {}
    )
    with pytest.raises(PlaybackConfigError, match="right start is not"):
        validate_replay_home_origin(
            records,
            [command],
            left_home=[0.0] * 7,
            right_home=[0.0] * 7,
            tolerance_rad=0.01,
        )


def test_replay_message_is_freshly_rebuilt_for_current_session() -> None:
    def duration() -> SimpleNamespace:
        return SimpleNamespace(sec=0, nanosec=0)

    message = SimpleNamespace(
        header=SimpleNamespace(frame_id=""),
        ttl=duration(),
    )
    point = SimpleNamespace(execute_after=duration())
    action = (
        0.001,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.001,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        *([100.0] * 12),
    )
    recorded = RecordedCommand(10, 99, 15, action)
    _fill_command_message(
        message,
        point,
        recorded,
        session_id="current-session",
        sequence=1234,
        ttl_s=0.2,
        episode_uuid="episode-id",
    )
    assert message.session_id == "current-session"
    assert message.source == "replay"
    assert message.sequence == 1234
    assert message.ttl.nanosec == 200_000_000
    assert message.trajectory == [point]
    assert point.left_hand_targets == [100.0] * 6
    assert message.metadata_values == ["episode-id", "99"]
