import sys
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from flexiv_inspire_isaac.cli import (
    _PolicyAuthorizationSupervisor,
    _commands,
    _conversion_manifest,
    _conversion_manifests,
    _load_policy_server_config,
    _policy_authorization_command,
    _policy_serve_commands,
    _run_convert,
)
from flexiv_inspire_isaac.system_config import (
    SystemConfigError,
    load_system_config,
    render_runtime_configs,
)


def _example() -> Path:
    return Path(__file__).parent / "system_mock.yaml"


def _example_document() -> dict:
    return load_system_config(_example()).document


def _site() -> Path:
    return Path(__file__).parents[1] / "config" / "site.yaml"


def _policy_server() -> Path:
    return Path(__file__).parents[1] / "config" / "policy_server.yaml"


def test_policy_server_stack_is_hardware_owner_only(tmp_path) -> None:
    config = load_system_config(_site())
    assert config.document["flexiv"]["policy_control"]["require_pedal"] is True
    rendered = render_runtime_configs(config, tmp_path / "rendered")
    settings = _load_policy_server_config(config, _policy_server())
    commands = _policy_serve_commands(config, rendered, settings)
    joined = [" ".join(command) for command in commands]

    assert settings["bind"] == "127.0.0.1"
    assert settings["rates"] == {
        "arm_hz": 200.0,
        "hand_hz": 200.0,
        "tactile_hz": 15.0,
        "camera_hz": 30.0,
        "action_hz": 30.0,
    }
    assert settings["auto_reset_before_serve"] is True
    assert settings["auto_authorize_policy"] is True
    authorization = _policy_authorization_command(
        config,
        Path(config.document["session"]["runtime_root"]) / "rdk.sock",
        clear_hold_latched=False,
    )
    assert authorization[0] == "flexiv-inspire-authorize-control"
    assert authorization[authorization.index("--source") + 1] == "policy"
    assert "--clear-hold-latched" not in authorization
    recovery_authorization = _policy_authorization_command(
        config,
        Path(config.document["session"]["runtime_root"]) / "rdk.sock",
        clear_hold_latched=True,
    )
    assert "--clear-hold-latched" in recovery_authorization
    assert len(commands) == 6
    assert any("flexiv-rdk-daemon" in command for command in joined)
    assert any("flexiv_inspire_control.node" in command for command in joined)
    assert any("flexiv-inspire-camera-node" in command for command in joined)
    assert any("flexiv-inspire-dftp-node" in command for command in joined)
    assert any("flexiv-inspire-pedal-router" in command for command in joined)
    assert not any("rerun_viz.cli" in command for command in joined)
    assert any("policy_api.ros_adapter" in command for command in joined)
    forbidden = (
        "teleop_input_node",
        "episode_control",
        "xr_raw",
        "run_manus_plugin",
    )
    assert not any(marker in command for marker in forbidden for command in joined)


def test_policy_authorizer_authorizes_once_per_pedal_press(monkeypatch, capsys) -> None:
    calls = []
    supervisor = _PolicyAuthorizationSupervisor.__new__(
        _PolicyAuthorizationSupervisor
    )
    supervisor._config = SimpleNamespace(document={"session": {"id": "test"}})
    supervisor._rdk_socket = Path("/tmp/rdk.sock")
    supervisor._require_pedal = True
    supervisor._pedal_pressed = False
    supervisor._control_state = ""
    supervisor._authorization_pending = False
    supervisor._last_attempt = 0.0
    supervisor._last_reported_error = ""
    monkeypatch.setattr(
        "flexiv_inspire_isaac.cli.time.monotonic", lambda: 20.0
    )
    monkeypatch.setattr(
        "flexiv_inspire_isaac.cli.subprocess.run",
        lambda command, **kwargs: calls.append((command, kwargs))
        or SimpleNamespace(returncode=0, stdout=""),
    )

    supervisor._on_control_state(SimpleNamespace(state_name="READY"))
    supervisor._on_pedal(SimpleNamespace(data=True))
    supervisor._refresh_if_needed()

    assert len(calls) == 1
    assert "--clear-hold-latched" not in calls[0][0]
    assert calls[0][1]["timeout"] == 15.0
    assert "policy 控制已自动授权" in capsys.readouterr().out

    # A successful request is not repeated while the bridge is transitioning.
    supervisor._last_attempt = 0.0
    supervisor._on_control_state(SimpleNamespace(state_name="POLICY_ARMED"))
    supervisor._refresh_if_needed()
    assert len(calls) == 1


def test_policy_authorizer_clears_hold_only_on_new_pedal_press(monkeypatch) -> None:
    calls = []
    supervisor = _PolicyAuthorizationSupervisor.__new__(
        _PolicyAuthorizationSupervisor
    )
    supervisor._config = SimpleNamespace(document={"session": {"id": "test"}})
    supervisor._rdk_socket = Path("/tmp/rdk.sock")
    supervisor._require_pedal = True
    supervisor._pedal_pressed = False
    supervisor._control_state = "HOLD_LATCHED"
    supervisor._authorization_pending = False
    supervisor._last_attempt = 0.0
    supervisor._last_reported_error = ""
    monkeypatch.setattr(
        "flexiv_inspire_isaac.cli.time.monotonic", lambda: 20.0
    )
    monkeypatch.setattr(
        "flexiv_inspire_isaac.cli.subprocess.run",
        lambda command, **kwargs: calls.append((command, kwargs))
        or SimpleNamespace(returncode=0, stdout=""),
    )

    supervisor._on_pedal(SimpleNamespace(data=True))
    supervisor._refresh_if_needed()

    assert len(calls) == 1
    assert "--clear-hold-latched" in calls[0][0]


def test_policy_authorizer_arms_direct_mode_without_pedal(monkeypatch, capsys) -> None:
    calls = []
    supervisor = _PolicyAuthorizationSupervisor.__new__(
        _PolicyAuthorizationSupervisor
    )
    supervisor._config = SimpleNamespace(document={"session": {"id": "test"}})
    supervisor._rdk_socket = Path("/tmp/rdk.sock")
    supervisor._require_pedal = False
    supervisor._pedal_pressed = True
    supervisor._control_state = ""
    supervisor._authorization_pending = True
    supervisor._last_attempt = 0.0
    supervisor._last_reported_error = ""
    monkeypatch.setattr(
        "flexiv_inspire_isaac.cli.time.monotonic", lambda: 20.0
    )
    monkeypatch.setattr(
        "flexiv_inspire_isaac.cli.subprocess.run",
        lambda command, **kwargs: calls.append((command, kwargs))
        or SimpleNamespace(returncode=0, stdout=""),
    )

    supervisor._on_control_state(SimpleNamespace(state_name="READY"))
    supervisor._refresh_if_needed()

    assert len(calls) == 1
    assert "--clear-hold-latched" not in calls[0][0]
    assert "RPC 策略动作可直接下发" in capsys.readouterr().out


def test_conversion_manifest_discovers_raw_and_legacy_episodes(tmp_path) -> None:
    dataset = tmp_path / "pick_place"
    legacy = dataset / "episode_000001"
    raw = dataset / "raw" / "episode_000002"
    for directory, index in ((legacy, 1), (raw, 2)):
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps(
                {"completed": True, "episode_index": index, "attempt": 1}
            ),
            encoding="utf-8",
        )

    class Config:
        document = {"recording": {"output_root": str(tmp_path), "dataset_name": "pick_place"}}

        @staticmethod
        def resolve(raw_path: str) -> Path:
            return Path(raw_path).resolve()

    assert _conversion_manifest(
        Config(), {"dataset_root": str(dataset), "episode": "latest"}
    ) == raw / "manifest.json"
    assert _conversion_manifest(
        Config(), {"dataset_root": str(dataset), "episode": "episode_000001"}
    ) == legacy / "manifest.json"
    assert _conversion_manifests(
        Config(), {"dataset_root": str(dataset), "episode": "all"}
    ) == [legacy / "manifest.json", raw / "manifest.json"]
    assert _conversion_manifests(
        Config(), {"dataset_root": "", "episode": "all"}
    ) == [legacy / "manifest.json", raw / "manifest.json"]


def test_batch_conversion_skips_existing_episode_output(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "sessions" / "pick_place"
    manifests = []
    for index in (1, 2):
        episode = dataset / "raw" / f"episode_{index:06d}"
        episode.mkdir(parents=True)
        manifest = episode / "manifest.json"
        manifest.write_text(
            json.dumps({"completed": True, "episode_index": index, "attempt": 1}),
            encoding="utf-8",
        )
        manifests.append(manifest)
    existing = dataset / "lerobot" / "episode_000001" / "sent_command"
    existing.mkdir(parents=True)
    (existing / "data.txt").write_text("already converted", encoding="utf-8")
    conversion = tmp_path / "conversion.yaml"
    conversion.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "source": {"dataset_root": str(dataset), "episode": "all"},
                "output": {
                    "root": str(dataset / "lerobot"),
                    "episode_subdirectory": True,
                },
                "lerobot_export": {"action": {"view": "sent_command"}},
            }
        ),
        encoding="utf-8",
    )
    exporter = tmp_path / "envs" / "data-py312" / "bin" / "flexiv-inspire-lerobot-export"
    exporter.parent.mkdir(parents=True)
    exporter.touch()
    calls = []
    monkeypatch.setattr(
        "flexiv_inspire_isaac.cli.subprocess.call",
        lambda command: calls.append(command) or 0,
    )

    class Config:
        root = tmp_path
        document = {
            "recording": {"dataset_name": "pick_place"},
            "lerobot_export": {"action": {"view": "sent_command"}},
        }

        @staticmethod
        def resolve(raw_path: str) -> Path:
            return Path(raw_path).resolve()

    status = _run_convert(
        Config(),
        SimpleNamespace(
            conversion_config=str(conversion),
            manifest="",
            action_view=None,
            output_root="",
            repo_id="",
        ),
    )

    assert status == 0
    assert len(calls) == 1
    assert calls[0][calls[0].index("--manifest") + 1] == str(manifests[1])
    assert calls[0][calls[0].index("--output-root") + 1] == str(
        dataset / "lerobot" / "episode_000002" / "sent_command"
    )


def test_profile_conversion_merges_manifests_with_one_exporter_call(
    tmp_path, monkeypatch
) -> None:
    dataset = tmp_path / "sessions" / "pick_place"
    manifests = []
    for index in (1, 2):
        episode = dataset / "raw" / f"episode_{index:06d}"
        episode.mkdir(parents=True)
        manifest = episode / "manifest.json"
        manifest.write_text(
            json.dumps(
                {"completed": True, "episode_index": index, "attempt": 1}
            ),
            encoding="utf-8",
        )
        manifests.append(manifest)
    output_root = dataset / "merged"
    conversion = tmp_path / "conversion-profile.yaml"
    conversion.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "source": {
                    "dataset_root": str(dataset),
                    "episode": "all",
                    "mcap_files": [],
                },
                "output": {
                    "root": str(output_root),
                    "episode_subdirectory": False,
                },
                "lerobot_export": {
                    "profile": "joint_proprio_cartesian_v1",
                    "action": {"view": "sent_command"},
                },
            }
        ),
        encoding="utf-8",
    )
    exporter = (
        tmp_path
        / "envs"
        / "data-py312"
        / "bin"
        / "flexiv-inspire-policy-profile-export"
    )
    exporter.parent.mkdir(parents=True)
    exporter.touch()
    calls = []
    monkeypatch.setattr(
        "flexiv_inspire_isaac.cli.subprocess.call",
        lambda command: calls.append(command) or 0,
    )

    class Config:
        root = tmp_path
        document = {
            "recording": {"dataset_name": "pick_place"},
            "lerobot_export": {"action": {"view": "sent_command"}},
        }

        @staticmethod
        def resolve(raw_path: str) -> Path:
            return Path(raw_path).resolve()

    status = _run_convert(
        Config(),
        SimpleNamespace(
            conversion_config=str(conversion),
            manifest="",
            action_view=None,
            output_root="",
            repo_id="",
        ),
    )

    assert status == 0
    assert len(calls) == 1
    command = calls[0]
    assert command[0] == str(exporter)
    assert [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--manifest"
    ] == [str(path) for path in manifests]
    assert command[command.index("--output-root") + 1] == str(output_root)


def test_example_system_config_renders_all_runtime_children(tmp_path):
    config = load_system_config(_example())
    rendered = render_runtime_configs(config, tmp_path)
    assert config.document["sampling"]["action_label"] == "sent_command"
    assert set(rendered) == {
        "camera.yaml",
        "xr_bridge.yaml",
        "isaac_camera_receiver.yaml",
        "dftp.yaml",
        "control_bridge.yaml",
        "teleop.yaml",
        "pedal.yaml",
        "lerobot_export.yaml",
        "snapshot",
    }
    camera = yaml.safe_load(rendered["camera.yaml"].read_text())
    assert camera["cameras"]["head"]["width"] == 424
    assert camera["recording"]["depth_enabled"] is True
    assert camera["cameras"]["head"]["pointcloud_enabled"] is True
    assert camera["cameras"]["head"]["pointcloud_stride"] == 2
    assert camera["cameras"]["head"]["depth_visual_preset"] == "high_density"
    assert camera["cameras"]["head"]["depth_emitter_enabled"] is True
    assert camera["cameras"]["head"]["depth_laser_power"] == 210
    assert camera["cameras"]["head"]["depth_spatial_filter_enabled"] is True
    assert camera["cameras"]["head"]["depth_temporal_filter_enabled"] is True
    assert camera["cameras"]["left_wrist"]["pointcloud_enabled"] is False
    receiver = yaml.safe_load(rendered["isaac_camera_receiver.yaml"].read_text())
    assert set(receiver["cameras"]) == {"head"}
    assert set(receiver["display"]["xr"]["planes"]) == {"head"}
    control = yaml.safe_load(rendered["control_bridge.yaml"].read_text())
    parameters = control["/**"]["ros__parameters"]
    assert len(parameters["joint_lower_limits_rad"]) == 7
    assert len(parameters["joint_upper_limits_rad"]) == 7
    assert parameters["joint_lower_limits_rad"][0] < 0.0
    assert parameters["joint_upper_limits_rad"][5] > 4.5
    assert parameters["software_safety_limits_enabled"] is False
    assert parameters["policy_pedal_required"] is True
    assert parameters["hand_daemon_sync_hz"] == 25.0
    assert parameters["hand_observation_timeout_ms"] == 500.0
    assert parameters["require_hands_for_arm_control"] is False
    assert parameters["cartesian_control_mode"] == "position"
    assert parameters["cartesian_impedance_stiffness"] == [
        1200,
        1200,
        1200,
        80,
        80,
        80,
    ]
    assert all(
        isinstance(value, float) for value in parameters["cartesian_position_stiffness"]
    )
    assert all(
        isinstance(value, float)
        for value in parameters["cartesian_impedance_stiffness"]
    )
    assert parameters["cartesian_damping_ratio"] == [0.7] * 6
    assert len(parameters["home_left_joints_rad"]) == 7
    assert (
        parameters["home_max_velocity_rad_s"] <= parameters["max_joint_velocity_rad_s"]
    )
    assert parameters["max_joint_velocity_rad_s"] == 2.0
    assert parameters["home_lift_enabled"] is True
    assert parameters["home_lift_left_target_x_m"] == 0.925266862
    assert parameters["home_lift_left_target_y_m"] == 0.345229030
    assert parameters["home_lift_left_safe_z_m"] == -0.177676
    assert parameters["home_lift_right_target_x_m"] == 0.952098012
    assert parameters["home_lift_right_target_y_m"] == -0.152109638
    assert parameters["home_lift_right_safe_z_m"] == -0.213296
    assert parameters["home_lift_max_linear_velocity_m_s"] == 0.12
    assert parameters["home_lift_parallel"] is False
    assert parameters["max_linear_velocity_m_s"] == 0.20
    assert parameters["max_angular_velocity_rad_s"] == 0.60
    assert parameters["frame_config"].endswith("/config/dual_arm_frames.yaml")
    dftp = yaml.safe_load(rendered["dftp.yaml"].read_text())[
        "flexiv_inspire_dftp_driver"
    ]["ros__parameters"]
    assert dftp["hardware_write_enabled"] is False
    assert dftp["local_write_confirmation"] == ""
    assert dftp["hand_reset_enabled"] is False
    assert dftp["hand_reset_open_angle"] == 1000
    assert dftp["hand_reset_closed_angle"] == 0
    assert dftp["hand_reset_open_tolerance"] == 30
    assert dftp["hand_reset_open_timeout_s"] == 10.0
    teleop = yaml.safe_load(rendered["teleop.yaml"].read_text())["/**"][
        "ros__parameters"
    ]
    assert teleop["deadman_source"] == "external_bool"
    assert teleop["foot_pedal"] == "name:input-remapper keyboard"
    assert teleop["enable_key_code"] == 57
    assert teleop["max_translation_step_m"] == 0.01
    assert teleop["max_rotation_step_rad"] == 0.10
    assert teleop["left_pose_index"] == 0
    assert teleop["right_pose_index"] == 1
    assert teleop["axis_rotation"] == [
        0.0,
        0.0,
        -1.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    assert teleop["translation_gain"] == 1.0
    assert teleop["rotation_gain"] == 1.0
    assert teleop["manus_left_ergonomics_topic"] == "/manus/left/ergonomics"
    assert teleop["manus_right_ergonomics_topic"] == "/manus/right/ergonomics"
    pedal = yaml.safe_load(rendered["pedal.yaml"].read_text())["/**"]["ros__parameters"]
    assert pedal["foot_pedal"] == "name:input-remapper keyboard"
    assert pedal["enable_key_code"] == 57
    commands = _commands(config, rendered, include_xr_receiver=False)
    rdk = commands[0]
    assert rdk[rdk.index("--config") + 1] == str(
        Path(__file__).parents[1] / "apps/flexiv_daemon/config/robots.yaml"
    )
    assert commands[1][:3] == [
        sys.executable,
        "-m",
        "flexiv_inspire_control.node",
    ]
    assert commands[2][:3] == [
        sys.executable,
        "-m",
        "flexiv_inspire_control.teleop_input_node",
    ]
    quest_source = next(
        command
        for command in commands
        if "flexiv_inspire_isaac.oculus_reader_ros_source" in command
    )
    assert "rate_hz:=60.0" in quest_source
    assert "stale_timeout_s:=0.12" in quest_source
    assert "package_name:=com.rail.oculus.teleop" in quest_source
    assert "auto_install_apk:=true" in quest_source
    assert any(command[0].endswith("run_manus_plugin.sh") for command in commands)
    ergonomics_source = next(
        command
        for command in commands
        if "flexiv_inspire_control.manus_ergonomics_source" in command
    )
    assert "udp_host:=127.0.0.1" in ergonomics_source
    assert "udp_port:=15053" in ergonomics_source
    manus_plugin = next(
        command for command in commands if command[0].endswith("run_manus_plugin.sh")
    )
    assert manus_plugin[-2:] == ["--ergonomics-udp", "127.0.0.1:15053"]
    episode = next(
        command
        for command in commands
        if "flexiv_inspire_isaac.episode_control" in command
    )
    assert "home_result_timeout_s:=60.0" in episode
    assert any(
        item.endswith("/apps/flexiv_daemon/config/tool_payload.yaml")
        for item in episode
    )
    assert any(item.endswith("/ft_zero_events.jsonl") for item in episode)
    assert not any(item.startswith("manus_calibration:=") for item in episode)
    assert not any(item.startswith("camera_head_extrinsics:=") for item in episode)
    assert f"dataset_name:={config.document['recording']['dataset_name']}" in episode
    assert f"task_name:={config.document['recording']['task_name']}" in episode
    assert f"episode_count:={config.document['recording']['episode_count']}" in episode
    assert any(
        item.startswith('task_description:="')
        and len(item) > len('task_description:=""')
        for item in episode
    )


def test_site_entry_composes_small_hardware_sensor_recording_runtime_files(tmp_path):
    config = load_system_config(_site())

    assert config.document["recording"]["dataset_name"] == "open_boxes_first_try"
    assert config.document["recording"]["task_name"] == "open_boxes"
    assert config.document["recording"]["episode_count"] == 20
    assert config.document["recording"]["auto_reset_before_record"] is True
    assert config.document["recording"]["ros_mcap_enabled"] is False
    assert config.document["recording"]["deviceio_profile"] == "training"
    assert config.document["recording"]["record_only_while_pedal_pressed"] is True
    assert config.document["sampling"]["arm_observation_hz"] == 300.0
    assert config.document["sampling"]["hand_state_hz"] == 200.0
    assert config.document["sampling"]["tactile_hz"] == 15.0
    assert config.document["sampling"]["camera_hz"] == 30.0
    assert config.document["recording"]["live_rerun"] == {
        "enabled": False,
        "viewer_port": 9876,
        "telemetry_hz": 5.0,
        "tactile_hz": 10.0,
        "image_hz": 10.0,
        "pointcloud_hz": 2.0,
    }
    assert config.document["flexiv"]["home"]["quest_button"] == ("right_primary_click")
    assert config.document["cameras"]["streams"]["head"]["serial"]
    assert config.document["xr_video"]["streams"]["head"]["enabled"] is True
    assert config.document["xr_video"]["streams"]["left_wrist"]["enabled"] is False
    assert config.document["xr_video"]["streams"]["right_wrist"]["enabled"] is False
    assert config.document["xr_video"]["transport"] == "usb_tcp"
    assert config.document["xr_video"]["cloudxr_client"] == {
        "per_eye_width": 1792,
        "per_eye_height": 1536,
        "frame_rate": 72,
        "max_bitrate_mbps": 80,
        "codec": "h264",
        "enable_tex_sub_image_2d": True,
    }
    assert config.document["xr_video"]["display"]["lock_mode"] == "world"
    assert config.document["teleop"]["quest_input"] == {
        "provider": "oculus_reader",
        "publish_rate_hz": 60.0,
        "stale_timeout_ms": 120.0,
        "oculus_reader": {
            "adb_serial": "",
            "package_name": "com.rail.oculus.teleop",
            "auto_install_apk": True,
        },
    }
    assert config.document["teleop"]["manus_ergonomics"] == {
        "enabled": True,
        "udp_host": "127.0.0.1",
        "udp_port": 15053,
        "left_topic": "/manus/left/ergonomics",
        "right_topic": "/manus/right/ergonomics",
    }
    assert "--mock" not in config.document["commands"]["rdk_daemon"]
    assert config.document["flexiv"]["safety"]["max_linear_velocity_m_s"] == 0.20
    assert (
        config.document["flexiv"]["safety"]["software_safety_limits_enabled"] is False
    )

    rendered = render_runtime_configs(config, tmp_path)
    camera = yaml.safe_load(rendered["camera.yaml"].read_text())
    assert camera["cameras"]["head"]["fps"] == 30
    assert camera["cameras"]["head"]["recording_hz"] == 30.0
    assert camera["cameras"]["left_wrist"]["recording_hz"] == 30.0
    lerobot = yaml.safe_load(rendered["lerobot_export.yaml"].read_text())
    assert lerobot["timeline"]["fps"] == 30.0
    assert lerobot["high_rate_arm_samples_per_frame"] == 20
    dftp = yaml.safe_load(rendered["dftp.yaml"].read_text())[
        "flexiv_inspire_dftp_driver"
    ]["ros__parameters"]
    assert dftp["hardware_write_enabled"] is True
    assert dftp["left_model"] == "rh56e2_2l_t1"
    assert dftp["right_model"] == "rh56e2_2r_t1"
    assert dftp["local_write_confirmation"] == ("DFTP-LOCAL-CONTROL-AUTHORIZED")
    assert dftp["hand_reset_enabled"] is True
    assert dftp["hand_reset_pause_s"] == 0.35
    rdk = _commands(config, rendered, include_xr_receiver=False)[0]
    assert "--hardware" in rdk
    assert "--allow-hardware-writes" in rdk
    assert "--local-permit-file" in rdk
    assert not any(
        "flexiv_inspire_isaac.rerun_viz.cli" in command
        for command in _commands(config, rendered, include_xr_receiver=False)
    )


def test_site_uses_validated_30hz_rates_and_disables_desktop_viewer():
    config = load_system_config(_site())

    assert config.document["sampling"]["teleop_command_hz"] == 30.0
    assert config.document["sampling"]["camera_hz"] == 30.0
    assert config.document["sampling"]["policy_observation_hz"] == 30.0
    assert config.document["sampling"]["training_timeline_hz"] == 30.0
    assert config.document["sampling"]["arm_observation_hz"] == 300.0
    assert config.document["sampling"]["hand_state_hz"] == 200.0
    assert config.document["teleop"]["quest_input"]["publish_rate_hz"] == 60.0
    assert config.document["flexiv"]["policy_control"]["require_pedal"] is True
    assert config.document["cameras"]["depth_fps"] == 30
    assert {
        name: stream["fps"]
        for name, stream in config.document["cameras"]["streams"].items()
    } == {"head": 30, "left_wrist": 30, "right_wrist": 30}
    assert config.document["recording"]["live_rerun"]["enabled"] is False


def test_system_config_rejects_non_executed_action_label(tmp_path):
    data = _example_document()
    data["sampling"]["action_label"] = "safe_command"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    try:
        load_system_config(path)
    except SystemConfigError as exc:
        assert "sent_command" in str(exc)
    else:
        raise AssertionError("unsafe dataset action label was accepted")


def test_quest_input_does_not_depend_on_xr_video_branch(tmp_path):
    config = load_system_config(_example())
    config.document["xr_video"]["enabled"] = False
    rendered = render_runtime_configs(config, tmp_path / "rendered")

    commands = _commands(config, rendered, include_xr_receiver=True)
    joined = [" ".join(command) for command in commands]
    assert any("oculus_reader_ros_source" in command for command in joined)
    assert any("run_manus_plugin.sh" in command for command in joined)
    assert not any("flexiv-inspire-xr-bridge" in command for command in joined)
    assert not any("run_isaac_camera_receiver.sh" in command for command in joined)


def test_isaac_openxr_remains_a_selectable_quest_input_provider(tmp_path):
    config = load_system_config(_example())
    config.document["teleop"]["quest_input"]["provider"] = "isaac_openxr"
    rendered = render_runtime_configs(config, tmp_path / "rendered")

    commands = _commands(config, rendered, include_xr_receiver=False)
    joined = [" ".join(command) for command in commands]
    assert any("run_xr_raw_source.sh" in command for command in joined)
    assert not any("oculus_reader_ros_source" in command for command in joined)


def test_legacy_config_without_quest_provider_keeps_isaac_openxr(tmp_path):
    config = load_system_config(_example())
    config.document["teleop"].pop("quest_input")
    rendered = render_runtime_configs(config, tmp_path / "rendered")

    joined = [
        " ".join(command)
        for command in _commands(config, rendered, include_xr_receiver=False)
    ]
    assert any("run_xr_raw_source.sh" in command for command in joined)
    assert not any("oculus_reader_ros_source" in command for command in joined)


def test_system_config_rejects_unknown_quest_input_provider(tmp_path):
    data = _example_document()
    root = Path(__file__).parents[1]
    data["flexiv"]["rdk_config"] = str(
        root / "apps/flexiv_daemon/config/robots.yaml"
    )
    data["flexiv"]["frame_config"] = str(root / "config/dual_arm_frames.yaml")
    data["teleop"]["quest_input"]["provider"] = "automatic"
    path = tmp_path / "bad-quest-provider.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(SystemConfigError, match="quest_input.provider"):
        load_system_config(path)


def test_system_config_rejects_non_boolean_xr_switch(tmp_path):
    data = _example_document()
    data["xr_video"]["enabled"] = "false"
    path = tmp_path / "bad-xr-switch.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    try:
        load_system_config(path)
    except SystemConfigError as exc:
        assert "xr_video.enabled must be boolean" in str(exc)
    else:
        raise AssertionError("string XR video switch was accepted")


def test_system_config_rejects_invalid_joint_limits(tmp_path):
    data = _example_document()
    data["flexiv"]["safety"]["joint_lower_limits_rad"][0] = 3.0
    path = tmp_path / "bad-limits.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    try:
        load_system_config(path)
    except SystemConfigError as exc:
        assert "joint lower limits" in str(exc)
    else:
        raise AssertionError("invalid joint limits were accepted")


def test_system_config_rejects_command_limit_above_daemon_ceiling(tmp_path):
    data = _example_document()
    daemon_source = Path(__file__).parents[1] / "apps/flexiv_daemon/config/robots.yaml"
    tool_source = (
        Path(__file__).parents[1] / "apps/flexiv_daemon/config/tool_payload.yaml"
    )
    daemon_data = yaml.safe_load(daemon_source.read_text(encoding="utf-8"))
    daemon_data["capture_frame_config"] = "frames.yaml"
    daemon_path = tmp_path / "robots.yaml"
    daemon_path.write_text(yaml.safe_dump(daemon_data), encoding="utf-8")
    (tmp_path / "tool_payload.yaml").write_text(
        tool_source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "frames.yaml").write_text(
        (Path(__file__).parents[1] / "config/dual_arm_frames.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    data["flexiv"]["rdk_config"] = str(daemon_path)
    data["flexiv"]["frame_config"] = str(tmp_path / "frames.yaml")
    data["flexiv"]["safety"]["max_linear_velocity_m_s"] = 0.36
    data["flexiv"]["safety"]["max_tcp_linear_speed_m_s"] = 0.50
    path = tmp_path / "bad-daemon-limit.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    try:
        load_system_config(path)
    except SystemConfigError as exc:
        assert "exceeds daemon ceiling" in str(exc)
    else:
        raise AssertionError("daemon Cartesian ceiling mismatch was accepted")


def test_system_config_rejects_missing_recording_prompt(tmp_path):
    data = _example_document()
    root = Path(__file__).parents[1]
    data["flexiv"]["rdk_config"] = str(
        root / "apps/flexiv_daemon/config/robots.yaml"
    )
    data["flexiv"]["frame_config"] = str(root / "config/dual_arm_frames.yaml")
    data["recording"]["task_description"] = ""
    path = tmp_path / "bad-recording.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    try:
        load_system_config(path)
    except SystemConfigError as exc:
        assert "task_description" in str(exc)
    else:
        raise AssertionError("empty VLA task description was accepted")


def test_system_config_rejects_invalid_recording_task_name(tmp_path):
    data = _example_document()
    root = Path(__file__).parents[1]
    data["flexiv"]["rdk_config"] = str(
        root / "apps/flexiv_daemon/config/robots.yaml"
    )
    data["flexiv"]["frame_config"] = str(root / "config/dual_arm_frames.yaml")
    data["recording"]["task_name"] = "pick/place"
    path = tmp_path / "bad-task-name.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    try:
        load_system_config(path)
    except SystemConfigError as exc:
        assert "task_name" in str(exc)
    else:
        raise AssertionError("unsafe recording task name was accepted")


def test_composed_config_rejects_duplicate_fragment_keys(tmp_path):
    (tmp_path / "a.yaml").write_text("recording: {dataset_name: first}\n")
    (tmp_path / "b.yaml").write_text("recording: {dataset_name: second}\n")
    entry = tmp_path / "site.yaml"
    entry.write_text(
        "schema_version: 1\nincludes: [a.yaml, b.yaml]\n", encoding="utf-8"
    )

    try:
        load_system_config(entry)
    except SystemConfigError as exc:
        assert "duplicate composed config key" in str(exc)
    else:
        raise AssertionError("duplicate composed keys were accepted")
