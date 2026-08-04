import sys
from pathlib import Path

import yaml
from flexiv_inspire_isaac.cli import _commands
from flexiv_inspire_isaac.system_config import (
    SystemConfigError,
    load_system_config,
    render_runtime_configs,
)


def _example() -> Path:
    return Path(__file__).parents[1] / "config" / "system.example.yaml"


def _site() -> Path:
    return Path(__file__).parents[1] / "config" / "site.yaml"


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
    assert parameters["home_lift_left_safe_z_m"] == -0.377676
    assert parameters["home_lift_right_safe_z_m"] == -0.413296
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
    xr_source = next(
        command for command in commands if command[0].endswith("run_xr_raw_source.sh")
    )
    assert xr_source[xr_source.index("--transport") + 1] == "lan"
    assert xr_source[xr_source.index("--wifi-connection") + 1] == "Deepybo-Prime"
    assert xr_source[xr_source.index("--client-per-eye-width") + 1] == "1792"
    assert xr_source[xr_source.index("--client-per-eye-height") + 1] == "1536"
    assert xr_source[xr_source.index("--client-frame-rate") + 1] == "72"
    assert xr_source[xr_source.index("--client-max-bitrate-mbps") + 1] == "80"
    assert xr_source[xr_source.index("--client-codec") + 1] == "h264"
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
    assert f"episode_count:={config.document['recording']['episode_count']}" in episode
    assert any(
        item.startswith('task_description:="')
        and len(item) > len('task_description:=""')
        for item in episode
    )


def test_site_entry_composes_small_hardware_sensor_recording_runtime_files(tmp_path):
    config = load_system_config(_site())

    assert config.document["recording"]["dataset_name"] == "pick_place_demo"
    assert config.document["recording"]["episode_count"] == 20
    assert config.document["recording"]["auto_reset_before_record"] is True
    assert config.document["recording"]["ros_mcap_enabled"] is False
    assert config.document["recording"]["deviceio_profile"] == "training"
    assert config.document["recording"]["record_only_while_pedal_pressed"] is True
    assert config.document["sampling"]["arm_observation_hz"] == 300.0
    assert config.document["sampling"]["hand_state_hz"] == 15.0
    assert config.document["sampling"]["tactile_hz"] == 15.0
    assert config.document["sampling"]["camera_hz"] == 15.0
    assert config.document["recording"]["live_rerun"] == {
        "enabled": True,
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
    assert camera["cameras"]["head"]["fps"] == 15
    assert camera["cameras"]["head"]["recording_hz"] == 15.0
    assert camera["cameras"]["left_wrist"]["recording_hz"] == 15.0
    lerobot = yaml.safe_load(rendered["lerobot_export.yaml"].read_text())
    assert lerobot["timeline"]["fps"] == 15.0
    assert lerobot["high_rate_arm_samples_per_frame"] == 20
    dftp = yaml.safe_load(rendered["dftp.yaml"].read_text())[
        "flexiv_inspire_dftp_driver"
    ]["ros__parameters"]
    assert dftp["hardware_write_enabled"] is True
    assert dftp["local_write_confirmation"] == ("DFTP-LOCAL-CONTROL-AUTHORIZED")
    assert dftp["hand_reset_enabled"] is True
    assert dftp["hand_reset_pause_s"] == 0.35
    rdk = _commands(config, rendered, include_xr_receiver=False)[0]
    assert "--hardware" in rdk
    assert "--allow-hardware-writes" in rdk
    assert "--local-permit-file" in rdk
    assert any(
        "flexiv_inspire_isaac.rerun_viz.cli" in command
        for command in _commands(config, rendered, include_xr_receiver=False)
    )


def test_system_config_rejects_non_executed_action_label(tmp_path):
    data = yaml.safe_load(_example().read_text())
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
    assert any("run_xr_raw_source.sh" in command for command in joined)
    assert any("run_manus_plugin.sh" in command for command in joined)
    assert not any("flexiv-inspire-xr-bridge" in command for command in joined)
    assert not any("run_isaac_camera_receiver.sh" in command for command in joined)


def test_system_config_rejects_invalid_joint_limits(tmp_path):
    data = yaml.safe_load(_example().read_text())
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
    data = yaml.safe_load(_example().read_text())
    daemon_source = Path(__file__).parents[1] / "apps/flexiv_daemon/config/robots.yaml"
    tool_source = (
        Path(__file__).parents[1] / "apps/flexiv_daemon/config/tool_payload.yaml"
    )
    daemon_data = yaml.safe_load(daemon_source.read_text(encoding="utf-8"))
    daemon_path = tmp_path / "robots.yaml"
    daemon_path.write_text(yaml.safe_dump(daemon_data), encoding="utf-8")
    (tmp_path / "tool_payload.yaml").write_text(
        tool_source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    data["flexiv"]["rdk_config"] = str(daemon_path)
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
    data = yaml.safe_load(_example().read_text())
    data["flexiv"]["rdk_config"] = str(
        Path(__file__).parents[1] / "apps/flexiv_daemon/config/robots.yaml"
    )
    data["recording"]["task_description"] = ""
    path = tmp_path / "bad-recording.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    try:
        load_system_config(path)
    except SystemConfigError as exc:
        assert "task_description" in str(exc)
    else:
        raise AssertionError("empty VLA task description was accepted")


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
