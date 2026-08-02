from pathlib import Path
import sys

import yaml

from flexiv_inspire_isaac.cli import _commands
from flexiv_inspire_isaac.system_config import SystemConfigError, load_system_config, render_runtime_configs


def _example() -> Path:
    return Path(__file__).parents[1] / "config" / "system.example.yaml"


def _site() -> Path:
    return Path(__file__).parents[1] / "config" / "site.yaml"


def test_example_system_config_renders_all_runtime_children(tmp_path):
    config = load_system_config(_example())
    rendered = render_runtime_configs(config, tmp_path)
    assert config.document["sampling"]["action_label"] == "sent_command"
    assert set(rendered) == {"camera.yaml", "xr_bridge.yaml", "isaac_camera_receiver.yaml", "dftp.yaml", "control_bridge.yaml", "teleop.yaml", "pedal.yaml", "lerobot_export.yaml", "snapshot"}
    camera = yaml.safe_load(rendered["camera.yaml"].read_text())
    assert camera["cameras"]["head"]["width"] == 424
    assert camera["recording"]["depth_enabled"] is True
    assert camera["cameras"]["head"]["pointcloud_enabled"] is True
    assert camera["cameras"]["head"]["pointcloud_stride"] == 2
    assert camera["cameras"]["left_wrist"]["pointcloud_enabled"] is False
    control = yaml.safe_load(rendered["control_bridge.yaml"].read_text())
    parameters = control["/**"]["ros__parameters"]
    assert len(parameters["joint_lower_limits_rad"]) == 7
    assert len(parameters["joint_upper_limits_rad"]) == 7
    assert parameters["joint_lower_limits_rad"][0] < 0.0
    assert parameters["joint_upper_limits_rad"][5] > 4.5
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
        isinstance(value, float)
        for value in parameters["cartesian_position_stiffness"]
    )
    assert all(
        isinstance(value, float)
        for value in parameters["cartesian_impedance_stiffness"]
    )
    assert parameters["cartesian_damping_ratio"] == [0.7] * 6
    assert len(parameters["home_left_joints_rad"]) == 7
    assert parameters["home_max_velocity_rad_s"] <= parameters[
        "max_joint_velocity_rad_s"
    ]
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
    pedal = yaml.safe_load(rendered["pedal.yaml"].read_text())["/**"][
        "ros__parameters"
    ]
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
        command for command in commands
        if command[0].endswith("run_xr_raw_source.sh")
    )
    assert xr_source[xr_source.index("--transport") + 1] == "lan"
    assert xr_source[xr_source.index("--wifi-connection") + 1] == "Deepybo-Prime"
    assert any(command[0].endswith("run_manus_plugin.sh") for command in commands)
    episode = next(
        command
        for command in commands
        if "flexiv_inspire_isaac.episode_control" in command
    )
    assert any(
        item.endswith("/apps/flexiv_daemon/config/tool_payload.yaml")
        for item in episode
    )
    assert any(item.endswith("/ft_zero_events.jsonl") for item in episode)
    assert not any(item.startswith("manus_calibration:=") for item in episode)
    assert not any(item.startswith("camera_head_extrinsics:=") for item in episode)
    assert (
        f"dataset_name:={config.document['recording']['dataset_name']}" in episode
    )
    assert (
        f"episode_count:={config.document['recording']['episode_count']}" in episode
    )
    assert any(
        item.startswith('task_description:="')
        and len(item) > len('task_description:=""')
        for item in episode
    )


def test_site_entry_composes_small_hardware_sensor_recording_runtime_files(tmp_path):
    config = load_system_config(_site())

    assert config.document["recording"]["dataset_name"] == "pick_place_demo"
    assert config.document["recording"]["episode_count"] == 20
    assert config.document["flexiv"]["home"]["quest_button"] == (
        "right_primary_click"
    )
    assert config.document["cameras"]["streams"]["head"]["serial"]
    assert "--mock" not in config.document["commands"]["rdk_daemon"]

    rendered = render_runtime_configs(config, tmp_path)
    dftp = yaml.safe_load(rendered["dftp.yaml"].read_text())[
        "flexiv_inspire_dftp_driver"
    ]["ros__parameters"]
    assert dftp["hardware_write_enabled"] is True
    assert dftp["local_write_confirmation"] == (
        "DFTP-LOCAL-CONTROL-AUTHORIZED"
    )
    assert dftp["hand_reset_enabled"] is True
    assert dftp["hand_reset_pause_s"] == 0.35
    rdk = _commands(config, rendered, include_xr_receiver=False)[0]
    assert "--hardware" in rdk
    assert "--allow-hardware-writes" in rdk
    assert "--local-permit-file" in rdk


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


def test_system_config_rejects_missing_recording_prompt(tmp_path):
    data = yaml.safe_load(_example().read_text())
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
