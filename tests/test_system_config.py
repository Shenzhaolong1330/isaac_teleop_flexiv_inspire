from pathlib import Path

import yaml

from flexiv_inspire_isaac.cli import _commands
from flexiv_inspire_isaac.system_config import SystemConfigError, load_system_config, render_runtime_configs


def _example() -> Path:
    return Path(__file__).parents[1] / "config" / "system.example.yaml"


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
    assert parameters["cartesian_damping_ratio"] == [0.7] * 6
    assert len(parameters["home_left_joints_rad"]) == 7
    assert parameters["home_max_velocity_rad_s"] <= parameters[
        "max_joint_velocity_rad_s"
    ]
    assert parameters["frame_config"].endswith("/config/dual_arm_frames.yaml")
    commands = _commands(config, rendered, include_xr_receiver=False)
    rdk = commands[0]
    assert rdk[rdk.index("--config") + 1] == str(
        Path(__file__).parents[1] / "apps/flexiv_daemon/config/robots.yaml"
    )
    episode = next(
        command
        for command in commands
        if command[0] == "flexiv-inspire-episode-controller"
    )
    assert any(
        item.endswith("/apps/flexiv_daemon/config/tool_payload.yaml")
        for item in episode
    )
    assert any(item.endswith("/ft_zero_events.jsonl") for item in episode)
    assert "manus_calibration:=" in episode


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
