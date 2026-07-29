from pathlib import Path

import yaml

from flexiv_inspire_isaac.system_config import SystemConfigError, load_system_config, render_runtime_configs


def _example() -> Path:
    return Path(__file__).parents[1] / "config" / "system.example.yaml"


def test_example_system_config_renders_all_runtime_children(tmp_path):
    config = load_system_config(_example())
    rendered = render_runtime_configs(config, tmp_path)
    assert config.document["sampling"]["action_label"] == "sent_command"
    assert set(rendered) == {"camera.yaml", "dftp.yaml", "control_bridge.yaml", "teleop.yaml", "pedal.yaml", "snapshot"}
    camera = yaml.safe_load(rendered["camera.yaml"].read_text())
    assert camera["cameras"]["head"]["width"] == 424
    assert camera["recording"]["depth_enabled"] is False


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
