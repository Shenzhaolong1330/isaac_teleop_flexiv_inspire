from pathlib import Path
import importlib.util
import sys

import pytest

from flexiv_inspire_isaac.cameras.config import load_camera_configs
from flexiv_inspire_isaac.cameras.smoke import DEFAULT_CONFIG as SMOKE_DEFAULT_CONFIG
from flexiv_inspire_isaac.cameras.verify import DEFAULT_CONFIG as VERIFY_DEFAULT_CONFIG


CONFIG = Path(__file__).parents[1] / "ros2_ws" / "src" / "flexiv_inspire_cameras" / "realsense_rgb.yaml"


def test_three_camera_config_is_rgb_only_424x240_at_30():
    cameras = load_camera_configs(CONFIG)
    assert set(cameras) == {"head", "left_wrist", "right_wrist"}
    assert len({camera.serial for camera in cameras.values()}) == 3
    for camera in cameras.values():
        assert (camera.width, camera.height, camera.fps) == (424, 240, 30)
        assert not camera.depth_enabled
        assert camera.jpeg_quality == 90


def test_camera_cli_defaults_follow_the_packaged_config():
    assert SMOKE_DEFAULT_CONFIG == CONFIG.resolve()
    assert VERIFY_DEFAULT_CONFIG == CONFIG.resolve()
    assert SMOKE_DEFAULT_CONFIG.is_file()


def test_camera_launch_uses_active_python_and_packaged_config():
    launch_module = pytest.importorskip("launch")
    if not hasattr(launch_module, "LaunchContext"):
        pytest.skip("ROS 2 launch Python package is unavailable")
    from launch import LaunchContext

    launch_file = CONFIG.parent / "launch" / "triple_rgb.launch.py"
    spec = importlib.util.spec_from_file_location("camera_launch_test", launch_file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    assert len(description.entities) == 1
    process = description.entities[0]
    context = LaunchContext()
    command = ["".join(item.perform(context) for item in part) for part in process.cmd]
    assert command[0] == sys.executable
    assert command[1:3] == ["-m", "flexiv_inspire_isaac.cameras.ros_node"]
    config_arg = next(value for value in command if value.startswith("config:="))
    assert Path(config_arg.removeprefix("config:=")).is_file()
    assert "envs/src" not in config_arg


def test_mixed_librealsense_version_is_rejected(tmp_path):
    content = CONFIG.read_text().replace("2.57.7", "2.58.0")
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    with pytest.raises(ValueError, match="mixing"):
        load_camera_configs(path)
