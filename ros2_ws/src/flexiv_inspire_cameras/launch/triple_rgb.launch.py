"""Launch the isolated triple RealSense RGB publisher."""

from pathlib import Path
import sys

from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description() -> LaunchDescription:
    config = Path(__file__).resolve().parents[1] / "realsense_rgb.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"camera configuration is missing: {config}")
    return LaunchDescription(
        [
            ExecuteProcess(
                cmd=[
                    sys.executable,
                    "-m",
                    "flexiv_inspire_isaac.cameras.ros_node",
                    "--ros-args",
                    "-p",
                    f"config:={config}",
                ],
                additional_env={
                    "ROS_LOCALHOST_ONLY": "1",
                    "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp",
                },
                output="screen",
            )
        ]
    )
