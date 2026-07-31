"""Launch two DFTP workers in one process, one socket owner per hand."""

from pathlib import Path
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    config_default = str(Path(__file__).resolve().parents[1] / "dual_hands.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=config_default),
            DeclareLaunchArgument("hardware_write_enabled", default_value="false"),
            DeclareLaunchArgument("local_session_id", default_value=""),
            DeclareLaunchArgument("local_write_confirmation", default_value=""),
            ExecuteProcess(
                cmd=[
                    sys.executable,
                    "-m",
                    "flexiv_inspire_isaac.dftp.ros_node",
                    "--ros-args",
                    "--params-file",
                    LaunchConfiguration("config"),
                    "-p",
                    [
                        "hardware_write_enabled:=",
                        LaunchConfiguration("hardware_write_enabled"),
                    ],
                    "-p",
                    ["local_session_id:=", LaunchConfiguration("local_session_id")],
                    "-p",
                    [
                        "local_write_confirmation:=",
                        LaunchConfiguration("local_write_confirmation"),
                    ],
                ],
                additional_env={
                    "ROS_LOCALHOST_ONLY": "1",
                    "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp",
                },
                output="screen",
            ),
        ]
    )
