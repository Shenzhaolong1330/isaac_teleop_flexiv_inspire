from pathlib import Path
import uuid

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile


def generate_launch_description():
    share = Path(get_package_share_directory("flexiv_inspire_control"))
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "session_id",
                default_value=f"hardware-{uuid.uuid4()}",
            ),
            DeclareLaunchArgument(
                "rdk_socket",
                default_value="/run/user/1000/isaac_teleop/rdk.sock",
            ),
            DeclareLaunchArgument(
                "foot_pedal",
                default_value="/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd",
            ),
            DeclareLaunchArgument("command_enabled", default_value="false"),
            DeclareLaunchArgument(
                "control_config",
                default_value=str(share / "config" / "control_bridge.yaml"),
            ),
            DeclareLaunchArgument(
                "teleop_config",
                default_value=str(share / "config" / "teleop.yaml"),
            ),
            Node(
                package="flexiv_inspire_control",
                executable="control_bridge",
                output="screen",
                parameters=[
                    ParameterFile(
                        LaunchConfiguration("control_config"),
                        allow_substs=True,
                    ),
                    {
                        "session_id": LaunchConfiguration("session_id"),
                        "rdk_socket": LaunchConfiguration("rdk_socket"),
                        "foot_pedal": LaunchConfiguration("foot_pedal"),
                    },
                ],
            ),
            Node(
                package="flexiv_inspire_control",
                executable="teleop_input",
                output="screen",
                parameters=[
                    ParameterFile(
                        LaunchConfiguration("teleop_config"),
                        allow_substs=True,
                    ),
                    {
                        "session_id": LaunchConfiguration("session_id"),
                        "command_enabled": LaunchConfiguration("command_enabled"),
                    },
                ],
            ),
        ]
    )
