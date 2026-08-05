"""Station-independent v2 channel/action descriptors."""

from __future__ import annotations

from policy_contracts import ActionSchema, ChannelDescriptor, SystemSchema, TensorDescriptor
from policy_contracts.profiles import HAND_ACTUATORS, JOINT_MINIMAL_PROFILE


ARM_JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 8))
POSE_XYZW_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw")
TWIST_NAMES = ("vx", "vy", "vz", "wx", "wy", "wz")
WRENCH_NAMES = ("fx", "fy", "fz", "mx", "my", "mz")
NATIVE_ACTION30_NAMES = tuple(
    [
        f"{side}_{name}"
        for side in ("left", "right")
        for name in (
            "dx", "dy", "dz", "dR00", "dR10", "dR20", "dR01", "dR11", "dR21"
        )
    ]
    + [
        f"{side}_hand_{actuator}"
        for side in ("left", "right")
        for actuator in HAND_ACTUATORS
    ]
)


def build_system_schema(
    *,
    arm_rate_hz: float = 300.0,
    hand_rate_hz: float = 200.0,
    tactile_rate_hz: float = 15.0,
    camera_rate_hz: float = 15.0,
    action_rate_hz: float = 30.0,
    image_shape: tuple[int, int, int] = (240, 424, 3),
) -> SystemSchema:
    channels: list[ChannelDescriptor] = []
    for side in ("left", "right"):
        arm_prefix = f"arm.{side}"
        channels.extend(
            (
                ChannelDescriptor(
                    f"{arm_prefix}.q", "joint_position",
                    TensorDescriptor("float64", (7,), ARM_JOINT_NAMES, "rad"),
                    arm_rate_hz, frame_id=f"{side}_base",
                ),
                ChannelDescriptor(
                    f"{arm_prefix}.dq", "joint_velocity",
                    TensorDescriptor("float64", (7,), ARM_JOINT_NAMES, "rad/s"),
                    arm_rate_hz, frame_id=f"{side}_base",
                ),
                ChannelDescriptor(
                    f"{arm_prefix}.tau", "measured_joint_torque",
                    TensorDescriptor("float64", (7,), ARM_JOINT_NAMES, "Nm"),
                    arm_rate_hz, frame_id=f"{side}_base",
                ),
                ChannelDescriptor(
                    f"{arm_prefix}.tau_des", "desired_joint_torque",
                    TensorDescriptor("float64", (7,), ARM_JOINT_NAMES, "Nm"),
                    arm_rate_hz, frame_id=f"{side}_base",
                ),
                ChannelDescriptor(
                    f"{arm_prefix}.tau_ext", "external_joint_torque",
                    TensorDescriptor("float64", (7,), ARM_JOINT_NAMES, "Nm"),
                    arm_rate_hz, frame_id=f"{side}_base",
                ),
                ChannelDescriptor(
                    f"{arm_prefix}.tau_interact", "interaction_joint_torque",
                    TensorDescriptor("float64", (7,), ARM_JOINT_NAMES, "Nm"),
                    arm_rate_hz, frame_id=f"{side}_base",
                ),
                ChannelDescriptor(
                    f"{arm_prefix}.tcp_pose", "cartesian_pose_xyzw",
                    TensorDescriptor("float64", (7,), POSE_XYZW_NAMES, "m+unit_quaternion"),
                    arm_rate_hz, frame_id="world",
                ),
                ChannelDescriptor(
                    f"{arm_prefix}.tcp_twist", "cartesian_twist",
                    TensorDescriptor("float64", (6,), TWIST_NAMES, "m/s+rad/s"),
                    arm_rate_hz, frame_id="world",
                ),
                ChannelDescriptor(
                    f"{arm_prefix}.raw_ft", "raw_force_torque",
                    TensorDescriptor("float64", (6,), WRENCH_NAMES, "N+Nm"),
                    arm_rate_hz, frame_id=f"{side}_tool",
                ),
                ChannelDescriptor(
                    f"{arm_prefix}.tcp_wrench", "tcp_wrench",
                    TensorDescriptor("float64", (6,), WRENCH_NAMES, "N+Nm"),
                    arm_rate_hz, frame_id=f"{side}_tcp",
                ),
            )
        )
        hand_prefix = f"hand.{side}"
        channels.extend(
            (
                ChannelDescriptor(
                    f"{hand_prefix}.angle", "hand_actuator_angle",
                    TensorDescriptor("float64", (6,), HAND_ACTUATORS, "raw_0_1000"),
                    hand_rate_hz, frame_id=f"{side}_hand",
                ),
                ChannelDescriptor(
                    f"{hand_prefix}.actual_force", "hand_actuator_force",
                    TensorDescriptor("float64", (6,), HAND_ACTUATORS, "g"),
                    hand_rate_hz, frame_id=f"{side}_hand",
                ),
                ChannelDescriptor(
                    f"{hand_prefix}.tactile", "tactile_taxels",
                    TensorDescriptor("uint16", (1062,), (), "raw_uint16"),
                    tactile_rate_hz, frame_id=f"{side}_hand",
                ),
            )
        )
    height, width, channels_count = image_shape
    for camera in ("head", "left_wrist", "right_wrist"):
        channels.append(
            ChannelDescriptor(
                f"camera.{camera}.rgb",
                "rgb_image",
                TensorDescriptor("uint8", image_shape, (), "srgb"),
                camera_rate_hz,
                frame_id=f"{camera}_color_optical_frame",
                encodings=("jpeg", "rgb8"),
            )
        )
    actions = (
        ActionSchema(
            "flexiv_inspire_native_rot6d_v1",
            TensorDescriptor("float32", (30,), NATIVE_ACTION30_NAMES),
            "world",
            "delta_xyz_rot6d+absolute_hand_0_1000",
            action_rate_hz,
            True,
        ),
        ActionSchema(
            "cartesian_delta_rotvec_v1",
            TensorDescriptor(
                "float32", (24,), JOINT_MINIMAL_PROFILE.action_names
            ),
            "world",
            "delta_xyz_rotvec+absolute_hand_0_1",
            action_rate_hz,
            True,
        ),
    )
    return SystemSchema(
        schema_version=2,
        system_id="flexiv-inspire-site",
        robot_type="flexiv_rizon4s_dual_inspire_dftp2",
        channels=tuple(channels),
        action_schemas=actions,
        metadata={
            "rotation6d_order": "R00,R10,R20,R01,R11,R21",
            "hardware_owner": "flexiv_inspire_control",
            "transport_metadata_outside_policy_tensor": "true",
        },
    )
