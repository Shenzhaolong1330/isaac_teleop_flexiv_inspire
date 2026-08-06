"""Single-site runtime configuration and immutable per-episode snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from flexiv_rdk_daemon.configuration import (
    CARTESIAN_LIMIT_KEYS,
    load_daemon_configuration,
)
from flexiv_rdk_daemon.configuration import (
    ConfigurationError as DaemonConfigurationError,
)


class SystemConfigError(ValueError):
    pass


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemConfigError(f"{name} must be a mapping")
    return value


def _positive(value: Any, name: str, *, upper: float = 1000.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemConfigError(f"{name} must be a number") from exc
    if not 0.0 < result <= upper:
        raise SystemConfigError(f"{name} must be in (0,{upper}]")
    return result


def _vector(value: Any, name: str, size: int) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise SystemConfigError(f"{name} must be a {size}-element list")
    result = [float(item) for item in value]
    if not all(abs(item) < 1e6 for item in result):
        raise SystemConfigError(f"{name} must contain finite practical values")
    return result


def canonical_hash(document: Mapping[str, Any]) -> str:
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _merge_document(target: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    for key, value in incoming.items():
        if key not in target:
            target[key] = value
        elif isinstance(target[key], dict) and isinstance(value, Mapping):
            _merge_document(target[key], value)
        else:
            raise SystemConfigError(f"duplicate composed config key: {key}")


def _load_composed_document(source: Path) -> dict[str, Any]:
    entry = _mapping(
        yaml.safe_load(source.read_text(encoding="utf-8")), "system config"
    )
    includes = entry.pop("includes", [])
    if not isinstance(includes, list) or not all(
        isinstance(item, str) and item.strip() for item in includes
    ):
        raise SystemConfigError("includes must be a list of non-empty paths")
    result: dict[str, Any] = {}
    for raw in includes:
        fragment_path = (source.parent / raw).resolve(strict=True)
        fragment = _mapping(
            yaml.safe_load(fragment_path.read_text(encoding="utf-8")),
            f"config fragment {raw}",
        )
        if "includes" in fragment or "schema_version" in fragment:
            raise SystemConfigError(
                f"config fragment {raw} cannot declare includes/schema_version"
            )
        _merge_document(result, fragment)
    _merge_document(result, entry)
    return result


@dataclass(frozen=True)
class SystemConfig:
    path: Path
    document: dict[str, Any]
    sha256: str

    @property
    def root(self) -> Path:
        candidate = self.path.parent.parent
        return (
            candidate if (candidate / "pyproject.toml").is_file() else self.path.parent
        )

    def resolve(self, raw: str) -> Path:
        candidate = Path(raw).expanduser()
        return (
            candidate if candidate.is_absolute() else (self.root / candidate).resolve()
        )


def load_system_config(path: str | Path) -> SystemConfig:
    source = Path(path).expanduser().resolve(strict=True)
    root = _load_composed_document(source)
    if int(root.get("schema_version", 0)) != 1:
        raise SystemConfigError("schema_version must be 1")
    session = _mapping(root.get("session"), "session")
    if not str(session.get("id", "")).strip():
        raise SystemConfigError("session.id is required")
    if (
        not str(session.get("runtime_root", "")).strip()
        or not str(session.get("sessions_root", "")).strip()
    ):
        raise SystemConfigError(
            "session.runtime_root and session.sessions_root are required"
        )
    sampling = _mapping(root.get("sampling"), "sampling")
    for key in (
        "arm_observation_hz",
        "teleop_command_hz",
        "hand_state_hz",
        "hand_daemon_sync_hz",
        "tactile_hz",
        "camera_hz",
        "policy_observation_hz",
        "training_timeline_hz",
    ):
        _positive(sampling.get(key), f"sampling.{key}")
    if sampling.get("action_label") != "sent_command":
        raise SystemConfigError(
            "sampling.action_label must be sent_command for an executed-action dataset"
        )
    pedal = _mapping(root.get("pedal"), "pedal")
    if not str(pedal.get("device", "")).strip():
        raise SystemConfigError("pedal.device is required")
    codes = [
        int(pedal.get(key, -1))
        for key in ("rerecord_key_code", "enable_key_code", "record_toggle_key_code")
    ]
    if len(set(codes)) != 3 or any(code <= 0 for code in codes):
        raise SystemConfigError(
            "pedal key codes must be three distinct positive Linux input codes"
        )
    flexiv = _mapping(root.get("flexiv"), "flexiv")
    for key in ("rdk_config", "tool_payload_config", "frame_config"):
        if not str(flexiv.get(key, "")).strip():
            raise SystemConfigError(f"flexiv.{key} is required")
    policy_control = _mapping(
        flexiv.get("policy_control"), "flexiv.policy_control"
    )
    if not isinstance(policy_control.get("require_pedal"), bool):
        raise SystemConfigError(
            "flexiv.policy_control.require_pedal must be a bool"
        )
    safety = _mapping(flexiv.get("safety"), "flexiv.safety")
    if not isinstance(safety.get("software_safety_limits_enabled"), bool):
        raise SystemConfigError(
            "flexiv.safety.software_safety_limits_enabled must be a bool"
        )
    if not isinstance(safety.get("require_hands_for_arm_control"), bool):
        raise SystemConfigError(
            "flexiv.safety.require_hands_for_arm_control must be a bool"
        )
    _positive(
        safety.get("hand_observation_timeout_ms"),
        "flexiv.safety.hand_observation_timeout_ms",
    )
    joint_lower = _vector(
        safety.get("joint_lower_limits_rad"),
        "flexiv.safety.joint_lower_limits_rad",
        7,
    )
    joint_upper = _vector(
        safety.get("joint_upper_limits_rad"),
        "flexiv.safety.joint_upper_limits_rad",
        7,
    )
    if any(lower >= upper for lower, upper in zip(joint_lower, joint_upper)):
        raise SystemConfigError(
            "flexiv.safety joint lower limits must be smaller than upper limits"
        )
    safety_limit_names = (
        "max_joint_velocity_rad_s",
        "max_tcp_linear_speed_m_s",
        "max_tcp_angular_speed_rad_s",
        "max_external_force_n",
        "max_external_torque_nm",
        "max_joint_temperature_c",
        "hand_reference_tolerance",
        "max_translation_step_m",
        "max_rotation_step_rad",
        "max_linear_velocity_m_s",
        "max_angular_velocity_rad_s",
        "max_linear_acceleration_m_s2",
        "max_angular_acceleration_rad_s2",
    )
    for key in safety_limit_names:
        _positive(safety.get(key), f"flexiv.safety.{key}")
    if float(safety["max_linear_velocity_m_s"]) > float(
        safety["max_tcp_linear_speed_m_s"]
    ):
        raise SystemConfigError(
            "flexiv.safety.max_linear_velocity_m_s exceeds max_tcp_linear_speed_m_s"
        )
    if float(safety["max_angular_velocity_rad_s"]) > float(
        safety["max_tcp_angular_speed_rad_s"]
    ):
        raise SystemConfigError(
            "flexiv.safety.max_angular_velocity_rad_s exceeds max_tcp_angular_speed_rad_s"
        )
    config_root = (
        source.parent.parent
        if (source.parent.parent / "pyproject.toml").is_file()
        else source.parent
    )
    daemon_path = Path(str(flexiv["rdk_config"])).expanduser()
    if not daemon_path.is_absolute():
        daemon_path = (config_root / daemon_path).resolve()
    try:
        daemon_config = load_daemon_configuration(daemon_path)
    except (OSError, DaemonConfigurationError, yaml.YAMLError) as exc:
        raise SystemConfigError(f"flexiv.rdk_config is invalid: {exc}") from exc
    site_cartesian_limits = tuple(float(safety[name]) for name in CARTESIAN_LIMIT_KEYS)
    for name, requested, ceiling in zip(
        CARTESIAN_LIMIT_KEYS,
        site_cartesian_limits,
        daemon_config.cartesian_limits,
        strict=True,
    ):
        if requested > ceiling:
            raise SystemConfigError(
                f"flexiv.safety.{name}={requested} exceeds daemon ceiling {ceiling}"
            )
    cartesian = _mapping(flexiv.get("cartesian_control"), "flexiv.cartesian_control")
    if cartesian.get("mode") not in {"position", "impedance"}:
        raise SystemConfigError(
            "flexiv.cartesian_control.mode must be position or impedance"
        )
    for key in ("position_stiffness", "impedance_stiffness"):
        values = _vector(cartesian.get(key), f"flexiv.cartesian_control.{key}", 6)
        if any(item < 0.0 for item in values):
            raise SystemConfigError(
                f"flexiv.cartesian_control.{key} must be non-negative"
            )
    damping_ratio = _vector(
        cartesian.get("damping_ratio"),
        "flexiv.cartesian_control.damping_ratio",
        6,
    )
    if any(item < 0.3 or item > 0.8 for item in damping_ratio):
        raise SystemConfigError(
            "flexiv.cartesian_control.damping_ratio must be in [0.3,0.8]"
        )
    home = _mapping(flexiv.get("home"), "flexiv.home")
    for side in ("left", "right"):
        positions = _vector(
            home.get(f"{side}_joints_rad"),
            f"flexiv.home.{side}_joints_rad",
            7,
        )
        if any(
            position < lower or position > upper
            for position, lower, upper in zip(positions, joint_lower, joint_upper)
        ):
            raise SystemConfigError(
                f"flexiv.home.{side}_joints_rad is outside configured safety limits"
            )
    home_velocity = _positive(
        home.get("max_velocity_rad_s"),
        "flexiv.home.max_velocity_rad_s",
        upper=0.75,
    )
    if home_velocity > float(safety["max_joint_velocity_rad_s"]):
        raise SystemConfigError(
            "flexiv.home.max_velocity_rad_s exceeds the configured joint velocity limit"
        )
    _positive(
        home.get("max_acceleration_rad_s2"),
        "flexiv.home.max_acceleration_rad_s2",
        upper=2.0,
    )
    _positive(
        home.get("tolerance_rad"),
        "flexiv.home.tolerance_rad",
        upper=0.1,
    )
    timeout = _positive(home.get("timeout_s"), "flexiv.home.timeout_s", upper=60.0)
    if timeout < 1.0:
        raise SystemConfigError("flexiv.home.timeout_s must be at least 1 second")
    if not str(home.get("quest_button", "")).strip():
        raise SystemConfigError("flexiv.home.quest_button is required")
    lift = _mapping(home.get("lift"), "flexiv.home.lift")
    if not isinstance(lift.get("enabled"), bool):
        raise SystemConfigError("flexiv.home.lift.enabled must be a bool")
    for side in ("left", "right"):
        for axis in ("x", "y"):
            target = float(lift.get(f"{side}_target_{axis}_m"))
            if not -2.0 <= target <= 2.0:
                raise SystemConfigError(
                    f"flexiv.home.lift.{side}_target_{axis}_m must be finite "
                    "and in [-2,2]"
                )
        safe_z = float(lift.get(f"{side}_safe_z_m"))
        if not -2.0 <= safe_z <= 2.0:
            raise SystemConfigError(
                f"flexiv.home.lift.{side}_safe_z_m must be finite and in [-2,2]"
            )
    lift_limits = (
        (
            "max_linear_velocity_m_s",
            "max_tcp_linear_speed_m_s",
        ),
        (
            "max_angular_velocity_rad_s",
            "max_tcp_angular_speed_rad_s",
        ),
        (
            "max_linear_acceleration_m_s2",
            "max_linear_acceleration_m_s2",
        ),
        (
            "max_angular_acceleration_rad_s2",
            "max_angular_acceleration_rad_s2",
        ),
    )
    for lift_key, safety_key in lift_limits:
        value = _positive(lift.get(lift_key), f"flexiv.home.lift.{lift_key}")
        if value > float(safety[safety_key]):
            raise SystemConfigError(
                f"flexiv.home.lift.{lift_key} exceeds flexiv.safety.{safety_key}"
            )
    _positive(
        lift.get("tolerance_m"),
        "flexiv.home.lift.tolerance_m",
        upper=0.05,
    )
    lift_timeout = _positive(
        lift.get("timeout_s"),
        "flexiv.home.lift.timeout_s",
        upper=30.0,
    )
    if lift_timeout < 1.0:
        raise SystemConfigError("flexiv.home.lift.timeout_s must be at least 1 second")
    if not isinstance(lift.get("parallel"), bool):
        raise SystemConfigError("flexiv.home.lift.parallel must be a bool")
    inspire = _mapping(root.get("inspire"), "inspire")
    for key in ("left_host", "right_host"):
        if not str(inspire.get(key, "")).strip():
            raise SystemConfigError(f"inspire.{key} is required")
    port = int(inspire.get("port", 0))
    if not 1 <= port <= 65535:
        raise SystemConfigError("inspire.port is invalid")
    if not isinstance(inspire.get("hardware_write_enabled"), bool):
        raise SystemConfigError("inspire.hardware_write_enabled must be a bool")
    hand_reset = _mapping(inspire.get("reset"), "inspire.reset")
    if not isinstance(hand_reset.get("enabled"), bool):
        raise SystemConfigError("inspire.reset.enabled must be a bool")
    open_angle = int(hand_reset.get("open_angle", -1))
    closed_angle = int(hand_reset.get("closed_angle", -1))
    if (
        not 0 <= open_angle <= 1000
        or not 0 <= closed_angle <= 1000
        or open_angle == closed_angle
    ):
        raise SystemConfigError(
            "inspire.reset open/closed angles must be distinct values in [0,1000]"
        )
    _positive(hand_reset.get("pause_s"), "inspire.reset.pause_s", upper=2.0)
    command_timeout = _positive(
        hand_reset.get("command_timeout_s"),
        "inspire.reset.command_timeout_s",
        upper=10.0,
    )
    if command_timeout < 1.0:
        raise SystemConfigError(
            "inspire.reset.command_timeout_s must be at least 1 second"
        )
    open_tolerance = int(hand_reset.get("open_tolerance", -1))
    if not 0 <= open_tolerance <= 200:
        raise SystemConfigError("inspire.reset.open_tolerance must be in [0,200]")
    open_timeout = _positive(
        hand_reset.get("open_timeout_s"),
        "inspire.reset.open_timeout_s",
        upper=30.0,
    )
    if open_timeout < 1.0:
        raise SystemConfigError(
            "inspire.reset.open_timeout_s must be at least 1 second"
        )
    teleop = _mapping(root.get("teleop"), "teleop")
    if teleop.get("deadman_source") not in {
        "pedal",
        "external_bool",
        "quest_squeeze_both",
        "quest_squeeze_either",
    }:
        raise SystemConfigError("teleop.deadman_source is unsupported")
    manus_ergonomics = _mapping(
        teleop.get("manus_ergonomics"), "teleop.manus_ergonomics"
    )
    if not isinstance(manus_ergonomics.get("enabled"), bool):
        raise SystemConfigError("teleop.manus_ergonomics.enabled must be a bool")
    if str(manus_ergonomics.get("udp_host", "")) not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise SystemConfigError("teleop.manus_ergonomics.udp_host must be loopback")
    ergonomics_port = int(manus_ergonomics.get("udp_port", 0))
    if not 1024 <= ergonomics_port <= 65535:
        raise SystemConfigError(
            "teleop.manus_ergonomics.udp_port must be in [1024,65535]"
        )
    ergonomics_topics = [
        str(manus_ergonomics.get(f"{side}_topic", "")) for side in ("left", "right")
    ]
    if len(set(ergonomics_topics)) != 2 or any(
        not topic.startswith("/") for topic in ergonomics_topics
    ):
        raise SystemConfigError(
            "teleop.manus_ergonomics left/right topics must be distinct absolute topics"
        )
    teleop_mapping = _mapping(teleop.get("mapping"), "teleop.mapping")
    axis_rotation = _vector(
        teleop_mapping.get("axis_rotation"),
        "teleop.mapping.axis_rotation",
        9,
    )
    axis = [axis_rotation[index : index + 3] for index in range(0, 9, 3)]
    for row in range(3):
        for column in range(3):
            dot = sum(axis[index][row] * axis[index][column] for index in range(3))
            expected = 1.0 if row == column else 0.0
            if abs(dot - expected) > 1.0e-7:
                raise SystemConfigError(
                    "teleop.mapping.axis_rotation must be orthonormal"
                )
    determinant = (
        axis[0][0] * (axis[1][1] * axis[2][2] - axis[1][2] * axis[2][1])
        - axis[0][1] * (axis[1][0] * axis[2][2] - axis[1][2] * axis[2][0])
        + axis[0][2] * (axis[1][0] * axis[2][1] - axis[1][1] * axis[2][0])
    )
    if abs(determinant - 1.0) > 1.0e-7:
        raise SystemConfigError(
            "teleop.mapping.axis_rotation must be a proper rotation"
        )
    _positive(
        teleop_mapping.get("translation_gain"),
        "teleop.mapping.translation_gain",
        upper=3.0,
    )
    _positive(
        teleop_mapping.get("rotation_gain"),
        "teleop.mapping.rotation_gain",
        upper=3.0,
    )
    left_pose_index = int(teleop_mapping.get("left_pose_index", -1))
    right_pose_index = int(teleop_mapping.get("right_pose_index", -1))
    if (
        left_pose_index < 0
        or right_pose_index < 0
        or left_pose_index == right_pose_index
    ):
        raise SystemConfigError(
            "teleop.mapping left/right pose indices must be distinct and non-negative"
        )
    export = _mapping(root.get("lerobot_export"), "lerobot_export")
    if int(export.get("schema_version", 0)) != 1:
        raise SystemConfigError("lerobot_export.schema_version must be 1")
    timeline = _mapping(export.get("timeline"), "lerobot_export.timeline")
    if not str(timeline.get("source", "")).strip():
        raise SystemConfigError("lerobot_export.timeline.source is required")
    _positive(timeline.get("fps"), "lerobot_export.timeline.fps")
    action = _mapping(export.get("action"), "lerobot_export.action")
    if action.get("view") not in {
        "sent_command",
        "absolute_joint_position",
        "absolute_cartesian_pose",
    }:
        raise SystemConfigError("lerobot_export.action.view is unsupported")
    high_rate_samples = int(export.get("high_rate_arm_samples_per_frame", 0))
    if not 0 <= high_rate_samples <= 128:
        raise SystemConfigError(
            "lerobot_export.high_rate_arm_samples_per_frame must be in [0,128]"
        )
    channels = _mapping(export.get("channels", {}), "lerobot_export.channels")
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in channels.items()
    ):
        raise SystemConfigError("lerobot_export.channels must contain string mappings")
    cameras = _mapping(root.get("cameras"), "cameras")
    depth_enabled = bool(cameras.get("depth_enabled", False))
    if depth_enabled:
        for key, lower, upper in (
            ("depth_width", 160, 1920),
            ("depth_height", 120, 1080),
            ("depth_fps", 1, 90),
        ):
            value = int(cameras.get(key, 0))
            if not lower <= value <= upper:
                raise SystemConfigError(f"cameras.{key} is outside supported bounds")
    elif bool(cameras.get("pointcloud_enabled", False)):
        raise SystemConfigError(
            "cameras.pointcloud_enabled requires cameras.depth_enabled"
        )
    if not 1 <= int(cameras.get("pointcloud_stride", 1)) <= 32:
        raise SystemConfigError("cameras.pointcloud_stride must be in [1,32]")
    if not 1 <= int(cameras.get("jpeg_quality", 0)) <= 100:
        raise SystemConfigError("cameras.jpeg_quality must be in [1,100]")
    streams = _mapping(cameras.get("streams"), "cameras.streams")
    if set(streams) != {"head", "left_wrist", "right_wrist"}:
        raise SystemConfigError(
            "cameras.streams must contain exactly head, left_wrist, right_wrist"
        )
    for name, stream_value in streams.items():
        stream = _mapping(stream_value, f"cameras.streams.{name}")
        recording = _mapping(
            stream.get("recording", {}), f"cameras.streams.{name}.recording"
        )
        if not bool(recording.get("rgb", True)):
            raise SystemConfigError(
                f"cameras.streams.{name}.recording.rgb must be true"
            )
        if bool(recording.get("pointcloud", False)) and not bool(
            recording.get("depth", depth_enabled)
        ):
            raise SystemConfigError(
                f"cameras.streams.{name}.recording.pointcloud requires depth"
            )
        if not str(stream.get("serial", "")).strip():
            raise SystemConfigError(f"cameras.streams.{name}.serial is required")
        if (
            not 160 <= int(stream.get("width", 0)) <= 1920
            or not 120 <= int(stream.get("height", 0)) <= 1080
        ):
            raise SystemConfigError(
                f"cameras.streams.{name} resolution is outside supported bounds"
            )
        _positive(stream.get("fps"), f"cameras.streams.{name}.fps", upper=90.0)
        if float(sampling["camera_hz"]) > float(stream.get("fps", 0.0)):
            raise SystemConfigError(
                f"sampling.camera_hz cannot exceed cameras.streams.{name}.fps"
            )
        if stream.get("pixel_format") != "rgb8":
            raise SystemConfigError(f"cameras.streams.{name}.pixel_format must be rgb8")
        preset = str(stream.get("depth_visual_preset", "unchanged"))
        if preset not in {
            "unchanged",
            "custom",
            "default",
            "hand",
            "high_accuracy",
            "high_density",
            "medium_density",
        }:
            raise SystemConfigError(
                f"cameras.streams.{name}.depth_visual_preset is unsupported"
            )
        for key in (
            "depth_spatial_filter_enabled",
            "depth_temporal_filter_enabled",
        ):
            if key in stream and not isinstance(stream[key], bool):
                raise SystemConfigError(f"cameras.streams.{name}.{key} must be a bool")
        if "depth_emitter_enabled" in stream and not isinstance(
            stream["depth_emitter_enabled"], bool
        ):
            raise SystemConfigError(
                f"cameras.streams.{name}.depth_emitter_enabled must be a bool"
            )
        laser_power = stream.get("depth_laser_power")
        if laser_power is not None and not 0.0 <= float(laser_power) <= 360.0:
            raise SystemConfigError(
                f"cameras.streams.{name}.depth_laser_power must be in [0,360]"
            )
        extrinsics = str(stream.get("extrinsics", "")).strip()
        if extrinsics and Path(extrinsics).suffix.lower() not in {".yaml", ".yml"}:
            raise SystemConfigError(
                f"cameras.streams.{name}.extrinsics must be a YAML file"
            )
    xr = _mapping(root.get("xr_video"), "xr_video")
    if xr.get("transport") not in {"lan", "usb_tcp"}:
        raise SystemConfigError("xr_video.transport must be lan or usb_tcp")
    if xr.get("transport") == "lan" and not str(xr.get("wifi_connection", "")).strip():
        raise SystemConfigError(
            "xr_video.wifi_connection is required for lan transport"
        )
    if xr.get("encoder") not in {"auto", "h264_nvenc", "libx264"}:
        raise SystemConfigError("xr_video.encoder must be auto, h264_nvenc, or libx264")
    if (
        not str(xr.get("ffmpeg", "")).strip()
        or not str(xr.get("receiver_host", "")).strip()
    ):
        raise SystemConfigError("xr_video.ffmpeg and receiver_host are required")
    _positive(xr.get("bitrate_mbps"), "xr_video.bitrate_mbps", upper=100.0)
    cloudxr_client = _mapping(xr.get("cloudxr_client"), "xr_video.cloudxr_client")
    per_eye_width = int(cloudxr_client.get("per_eye_width", 0))
    per_eye_height = int(cloudxr_client.get("per_eye_height", 0))
    if per_eye_width < 128 or per_eye_width % 16:
        raise SystemConfigError(
            "xr_video.cloudxr_client.per_eye_width must be >=128 and divisible by 16"
        )
    if per_eye_height < 128 or per_eye_height % 64:
        raise SystemConfigError(
            "xr_video.cloudxr_client.per_eye_height must be >=128 and divisible by 64"
        )
    if int(cloudxr_client.get("frame_rate", 0)) not in {72, 90, 120}:
        raise SystemConfigError(
            "xr_video.cloudxr_client.frame_rate must be 72, 90, or 120"
        )
    if int(cloudxr_client.get("max_bitrate_mbps", 0)) not in {
        80,
        100,
        120,
        150,
        180,
        200,
    }:
        raise SystemConfigError(
            "xr_video.cloudxr_client.max_bitrate_mbps must be a supported WebXR option"
        )
    if cloudxr_client.get("codec") not in {"h264", "h265", "av1"}:
        raise SystemConfigError(
            "xr_video.cloudxr_client.codec must be h264, h265, or av1"
        )
    if not isinstance(cloudxr_client.get("enable_tex_sub_image_2d"), bool):
        raise SystemConfigError(
            "xr_video.cloudxr_client.enable_tex_sub_image_2d must be boolean"
        )
    if int(xr.get("payload_type", -1)) != 96:
        raise SystemConfigError("xr_video.payload_type must be 96 for IsaacTeleop")
    xr_streams = _mapping(xr.get("streams"), "xr_video.streams")
    if set(xr_streams) != set(streams):
        raise SystemConfigError("xr_video.streams must match cameras.streams")
    ports = []
    for name, stream_value in xr_streams.items():
        stream = _mapping(stream_value, f"xr_video.streams.{name}")
        port = int(stream.get("port", 0))
        if not 1024 <= port <= 65535:
            raise SystemConfigError(f"xr_video.streams.{name}.port is invalid")
        ports.append(port)
    if len(set(ports)) != len(ports):
        raise SystemConfigError("xr_video RTP ports must be unique")
    if not any(bool(stream.get("enabled", False)) for stream in xr_streams.values()):
        raise SystemConfigError("xr_video.streams must enable at least one camera")
    display = _mapping(xr.get("display"), "xr_video.display")
    if display.get("mode") not in {"xr", "monitor"}:
        raise SystemConfigError("xr_video.display.mode must be xr or monitor")
    for name in streams:
        plane = _mapping(display.get(name), f"xr_video.display.{name}")
        for key in ("distance", "width"):
            _positive(plane.get(key), f"xr_video.display.{name}.{key}", upper=10.0)
    recording = _mapping(root.get("recording"), "recording")
    dataset_name = str(recording.get("dataset_name", "")).strip()
    if (
        not dataset_name
        or len(dataset_name) > 96
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in dataset_name
        )
    ):
        raise SystemConfigError("recording.dataset_name is invalid")
    task_name = str(recording.get("task_name", "")).strip()
    if (
        not task_name
        or len(task_name) > 96
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in task_name
        )
    ):
        raise SystemConfigError("recording.task_name is invalid")
    if not str(recording.get("output_root", "")).strip():
        raise SystemConfigError("recording.output_root is required")
    count = int(recording.get("episode_count", 0))
    if not 1 <= count <= 100_000:
        raise SystemConfigError("recording.episode_count must be in [1,100000]")
    if not str(recording.get("task_description", "")).strip():
        raise SystemConfigError("recording.task_description is required")
    if recording.get("camera_recording_mode") != "jpeg":
        raise SystemConfigError("recording.camera_recording_mode must be jpeg")
    if recording.get("deviceio_mode") != "native":
        raise SystemConfigError("recording.deviceio_mode must be native")
    if not isinstance(recording.get("ros_mcap_enabled"), bool):
        raise SystemConfigError("recording.ros_mcap_enabled must be a bool")
    if recording.get("deviceio_profile") not in {"training", "full"}:
        raise SystemConfigError("recording.deviceio_profile must be training or full")
    if not isinstance(recording.get("record_only_while_pedal_pressed"), bool):
        raise SystemConfigError(
            "recording.record_only_while_pedal_pressed must be a bool"
        )
    if not isinstance(recording.get("auto_reset_before_record"), bool):
        raise SystemConfigError("recording.auto_reset_before_record must be a bool")
    live_rerun = _mapping(recording.get("live_rerun", {}), "recording.live_rerun")
    if not isinstance(live_rerun.get("enabled"), bool):
        raise SystemConfigError("recording.live_rerun.enabled must be a bool")
    viewer_port = int(live_rerun.get("viewer_port", 0))
    if not 1 <= viewer_port <= 65535:
        raise SystemConfigError("recording.live_rerun.viewer_port must be a TCP port")
    for field in (
        "telemetry_hz",
        "tactile_hz",
        "image_hz",
        "pointcloud_hz",
    ):
        value = float(live_rerun.get(field, 0.0))
        if not 0.1 <= value <= 120.0:
            raise SystemConfigError(
                f"recording.live_rerun.{field} must be in [0.1,120]"
            )
    return SystemConfig(source, root, canonical_hash(root))


def render_runtime_configs(config: SystemConfig, output: str | Path) -> dict[str, Path]:
    """Materialize child configs; their only source of operator parameters is config."""
    out = Path(output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = config.document
    sampling, pedal, flexiv = root["sampling"], root["pedal"], root["flexiv"]
    safety = flexiv["safety"]
    cameras, inspire, teleop = root["cameras"], root["inspire"], root["teleop"]
    xr = root["xr_video"]
    session = root["session"]
    result: dict[str, Path] = {}
    camera = {
        "schema_version": 1,
        "librealsense_version": "2.57.7",
        "firmware_policy": "preserve",
        "recording": {
            "encoding": cameras["recording_encoding"],
            "jpeg_quality": cameras["jpeg_quality"],
            "depth_enabled": bool(cameras["depth_enabled"]),
        },
        "cameras": {},
    }
    for name, stream in cameras["streams"].items():
        recording = stream.get("recording", {})
        extrinsics = str(stream.get("extrinsics", "")).strip()
        camera["cameras"][name] = {
            **stream,
            "extrinsics": str(config.resolve(extrinsics)) if extrinsics else "",
            "jpeg_quality": cameras["jpeg_quality"],
            "depth_enabled": bool(recording.get("depth", cameras["depth_enabled"]))
            or bool(
                recording.get("pointcloud", cameras.get("pointcloud_enabled", False))
            ),
            "depth_width": int(cameras.get("depth_width", stream["width"])),
            "depth_height": int(cameras.get("depth_height", stream["height"])),
            "depth_fps": int(cameras.get("depth_fps", stream["fps"])),
            "pointcloud_enabled": bool(
                recording.get("pointcloud", cameras.get("pointcloud_enabled", False))
            ),
            "pointcloud_stride": int(
                recording.get("pointcloud_stride", cameras.get("pointcloud_stride", 2))
            ),
            "fps": int(stream["fps"]),
            "recording_hz": float(sampling["camera_hz"]),
        }
    xr_params = {
        "enabled": bool(xr["enabled"]),
        "ffmpeg": xr["ffmpeg"],
        "encoder": xr["encoder"],
        "receiver_host": xr["receiver_host"],
        "bitrate_mbps": float(xr["bitrate_mbps"]),
        "keyframe_interval_frames": int(xr["keyframe_interval_frames"]),
        "packet_size": int(xr["packet_size"]),
        "payload_type": int(xr["payload_type"]),
    }
    isaac_cameras = {}
    for name, camera_stream in cameras["streams"].items():
        xr_stream = xr["streams"][name]
        xr_params[f"streams.{name}.enabled"] = bool(xr_stream["enabled"])
        xr_params[f"streams.{name}.topic"] = f"/camera/{name}/color/frame"
        xr_params[f"streams.{name}.port"] = int(xr_stream["port"])
        xr_params[f"streams.{name}.fps"] = float(sampling["camera_hz"])
        if bool(xr_stream["enabled"]):
            isaac_cameras[name] = {
                "type": "v4l2",
                "stereo": False,
                "device": "/dev/null",
                "width": int(camera_stream["width"]),
                "height": int(camera_stream["height"]),
                "fps": int(sampling["camera_hz"]),
                "streams": {
                    "mono": {
                        "stream_id": len(isaac_cameras),
                        "port": int(xr_stream["port"]),
                        "bitrate_mbps": float(xr["bitrate_mbps"]),
                    }
                },
            }
    display = xr["display"]
    receiver = {
        "source": "rtp",
        "streaming": {"host": xr["receiver_host"]},
        "cameras": isaac_cameras,
        "display": {
            "mode": display["mode"],
            "cuda_device": int(display["cuda_device"]),
            "monitor": {
                "width": 1920,
                "height": 1080,
                "title": "Flexiv Inspire Teleop",
                "padding": 4,
                "stream_timeout": float(xr["stream_timeout_s"]),
            },
            "xr": {
                "planes": {name: display[name] for name in isaac_cameras},
                "lock_mode": display["lock_mode"],
                "look_away_angle": float(display["look_away_angle"]),
                "reposition_distance": float(display["reposition_distance"]),
                "reposition_delay": float(display["reposition_delay"]),
                "transition_duration": float(display["transition_duration"]),
            },
        },
    }
    payloads = {
        "camera.yaml": camera,
        "xr_bridge.yaml": {"flexiv_inspire_xr_bridge": {"ros__parameters": xr_params}},
        "isaac_camera_receiver.yaml": receiver,
        "dftp.yaml": {
            "flexiv_inspire_dftp_driver": {
                "ros__parameters": {
                    "left_host": inspire["left_host"],
                    "right_host": inspire["right_host"],
                    "port": inspire["port"],
                    "state_hz": sampling["hand_state_hz"],
                    "tactile_hz": sampling["tactile_hz"],
                    "hardware_write_enabled": bool(inspire["hardware_write_enabled"]),
                    "local_session_id": session["id"],
                    "local_write_confirmation": (
                        "DFTP-LOCAL-CONTROL-AUTHORIZED"
                        if bool(inspire["hardware_write_enabled"])
                        else ""
                    ),
                    "hand_reset_enabled": bool(inspire["reset"]["enabled"]),
                    "hand_reset_open_angle": int(inspire["reset"]["open_angle"]),
                    "hand_reset_closed_angle": int(inspire["reset"]["closed_angle"]),
                    "hand_reset_pause_s": float(inspire["reset"]["pause_s"]),
                    "hand_reset_command_timeout_s": float(
                        inspire["reset"]["command_timeout_s"]
                    ),
                    "hand_reset_open_tolerance": int(
                        inspire["reset"]["open_tolerance"]
                    ),
                    "hand_reset_open_timeout_s": float(
                        inspire["reset"]["open_timeout_s"]
                    ),
                    # State is still acquired at the configured native rate;
                    # this only tolerates short scheduler/Modbus jitter.
                    "hand_state_timeout_ms": float(
                        safety["hand_observation_timeout_ms"]
                    ),
                }
            }
        },
        "control_bridge.yaml": {
            "/**": {
                "ros__parameters": {
                    "session_id": session["id"],
                    "frame_config": str(config.resolve(flexiv["frame_config"])),
                    "foot_pedal": pedal["device"],
                    "observation_rate_hz": sampling["arm_observation_hz"],
                    "hand_daemon_sync_hz": sampling["hand_daemon_sync_hz"],
                    "hand_observation_timeout_ms": float(
                        safety["hand_observation_timeout_ms"]
                    ),
                    "require_hands_for_arm_control": bool(
                        safety["require_hands_for_arm_control"]
                    ),
                    "software_safety_limits_enabled": bool(
                        safety["software_safety_limits_enabled"]
                    ),
                    "policy_pedal_required": bool(
                        flexiv["policy_control"]["require_pedal"]
                    ),
                    "joint_lower_limits_rad": safety["joint_lower_limits_rad"],
                    "joint_upper_limits_rad": safety["joint_upper_limits_rad"],
                    "max_joint_velocity_rad_s": safety["max_joint_velocity_rad_s"],
                    "max_tcp_linear_speed_m_s": safety["max_tcp_linear_speed_m_s"],
                    "max_tcp_angular_speed_rad_s": safety[
                        "max_tcp_angular_speed_rad_s"
                    ],
                    "max_external_force_n": safety["max_external_force_n"],
                    "max_external_torque_nm": safety["max_external_torque_nm"],
                    "max_joint_temperature_c": safety["max_joint_temperature_c"],
                    "hand_reference_tolerance": safety["hand_reference_tolerance"],
                    "max_translation_step_m": safety["max_translation_step_m"],
                    "max_rotation_step_rad": safety["max_rotation_step_rad"],
                    "max_linear_velocity_m_s": safety["max_linear_velocity_m_s"],
                    "max_angular_velocity_rad_s": safety["max_angular_velocity_rad_s"],
                    "max_linear_acceleration_m_s2": safety[
                        "max_linear_acceleration_m_s2"
                    ],
                    "max_angular_acceleration_rad_s2": safety[
                        "max_angular_acceleration_rad_s2"
                    ],
                    "enable_key_code": pedal["enable_key_code"],
                    "cartesian_control_mode": flexiv["cartesian_control"]["mode"],
                    "cartesian_position_stiffness": [
                        float(value)
                        for value in flexiv["cartesian_control"]["position_stiffness"]
                    ],
                    "cartesian_impedance_stiffness": [
                        float(value)
                        for value in flexiv["cartesian_control"]["impedance_stiffness"]
                    ],
                    "cartesian_damping_ratio": [
                        float(value)
                        for value in flexiv["cartesian_control"]["damping_ratio"]
                    ],
                    "home_left_joints_rad": flexiv["home"]["left_joints_rad"],
                    "home_right_joints_rad": flexiv["home"]["right_joints_rad"],
                    "home_max_velocity_rad_s": flexiv["home"]["max_velocity_rad_s"],
                    "home_max_acceleration_rad_s2": flexiv["home"][
                        "max_acceleration_rad_s2"
                    ],
                    "home_tolerance_rad": flexiv["home"]["tolerance_rad"],
                    "home_timeout_s": flexiv["home"]["timeout_s"],
                    "home_lift_enabled": bool(flexiv["home"]["lift"]["enabled"]),
                    "home_lift_left_target_x_m": float(
                        flexiv["home"]["lift"]["left_target_x_m"]
                    ),
                    "home_lift_left_target_y_m": float(
                        flexiv["home"]["lift"]["left_target_y_m"]
                    ),
                    "home_lift_left_safe_z_m": float(
                        flexiv["home"]["lift"]["left_safe_z_m"]
                    ),
                    "home_lift_right_target_x_m": float(
                        flexiv["home"]["lift"]["right_target_x_m"]
                    ),
                    "home_lift_right_target_y_m": float(
                        flexiv["home"]["lift"]["right_target_y_m"]
                    ),
                    "home_lift_right_safe_z_m": float(
                        flexiv["home"]["lift"]["right_safe_z_m"]
                    ),
                    "home_lift_max_linear_velocity_m_s": float(
                        flexiv["home"]["lift"]["max_linear_velocity_m_s"]
                    ),
                    "home_lift_max_angular_velocity_rad_s": float(
                        flexiv["home"]["lift"]["max_angular_velocity_rad_s"]
                    ),
                    "home_lift_max_linear_acceleration_m_s2": float(
                        flexiv["home"]["lift"]["max_linear_acceleration_m_s2"]
                    ),
                    "home_lift_max_angular_acceleration_rad_s2": float(
                        flexiv["home"]["lift"]["max_angular_acceleration_rad_s2"]
                    ),
                    "home_lift_tolerance_m": float(
                        flexiv["home"]["lift"]["tolerance_m"]
                    ),
                    "home_lift_timeout_s": float(flexiv["home"]["lift"]["timeout_s"]),
                    "home_lift_parallel": bool(flexiv["home"]["lift"]["parallel"]),
                }
            }
        },
        "teleop.yaml": {
            "/**": {
                "ros__parameters": {
                    "session_id": session["id"],
                    "command_enabled": bool(teleop["control_enabled"]),
                    "control_rate_hz": sampling["teleop_command_hz"],
                    "manus_calibration": str(
                        config.resolve(teleop["manus_calibration"])
                    )
                    if str(teleop["manus_calibration"]).strip()
                    else "",
                    "manus_left_ergonomics_topic": str(
                        teleop["manus_ergonomics"]["left_topic"]
                    ),
                    "manus_right_ergonomics_topic": str(
                        teleop["manus_ergonomics"]["right_topic"]
                    ),
                    "deadman_source": teleop["deadman_source"],
                    "foot_pedal": pedal["device"],
                    "enable_key_code": pedal["enable_key_code"],
                    "left_pose_index": int(teleop["mapping"]["left_pose_index"]),
                    "right_pose_index": int(teleop["mapping"]["right_pose_index"]),
                    "axis_rotation": [
                        float(value) for value in teleop["mapping"]["axis_rotation"]
                    ],
                    "translation_gain": float(teleop["mapping"]["translation_gain"]),
                    "rotation_gain": float(teleop["mapping"]["rotation_gain"]),
                    "max_translation_step_m": safety["max_translation_step_m"],
                    "max_rotation_step_rad": safety["max_rotation_step_rad"],
                    "home_button_key": flexiv["home"]["quest_button"],
                    "home_topic": "/episode/control",
                }
            }
        },
        "pedal.yaml": {
            "/**": {
                "ros__parameters": {
                    "foot_pedal": pedal["device"],
                    "rerecord_key_code": pedal["rerecord_key_code"],
                    "enable_key_code": pedal["enable_key_code"],
                    "record_toggle_key_code": pedal["record_toggle_key_code"],
                }
            }
        },
        "lerobot_export.yaml": {
            **root["lerobot_export"],
            "timeline": {
                **root["lerobot_export"]["timeline"],
                "fps": float(sampling["training_timeline_hz"]),
            },
        },
    }
    for name, payload in payloads.items():
        path = out / name
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        result[name] = path
    snapshot = out / "system.resolved.yaml"
    snapshot.write_text(yaml.safe_dump(root, sort_keys=False), encoding="utf-8")
    (out / "system.sha256").write_text(config.sha256 + "\n", encoding="utf-8")
    result["snapshot"] = snapshot
    return result
