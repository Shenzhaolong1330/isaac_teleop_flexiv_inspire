"""Configuration parsing and validation for the isolated bridge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _rotation_matrix(value: Any, name: str) -> tuple[tuple[float, ...], ...]:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must be a 3x3 matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} must be orthonormal")
    determinant = float(np.linalg.det(matrix))
    if not np.isclose(determinant, 1.0, atol=1e-6):
        raise ValueError(
            f"{name} must be a proper rotation (det=+1), got det={determinant:.6f}"
        )
    return tuple(tuple(float(v) for v in row) for row in matrix)


@dataclass(frozen=True)
class SideMapping:
    pose_index: int
    translation_gain: float
    rotation_gain: float
    axis_rotation: tuple[tuple[float, ...], ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any], name: str) -> "SideMapping":
        pose_index = int(raw.get("pose_index", 0))
        if pose_index not in (0, 1):
            raise ValueError(f"{name}.pose_index must be 0 or 1")
        translation_gain = float(raw.get("translation_gain", 1.0))
        rotation_gain = float(raw.get("rotation_gain", 1.0))
        if not 0.0 < translation_gain <= 3.0:
            raise ValueError(f"{name}.translation_gain must be in (0, 3]")
        if not 0.0 < rotation_gain <= 3.0:
            raise ValueError(f"{name}.rotation_gain must be in (0, 3]")
        axis_rotation = _rotation_matrix(
            raw.get("axis_rotation", np.eye(3).tolist()),
            f"{name}.axis_rotation",
        )
        return cls(
            pose_index=pose_index,
            translation_gain=translation_gain,
            rotation_gain=rotation_gain,
            axis_rotation=axis_rotation,
        )


@dataclass(frozen=True)
class MappingConfig:
    left: SideMapping
    right: SideMapping

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MappingConfig":
        left = SideMapping.from_dict(
            _mapping(raw.get("left", {}), "mapping.left"), "mapping.left"
        )
        right = SideMapping.from_dict(
            _mapping(raw.get("right", {}), "mapping.right"), "mapping.right"
        )
        if left.pose_index == right.pose_index:
            raise ValueError("left and right pose_index must be different")
        return cls(left=left, right=right)


@dataclass(frozen=True)
class SafetyConfig:
    warmup_samples: int
    max_input_age_s: float
    max_tf_age_s: float
    max_deadman_age_s: float
    max_control_dt_s: float
    ack_timeout_s: float
    tracking_jump_translation_m: float
    tracking_jump_rotation_rad: float
    max_anchor_translation_m: float
    max_anchor_rotation_rad: float
    max_tracking_lag_translation_m: float
    max_tracking_lag_rotation_rad: float
    max_linear_speed_m_s: float
    max_angular_speed_rad_s: float
    max_linear_accel_m_s2: float
    max_angular_accel_rad_s2: float
    max_translation_step_m: float
    max_rotation_step_rad: float
    gateway_watchdog_s: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SafetyConfig":
        values = cls(
            warmup_samples=int(raw.get("warmup_samples", 5)),
            max_input_age_s=float(raw.get("max_input_age_s", 0.12)),
            max_tf_age_s=float(raw.get("max_tf_age_s", 0.12)),
            max_deadman_age_s=float(raw.get("max_deadman_age_s", 0.15)),
            max_control_dt_s=float(raw.get("max_control_dt_s", 0.05)),
            ack_timeout_s=float(raw.get("ack_timeout_s", 0.08)),
            tracking_jump_translation_m=float(
                raw.get("tracking_jump_translation_m", 0.12)
            ),
            tracking_jump_rotation_rad=float(
                raw.get("tracking_jump_rotation_rad", 0.60)
            ),
            max_anchor_translation_m=float(
                raw.get("max_anchor_translation_m", 0.35)
            ),
            max_anchor_rotation_rad=float(raw.get("max_anchor_rotation_rad", 2.2)),
            max_tracking_lag_translation_m=float(
                raw.get("max_tracking_lag_translation_m", 0.12)
            ),
            max_tracking_lag_rotation_rad=float(
                raw.get("max_tracking_lag_rotation_rad", 0.70)
            ),
            max_linear_speed_m_s=float(raw.get("max_linear_speed_m_s", 0.18)),
            max_angular_speed_rad_s=float(
                raw.get("max_angular_speed_rad_s", 0.60)
            ),
            max_linear_accel_m_s2=float(
                raw.get("max_linear_accel_m_s2", 0.80)
            ),
            max_angular_accel_rad_s2=float(
                raw.get("max_angular_accel_rad_s2", 2.5)
            ),
            max_translation_step_m=float(
                raw.get("max_translation_step_m", 0.015)
            ),
            max_rotation_step_rad=float(raw.get("max_rotation_step_rad", 0.030)),
            gateway_watchdog_s=float(raw.get("gateway_watchdog_s", 0.10)),
        )
        if values.warmup_samples < 1:
            raise ValueError("safety.warmup_samples must be >= 1")
        for field_name, value in values.__dict__.items():
            if field_name == "warmup_samples":
                continue
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"safety.{field_name} must be finite and > 0")
        if values.max_translation_step_m > 0.02 + 1e-12:
            raise ValueError(
                "safety.max_translation_step_m must not exceed the existing "
                "Flexiv limit of 0.02 m"
            )
        if values.max_rotation_step_rad > 0.04 + 1e-12:
            raise ValueError(
                "safety.max_rotation_step_rad must not exceed the existing "
                "Flexiv limit of 0.04 rad"
            )
        return values


@dataclass(frozen=True)
class DeadmanConfig:
    source: str
    squeeze_threshold: float
    external_topic: str
    gateway_pedal_required: bool
    gateway_pedal_device: str
    gateway_pedal_keys: tuple[str, ...]
    gateway_pedal_grab: bool

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DeadmanConfig":
        source = str(raw.get("source", "quest_squeeze_both")).strip().lower()
        allowed = {
            "quest_squeeze_both",
            "quest_squeeze_either",
            "external_bool",
        }
        if source not in allowed:
            raise ValueError(f"deadman.source must be one of {sorted(allowed)}")
        threshold = float(raw.get("squeeze_threshold", 0.65))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("deadman.squeeze_threshold must be in [0, 1]")
        keys = raw.get("gateway_pedal_keys", [])
        if not isinstance(keys, list):
            raise ValueError("deadman.gateway_pedal_keys must be a list")
        return cls(
            source=source,
            squeeze_threshold=threshold,
            external_topic=str(
                raw.get("external_topic", "/isaac_flexiv/deadman")
            ),
            gateway_pedal_required=bool(
                raw.get("gateway_pedal_required", True)
            ),
            gateway_pedal_device=str(raw.get("gateway_pedal_device", "")),
            gateway_pedal_keys=tuple(str(value) for value in keys),
            gateway_pedal_grab=bool(raw.get("gateway_pedal_grab", False)),
        )


@dataclass(frozen=True)
class RosConfig:
    control_rate_hz: float
    ee_topic: str
    controller_topic: str
    tf_topic: str
    world_frame: str
    left_wrist_frame: str
    right_wrist_frame: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RosConfig":
        rate = float(raw.get("control_rate_hz", 60.0))
        if not 1.0 <= rate <= 200.0:
            raise ValueError("ros.control_rate_hz must be in [1, 200]")
        return cls(
            control_rate_hz=rate,
            ee_topic=str(raw.get("ee_topic", "/xr_teleop/ee_poses")),
            controller_topic=str(
                raw.get("controller_topic", "/xr_teleop/controller_data")
            ),
            tf_topic=str(raw.get("tf_topic", "/tf")),
            world_frame=str(raw.get("world_frame", "world")),
            left_wrist_frame=str(raw.get("left_wrist_frame", "left_wrist")),
            right_wrist_frame=str(raw.get("right_wrist_frame", "right_wrist")),
        )


@dataclass(frozen=True)
class IpcConfig:
    gateway_socket: str
    ros_socket: str
    max_packet_bytes: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "IpcConfig":
        max_packet_bytes = int(raw.get("max_packet_bytes", 65536))
        if not 1024 <= max_packet_bytes <= 1_048_576:
            raise ValueError("ipc.max_packet_bytes must be in [1024, 1048576]")
        return cls(
            gateway_socket=str(
                raw.get("gateway_socket", "/tmp/isaac_flexiv_gateway.sock")
            ),
            ros_socket=str(raw.get("ros_socket", "/tmp/isaac_flexiv_ros.sock")),
            max_packet_bytes=max_packet_bytes,
        )


@dataclass(frozen=True)
class ExistingStackConfig:
    workspace: str
    source_repo: str
    robot_config: str
    conda_env: str
    disable_startup_motion: bool

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExistingStackConfig":
        return cls(
            workspace=str(raw.get("workspace", "/home/hb/flexiv_inspire_ws")),
            source_repo=str(
                raw.get(
                    "source_repo",
                    "/home/hb/flexiv_inspire_ws/src/dual_arm_teleop",
                )
            ),
            robot_config=str(
                raw.get(
                    "robot_config",
                    "/home/hb/flexiv_inspire_ws/src/dual_arm_teleop/"
                    "scripts/config/robots/flexiv_config.yaml",
                )
            ),
            conda_env=str(raw.get("conda_env", "flexiv_teleop")),
            disable_startup_motion=bool(raw.get("disable_startup_motion", True)),
        )


@dataclass(frozen=True)
class BridgeConfig:
    command_enabled: bool
    mapping: MappingConfig
    safety: SafetyConfig
    deadman: DeadmanConfig
    ros: RosConfig
    ipc: IpcConfig
    existing_stack: ExistingStackConfig
    source_path: Path

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], source_path: Path | None = None
    ) -> "BridgeConfig":
        return cls(
            command_enabled=bool(raw.get("command_enabled", False)),
            mapping=MappingConfig.from_dict(
                _mapping(raw.get("mapping", {}), "mapping")
            ),
            safety=SafetyConfig.from_dict(
                _mapping(raw.get("safety", {}), "safety")
            ),
            deadman=DeadmanConfig.from_dict(
                _mapping(raw.get("deadman", {}), "deadman")
            ),
            ros=RosConfig.from_dict(_mapping(raw.get("ros", {}), "ros")),
            ipc=IpcConfig.from_dict(_mapping(raw.get("ipc", {}), "ipc")),
            existing_stack=ExistingStackConfig.from_dict(
                _mapping(raw.get("existing_stack", {}), "existing_stack")
            ),
            source_path=source_path or Path("<memory>"),
        )


def load_config(path: str | Path) -> BridgeConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"bridge config must be a mapping: {config_path}")
    return BridgeConfig.from_dict(raw, source_path=config_path)

