"""Strict local configuration and tool/payload identity handling.

The daemon owns these files. Remote clients may echo an expected digest, but
they cannot select or redefine the configuration used by the hardware process.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    pass


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ConfigurationError(f"{name} must be finite and non-negative")
    return result


CARTESIAN_LIMIT_KEYS = (
    "max_linear_velocity_m_s",
    "max_angular_velocity_rad_s",
    "max_linear_acceleration_m_s2",
    "max_angular_acceleration_rad_s2",
)


@dataclass(frozen=True)
class ConfiguredRobot:
    side: str
    serial: str
    expected_model: str
    expected_active_tool: str
    require_ft_sensor: bool
    expected_software_prefix: str
    required_license: str


@dataclass(frozen=True)
class CaptureBaseTransform:
    """``world_T_base`` used by the native DeviceIO capture path.

    The RDK reports Cartesian quantities in each controller's local base.
    Keeping this small representation in the RDK-only package lets the raw
    recorder add a shared frame without depending on the ROS control bridge.
    """

    translation_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class DaemonConfiguration:
    source_path: Path
    schema_version: int
    robots: tuple[ConfiguredRobot, ConfiguredRobot]
    target_rate_hz: float
    cartesian_limits: tuple[float, float, float, float]
    socket_name: str
    max_packet_bytes: int
    tool_payload_path: Path
    ft_zero: dict[str, Any]
    capture_frame_path: Path
    world_frame: str
    world_from_base: dict[str, CaptureBaseTransform]


@dataclass(frozen=True)
class ToolPayloadIdentity:
    """Canonical digest plus file identity used to invalidate F/T-zero state."""

    sha256: str
    fingerprint: str
    mtime_ns: int
    size: int
    source_path: Path
    canonical_json: str


def _finite_vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ConfigurationError(f"{name} must contain {size} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ConfigurationError(f"{name} must contain only finite values")
    return result


def _load_capture_frames(
    source: Path, root: dict[str, Any]
) -> tuple[Path, str, dict[str, CaptureBaseTransform]]:
    """Read the same measured base transforms used by the ROS bridge.

    The path is deliberately relative to ``robots.yaml`` so a site deployment
    remains self-contained and the hardware daemon never accepts frame data
    from a client request.
    """

    frame_value = str(root.get("capture_frame_config", "")).strip()
    if not frame_value:
        raise ConfigurationError("capture_frame_config is required")
    frame_path = Path(frame_value)
    if frame_path.is_absolute():
        raise ConfigurationError("capture_frame_config must be relative")
    resolved = (source.parent / frame_path).resolve(strict=True)
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    document = _mapping(raw, "capture frame configuration")
    if int(document.get("schema_version", 0)) != 1:
        raise ConfigurationError("capture frame config must be schema_version: 1")
    world_frame = str(document.get("world_frame", "")).strip()
    bases = _mapping(document.get("bases"), "capture frame bases")
    if not world_frame or set(bases) != {"left", "right"}:
        raise ConfigurationError(
            "capture frame config requires world_frame and left/right bases"
        )
    transforms: dict[str, CaptureBaseTransform] = {}
    for side in ("left", "right"):
        entry = _mapping(bases.get(side), f"capture frame bases.{side}")
        translation = _finite_vector(
            entry.get("translation_m"), 3, f"capture frame bases.{side}.translation_m"
        )
        quaternion = _finite_vector(
            entry.get("rotation_xyzw"), 4, f"capture frame bases.{side}.rotation_xyzw"
        )
        norm = math.sqrt(sum(component * component for component in quaternion))
        if norm <= 1e-12:
            raise ConfigurationError(
                f"capture frame bases.{side}.rotation_xyzw must be non-zero"
            )
        transforms[side] = CaptureBaseTransform(
            translation_m=(translation[0], translation[1], translation[2]),
            rotation_xyzw=tuple(component / norm for component in quaternion),
        )
    return resolved, world_frame, transforms


def load_daemon_configuration(path: str | Path) -> DaemonConfiguration:
    source = Path(path).expanduser().resolve(strict=True)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "daemon configuration")
    schema = int(root.get("schema_version", 0))
    if schema != 1:
        raise ConfigurationError(f"unsupported daemon schema_version {schema}")
    capture_frame_path, world_frame, world_from_base = _load_capture_frames(source, root)

    robots: list[ConfiguredRobot] = []
    serials: set[str] = set()
    for side in ("left", "right"):
        item = _mapping(root.get(side), side)
        serial = str(item.get("serial", "")).strip()
        if not serial:
            raise ConfigurationError(f"{side}.serial is required")
        if serial in serials:
            raise ConfigurationError("left/right robot serials must differ")
        serials.add(serial)
        expected_model = str(item.get("expected_model", "")).strip()
        if not expected_model:
            raise ConfigurationError(f"{side}.expected_model is required")
        expected_active_tool = str(item.get("expected_active_tool", "")).strip()
        if not expected_active_tool:
            raise ConfigurationError(f"{side}.expected_active_tool is required")
        expected_software_prefix = str(
            item.get("expected_software_prefix", "")
        ).strip()
        if not expected_software_prefix:
            raise ConfigurationError(
                f"{side}.expected_software_prefix is required"
            )
        required_license = str(item.get("required_license", "")).strip()
        if not required_license:
            raise ConfigurationError(f"{side}.required_license is required")
        robots.append(
            ConfiguredRobot(
                side=side,
                serial=serial,
                expected_model=expected_model,
                expected_active_tool=expected_active_tool,
                require_ft_sensor=bool(item.get("require_ft_sensor", True)),
                expected_software_prefix=expected_software_prefix,
                required_license=required_license,
            )
        )

    control = _mapping(root.get("control"), "control")
    rate = float(control.get("target_rate_hz", 0.0))
    if not math.isfinite(rate) or not 1.0 <= rate <= 1000.0:
        raise ConfigurationError("control.target_rate_hz must be in [1,1000]")
    if str(control.get("default_mode", "")) != "observe_only":
        raise ConfigurationError("control.default_mode must be observe_only")
    if bool(control.get("hardware_writes_enabled", False)):
        raise ConfigurationError(
            "config cannot enable hardware writes; use the guarded local CLI flow"
        )
    raw_cartesian_limits = _mapping(
        control.get("cartesian_limits"), "control.cartesian_limits"
    )
    cartesian_limits = tuple(
        _finite_nonnegative(
            raw_cartesian_limits.get(name), f"control.cartesian_limits.{name}"
        )
        for name in CARTESIAN_LIMIT_KEYS
    )
    if any(value <= 0.0 for value in cartesian_limits):
        raise ConfigurationError(
            "control.cartesian_limits values must be positive"
        )
    practical_maxima = (1.0, 3.0, 5.0, 10.0)
    if any(
        value > maximum
        for value, maximum in zip(
            cartesian_limits, practical_maxima, strict=True
        )
    ):
        raise ConfigurationError(
            "control.cartesian_limits exceeds the daemon practical ceiling"
        )

    ipc = _mapping(root.get("ipc"), "ipc")
    socket_name = str(ipc.get("socket_name", "")).strip()
    if not socket_name or "/" in socket_name:
        raise ConfigurationError("ipc.socket_name must be a plain filename")
    max_packet_bytes = int(ipc.get("max_packet_bytes", 0))
    if not 4096 <= max_packet_bytes <= 1_048_576:
        raise ConfigurationError("ipc.max_packet_bytes outside safe range")

    tool_payload_value = str(root.get("tool_payload_config", "")).strip()
    if not tool_payload_value:
        raise ConfigurationError("tool_payload_config is required")
    tool_payload_path = (source.parent / tool_payload_value).resolve(strict=True)
    if tool_payload_path.parent != source.parent:
        raise ConfigurationError("tool_payload_config must stay in config directory")

    ft_zero = _mapping(root.get("ft_zero"), "ft_zero").copy()
    if not isinstance(ft_zero.get("external_contact_check_enabled"), bool):
        raise ConfigurationError(
            "ft_zero.external_contact_check_enabled must be a bool"
        )
    for name in (
        "sample_window_s",
        "sample_rate_hz",
        "operational_timeout_s",
        "primitive_timeout_s",
        "enable_settle_timeout_s",
        "enable_settle_window_s",
        "max_joint_velocity_norm",
        "max_tcp_velocity_norm",
        "max_wrench_std_force_n",
        "max_wrench_std_torque_nm",
        "max_residual_force_n",
        "max_residual_torque_nm",
        "max_pre_external_mean_force_n",
        "max_pre_external_mean_torque_nm",
        "max_pre_external_peak_force_n",
        "max_pre_external_peak_torque_nm",
        "max_hand_delta",
    ):
        ft_zero[name] = _finite_nonnegative(ft_zero.get(name), f"ft_zero.{name}")
    if ft_zero["sample_rate_hz"] <= 0.0:
        raise ConfigurationError("ft_zero.sample_rate_hz must be positive")
    if str(ft_zero.get("confirmation_token", "")) != "FLEXIV-FT-UNLOADED":
        raise ConfigurationError("unsafe ft_zero.confirmation_token")

    return DaemonConfiguration(
        source_path=source,
        schema_version=schema,
        robots=(robots[0], robots[1]),
        target_rate_hz=rate,
        cartesian_limits=cartesian_limits,
        socket_name=socket_name,
        max_packet_bytes=max_packet_bytes,
        tool_payload_path=tool_payload_path,
        ft_zero=ft_zero,
        capture_frame_path=capture_frame_path,
        world_frame=world_frame,
        world_from_base=world_from_base,
    )


def read_tool_payload_identity(path: str | Path) -> ToolPayloadIdentity:
    source = Path(path).expanduser().resolve(strict=True)
    raw_bytes = source.read_bytes()
    root = _mapping(yaml.safe_load(raw_bytes), "tool/payload configuration")
    if int(root.get("schema_version", 0)) != 1:
        raise ConfigurationError("unsupported tool/payload schema_version")
    arms = _mapping(root.get("arms"), "tool/payload arms")
    for side in ("left", "right"):
        arm = _mapping(arms.get(side), f"arms.{side}")
        tool = _mapping(arm.get("tool"), f"arms.{side}.tool")
        payload = _mapping(arm.get("payload"), f"arms.{side}.payload")
        if not str(tool.get("name", "")).strip():
            raise ConfigurationError(f"arms.{side}.tool.name is required")
        _finite_nonnegative(payload.get("mass_kg"), f"arms.{side}.payload.mass_kg")
        for field in (
            "center_of_mass_m",
            "inertia_kg_m2",
            "tcp_location_xyz_wxyz",
        ):
            values = payload.get(field)
            expected = {
                "center_of_mass_m": 3,
                "inertia_kg_m2": 6,
                "tcp_location_xyz_wxyz": 7,
            }[field]
            if not isinstance(values, list) or len(values) != expected:
                raise ConfigurationError(
                    f"arms.{side}.payload.{field} must have {expected} values"
                )
            for index, value in enumerate(values):
                number = float(value)
                if not math.isfinite(number):
                    raise ConfigurationError(
                        f"arms.{side}.payload.{field}[{index}] must be finite"
                    )
        if not bool(arm.get("locally_verified", False)):
            raise ConfigurationError(
                f"arms.{side}.locally_verified must be true after local audit"
            )

    canonical = json.dumps(
        root,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    stat_result = source.stat()
    material = (
        f"{source}:{stat_result.st_dev}:{stat_result.st_ino}:"
        f"{stat_result.st_mtime_ns}:{stat_result.st_size}:{digest}"
    )
    return ToolPayloadIdentity(
        sha256=digest,
        fingerprint=hashlib.sha256(material.encode("utf-8")).hexdigest(),
        mtime_ns=stat_result.st_mtime_ns,
        size=stat_result.st_size,
        source_path=source,
        canonical_json=canonical,
    )
