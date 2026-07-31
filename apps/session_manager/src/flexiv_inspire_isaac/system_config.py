"""Single-site runtime configuration and immutable per-episode snapshots."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


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
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class SystemConfig:
    path: Path
    document: dict[str, Any]
    sha256: str

    @property
    def root(self) -> Path:
        return self.path.parent

    def resolve(self, raw: str) -> Path:
        candidate = Path(raw).expanduser()
        return candidate if candidate.is_absolute() else (self.root / candidate).resolve()


def load_system_config(path: str | Path) -> SystemConfig:
    source = Path(path).expanduser().resolve(strict=True)
    root = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "system config")
    if int(root.get("schema_version", 0)) != 1:
        raise SystemConfigError("schema_version must be 1")
    session = _mapping(root.get("session"), "session")
    if not str(session.get("id", "")).strip():
        raise SystemConfigError("session.id is required")
    if not str(session.get("runtime_root", "")).strip() or not str(session.get("sessions_root", "")).strip():
        raise SystemConfigError("session.runtime_root and session.sessions_root are required")
    sampling = _mapping(root.get("sampling"), "sampling")
    for key in ("arm_observation_hz", "teleop_command_hz", "hand_state_hz", "tactile_hz", "camera_hz", "policy_observation_hz", "training_timeline_hz"):
        _positive(sampling.get(key), f"sampling.{key}")
    if sampling.get("action_label") != "sent_command":
        raise SystemConfigError("sampling.action_label must be sent_command for an executed-action dataset")
    pedal = _mapping(root.get("pedal"), "pedal")
    if not str(pedal.get("device", "")).strip():
        raise SystemConfigError("pedal.device is required")
    codes = [int(pedal.get(key, -1)) for key in ("rerecord_key_code", "enable_key_code", "record_toggle_key_code")]
    if len(set(codes)) != 3 or any(code <= 0 for code in codes):
        raise SystemConfigError("pedal key codes must be three distinct positive Linux input codes")
    flexiv = _mapping(root.get("flexiv"), "flexiv")
    cartesian = _mapping(flexiv.get("cartesian_control"), "flexiv.cartesian_control")
    if cartesian.get("mode") not in {"position", "impedance"}:
        raise SystemConfigError("flexiv.cartesian_control.mode must be position or impedance")
    for key in ("position_stiffness", "impedance_stiffness", "damping"):
        if any(item < 0.0 for item in _vector(cartesian.get(key), f"flexiv.cartesian_control.{key}", 6)):
            raise SystemConfigError(f"flexiv.cartesian_control.{key} must be non-negative")
    home = _mapping(flexiv.get("home"), "flexiv.home")
    _vector(home.get("left_joints_rad"), "flexiv.home.left_joints_rad", 7)
    _vector(home.get("right_joints_rad"), "flexiv.home.right_joints_rad", 7)
    for key in ("max_velocity_rad_s", "max_acceleration_rad_s2", "tolerance_rad", "timeout_s"):
        _positive(home.get(key), f"flexiv.home.{key}")
    if not str(home.get("quest_button", "")).strip():
        raise SystemConfigError("flexiv.home.quest_button is required")
    export = _mapping(root.get("lerobot_export"), "lerobot_export")
    if int(export.get("schema_version", 0)) != 1:
        raise SystemConfigError("lerobot_export.schema_version must be 1")
    timeline = _mapping(export.get("timeline"), "lerobot_export.timeline")
    if not str(timeline.get("source", "")).strip():
        raise SystemConfigError("lerobot_export.timeline.source is required")
    _positive(timeline.get("fps"), "lerobot_export.timeline.fps")
    action = _mapping(export.get("action"), "lerobot_export.action")
    if action.get("view") not in {"sent_command", "absolute_joint_position", "absolute_cartesian_pose"}:
        raise SystemConfigError("lerobot_export.action.view is unsupported")
    channels = _mapping(export.get("channels", {}), "lerobot_export.channels")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in channels.items()):
        raise SystemConfigError("lerobot_export.channels must contain string mappings")
    cameras = _mapping(root.get("cameras"), "cameras")
    if bool(cameras.get("depth_enabled", False)):
        raise SystemConfigError("depth must remain disabled in this RGB-only first release")
    if not 1 <= int(cameras.get("jpeg_quality", 0)) <= 100:
        raise SystemConfigError("cameras.jpeg_quality must be in [1,100]")
    streams = _mapping(cameras.get("streams"), "cameras.streams")
    if set(streams) != {"head", "left_wrist", "right_wrist"}:
        raise SystemConfigError("cameras.streams must contain exactly head, left_wrist, right_wrist")
    for name, stream_value in streams.items():
        stream = _mapping(stream_value, f"cameras.streams.{name}")
        if not str(stream.get("serial", "")).strip():
            raise SystemConfigError(f"cameras.streams.{name}.serial is required")
        if not 160 <= int(stream.get("width", 0)) <= 1920 or not 120 <= int(stream.get("height", 0)) <= 1080:
            raise SystemConfigError(f"cameras.streams.{name} resolution is outside supported bounds")
        _positive(stream.get("fps"), f"cameras.streams.{name}.fps", upper=90.0)
        if stream.get("pixel_format") != "rgb8":
            raise SystemConfigError(f"cameras.streams.{name}.pixel_format must be rgb8")
    xr = _mapping(root.get("xr_video"), "xr_video")
    if xr.get("encoder") not in {"auto", "h264_nvenc", "libx264"}:
        raise SystemConfigError("xr_video.encoder must be auto, h264_nvenc, or libx264")
    if not str(xr.get("ffmpeg", "")).strip() or not str(xr.get("receiver_host", "")).strip():
        raise SystemConfigError("xr_video.ffmpeg and receiver_host are required")
    _positive(xr.get("bitrate_mbps"), "xr_video.bitrate_mbps", upper=100.0)
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
    display = _mapping(xr.get("display"), "xr_video.display")
    if display.get("mode") not in {"xr", "monitor"}:
        raise SystemConfigError("xr_video.display.mode must be xr or monitor")
    for name in streams:
        plane = _mapping(display.get(name), f"xr_video.display.{name}")
        for key in ("distance", "width"):
            _positive(plane.get(key), f"xr_video.display.{name}.{key}", upper=10.0)
    return SystemConfig(source, root, canonical_hash(root))


def render_runtime_configs(config: SystemConfig, output: str | Path) -> dict[str, Path]:
    """Materialize child configs; their only source of operator parameters is config."""
    out = Path(output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = config.document
    sampling, pedal, flexiv = root["sampling"], root["pedal"], root["flexiv"]
    cameras, inspire, teleop = root["cameras"], root["inspire"], root["teleop"]
    xr = root["xr_video"]
    session = root["session"]
    result: dict[str, Path] = {}
    camera = {"schema_version": 1, "librealsense_version": "2.57.7", "firmware_policy": "preserve", "recording": {"encoding": cameras["recording_encoding"], "jpeg_quality": cameras["jpeg_quality"], "calibration_encoding": "raw_rgb", "depth_enabled": False}, "cameras": {}}
    for name, stream in cameras["streams"].items():
        camera["cameras"][name] = {**stream, "jpeg_quality": cameras["jpeg_quality"], "depth_enabled": False, "fps": int(stream["fps"])}
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
        xr_params[f"streams.{name}.fps"] = float(camera_stream["fps"])
        isaac_cameras[name] = {
            "enabled": bool(xr_stream["enabled"]), "type": "v4l2", "stereo": False,
            "device": "/dev/null", "width": int(camera_stream["width"]),
            "height": int(camera_stream["height"]), "fps": int(camera_stream["fps"]),
            "streams": {"mono": {"stream_id": len(isaac_cameras), "port": int(xr_stream["port"]),
                                   "bitrate_mbps": float(xr["bitrate_mbps"])}}
        }
    display = xr["display"]
    receiver = {
        "source": "rtp", "streaming": {"host": xr["receiver_host"]},
        "cameras": isaac_cameras,
        "display": {
            "mode": display["mode"], "cuda_device": int(display["cuda_device"]),
            "monitor": {"width": 1920, "height": 1080, "title": "Flexiv Inspire Teleop",
                        "padding": 4, "stream_timeout": float(xr["stream_timeout_s"])},
            "xr": {"planes": {name: display[name] for name in cameras["streams"]},
                   "lock_mode": display["lock_mode"], "look_away_angle": float(display["look_away_angle"]),
                   "reposition_distance": float(display["reposition_distance"]),
                   "reposition_delay": float(display["reposition_delay"]),
                   "transition_duration": float(display["transition_duration"])},
        },
    }
    payloads = {
        "camera.yaml": camera,
        "xr_bridge.yaml": {"flexiv_inspire_xr_bridge": {"ros__parameters": xr_params}},
        "isaac_camera_receiver.yaml": receiver,
        "dftp.yaml": {"flexiv_inspire_dftp_driver": {"ros__parameters": {"left_host": inspire["left_host"], "right_host": inspire["right_host"], "port": inspire["port"], "state_hz": sampling["hand_state_hz"], "tactile_hz": sampling["tactile_hz"], "hardware_write_enabled": bool(inspire["hardware_write_enabled"]), "local_session_id": session["id"], "local_write_confirmation": ""}}},
        "control_bridge.yaml": {"/**": {"ros__parameters": {"session_id": session["id"], "foot_pedal": pedal["device"], "observation_rate_hz": sampling["arm_observation_hz"], "enable_key_code": pedal["enable_key_code"], "cartesian_control_mode": flexiv["cartesian_control"]["mode"], "cartesian_position_stiffness": flexiv["cartesian_control"]["position_stiffness"], "cartesian_impedance_stiffness": flexiv["cartesian_control"]["impedance_stiffness"], "cartesian_damping": flexiv["cartesian_control"]["damping"], "home_left_joints_rad": flexiv["home"]["left_joints_rad"], "home_right_joints_rad": flexiv["home"]["right_joints_rad"], "home_max_velocity_rad_s": flexiv["home"]["max_velocity_rad_s"], "home_max_acceleration_rad_s2": flexiv["home"]["max_acceleration_rad_s2"], "home_tolerance_rad": flexiv["home"]["tolerance_rad"], "home_timeout_s": flexiv["home"]["timeout_s"]}}},
        "teleop.yaml": {"/**": {"ros__parameters": {"session_id": session["id"], "command_enabled": bool(teleop["control_enabled"]), "control_rate_hz": sampling["teleop_command_hz"], "manus_calibration": teleop["manus_calibration"], "home_button_key": flexiv["home"]["quest_button"], "home_topic": "/control/home_request"}}},
        "pedal.yaml": {"/**": {"ros__parameters": {"foot_pedal": pedal["device"], "rerecord_key_code": pedal["rerecord_key_code"], "enable_key_code": pedal["enable_key_code"], "record_toggle_key_code": pedal["record_toggle_key_code"]}}},
        "lerobot_export.yaml": root["lerobot_export"],
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
