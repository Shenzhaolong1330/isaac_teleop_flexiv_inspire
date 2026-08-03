"""Executable LeRobot 0.6.0 Dataset v3 writer for aligned episode rows."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from isaac_teleop_core.rotation6d import rotation6d_to_matrix
from .export_spec import ActionView, CORE_LEROBOT_FIELDS


FPS = 30
CAMERAS = ("head", "left_wrist", "right_wrist")
VALIDITY_FIELDS = (
    "images.head",
    "images.left_wrist",
    "images.right_wrist",
    "left_arm.state",
    "right_arm.state",
    "left_arm.tcp_twist",
    "right_arm.tcp_twist",
    "left_arm.pose",
    "right_arm.pose",
    "left_arm.raw_ft",
    "right_arm.raw_ft",
    "left_arm.tcp_wrench",
    "right_arm.tcp_wrench",
    "left_hand.state",
    "right_hand.state",
    "left_hand.tactile",
    "right_hand.tactile",
    "action",
)

HAND_FIELDS = (
    "angle",
    "position",
    "actual_force",
    "current",
    "temperature",
    "error",
    "status",
)
HAND_FIELD_TIMING_NAMES = tuple(
    f"{side}_{field}" for side in ("left", "right") for field in HAND_FIELDS
)
ARM_HIGH_RATE_WIDTH_PER_ARM = 74


def lerobot_features(
    action: ActionView = ActionView(),
    image_shapes: Mapping[str, tuple[int, int, int]] | None = None,
    high_rate_arm_samples_per_frame: int = 0,
    fields: Sequence[str] | None = None,
    depth_shapes: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, dict]:
    selected = set(CORE_LEROBOT_FIELDS if fields is None else fields)
    features: dict[str, dict] = {
        "observation.arm_pose": {
            "dtype": "float32",
            "shape": (18,),
            "names": [
                f"{side}_{name}"
                for side in ("left", "right")
                for name in (
                    "x",
                    "y",
                    "z",
                    "R00",
                    "R10",
                    "R20",
                    "R01",
                    "R11",
                    "R21",
                )
            ],
        },
        "observation.hand_state": {
            "dtype": "float32",
            "shape": (12,),
            "names": [
                f"{side}_{actuator}"
                for side in ("left", "right")
                for actuator in (
                    "little",
                    "ring",
                    "middle",
                    "index",
                    "thumb_bend",
                    "thumb_rotate",
                )
            ],
        },
        "observation.hand_field_valid": {
            "dtype": "bool",
            "shape": (len(HAND_FIELD_TIMING_NAMES),),
            "names": list(HAND_FIELD_TIMING_NAMES),
        },
        "observation.hand_field_age_s": {
            "dtype": "float32",
            "shape": (len(HAND_FIELD_TIMING_NAMES),),
            "names": list(HAND_FIELD_TIMING_NAMES),
        },
        "observation.force_torque": {
            "dtype": "float32",
            "shape": (24,),
            "names": [
                f"{side}_{sensor}_{axis}"
                for side in ("left", "right")
                for sensor in ("raw_ft", "tcp_wrench")
                for axis in ("fx", "fy", "fz", "mx", "my", "mz")
            ],
        },
        "observation.tactile": {
            "dtype": "uint16",
            "shape": (2124,),
            "names": None,
        },
        "observation.valid": {
            "dtype": "bool",
            "shape": (len(VALIDITY_FIELDS),),
            "names": list(VALIDITY_FIELDS),
        },
        "observation.age_s": {
            "dtype": "float32",
            "shape": (len(VALIDITY_FIELDS),),
            "names": list(VALIDITY_FIELDS),
        },
        "action": {
            "dtype": "float32",
            "shape": (30,),
            "names": [
                *[
                    f"{side}_{name}"
                    for side in ("left", "right")
                    for name in (
                        "dx",
                        "dy",
                        "dz",
                        "dR00",
                        "dR10",
                        "dR20",
                        "dR01",
                        "dR11",
                        "dR21",
                    )
                ],
                *[
                    f"{side}_hand_{actuator}"
                    for side in ("left", "right")
                    for actuator in (
                        "little",
                        "ring",
                        "middle",
                        "index",
                        "thumb_bend",
                        "thumb_rotate",
                    )
                ],
            ],
        },
    }
    # Action is deliberately configurable while the raw MCAP remains unchanged.
    features["action"] = {
        "dtype": "float32", "shape": (action.shape,), "names": action.names,
    }
    features["observation.arm_quaternion_xyzw"] = {
        "dtype": "float32",
        "shape": (8,),
        "names": [
            f"{side}_q{component}"
            for side in ("left", "right")
            for component in ("x", "y", "z", "w")
        ],
    }
    arm_joint_names = [
        f"{side}_j{joint}"
        for side in ("left", "right")
        for joint in range(7)
    ]
    for field in ("q", "dq", "tau", "tau_des", "tau_ext", "tau_interact"):
        features[f"observation.arm_{field}"] = {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"{field}_{name}" for name in arm_joint_names],
        }
    features["observation.tcp_twist"] = {
        "dtype": "float32",
        "shape": (12,),
        "names": [
            f"{side}_{axis}"
            for side in ("left", "right")
            for axis in ("vx", "vy", "vz", "wx", "wy", "wz")
        ],
    }
    features["observation.arm_temperature"] = {
        "dtype": "float32",
        "shape": (14,),
        "names": [f"temperature_{name}" for name in arm_joint_names],
    }
    hand_actuator_names = [
        f"{side}_{actuator}"
        for side in ("left", "right")
        for actuator in ("little", "ring", "middle", "index", "thumb_bend", "thumb_rotate")
    ]
    for field in ("position", "actual_force", "current", "temperature", "error", "status"):
        dtype = "uint16" if field in {"error", "status"} else "float32"
        features[f"observation.hand_{field}"] = {
            "dtype": dtype,
            "shape": (12,),
            "names": [f"{field}_{name}" for name in hand_actuator_names],
        }
    for camera in CAMERAS:
        if f"observation.images.{camera}" not in selected:
            continue
        shape = (240, 424, 3) if image_shapes is None else image_shapes[camera]
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channel"],
        }
    if high_rate_arm_samples_per_frame:
        features["observation.arm_high_rate"] = {
            "dtype": "float32",
            "shape": (
                high_rate_arm_samples_per_frame
                * ARM_HIGH_RATE_WIDTH_PER_ARM
                * 2,
            ),
            "names": None,
        }
        features["observation.arm_high_rate_valid"] = {
            "dtype": "bool",
            "shape": (high_rate_arm_samples_per_frame * 2,),
            "names": None,
        }
        features["observation.arm_high_rate_age_s"] = {
            "dtype": "float32",
            "shape": (high_rate_arm_samples_per_frame * 2,),
            "names": None,
        }
    for camera, shape in (depth_shapes or {}).items():
        features[f"observation.depth.{camera}"] = {
            "dtype": "uint16",
            "shape": shape,
            "names": None,
        }
        features[f"observation.depth_scale_m.{camera}"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["meters_per_z16_unit"],
        }
        features[f"observation.depth_intrinsics.{camera}"] = {
            "dtype": "float32",
            "shape": (4,),
            "names": ["fx", "fy", "ppx", "ppy"],
        }
    features.update(
        {
            "observation.source_timestamp_ns": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["mapped_host_time_ns"],
            },
            "observation.source_gap_s": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["seconds_since_previous_source_frame"],
            },
            "observation.capture_segment": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["capture_segment"],
            },
            "observation.frame_in_segment": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["frame_in_segment"],
            },
            "observation.capture_segment_start": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["capture_segment_start"],
            },
        }
    )
    return {name: feature for name, feature in features.items() if name in selected}


def _vector(value: Any, length: int, *, dtype=np.float32) -> np.ndarray:
    if value is None:
        fill = 65535 if np.dtype(dtype) == np.dtype(np.uint16) else np.nan
        return np.full(length, fill, dtype=dtype)
    if hasattr(value, "flattened"):
        value = value.flattened
    if isinstance(value, Mapping):
        for key in ("values", "value", "angle", "actuator_angle", "taxels"):
            if key in value:
                value = value[key]
                break
    result = np.asarray(value, dtype=dtype).reshape(-1)
    if result.size != length:
        raise ValueError(f"expected vector length {length}, got {result.size}")
    return result


_FIELD_ALIASES = {
    "q": ("q", "position"),
    "dq": ("dq", "velocity"),
    "tau": ("tau", "effort"),
    "tau_des": ("tau_des",),
    "tau_ext": ("tau_ext",),
    "tau_interact": ("tau_interact",),
    "temperature": ("temperature", "temperature_c"),
    "position": ("position", "actuator_position"),
    "angle": ("angle", "actuator_angle"),
    "actual_force": ("actual_force", "actual_force_g"),
    "current": ("current", "current_ma"),
    "error": ("error", "error_code"),
    "status": ("status", "status_code"),
    "tcp_twist": ("tcp_twist", "twist"),
    "raw_ft": ("raw_ft", "values", "value"),
    "tcp_wrench": ("tcp_wrench", "values", "value"),
}


def _field_vector(value: Any, field: str, length: int, *, dtype=np.float32) -> np.ndarray:
    if value is None:
        return _vector(None, length, dtype=dtype)
    for alias in _FIELD_ALIASES[field]:
        if isinstance(value, Mapping) and alias in value:
            return _vector(value[alias], length, dtype=dtype)
        if hasattr(value, alias):
            return _vector(getattr(value, alias), length, dtype=dtype)
    if not isinstance(value, Mapping):
        return _vector(value, length, dtype=dtype)
    return _vector(None, length, dtype=dtype)


def _native_arm_state_vector(value: Any) -> np.ndarray:
    """Pack one complete native arm state into a stable 74-D vector."""

    if not isinstance(value, Mapping):
        return np.full(ARM_HIGH_RATE_WIDTH_PER_ARM, np.nan, dtype=np.float32)
    joint_fields = (
        "q",
        "dq",
        "tau",
        "tau_des",
        "tau_ext",
        "tau_interact",
        "temperature",
    )
    parts = [_field_vector(value, field, 7) for field in joint_fields]
    parts.extend(
        (
            _vector(value.get("tcp_pose_rdk_xyz_wxyz"), 7),
            _vector(value.get("tcp_velocity"), 6),
            _vector(value.get("raw_ft"), 6),
            _vector(value.get("external_wrench"), 6),
        )
    )
    result = np.concatenate(parts).astype(np.float32, copy=False)
    if result.shape != (ARM_HIGH_RATE_WIDTH_PER_ARM,):
        raise ValueError("native arm high-rate state is not 74-D")
    return result


def _arm_high_rate_frame(
    row: Mapping[str, Any], samples_per_frame: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    packed: list[np.ndarray] = []
    valid_flat: list[bool] = []
    age_flat: list[float] = []
    histories = {
        side: list(row.get(f"observation.{side}_arm.high_rate", ()))
        for side in ("left", "right")
    }
    validities = {
        side: list(row.get(f"observation.{side}_arm.high_rate.valid", ()))
        for side in ("left", "right")
    }
    ages = {
        side: list(row.get(f"observation.{side}_arm.high_rate.age_ns", ()))
        for side in ("left", "right")
    }
    for side in ("left", "right"):
        if not all(
            len(values) == samples_per_frame
            for values in (histories[side], validities[side], ages[side])
        ):
            raise ValueError(f"{side} arm high-rate history length mismatch")
    for index in range(samples_per_frame):
        for side in ("left", "right"):
            is_valid = bool(validities[side][index])
            packed.append(
                _native_arm_state_vector(histories[side][index])
                if is_valid
                else np.full(ARM_HIGH_RATE_WIDTH_PER_ARM, np.nan, dtype=np.float32)
            )
            valid_flat.append(is_valid)
            age_ns = ages[side][index]
            age_flat.append(np.nan if age_ns is None else float(age_ns) / 1e9)
    return (
        np.concatenate(packed).astype(np.float32, copy=False),
        np.asarray(valid_flat, dtype=bool),
        np.asarray(age_flat, dtype=np.float32),
    )




def _tactile_vector(value: Any) -> np.ndarray:
    if isinstance(value, Mapping) and "surfaces" in value:
        flattened = []
        for surface in value["surfaces"]:
            if isinstance(surface, Mapping):
                flattened.extend(surface.get("taxels", surface.get("values", ())))
            else:
                flattened.extend(surface.values)
        value = flattened
    return _vector(value, 1062, dtype=np.uint16)


def _validated_action(value: Any, action_view: ActionView = ActionView()) -> np.ndarray:
    action = _vector(value, action_view.shape)
    if not np.all(np.isfinite(action)):
        raise ValueError("action contains NaN or Inf")
    if action_view.name == "sent_command":
        for start in (3, 12):
            rotation6d_to_matrix(action[start : start + 6])
        if np.any(action[18:] < 0.0) or np.any(action[18:] > 1000.0):
            raise ValueError("hand targets must be in 0..1000")
    elif action_view.name == "absolute_cartesian_pose":
        for start in (3, 12):
            rotation6d_to_matrix(action[start : start + 6])
    return action
def _pose_vector(value: Any) -> np.ndarray:
    if value is None:
        return np.full(9, np.nan, dtype=np.float32)
    if hasattr(value, "pose") and hasattr(value, "rotation6d"):
        xyz = value.pose.xyz
        rotation6d = value.rotation6d
    elif isinstance(value, Mapping):
        xyz = value["xyz"]
        rotation6d = value["rotation6d"]
    else:
        raise ValueError("pose must provide xyz and rotation6d")
    return np.asarray((*xyz, *rotation6d), dtype=np.float32)


def _pose_quaternion_vector(value: Any) -> np.ndarray:
    if value is None:
        return _vector(None, 4)
    if hasattr(value, "pose") and hasattr(value.pose, "quaternion_xyzw"):
        quaternion = value.pose.quaternion_xyzw
    elif isinstance(value, Mapping) and "quaternion_xyzw" in value:
        quaternion = value["quaternion_xyzw"]
    else:
        raise ValueError("pose must preserve quaternion_xyzw beside rotation6d")
    return _vector(quaternion, 4)


def _decode_image(
    value: Any, expected_shape: tuple[int, int, int] | None = None
) -> np.ndarray:
    if isinstance(value, np.ndarray):
        image = value
    else:
        if isinstance(value, Mapping):
            if "jpeg_b64" in value:
                value = base64.b64decode(value["jpeg_b64"])
            elif "jpeg" in value:
                value = value["jpeg"]
        if not isinstance(value, (bytes, bytearray)):
            raise ValueError("image is not JPEG bytes or an ndarray")
        image = np.asarray(Image.open(BytesIO(value)).convert("RGB"))
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"image must be uint8 HWC with 3 channels, got {image.shape}/{image.dtype}"
        )
    if expected_shape is not None and image.shape != expected_shape:
        raise ValueError(
            f"image shape changed within episode: expected {expected_shape}, "
            f"got {image.shape}"
        )
    return image


def _decode_depth(
    value: Any, expected_shape: tuple[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(value, Mapping):
        raise ValueError("depth is not a decoded Z16 mapping")
    try:
        width = int(value["width"])
        height = int(value["height"])
        raw = value["z16"]
        scale_m = float(value["scale_m"])
        intrinsics = np.asarray(value["intrinsics"], dtype=np.float32).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"depth metadata is invalid: {exc}") from exc
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError("depth Z16 payload is not bytes")
    image = np.frombuffer(raw, dtype="<u2")
    if image.size != width * height:
        raise ValueError("depth Z16 payload size does not match dimensions")
    image = image.reshape(height, width).astype(np.uint16, copy=False)
    if expected_shape is not None and image.shape != expected_shape:
        raise ValueError(
            f"depth shape changed within episode: expected {expected_shape}, "
            f"got {image.shape}"
        )
    if not np.isfinite(scale_m) or scale_m <= 0.0:
        raise ValueError("depth scale must be finite and positive")
    if intrinsics.shape != (4,) or not np.all(np.isfinite(intrinsics)):
        raise ValueError("depth intrinsics must be finite [fx,fy,ppx,ppy]")
    return (
        image,
        np.asarray([scale_m], dtype=np.float32),
        intrinsics,
    )


def _ros_time_ns(value: Any) -> int | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return int(value.get("sec", 0)) * 1_000_000_000 + int(
            value.get("nanosec", 0)
        )
    except (TypeError, ValueError):
        return None


def _hand_field_timing(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    reference_ns = int(row.get("timestamp_ns", 0))
    valid_values: list[bool] = []
    age_values: list[float] = []
    for side in ("left", "right"):
        state = row.get(f"observation.{side}_hand.state")
        aggregate_valid = bool(
            row.get(f"observation.{side}_hand.state.valid", False)
        )
        aggregate_age = row.get(f"observation.{side}_hand.state.age_ns")
        for field in HAND_FIELDS:
            timing = (
                state.get(f"{field}_timing")
                if isinstance(state, Mapping)
                else None
            )
            if not isinstance(timing, Mapping):
                valid_values.append(aggregate_valid)
                age_values.append(
                    np.nan if aggregate_age is None else float(aggregate_age) / 1e9
                )
                continue
            mapped_ns = _ros_time_ns(timing.get("mapped_host_time"))
            valid = bool(timing.get("valid", False)) and bool(
                timing.get("timing_valid", False)
            )
            if mapped_ns is None or reference_ns < mapped_ns:
                valid = False
                age_s = np.nan
            else:
                age_s = (reference_ns - mapped_ns) / 1e9
            valid_values.append(valid)
            age_values.append(age_s)
    return (
        np.asarray(valid_values, dtype=bool),
        np.asarray(age_values, dtype=np.float32),
    )


def aligned_row_to_frame(
    row: Mapping[str, Any],
    task: str,
    action: ActionView = ActionView(),
    image_shapes: Mapping[str, tuple[int, int, int]] | None = None,
    high_rate_arm_samples_per_frame: int = 0,
    fields: Sequence[str] | None = None,
    depth_shapes: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    if not task:
        raise ValueError("LeRobot frames require a non-empty task")
    selected = set(CORE_LEROBOT_FIELDS if fields is None else fields)
    images = {}
    for camera in CAMERAS:
        key = f"observation.images.{camera}"
        if key not in selected:
            continue
        if not row.get(f"{key}.valid", False):
            # Video features cannot represent a missing frame. Drop the aligned
            # row at the caller rather than inserting a black image.
            raise ValueError(f"required image {camera} is invalid")
        expected_shape = None if image_shapes is None else image_shapes[camera]
        images[key] = _decode_image(row[key], expected_shape)

    depths: dict[str, Any] = {}
    for camera, expected_shape in (depth_shapes or {}).items():
        key = f"observation.depth.{camera}"
        if key not in selected:
            continue
        if not row.get(f"{key}.valid", False):
            raise ValueError(f"required depth {camera} is invalid")
        depth, scale, intrinsics = _decode_depth(row.get(key), expected_shape)
        depths[key] = depth
        depths[f"observation.depth_scale_m.{camera}"] = scale
        depths[f"observation.depth_intrinsics.{camera}"] = intrinsics

    arm_pose = np.concatenate(
        [
            _pose_vector(row.get(f"observation.{side}_arm.pose"))
            for side in ("left", "right")
        ]
    )
    arm_quaternion_xyzw = np.concatenate(
        [
            _pose_quaternion_vector(row.get(f"observation.{side}_arm.pose"))
            for side in ("left", "right")
        ]
    )
    arm_states = [
        row.get(f"observation.{side}_arm.state")
        for side in ("left", "right")
    ]
    arm_field_vectors = {
        field: np.concatenate(
            [_field_vector(state, field, 7) for state in arm_states]
        )
        for field in ("q", "dq", "tau", "tau_des", "tau_ext", "tau_interact", "temperature")
    }
    tcp_twist = np.concatenate(
        [
            _field_vector(
                row.get(f"observation.{side}_arm.tcp_twist")
                if row.get(f"observation.{side}_arm.tcp_twist") is not None
                else arm_states[index],
                "tcp_twist",
                6,
            )
            for index, side in enumerate(("left", "right"))
        ]
    )
    hand_state = np.concatenate(
        [
            _vector(row.get(f"observation.{side}_hand.state"), 6)
            for side in ("left", "right")
        ]
    )
    hand_states = [
        row.get(f"observation.{side}_hand.state")
        for side in ("left", "right")
    ]
    hand_field_valid, hand_field_age_s = _hand_field_timing(row)
    hand_field_vectors = {}
    for field in ("position", "actual_force", "current", "temperature", "error", "status"):
        dtype = np.uint16 if field in {"error", "status"} else np.float32
        hand_field_vectors[field] = np.concatenate(
            [
                _field_vector(state, field, 6, dtype=dtype)
                for state in hand_states
            ]
        )
    force_torque = np.concatenate(
        [
            _field_vector(row.get(f"observation.{side}_arm.{sensor}"), sensor, 6)
            for side in ("left", "right")
            for sensor in ("raw_ft", "tcp_wrench")
        ]
    )
    tactile = np.concatenate(
        [
            _tactile_vector(row.get(f"observation.{side}_hand.tactile"))
            for side in ("left", "right")
        ]
    )
    valid = np.asarray(
        [
            bool(row.get(f"observation.{name}.valid", row.get(f"{name}.valid", False)))
            if name != "action"
            else bool(row.get("action.valid", False))
            for name in VALIDITY_FIELDS
        ],
        dtype=bool,
    )
    age_s = np.asarray(
        [
            (
                np.nan
                if row.get(f"observation.{name}.age_ns", row.get(f"{name}.age_ns"))
                is None
                else row.get(
                    f"observation.{name}.age_ns", row.get(f"{name}.age_ns")
                )
                / 1e9
            )
            for name in VALIDITY_FIELDS
        ],
        dtype=np.float32,
    )
    frame = {
        "task": task,
        "observation.arm_pose": arm_pose,
        "observation.arm_quaternion_xyzw": arm_quaternion_xyzw,
        **{
            f"observation.arm_{field}": value
            for field, value in arm_field_vectors.items()
        },
        "observation.tcp_twist": tcp_twist,
        "observation.hand_state": hand_state,
        "observation.hand_field_valid": hand_field_valid,
        "observation.hand_field_age_s": hand_field_age_s,
        **{
            f"observation.hand_{field}": value
            for field, value in hand_field_vectors.items()
        },
        "observation.force_torque": force_torque,
        "observation.tactile": tactile,
        "observation.valid": valid,
        "observation.age_s": age_s,
        "action": _validated_action(row.get("action"), action),
        **images,
        **depths,
        "observation.source_timestamp_ns": np.asarray(
            [int(row.get("observation.source_timestamp_ns", row.get("timestamp_ns", 0)))],
            dtype=np.int64,
        ),
        "observation.source_gap_s": np.asarray(
            [float(row.get("observation.source_gap_s", 0.0))],
            dtype=np.float32,
        ),
        "observation.capture_segment": np.asarray(
            [int(row.get("observation.capture_segment", 0))], dtype=np.int64
        ),
        "observation.frame_in_segment": np.asarray(
            [int(row.get("observation.frame_in_segment", 0))], dtype=np.int64
        ),
        "observation.capture_segment_start": np.asarray(
            [bool(row.get("observation.capture_segment_start", False))],
            dtype=bool,
        ),
    }
    if high_rate_arm_samples_per_frame:
        history, history_valid, history_age_s = _arm_high_rate_frame(
            row, high_rate_arm_samples_per_frame
        )
        frame.update(
            {
                "observation.arm_high_rate": history,
                "observation.arm_high_rate_valid": history_valid,
                "observation.arm_high_rate_age_s": history_age_s,
            }
        )
    return {
        "task": task,
        **{name: value for name, value in frame.items() if name in selected},
    }


@dataclass(frozen=True)
class ExportResult:
    output_root: str
    frames_written: int
    frames_dropped_invalid_action: int
    frames_dropped_invalid_image: int
    frames_dropped_invalid_depth: int
    episodes_written: int
    reload_length: int


def export_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_root: str | Path,
    repo_id: str,
    task: str,
    dataset_class=None,
    action: ActionView = ActionView(),
    fps: float = FPS,
    high_rate_arm_samples_per_frame: int = 0,
    fields: Sequence[str] | None = None,
    depth_cameras: Sequence[str] = (),
    split_episodes: bool = False,
) -> ExportResult:
    selected = set(CORE_LEROBOT_FIELDS if fields is None else fields)
    selected_image_cameras = tuple(
        camera
        for camera in CAMERAS
        if f"observation.images.{camera}" in selected
    )
    image_shapes: dict[str, tuple[int, int, int]] = {}
    if selected_image_cameras:
        for row in rows:
            try:
                candidate = {
                    camera: tuple(
                        int(value)
                        for value in _decode_image(
                            row[f"observation.images.{camera}"]
                        ).shape
                    )
                    for camera in selected_image_cameras
                    if row.get(f"observation.images.{camera}.valid", False)
                }
            except (KeyError, ValueError):
                continue
            if len(candidate) == len(selected_image_cameras):
                image_shapes = candidate
                break
        if len(image_shapes) != len(selected_image_cameras):
            raise ValueError("no fully valid aligned image frames to export")

    depth_shapes: dict[str, tuple[int, int]] = {}
    for camera in depth_cameras:
        key = f"observation.depth.{camera}"
        if key not in selected:
            continue
        for row in rows:
            if not row.get(f"{key}.valid", False):
                continue
            try:
                depth, _, _ = _decode_depth(row.get(key))
            except ValueError:
                continue
            depth_shapes[camera] = tuple(int(value) for value in depth.shape)
            break
        if camera not in depth_shapes:
            raise ValueError(f"no valid {camera} depth frames to export")

    if dataset_class is None:
        try:
            import lerobot
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot is unavailable; run with envs/data-py312 where "
                "lerobot==0.6.0 is pinned"
            ) from exc
        version = str(getattr(lerobot, "__version__", ""))
        if version != "0.6.0":
            raise RuntimeError(f"expected lerobot==0.6.0, loaded {version or 'unknown'}")
        dataset_class = LeRobotDataset

    root = Path(output_root)
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite existing dataset root: {root}"
            )
        # LeRobotDataset.create() deliberately requires the root not to exist.
        # Accept a conventional empty mktemp/mkdir target, but remove only the
        # exact empty leaf directory immediately before handing it to LeRobot.
        root.rmdir()
    dataset = dataset_class.create(
        repo_id=repo_id,
        fps=int(fps),
        features=lerobot_features(
            action,
            image_shapes,
            high_rate_arm_samples_per_frame,
            fields,
            depth_shapes,
        ),
        root=root,
        robot_type="flexiv_rizon4s_dual_inspire_dftp2",
        use_videos=True,
    )
    written = 0
    dropped_action = 0
    dropped_image = 0
    dropped_depth = 0
    episodes_written = 0
    frames_in_episode = 0
    active_segment: int | None = None
    try:
        for row in rows:
            validity = row.get("valid", {})
            action_explicitly_invalid = (
                isinstance(validity, Mapping)
                and "action" in validity
                and not bool(validity["action"])
            )
            if row.get("action") is None or action_explicitly_invalid:
                dropped_action += 1
                continue
            try:
                frame = aligned_row_to_frame(
                    row,
                    task,
                    action,
                    image_shapes,
                    high_rate_arm_samples_per_frame,
                    fields,
                    depth_shapes,
                )
            except ValueError as exc:
                if "required image" in str(exc):
                    dropped_image += 1
                    continue
                if "required depth" in str(exc):
                    dropped_depth += 1
                    continue
                raise
            row_segment = int(row.get("observation.capture_segment", 0))
            if (
                split_episodes
                and active_segment is not None
                and row_segment != active_segment
                and frames_in_episode > 0
            ):
                dataset.save_episode()
                episodes_written += 1
                frames_in_episode = 0
            dataset.add_frame(frame)
            written += 1
            frames_in_episode += 1
            active_segment = row_segment
        if written == 0:
            raise ValueError("no fully valid aligned frames to export")
        if frames_in_episode > 0:
            dataset.save_episode()
            episodes_written += 1
        dataset.finalize()
    except Exception:
        # finalize is idempotent in 0.6.0 and closes parquet/video writers.
        dataset.finalize()
        raise

    reloaded = dataset_class(repo_id=repo_id, root=root)
    reload_length = len(reloaded)
    if reload_length != written:
        raise RuntimeError(
            f"LeRobot reload validation failed: wrote {written}, loaded {reload_length}"
        )
    sample = reloaded[0]
    if tuple(np.asarray(sample["action"]).shape) != (action.shape,):
        raise RuntimeError(f"reloaded action shape is not {action.shape}")
    expected_shapes = {
        "observation.arm_pose": (18,),
        "observation.arm_quaternion_xyzw": (8,),
        "observation.tcp_twist": (12,),
        "observation.hand_state": (12,),
        "observation.force_torque": (24,),
        "observation.tactile": (2124,),
        "observation.valid": (len(VALIDITY_FIELDS),),
        "observation.age_s": (len(VALIDITY_FIELDS),),
    }
    if high_rate_arm_samples_per_frame:
        expected_shapes.update(
            {
                "observation.arm_high_rate": (
                    high_rate_arm_samples_per_frame
                    * ARM_HIGH_RATE_WIDTH_PER_ARM
                    * 2,
                ),
                "observation.arm_high_rate_valid": (
                    high_rate_arm_samples_per_frame * 2,
                ),
                "observation.arm_high_rate_age_s": (
                    high_rate_arm_samples_per_frame * 2,
                ),
            }
        )
    expected_shapes.update({
        f"observation.arm_{field}": (14,)
        for field in ("q", "dq", "tau", "tau_des", "tau_ext", "tau_interact", "temperature")
    })
    expected_shapes.update({
        f"observation.hand_{field}": (12,)
        for field in ("position", "actual_force", "current", "temperature", "error", "status")
    })
    expected_shapes.update(
        {
            f"observation.depth.{camera}": shape
            for camera, shape in depth_shapes.items()
        }
    )
    expected_shapes.update(
        {
            f"observation.depth_scale_m.{camera}": (1,)
            for camera in depth_shapes
        }
    )
    expected_shapes.update(
        {
            f"observation.depth_intrinsics.{camera}": (4,)
            for camera in depth_shapes
        }
    )
    expected_shapes.update(
        {
            "observation.source_timestamp_ns": (1,),
            "observation.source_gap_s": (1,),
            "observation.capture_segment": (1,),
            "observation.frame_in_segment": (1,),
            "observation.capture_segment_start": (1,),
        }
    )
    for key, expected in expected_shapes.items():
        if key not in selected:
            continue
        actual = tuple(np.asarray(sample[key]).shape)
        # LeRobot/HuggingFace stores declared one-element numeric features as
        # scalar Value columns and therefore reloads them with shape ().
        accepted = {expected, ()} if expected == (1,) else {expected}
        if actual not in accepted:
            raise RuntimeError(f"reloaded {key} shape is not {expected}")
    if "observation.tactile" in selected:
        tactile_reload = np.asarray(sample["observation.tactile"])
        if (
            not np.issubdtype(tactile_reload.dtype, np.integer)
            or np.any(tactile_reload < 0)
            or np.any(tactile_reload > 65535)
        ):
            raise RuntimeError(
                "reloaded tactile values must preserve the uint16 value domain"
            )
    return ExportResult(
        str(root.resolve()),
        written,
        dropped_action,
        dropped_image,
        dropped_depth,
        episodes_written,
        reload_length,
    )
