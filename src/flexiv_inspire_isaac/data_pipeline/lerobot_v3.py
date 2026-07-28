"""Executable LeRobot 0.6.0 Dataset v3 writer for aligned episode rows."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from isaac_teleop_core.rotation6d import rotation6d_to_matrix


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


def lerobot_features() -> dict[str, dict]:
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
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (240, 424, 3),
            "names": ["height", "width", "channel"],
        }
    return features


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


def _validated_action(value: Any) -> np.ndarray:
    action = _vector(value, 30)
    if not np.all(np.isfinite(action)):
        raise ValueError("action contains NaN or Inf")
    for start in (3, 12):
        # Keep ROS, gRPC and dataset validation on the exact same
        # scale-invariant Rotation-6D decoder.
        rotation6d_to_matrix(action[start : start + 6])
    if np.any(action[18:] < 0.0) or np.any(action[18:] > 1000.0):
        raise ValueError("hand targets must be in 0..1000")
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


def _decode_image(value: Any) -> np.ndarray:
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
    if image.shape != (240, 424, 3) or image.dtype != np.uint8:
        raise ValueError(
            f"image must be uint8 HWC 240x424x3, got {image.shape}/{image.dtype}"
        )
    return image


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


def aligned_row_to_frame(row: Mapping[str, Any], task: str) -> dict[str, Any]:
    if not task:
        raise ValueError("LeRobot frames require a non-empty task")
    images = {}
    for camera in CAMERAS:
        key = f"observation.images.{camera}"
        if not row.get(f"{key}.valid", False):
            # Video features cannot represent a missing frame. Drop the aligned
            # row at the caller rather than inserting a black image.
            raise ValueError(f"required image {camera} is invalid")
        images[key] = _decode_image(row[key])

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
        # EpisodeAligner selects control/safe_command for this field.
        "action": _validated_action(row.get("action")),
        **images,
    }
    return frame


@dataclass(frozen=True)
class ExportResult:
    output_root: str
    frames_written: int
    frames_dropped_invalid_image: int
    reload_length: int


def export_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_root: str | Path,
    repo_id: str,
    task: str,
    dataset_class=None,
) -> ExportResult:
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
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty dataset root: {root}")
    dataset = dataset_class.create(
        repo_id=repo_id,
        fps=FPS,
        features=lerobot_features(),
        root=root,
        robot_type="flexiv_rizon4s_dual_inspire_dftp2",
        use_videos=True,
    )
    written = 0
    dropped = 0
    try:
        for row in rows:
            validity = row.get("valid", {})
            action_explicitly_invalid = (
                isinstance(validity, Mapping)
                and "action" in validity
                and not bool(validity["action"])
            )
            if row.get("action") is None or action_explicitly_invalid:
                dropped += 1
                continue
            try:
                frame = aligned_row_to_frame(row, task)
            except ValueError as exc:
                if "required image" in str(exc):
                    dropped += 1
                    continue
                raise
            dataset.add_frame(frame)
            written += 1
        if written == 0:
            raise ValueError("no fully valid aligned frames to export")
        dataset.save_episode()
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
    if tuple(np.asarray(sample["action"]).shape) != (30,):
        raise RuntimeError("reloaded action is not 30-dimensional")
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
    expected_shapes.update({
        f"observation.arm_{field}": (14,)
        for field in ("q", "dq", "tau", "tau_des", "tau_ext", "tau_interact", "temperature")
    })
    expected_shapes.update({
        f"observation.hand_{field}": (12,)
        for field in ("position", "actual_force", "current", "temperature", "error", "status")
    })
    for key, expected in expected_shapes.items():
        if tuple(np.asarray(sample[key]).shape) != expected:
            raise RuntimeError(f"reloaded {key} shape is not {expected}")
    tactile_reload = np.asarray(sample["observation.tactile"])
    if (
        not np.issubdtype(tactile_reload.dtype, np.integer)
        or np.any(tactile_reload < 0)
        or np.any(tactile_reload > 65535)
    ):
        raise RuntimeError(
            "reloaded tactile values must preserve the uint16 value domain"
        )
    return ExportResult(str(root.resolve()), written, dropped, reload_length)
