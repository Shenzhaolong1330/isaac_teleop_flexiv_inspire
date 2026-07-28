"""Generate a deterministic Rerun recording without ROS or hardware."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from .runtime import ROT6D_IDENTITY, RerunVisualizer


def _jpeg(camera_index: int, step: int) -> bytes:
    from PIL import Image

    height, width = 240, 424
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = ((x + 23 * camera_index + step) % 256).astype(np.uint8)
    rgb[:, :, 1] = ((y + 41 * camera_index + 2 * step) % 256).astype(np.uint8)
    rgb[:, :, 2] = ((x // 2 + y // 2 + 3 * step) % 256).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _acquisition(stamp_ns: int, sequence: int, *, valid: bool = True) -> dict[str, Any]:
    return {
        "source_time_ns": stamp_ns - 1_000_000,
        "host_receive_time_ns": stamp_ns,
        "acquisition_start_ns": stamp_ns - 2_000_000,
        "acquisition_end_ns": stamp_ns - 500_000,
        "sequence": sequence,
        "valid": valid,
        "age_ns": 1_000_000,
        "invalid_reason": "",
        "source_clock_domain": "synthetic",
        "host_clock_domain": "host_monotonic",
        "mapped_host_time_ns": stamp_ns - 1_000_000,
        "timing_valid": True,
    }


def _surfaces(step: int) -> list[dict[str, Any]]:
    layout = (
        *(
            (f"{finger}_{suffix}", rows, columns)
            for finger in ("little", "ring", "middle", "index")
            for suffix, rows, columns in (("end", 3, 3), ("tip", 12, 8), ("pad", 10, 8))
        ),
        ("thumb_end", 3, 3),
        ("thumb_tip", 12, 8),
        ("thumb_middle", 3, 3),
        ("thumb_pad", 12, 8),
        ("palm", 8, 14),
    )
    result = []
    offset = 0
    for index, (name, rows, columns) in enumerate(layout):
        values = (
            np.arange(rows * columns, dtype=np.uint16) + offset + step * 7
        ) % np.uint16(4096)
        if index == len(layout) - 1:
            values[-1] = np.uint16(65535)
        acquisition_start_ns = 1_000_000_000 + index * 100_000
        acquisition_end_ns = acquisition_start_ns + 50_000
        surface_acquisition = _acquisition(acquisition_end_ns, index)
        surface_acquisition["acquisition_start_ns"] = acquisition_start_ns
        surface_acquisition["acquisition_end_ns"] = acquisition_end_ns
        result.append(
            {
                "name": name,
                "rows": rows,
                "columns": columns,
                "taxels": values.tolist(),
                "acquisition": surface_acquisition,
                "valid": True,
                "invalid_reason": "",
                "acquisition_start_ns": acquisition_start_ns,
                "acquisition_end_ns": acquisition_end_ns,
            }
        )
        offset += rows * columns
    return result


def _command(stamp_ns: int, sequence: int, delta: float) -> dict[str, Any]:
    point = {
        "execute_after_ns": 0,
        "left_delta_xyz": [delta, 0.0, 0.0],
        "left_delta_rotation6d": ROT6D_IDENTITY.tolist(),
        "right_delta_xyz": [-delta, 0.0, 0.0],
        "right_delta_rotation6d": ROT6D_IDENTITY.tolist(),
        "left_delta_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "right_delta_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "left_arm_joint_positions": [],
        "right_arm_joint_positions": [],
        "left_hand_targets": [0.1] * 6,
        "right_hand_targets": [0.2] * 6,
    }
    return {
        "stamp_ns": stamp_ns,
        "frame_id": "world",
        "schema_version": 1,
        "session_id": "synthetic-session",
        "source": "synthetic",
        "sequence": sequence,
        "ttl_ns": 100_000_000,
        "representation": 1,
        "representation_name": "CARTESIAN_ROT6D",
        "rotation_order": "R00,R10,R20,R01,R11,R21",
        "valid_mask": 15,
        "deadman": True,
        "trajectory": [point],
    }


def run_synthetic(path: str | Path, *, frames: int = 12) -> Path:
    output = Path(path).expanduser().resolve()
    if frames < 2:
        raise ValueError("synthetic smoke needs at least two frames")
    with RerunVisualizer(save_path=output) as visualizer:
        base_ns = 1_800_000_000_000_000_000
        for step in range(frames):
            stamp_ns = base_ns + step * 33_333_333
            acquisition = _acquisition(stamp_ns, step)
            for camera_index, camera in enumerate(("head", "left_wrist", "right_wrist")):
                visualizer.log_camera(
                    {
                        "camera": camera,
                        "stamp_ns": stamp_ns,
                        "sequence": step,
                        "width": 424,
                        "height": 240,
                        "format": "jpeg; rgb8",
                        "jpeg": _jpeg(camera_index, step),
                        "acquisition": acquisition,
                    }
                )
            for side_index, side in enumerate(("left", "right")):
                phase = step * 0.03 + side_index * 0.2
                arm = {
                    "stamp_ns": stamp_ns,
                    "side": side,
                    "acquisition": acquisition,
                    "q": [phase + index * 0.01 for index in range(7)],
                    "dq": [0.03] * 7,
                    "tau": [1.0 + index for index in range(7)],
                    "tau_des": [1.1 + index for index in range(7)],
                    "tau_ext": [0.1] * 7,
                    "tau_interact": [0.2] * 7,
                    "tcp_pose": {
                        "position": [0.4, (-0.25 if side == "left" else 0.25), 0.3],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "tcp_twist": [0.01, 0.0, 0.0, 0.0, 0.0, 0.01],
                    "raw_ft": [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
                    "tcp_wrench": [0.5, 0.4, 0.3, 0.05, 0.04, 0.03],
                    "temperature": [35.0 + side_index] * 7,
                    "generation": 1,
                    "connected": True,
                    "fault": "",
                }
                visualizer.log_arm(arm)
                hand = {
                    "stamp_ns": stamp_ns,
                    "side": side,
                    "sequence": step,
                    "acquisition": acquisition,
                    "angle": [0.1 + phase] * 6,
                    "position": [100 + step] * 6,
                    "actual_force": [20 + step] * 6,
                    "current": [30 + step] * 6,
                    "temperature": [31 + side_index] * 6,
                    "error": [0] * 6,
                    "status": [1] * 6,
                    "connected": True,
                    "fault": False,
                    "fault_reason": "",
                }
                visualizer.log_hand(hand)
                surfaces = _surfaces(step)
                visualizer.log_tactile(
                    {
                        "stamp_ns": stamp_ns,
                        "side": side,
                        "sequence": step,
                        "acquisition": acquisition,
                        "surfaces": surfaces,
                        "taxel_count": 1062,
                        "valid": True,
                        "invalid_reason": "",
                    }
                )
            requested = _command(stamp_ns, step, 0.01)
            safe = _command(stamp_ns, step, 0.008)
            sent = _command(stamp_ns, step, 0.0075)
            visualizer.log_trace(
                {
                    "stamp_ns": stamp_ns,
                    "trace_sequence": step,
                    "requested": requested,
                    "safe": safe,
                    "sent": sent,
                    "requested_valid": True,
                    "safe_valid": True,
                    "sent_valid": True,
                    "rejection_reason": "",
                    "validation_latency_ns": 500_000,
                    "send_latency_ns": 300_000,
                }
            )
            visualizer.log_control_state(
                {
                    "stamp_ns": stamp_ns,
                    "state": 7,
                    "state_name": "ACTIVE",
                    "session_id": "synthetic-session",
                    "active_source": "teleop",
                    "hold_reason": "",
                    "local_permission": True,
                    "physical_pedal": True,
                    "ft_zeroed_for_session": True,
                    "arms_online": True,
                    "hands_online": True,
                    "generation": 1,
                }
            )
    return output
