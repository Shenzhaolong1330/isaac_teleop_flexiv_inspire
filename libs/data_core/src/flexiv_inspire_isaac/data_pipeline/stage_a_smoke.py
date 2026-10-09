"""Hardware-free production recorder -> MCAP -> archive -> command consumer closure."""

from __future__ import annotations
import argparse
import base64
from io import BytesIO
from PIL import Image
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from .action_contract import (
    STEP_NS,
    REQUIRED,
    OPTIONAL,
    export_supervision,
    records_from_mcap,
    reconstruct,
    windows,
)
from .episode_manager import EpisodeSession
from .manifest import canonical_yaml_sha256
from .recorder import RecordEnvelope
from .provenance import verify_archive
from isaac_teleop_core.evidence import receipt
from isaac_teleop_core.command import CommandPoint, ControlRepresentation
from flexiv_inspire_control.conversion import cartesian_target_from_point
from .mcap_input import _flatten_safe_command_checked
from isaac_teleop_core.rotation6d import matrix_to_rotation6d


def command_payload(index, *, rotation=None):
    rotation = np.eye(3) if rotation is None else rotation
    rot = matrix_to_rotation6d(rotation).tolist()
    return {
        "schema_version": 1,
        "session_id": "synthetic",
        "source": "teleop",
        "sequence": index,
        "frame_id": "world",
        "valid_mask": 15,
        "deadman": True,
        "representation": 1,
        "rotation_order": "FIRST_TWO_COLUMNS:R00,R10,R20,R01,R11,R21",
        "trajectory": [
            {
                "left_delta_xyz": [0.001, 0, 0],
                "right_delta_xyz": [0, 0.002, 0],
                "left_delta_rotation6d": rot,
                "right_delta_rotation6d": rot,
                "left_hand_targets": [100 + index] * 6,
                "right_hand_targets": [200 + index] * 6,
            }
        ],
    }


def envelope(topic, t, sequence, payload, *, valid=True, receive=None):
    return RecordEnvelope(
        topic="/" + topic.lstrip("/"),
        source_time_ns=t,
        host_receive_time_ns=t + 1 if receive is None else receive,
        mapped_host_time_ns=t,
        sequence=sequence,
        valid=valid,
        payload=payload,
    )


def run(root, count=60):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    inputs.mkdir()
    tool = inputs / "tool.yaml"
    tool.write_text("tool: synthetic-only\npayload: {mass: 0.1}\n")
    calibration = inputs / "frames.yaml"
    calibration.write_text("synthetic: true\nworld_from_base: [1, 0, 0, 0]\n")
    ft = inputs / "ft.json"
    ft.write_text(
        json.dumps(
            {
                "event_type": "ft_zero_completed",
                "session_id": "synthetic",
                "success": True,
                "connection_generation": 1,
                "tool_payload_config_hash": canonical_yaml_sha256(tool),
            }
        )
    )
    clock = [1_000_000_000]
    config = {
        "sampling": {
            "arm_observation_hz": 300,
            "hand_state_hz": 200,
            "tactile_hz": 15,
            "camera_hz": 15,
            "teleop_command_hz": 60,
        },
        "task": "synthetic only",
        "api_token": "test-secret-must-not-archive",
    }
    session = EpisodeSession(
        root / "episodes",
        "synthetic",
        {"frames": calibration},
        tool,
        ft,
        record_ros_mcap=False,
        deviceio_profile="training",
        effective_config=config,
        clock_ns=lambda: clock[0],
        task_name="stage_a_synthetic",
        task_description="Offline engineering evidence only",
    )
    jpeg = BytesIO()
    Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(jpeg, format="JPEG")
    jpeg_b64 = base64.b64encode(jpeg.getvalue()).decode()
    t0 = clock[0] + STEP_NS
    accepted_targets = {
        side: np.array([0, 0, 0, 1, 0, 0, 0], dtype=float) for side in ("left", "right")
    }
    for i in range(count):
        t = t0 + i * STEP_NS
        clock[0] = t
        for name in REQUIRED:
            if name.startswith("camera/"):
                if i % 4:
                    continue
                value = {"jpeg_b64": jpeg_b64, "camera": name.split("/")[1]}
                sample_t = t
            else:
                value = (
                    {
                        **{
                            field: [9.0 if field == "q" else 0.0] * 7
                            for field in (
                                "q",
                                "dq",
                                "tau",
                                "tau_des",
                                "tau_ext",
                                "tau_interact",
                                "temperature",
                            )
                        },
                        "tcp_pose_world_xyz_wxyz": [5.0, 0, 0, 1, 0, 0, 0],
                        "tcp_velocity_world": [0.0] * 6,
                        "raw_ft": [0.0] * 6,
                        "external_wrench": [0.0] * 6,
                    }
                    if "arm" in name
                    else {
                        "angle": [300.0] * 6,
                        "position": [300.0] * 6,
                        "actual_force": [0.0] * 6,
                        "current": [0.0] * 6,
                        "temperature": [25.0] * 6,
                        "error": [0] * 6,
                        "status": [0] * 6,
                    }
                )
                sample_t = t - 3_000_000  # deliberately asynchronous, causal feedback
            session.submit(envelope(name, sample_t, i, value))
        if i % 4 == 0:
            for name in OPTIONAL:
                session.submit(envelope(name, t, i // 4, {"values": [0] * 1062}))
        payload = command_payload(i)
        session.submit(envelope("control/sent_command", t, i, payload), critical=True)
        vector, _ = _flatten_safe_command_checked(payload)
        point = CommandPoint.from_policy_vector(vector)
        for side in ("left", "right"):
            target, _ = cartesian_target_from_point(
                point,
                side=side,
                representation=ControlRepresentation.CARTESIAN_ROT6D,
                previous_safe_pose_rdk=accepted_targets[side],
                previous_output_quaternion_xyzw=None,
                max_translation_step_m=0.01,
                max_rotation_step_rad=0.1,
                enforce_step_limits=False,
            )
            accepted_targets[side] = target
        cid = f"synthetic:teleop:{i}"
        for actuator in ("left_arm", "right_arm", "left_hand", "right_hand"):
            session.submit(
                envelope(
                    "control/actuator_evidence",
                    t + 1,
                    i,
                    receipt(
                        cid,
                        actuator,
                        "sdk_returned",
                        now_ns=t + 1,
                        detail="synthetic executor using production target conversion only",
                        target=(
                            {
                                "tcp_pose_rdk": accepted_targets[
                                    actuator.split("_")[0]
                                ].tolist(),
                                "frame": actuator.split("_")[0] + "_base",
                                "quaternion": "wxyz",
                            }
                            if actuator.endswith("_arm")
                            else None
                        ),
                    ),
                ),
                critical=True,
            )
    clock[0] += 2 * STEP_NS
    session.finish(SimpleNamespace(enabled=False), reason="synthetic-complete")
    verify_archive(session.directory)
    records = records_from_mcap(session.device_path)
    document = export_supervision(records, episode_id=session.episode_uuid)
    (root / "command_supervision_v2.json").write_text(json.dumps(document, indent=2))
    assert len(document["rows"]) == count
    assert document["quality"]["valid_samples"] == count, document["quality"]
    xyz, rotation, hands = reconstruct(document["rows"])
    np.testing.assert_allclose(xyz, [[count * 0.001, 0, 0], [0, count * 0.002, 0]])
    np.testing.assert_allclose(
        xyz, [accepted_targets[side][:3] for side in ("left", "right")]
    )
    np.testing.assert_allclose(hands, [[100 + count - 1] * 6, [200 + count - 1] * 6])
    result = {
        "status": "PASS",
        "engineering_only": True,
        "hardware": "NOT_RUN",
        "commands": count,
        "camera_frames_per_stream": (count + 3) // 4,
        "valid_samples": document["quality"]["valid_samples"],
        "horizon_8_windows": len(list(windows(document, 8))),
        "manifest": str(session.manifest_path),
        "reconstructed_xyz": xyz.tolist(),
        "hand_targets": hands.tolist(),
    }
    (root / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    options = parser.parse_args(argv)
    print(json.dumps(run(options.output), indent=2))


if __name__ == "__main__":
    main()
