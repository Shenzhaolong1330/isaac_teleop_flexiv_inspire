"""Synthetic Stage B mapping/control/dispatcher -> production MCAP/archive/export.

No ROS runtime, RDK SDK or hardware. Simulated measured tracking is explicit.
"""

from __future__ import annotations
import argparse
import base64
from io import BytesIO
from PIL import Image
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace as NS
import time
import numpy as np
import yaml

from isaac_teleop_core.world_teleop import (
    FixedMapping,
    WorldController,
    ApprovedRegion,
    Limits,
    calibrate,
    identity,
)
from isaac_teleop_core.rotation6d import matrix_to_quaternion_xyzw
from flexiv_rdk_daemon.mock_backend import MockBackend
from flexiv_rdk_daemon.server import (
    DaemonInterlock,
    HandObservationCache,
    RDKRequestDispatcher,
)
from flexiv_rdk_daemon.typed_ipc import TypedEnvelopeCodec
from flexiv_inspire_control.teleop_mapping import (
    QuestSE3Mapper,
    TrackedPose,
    TrackingSample,
)
from .stage_a_smoke import envelope, command_payload
from .episode_manager import EpisodeSession
from .manifest import canonical_yaml_sha256
from .action_contract import (
    STEP_NS,
    REQUIRED,
    export_supervision,
    records_from_mcap,
    windows,
    reconstruct,
)
from .provenance import verify_archive, resolve_asset
from .mcap_input import load_json_mcap_streams


def diagnostic_assets():
    cfg = {
        "schema_version": 1,
        "mode": "world_absolute",
        "world_frame": "world",
        "input_backend": "synthetic",
        "tracking_origin_id": "synthetic-origin",
        "input_session_id": "synthetic-boot",
        "origin_reported": False,
        "tool_identity": "synthetic-tool",
        "axis_rotation": np.eye(3).tolist(),
        "translation_scale": 1.0,
        **{
            name: {s: [0, 0, 0, 0, 0, 0, 1] for s in ("left", "right")}
            for name in (
                "world_from_base",
                "controller_from_reference",
                "tcp_from_reference",
            )
        },
        "calibration_limits": {
            "age_s": 0.1,
            "pair_skew_s": 0.02,
            "stable_position_m": 0.005,
            "stable_rotation_rad": 0.03,
            "relative_position_m": 0.01,
            "relative_rotation_rad": 0.05,
        },
    }
    rows = [
        {
            "capture_ns": 1_000_000_000 + i * 30_000_000,
            "identity": identity(cfg),
            **{
                name: {
                    "left": [0, 0, 0, 0, 0, 0, 1],
                    "right": [0, 0, 0, 0, 0, 0, 1],
                    "left_time_ns": 1_000_000_000 + i * 30_000_000,
                    "right_time_ns": 1_000_000_000 + i * 30_000_000,
                }
                for name in ("human", "robot")
            },
        }
        for i in range(12)
    ]
    cal = calibrate(cfg, rows)
    cal["reference_samples"] = "reference_samples.json"
    region = {
        "approved_by": "synthetic-test-only",
        "tool_identity": cfg["tool_identity"],
        "mapping_epoch": cal["mapping_epoch"],
        **{s: {"lower_m": [-1] * 3, "upper_m": [1] * 3} for s in ("left", "right")},
    }
    return cfg, cal, region, rows


def tracking(cfg, t, i, x=0):
    return {
        "identity": identity(cfg),
        "receive_ns": t,
        "source_ns": t,
        "sequence": i,
        "frame_id": "world",
        **{s: [x, 0, 0, 0, 0, 0, 1] for s in ("left", "right")},
    }


def compare(mapper):
    incremental = QuestSE3Mapper(
        max_translation_jump_m=0.2, max_translation_step_m=0.02
    )
    accepted = np.zeros(3)
    desired = []
    xs = [0, 0.08, 0.09, 0, 0, 0.08, 0.09, 0]
    for i, x in enumerate(xs):
        t = 1_000_000_000 + i * STEP_NS
        sample = TrackingSample(
            TrackedPose.make([x, 0, 0], [0, 0, 0, 1]),
            TrackedPose.make([x, 0, 0], [0, 0, 0, 1]),
            i,
            t,
            t,
            "world",
        )
        delta = incremental.update(sample, deadman=True, now_ns=t)
        accepted += delta.left_xyz
        desired.append(mapper.targets(tracking(mapper.raw, t, i, x))["left"].p.tolist())
    return {
        "input": "synthetic-fixed-diagnostic-v1",
        "positions_m": xs,
        "incremental_revisit_error_m": float(np.linalg.norm(accepted)),
        "world_absolute_revisit_error_m": float(
            np.linalg.norm(np.asarray(desired[-1]) - desired[0])
        ),
        "world_absolute_mapping_drift_m": 0.0,
        "incremental_final_target_m": accepted.tolist(),
        "explanation": "incremental frame clipping loses displacement; fixed mapping revisits exactly; execution lag is separate",
        "nullspace_physical_benefit": "NOT_RUN",
    }


def run(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    assets = root / "inputs"
    assets.mkdir()
    cfg, cal, region, reference_rows = diagnostic_assets()
    (assets / "reference_samples.json").write_text(json.dumps(reference_rows, indent=2))
    mapper = FixedMapping(cfg, cal)
    c = WorldController(mapper, Limits(dwell=0.1), ApprovedRegion(region))
    for name, value in (
        ("installation", cfg),
        ("calibration", cal),
        (
            "execution",
            {
                "schema_version": 1,
                "limits": asdict(c.limits),
                "approved_region": region,
            },
        ),
    ):
        (assets / (name + ".yaml")).write_text(yaml.safe_dump(value))
    tool = assets / "tool.yaml"
    tool.write_text("tool: synthetic-only\n")
    ft = assets / "ft.json"
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
    clock = [2_000_000_000]
    effective = {
        "teleop": {
            "mapping_mode": "world_absolute",
            **{
                "world_" + key: str(assets / (name + ".yaml"))
                for key, name in (
                    ("installation", "installation"),
                    ("calibration", "calibration"),
                    ("execution_config", "execution"),
                )
            },
        },
        "sampling": {
            "teleop_command_hz": 60,
            "camera_hz": 15,
            "arm_observation_hz": 60,
            "hand_state_hz": 60,
        },
    }
    session = EpisodeSession(
        root / "episodes",
        "synthetic",
        {"world": assets / "calibration.yaml"},
        tool,
        ft,
        record_ros_mcap=False,
        deviceio_profile="training",
        effective_config=effective,
        clock_ns=lambda: clock[0],
        task_name="stage_b_synthetic",
        task_description="No hardware; fixed mapping and Home continuity diagnostic",
    )

    class Emitter:
        def emit(self, document, **kwargs):
            payload = document["payload"]
            session.submit(
                envelope(document["topic"], clock[0], clock[0], payload), critical=True
            )
            return True

    backend = MockBackend()
    interlock = DaemonInterlock()
    interlock.ready_generation = backend.connection_generation
    dispatcher = RDKRequestDispatcher(
        backend,
        NS(is_zeroed_for=lambda s: True),
        HandObservationCache(),
        interlock,
        deviceio_emitter=Emitter(),
        control_authorizations=NS(consume=lambda *a, **kw: None),
        world_absolute_enabled=True,
    )
    codec = TypedEnvelopeCodec.load()
    now = clock[0]
    actual = {"receive_ns": now, **cal["robot_reference"]}
    c.start(tracking(cfg, now, 0), actual, 1, now)
    image = BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image, format="JPEG")
    jpeg_b64 = base64.b64encode(image.getvalue()).decode()
    state_before = None
    anchors = {}
    max_speed = 0
    max_acc = 0
    prev_v = np.zeros(3)
    try:
        for i in range(1, 181):
            t = 2_000_000_000 + i * STEP_NS
            clock[0] = t
            if i == 90:
                c.home()
                session.submit(
                    envelope(
                        "episode/events",
                        t - 2,
                        90,
                        {"event_type": "home_started", "effective_time_ns": t - 2},
                    ),
                    critical=True,
                )
                c.home_complete(True)
                session.submit(
                    envelope(
                        "episode/events",
                        t - 1,
                        91,
                        {"event_type": "home_completed", "effective_time_ns": t - 1},
                    ),
                    critical=True,
                )
                actual = (
                    {"receive_ns": t, **{s: p.wire() for s, p in c.accepted.items()}}
                    if c.accepted
                    else {"receive_ns": t, **cal["robot_reference"]}
                )
                c.last_tracking = None
                c.start(tracking(cfg, t, 90), actual, 1, t)
                continue
            actual = {"receive_ns": t, **{s: p.wire() for s, p in c.accepted.items()}}
            x = 0.003 if (25 <= i < 65 or 120 <= i < 160) else 0
            point = c.propose(tracking(cfg, t, i, x), actual, 1, t)
            state = c.state
            if state != state_before:
                session.submit(
                    envelope(
                        "episode/events",
                        t - 1,
                        i,
                        {
                            "event_type": "world_control_changed",
                            "effective_time_ns": t - 1,
                            "discontinuity": True,
                            "world": c.audit(),
                        },
                    ),
                    critical=True,
                )
                state_before = state
            session.submit(
                envelope("maintenance/world_state", t, i, c.audit()), critical=True
            )
            anchors[i] = [c.accepted[s].p.tolist() for s in ("left", "right")]
            # Production daemon validates targets and records each real simulated SDK boundary.
            targets = {}
            for s in ("left", "right"):
                p = c.limited[s]
                q = matrix_to_quaternion_xyzw(p.r)
                targets[s] = {
                    "tcp_pose_rdk": [*p.p.tolist(), q[3], *q[:3].tolist()],
                    "control_mode": "impedance",
                    "max_linear_velocity": 0.05,
                    "max_angular_velocity": 0.2,
                    "max_linear_acceleration": 0.2,
                    "max_angular_acceleration": 0.5,
                    "cartesian_stiffness": [1200] * 3 + [80] * 3,
                    "cartesian_damping_ratio": [0.7] * 6,
                }
            wire = {
                "session_id": "synthetic",
                "source": "teleop",
                "source_sequence": str(i),
                "expires_monotonic_ns": str(time.monotonic_ns() + 100_000_000),
                "valid_mask": 3,
                "safety_validated": True,
                "local_permission": True,
                "physical_pedal": False,
                "world_session_authority": True,
                "mapping_epoch": cal["mapping_epoch"],
                **targets,
            }
            _, _, wire = codec.decode(codec.encode("cartesian_command", i, wire))
            kind, ack = dispatcher("cartesian_command", i, wire, (222, 0, 0))
            _, _, ack = codec.decode(codec.encode(kind, i, ack))
            c.acknowledge(ack["sdk_returned_sides"], complete=ack["accepted"])
            v = point.left_delta_xyz / (STEP_NS / 1e9)
            max_speed = max(max_speed, float(np.linalg.norm(v)))
            max_acc = max(max_acc, float(np.linalg.norm(v - prev_v) / (STEP_NS / 1e9)))
            prev_v = v
            # Deliberately distinct causal measured state, finite and small tracking lag.
            for name in REQUIRED:
                if name.startswith("camera/"):
                    if i % 4:
                        continue
                    value = {"jpeg_b64": jpeg_b64, "synthetic_image": True}
                elif "arm" in name:
                    side = name.split("/")[1].split("_")[0]
                    value = {
                        "q": [0.2] * 7,
                        "tcp_pose_world_xyz_wxyz": [
                            c.accepted[side].p[0] - 0.0001,
                            0,
                            0,
                            1,
                            0,
                            0,
                            0,
                        ],
                    }
                else:
                    value = {"angle": [300] * 6, "position": [300] * 6}
                session.submit(envelope(name, t - 1_000_000, i, value))
            payload = command_payload(i)
            payload["metadata_keys"] = [
                "world_state",
                "mapping_epoch",
                "execution_epoch",
            ]
            payload["metadata_values"] = [
                state,
                cal["mapping_epoch"],
                c.execution_epoch,
            ]
            for s in ("left", "right"):
                payload["trajectory"][0][s + "_delta_xyz"] = getattr(
                    point, s + "_delta_xyz"
                ).tolist()
                payload["trajectory"][0][s + "_delta_rotation6d"] = getattr(
                    point, s + "_delta_rotation6d"
                ).tolist()
                payload["trajectory"][0][s + "_hand_targets"] = [300] * 6
            session.submit(
                envelope("control/sent_command", t, i, payload), critical=True
            )
        clock[0] += 2 * STEP_NS
        session.finish(NS(enabled=False), reason="synthetic-complete")
    finally:
        dispatcher.close()
    verify_archive(session.directory)
    resolve_asset(session.directory, "effective_config")
    assert (
        len(
            yaml.safe_load(
                resolve_asset(session.directory, "reference_samples.json").read_text()
            )
        )
        == cal["capture_count"]
    )
    records = records_from_mcap(session.device_path)
    document = export_supervision(records, episode_id=session.episode_uuid)
    (root / "command_supervision_v2.json").write_text(json.dumps(document, indent=2))
    assert document["quality"]["valid_samples"] > 100, document["quality"]
    segments = {}
    for row in document["rows"]:
        if row["valid"]:
            segments.setdefault(row["segment_id"], []).append(row)
    # Every valid continuous segment has its own execution anchor; no cross-epoch integration.
    for rows in segments.values():
        first = int(rows[0]["command_id"].split(":")[-1])
        reconstruct(rows, initial_xyz=np.asarray(anchors[first]))
    streams = load_json_mcap_streams([session.device_path])
    assert any(
        not r.valid and r.invalid_reason == "world-maintenance-not-follow"
        for r in streams["control/sent_command"]
    )
    comparison = compare(mapper)
    (root / "mode_comparison.json").write_text(json.dumps(comparison, indent=2))
    assert (
        comparison["world_absolute_revisit_error_m"] < 1e-12
        and comparison["incremental_revisit_error_m"] > 0.01
    )
    result = {
        "status": "PASS",
        "hardware": "NOT_RUN",
        "synthetic": True,
        "raw_commands": document["quality"]["raw_commands"],
        "valid_follow_commands": document["quality"]["valid_samples"],
        "continuous_segments": len(segments),
        "horizon_8_windows": len(list(windows(document, 8))),
        "max_limited_linear_speed_m_s": max_speed,
        "max_limited_linear_acceleration_m_s2": max_acc,
        "mapping_epoch": cal["mapping_epoch"],
        "manifest": str(session.manifest_path),
        "mode_comparison": comparison,
    }
    (root / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
