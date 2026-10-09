"""Executable command-rate supervision. No ROS, policy framework or hardware imports."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from isaac_teleop_core.rotation6d import rotation6d_to_matrix
from .mcap_input import _flatten_safe_command_checked

VERSION = "command-supervision-v2"
ACTION_HZ = 60.0
STEP_NS = round(1e9 / ACTION_HZ)
CONTRACT = {
    "schema": VERSION,
    "command_hz": ACTION_HZ,
    "action_duration_s": 1 / ACTION_HZ,
    "arm": "world translation delta in metres; R_next = R_delta @ R_previous",
    "rotation": "first two matrix columns",
    "hand": "six absolute targets per hand; never summed",
    "observation": "latest valid source at or before command start, within age budget and segment",
    "policy_call_hz": "independent; chunks allowed",
    "servo_hz": "controller internal, unknown",
    "missing_command": "explicit invalid slots; never repeat increments",
    "execution_evidence": "bridge publication is not all-actuator success",
}
REQUIRED = {
    "robot/left_arm/state": 20_000_000,
    "robot/right_arm/state": 20_000_000,
    "robot/left_hand/state": 150_000_000,
    "robot/right_hand/state": 150_000_000,
    "camera/head/color/image_raw/compressed": 100_000_000,
    "camera/left_wrist/color/image_raw/compressed": 100_000_000,
    "camera/right_wrist/color/image_raw/compressed": 100_000_000,
}
OPTIONAL = {f"robot/{side}_hand/tactile_raw": 100_000_000 for side in ("left", "right")}
BOUNDARIES = {
    "episode_started",
    "episode_paused",
    "episode_resumed",
    "episode_stopped",
    "episode_failed",
    "pedal_pressed",
    "pedal_released",
    "home_started",
    "home_completed",
    "home_failed",
    "fault",
    "recovered",
    "control_changed",
    "world_control_changed",
    "recording_started",
    "recording_stopped",
}


def command_id(session, source, sequence):
    return f"{session}:{source}:{int(sequence)}"


def timestamp(record):
    if record.get("timing_valid") is False:
        return None
    mapped = record.get("mapped_host_time_ns")
    if mapped is not None:
        return int(mapped)
    if record.get("source_clock_domain", "host_monotonic") == record.get(
        "host_clock_domain", "host_monotonic"
    ):
        return int(record["source_time_ns"])
    return None


def records_from_mcap(path):
    from mcap.reader import make_reader

    with Path(path).open("rb") as handle:
        return [
            json.loads(message.data)
            for _, _, message in make_reader(handle).iter_messages()
        ]


def event_time(record):
    payload = record.get("payload", {})
    return int(
        payload.get(
            "effective_time_ns", timestamp(record) or record["host_receive_time_ns"]
        )
    )


def boundaries(records):
    """Use event time, never writer ordering. Keep event IDs for provenance."""
    result = []
    for record in records:
        topic = record["topic"].lstrip("/")
        payload = record.get("payload", {})
        if topic not in {
            "episode/events",
            "maintenance/events",
            "control/state",
            "teleop/deadman",
        }:
            continue
        kind = payload.get("event_type", "")
        if kind in {"pedal_pressed", "pedal_released"} and payload.get("recording_gate") is False:
            continue
        if kind in BOUNDARIES or payload.get("discontinuity", False):
            result.append(
                (
                    event_time(record),
                    str(payload.get("event_id", f"{topic}:{record['sequence']}")),
                    kind,
                )
            )
    return sorted(set(result))


def _finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def modality_valid(name, sample, *, time_ns=None, max_age_ns=None):
    if not sample or not sample.get("valid") or not _finite(sample["payload"]):
        return False
    value = sample["payload"]
    if name.endswith("_arm/state"):
        if not isinstance(value, dict):
            return False
        try:
            q = np.asarray(value.get("q", []), dtype=float)
            return q.shape == (7,) and bool(np.all(np.isfinite(q)))
        except (TypeError, ValueError):
            return False
    if name.endswith("_hand/state"):
        if not isinstance(value, dict):
            return False
        try:
            angles = np.asarray(value.get("angle", []), dtype=float)
            if angles.shape != (6,) or not np.all(np.isfinite(angles)):
                return False
        except (TypeError, ValueError):
            return False
        timing = value.get("angle_timing")
        if timing is not None:
            if not timing.get("valid") or not timing.get("timing_valid"):
                return False
            mapped = timing.get("mapped_host_time", {})
            mapped_ns = int(mapped.get("sec", 0)) * 1_000_000_000 + int(
                mapped.get("nanosec", 0)
            )
            if time_ns is not None and not 0 <= time_ns - mapped_ns <= max_age_ns:
                return False
    if name.endswith("/image_raw/compressed"):
        return (
            isinstance(value, dict)
            and isinstance(value.get("jpeg_b64"), str)
            and bool(value["jpeg_b64"])
        )
    if name.endswith("tactile_raw") and isinstance(value, dict) and "surfaces" in value:
        surfaces = value["surfaces"]
        return (
            len(surfaces) == 17
            and sum(len(surface.get("taxels", [])) for surface in surfaces) == 1062
            and all(s.get("acquisition", {}).get("valid", False) for s in surfaces)
        )
    return True


def export_supervision(
    records, *, episode_id, required=None, optional=None, allow_legacy=False
):
    """One label per observed command; gap slots explicit; full raw stream retained by MCAP."""
    if any(
        record["topic"].lstrip("/") == "control/sent_command"
        and timestamp(record) is None
        for record in records
    ):
        raise ValueError(
            "command timing unmapped: cannot establish ordering or valid supervision; retain raw episode for diagnosis"
        )
    required = REQUIRED if required is None else required
    optional = OPTIONAL if optional is None else optional
    cuts = boundaries(records)
    recording_events = sorted(
        (
            (event_time(r), r["payload"])
            for r in records
            if r["topic"].lstrip("/") == "episode/events"
            and isinstance(r.get("payload"), dict)
        ),
        key=lambda item: item[0],
    )

    def recording_active(t):
        paused = False
        pedal = True
        world_follow = True
        for event_t, event in recording_events:
            if event_t > t:
                break
            kind = event.get("event_type")
            if kind == "world_control_changed":
                world_follow = event.get("world", {}).get("state") == "FOLLOW"
            if kind in {"episode_paused", "episode_stopped", "episode_failed"}:
                paused = True
            elif kind in {"episode_started", "episode_resumed"}:
                paused = False
            elif kind == "pedal_released" and event.get("recording_gate", True):
                pedal = False
            elif kind == "pedal_pressed":
                pedal = True
        return not paused and pedal and world_follow

    if not cuts and not allow_legacy:
        raise ValueError(
            "events missing: use explicit --allow-legacy; segment certainty remains inferred"
        )
    cut_times = [cut[0] for cut in cuts]
    streams = {}
    for record in records:
        time_ns = timestamp(record)
        if time_ns is not None:
            streams.setdefault(record["topic"].lstrip("/"), []).append(
                (time_ns, record)
            )
    for values in streams.values():
        values.sort(key=lambda item: (item[0], item[1]["sequence"]))
    indices = {name: [item[0] for item in values] for name, values in streams.items()}
    commands = streams.get("control/sent_command", [])
    receipts = {}
    for _, record in streams.get("control/actuator_evidence", []):
        payload = record.get("payload", {})
        if payload.get("schema") == "actuator-evidence-v2":
            receipts.setdefault(payload.get("command_id"), []).append(payload)
    for values in receipts.values():
        values.sort(key=lambda value: value["time_ns"])
    rows, rejected, seen = [], Counter(), set()
    previous_time, previous_segment = None, None
    inferred_segment = 0
    for time_ns, record in commands:
        payload = record["payload"]
        cid = command_id(
            payload.get("session_id", "unknown"),
            payload.get("source", "unknown"),
            payload.get("sequence", record["sequence"]),
        )
        if cid in seen:
            rejected["duplicate-command-id"] += 1
            continue
        seen.add(cid)
        segment = bisect_right(cut_times, time_ns)
        if not cuts:
            if previous_time is not None and time_ns - previous_time > 250_000_000:
                inferred_segment += 1
            segment = inferred_segment
        start_ns = cut_times[segment - 1] if cuts and segment else 0
        reasons = []
        if (
            not payload.get("session_id")
            or payload.get("source") not in {"teleop", "policy", "replay"}
            or "sequence" not in payload
        ):
            reasons.append("command-identity-missing-or-invalid")
        metadata = dict(zip(payload.get("metadata_keys", []), payload.get("metadata_values", [])))
        if metadata.get("world_state") and metadata["world_state"] != "FOLLOW":
            reasons.append("world-maintenance-not-follow")
        if not recording_active(time_ns):
            reasons.append("inactive-recording-interval")
        action, reason = _flatten_safe_command_checked(payload)
        if reason or not record.get("valid", False) or not _finite(action):
            reasons.append(reason or record.get("invalid_reason") or "invalid-command")
        if len(payload.get("trajectory", [])) != 1:
            reasons.append("sent-command-must-contain-one-executed-point")
        if action is not None:
            try:
                rotation6d_to_matrix(action[3:9])
                rotation6d_to_matrix(action[12:18])
            except ValueError:
                reasons.append("invalid-command-rotation")
        if previous_time is not None and previous_segment == segment:
            gap = time_ns - previous_time
            if gap < STEP_NS // 2:
                reasons.append("command-step-too-short")
            elif gap > STEP_NS * 1.5:
                missing = max(1, round(gap / STEP_NS) - 1)
                for index in range(missing):
                    rows.append(
                        {
                            "schema": VERSION,
                            "episode_id": episode_id,
                            "segment_id": segment,
                            "start_ns": previous_time + (index + 1) * STEP_NS,
                            "end_ns": previous_time + (index + 2) * STEP_NS,
                            "command_id": None,
                            "action": None,
                            "valid": False,
                            "reasons": ["missing-command"],
                            "observations": {},
                        }
                    )
                rejected["missing-command"] += missing
        observations = {}
        for name, max_age in {**optional, **required}.items():
            values = streams.get(name, [])
            i = bisect_right(indices.get(name, []), time_ns) - 1
            sample = values[i][1] if i >= 0 else None
            sample_time = values[i][0] if i >= 0 else None
            age = None if sample_time is None else time_ns - sample_time
            valid = bool(
                modality_valid(name, sample, time_ns=time_ns, max_age_ns=max_age)
                and sample_time >= start_ns
                and age <= max_age
                and _finite(sample["payload"])
            )
            why = (
                ""
                if valid
                else ("missing" if sample is None else "invalid-stale-or-cross-segment")
            )
            observations[name] = {
                "value": sample["payload"] if valid else None,
                "valid": valid,
                "source_time_ns": None if sample is None else sample["source_time_ns"],
                "mapped_time_ns": sample_time,
                "age_ns": age,
                "sequence": None if sample is None else sample["sequence"],
                "reason": why,
            }
            if name in required and not valid:
                reasons.append(f"required:{name}:{why}")
        for why in reasons:
            rejected[why] += 1
        rows.append(
            {
                "schema": VERSION,
                "episode_id": episode_id,
                "segment_id": segment,
                "boundary_basis": "explicit-event"
                if cuts
                else "legacy-gap-inference-uncertain",
                "event_ids": [cut[1] for cut in cuts if cut[0] == start_ns],
                "start_ns": time_ns,
                "end_ns": time_ns + STEP_NS,
                "source_time_ns": record["source_time_ns"],
                "command_id": cid,
                "action": action,
                "valid": not reasons,
                "reasons": reasons,
                "observations": observations,
                "actuator_evidence": {
                    unit: [
                        r for r in receipts.get(cid, []) if r.get("actuator") == unit
                    ]
                    for unit in ("left_arm", "right_arm", "left_hand", "right_hand")
                },
                "missing_receipts": [
                    unit
                    for unit in ("left_arm", "right_arm", "left_hand", "right_hand")
                    if not any(r.get("actuator") == unit for r in receipts.get(cid, []))
                ],
                "physical_arrival": "unknown",
            }
        )
        previous_time, previous_segment = time_ns, segment
    # A command interval must not overlap an explicit discontinuity.
    for row in rows:
        i = bisect_right(cut_times, row["start_ns"])
        if i < len(cut_times) and cut_times[i] < row["end_ns"]:
            row["valid"] = False
            row["reasons"].append("interval-crosses-boundary")
            rejected["interval-crosses-boundary"] += 1
    return {
        "contract": CONTRACT,
        "episode_id": episode_id,
        "rows": rows,
        "quality": {
            "raw_commands": len(commands),
            "valid_samples": sum(row["valid"] for row in rows),
            "rejections": dict(rejected),
        },
        "events": [dict(time_ns=t, event_id=e, event_type=k) for t, e, k in cuts],
    }


def windows(document, horizon):
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    rows = document["rows"]
    if document.get("contract", {}).get("schema") != VERSION:
        raise ValueError("unsupported supervision contract")
    for start in range(len(rows) - horizon + 1):
        chunk = rows[start : start + horizon]
        if (
            all(row["valid"] for row in chunk)
            and len({row["segment_id"] for row in chunk}) == 1
        ):
            yield chunk


def reconstruct(rows, *, initial_xyz=None, initial_rotation=None):
    """Reference consumer; the supplied anchors are control targets, never measured substitutes."""
    xyz = (
        np.zeros((2, 3))
        if initial_xyz is None
        else np.array(initial_xyz, dtype=float).copy()
    )
    rotation = (
        np.repeat(np.eye(3)[None], 2, axis=0)
        if initial_rotation is None
        else np.array(initial_rotation, dtype=float).copy()
    )
    hands = None
    seen = set()
    segments = set()
    previous_time = None
    for row in rows:
        if row.get("schema") != VERSION or not row["valid"]:
            raise ValueError("consumer requires valid command-supervision-v2 rows")
        if (
            previous_time is not None
            and not STEP_NS // 2 <= row["start_ns"] - previous_time <= STEP_NS * 1.5
        ):
            raise ValueError(
                "consumer requires contiguous commands or a new control anchor"
            )
        previous_time = row["start_ns"]
        segments.add((row["episode_id"], row["segment_id"]))
        if len(segments) > 1:
            raise ValueError(
                "consumer requires independent control anchors per segment"
            )
        if row["command_id"] in seen:
            raise ValueError("duplicate command")
        seen.add(row["command_id"])
        a = row["action"]
        for side, offset in enumerate((0, 9)):
            xyz[side] += a[offset : offset + 3]
            rotation[side] = (
                rotation6d_to_matrix(a[offset + 3 : offset + 9]) @ rotation[side]
            )
        hands = np.asarray(a[18:30]).reshape(2, 6)
    return xyz, rotation, hands


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export command-rate supervision, independent of LeRobot camera rows"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-legacy", action="store_true")
    options = parser.parse_args(argv)
    manifest = json.loads(options.manifest.read_text())
    if not manifest.get("completed"):
        raise ValueError("incomplete capture: cannot export training supervision")
    from .provenance import verify_archive

    if manifest.get("provenance"):
        verify_archive(options.manifest.parent)
    records = records_from_mcap(options.manifest.parent / manifest["deviceio_mcap"])
    document = export_supervision(
        records, episode_id=manifest["episode_uuid"], allow_legacy=options.allow_legacy
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(document, indent=2, allow_nan=False))
    print(json.dumps(document["quality"]))


if __name__ == "__main__":
    main()
