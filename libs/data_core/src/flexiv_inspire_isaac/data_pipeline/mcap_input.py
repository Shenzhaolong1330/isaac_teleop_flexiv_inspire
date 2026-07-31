"""Read JSON RecordEnvelope channels from episode MCAP truth files."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Iterable

from .alignment import Pose, TimedSample
from isaac_teleop_core.rotation6d import quaternion_xyzw_to_rotation6d
ROT6D_ORDER = "R00,R10,R20,R01,R11,R21"
QUATERNION_ORDER = "qx,qy,qz,qw"
FULL_VALID_MASK = 1 | 2 | 4 | 8


def _flatten_safe_command_checked(payload):
    if not isinstance(payload, dict):
        return None, "sent-command-payload-not-mapping"
    try:
        schema_version = int(payload.get("schema_version", 0))
        valid_mask = int(payload.get("valid_mask", 0))
    except (TypeError, ValueError):
        return None, "sent-command-schema-or-mask-not-integer"
    if (
        schema_version != 1
        or payload.get("frame_id") != "world"
        or valid_mask != FULL_VALID_MASK
        or not bool(payload.get("deadman", False))
    ):
        return None, "sent-command-schema-frame-mask-or-deadman-invalid"
    representation = payload.get("representation")
    if representation in (3, "JOINT_POSITION"):
        return None, "joint-position-sent-command-not-exportable-as-30d-cartesian"
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        return None, "sent-command-trajectory-missing"
    point = trajectory[-1]
    if not isinstance(point, dict):
        return None, "sent-command-point-not-mapping"
    try:
        left_xyz = list(point["left_delta_xyz"])
        right_xyz = list(point["right_delta_xyz"])
        left_hand = list(point["left_hand_targets"])
        right_hand = list(point["right_hand_targets"])
        if representation in (1, "CARTESIAN_ROT6D"):
            if payload.get("rotation_order") != ROT6D_ORDER:
                return None, "sent-command-rot6d-order-invalid"
            left_rotation = list(point["left_delta_rotation6d"])
            right_rotation = list(point["right_delta_rotation6d"])
        elif representation in (2, "CARTESIAN_QUATERNION"):
            if payload.get("rotation_order") != QUATERNION_ORDER:
                return None, "sent-command-quaternion-order-invalid"
            left_rotation = quaternion_xyzw_to_rotation6d(
                point["left_delta_quaternion_xyzw"]
            ).tolist()
            right_rotation = quaternion_xyzw_to_rotation6d(
                point["right_delta_quaternion_xyzw"]
            ).tolist()
        else:
            return None, "sent-command-representation-unsupported"
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"sent-command-conversion-failed:{exc}"
    parts = (left_xyz, left_rotation, right_xyz, right_rotation, left_hand, right_hand)
    if tuple(len(part) for part in parts) != (3, 6, 3, 6, 6, 6):
        return None, "sent-command-field-dimensions-invalid"
    try:
        values = [float(value) for part in parts for value in part]
    except (TypeError, ValueError) as exc:
        return None, f"sent-command-nonnumeric:{exc}"
    return (values, "") if len(values) == 30 else (None, "sent-command-not-30d")


def _flatten_safe_command(payload):
    values, _ = _flatten_safe_command_checked(payload)
    return values


def _payload_value(topic: str, payload):
    if topic.rstrip("/").endswith("control/sent_command"):
        return _flatten_safe_command(payload)
    if topic.endswith("/tcp_pose") and isinstance(payload, dict):
        return Pose(tuple(payload["xyz"]), tuple(payload["quaternion_xyzw"]))
    if "/camera/" in f"/{topic}" or topic.startswith("camera/"):
        if isinstance(payload, dict) and "jpeg_b64" in payload:
            return base64.b64decode(payload["jpeg_b64"])
    return payload


def _stream_name(topic: str) -> str:
    normalized = topic.lstrip("/")
    parts = normalized.split("/")
    if (
        len(parts) == 5
        and parts[0] == "camera"
        and parts[2:] == ["color", "image_raw", "compressed"]
    ):
        return f"camera/{parts[1]}/jpeg"
    return normalized

def load_json_mcap_streams(paths: Iterable[str | Path]) -> dict[str, list[TimedSample]]:
    try:
        from mcap.reader import make_reader
    except ImportError as exc:
        raise RuntimeError("install 'mcap' in envs/data-py312") from exc
    streams: dict[str, list[TimedSample]] = {}
    unsupported_channels: set[str] = set()
    for path in paths:
        with Path(path).open("rb") as handle:
            reader = make_reader(handle)
            for schema, channel, message in reader.iter_messages():
                if channel.message_encoding != "json":
                    unsupported_channels.add(channel.topic)
                    continue
                document = json.loads(message.data)
                if "source_time_ns" not in document or "payload" not in document:
                    continue
                raw_topic = str(document.get("topic", channel.topic)).lstrip("/")
                topic = _stream_name(raw_topic)
                if raw_topic.rstrip("/").endswith("control/sent_command"):
                    payload, safe_reason = _flatten_safe_command_checked(
                        document["payload"]
                    )
                    if payload is None:
                        document["valid"] = False
                        document["invalid_reason"] = ";".join(filter(None, (
                            str(document.get("invalid_reason", "")), safe_reason
                        )))
                else:
                    payload = _payload_value(raw_topic, document["payload"])
                source_domain = str(document.get("source_clock_domain", "unknown"))
                host_domain = str(document.get("host_clock_domain", "unknown"))
                mapped_raw = document.get("mapped_host_time_ns")
                mapped = None if mapped_raw is None else int(mapped_raw)
                timing_valid = bool(document.get("timing_valid", mapped is not None))
                effective_valid = bool(document["valid"])
                reason = str(document.get("invalid_reason", ""))
                if mapped is None or not timing_valid:
                    timing_reason = str(document.get("timing_invalid_reason", "timing-unmapped"))
                    reason = ";".join(filter(None, (reason, timing_reason)))
                    effective_valid = False
                streams.setdefault(topic, []).append(
                    TimedSample(
                        value=payload,
                        source_time_ns=int(document["source_time_ns"]),
                        host_receive_time_ns=int(document["host_receive_time_ns"]),
                        sequence=int(document["sequence"]),
                        valid=effective_valid,
                        invalid_reason=reason,
                        source_clock_domain=source_domain,
                        host_clock_domain=host_domain,
                        mapped_host_time_ns=mapped,
                        require_explicit_mapping=True,
                    )
                )
    for samples in streams.values():
        samples.sort(
            key=lambda sample: (
                sample.alignment_time_ns is None,
                sample.alignment_time_ns
                if sample.alignment_time_ns is not None
                else sample.host_receive_time_ns,
            )
        )
    if unsupported_channels and not streams:
        raise RuntimeError(
            "MCAP contains only non-JSON ROS CDR channels. Install the ROS "
            "decoder/export mirror and convert canonical observation topics to "
            "RecordEnvelope JSON before LeRobot export. Unsupported: "
            + ", ".join(sorted(unsupported_channels)[:10])
        )
    return streams
