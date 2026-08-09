"""Capture and finalize site-specific MANUS-to-Inspire endpoint calibration."""

from __future__ import annotations

import argparse
import copy
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from .manus_pose import pose_message_values, split_bimanual_pose_array

CAPTURE_SCHEMA = "MANUS_OPENXR_FEATURE_CAPTURE_V1"
CALIBRATION_SOURCE = "ISAAC_XR_HAND_POSEARRAY_LEFT25_RIGHT25"
ERGONOMICS_CAPTURE_SCHEMA = "MANUS_SDK_ERGONOMICS_CAPTURE_V1"
ERGONOMICS_CALIBRATION_SOURCE = "MANUS_SDK_ERGONOMICS_RADIANS"


def _write_new_yaml(path: Path, document: dict) -> None:
    target = path.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def summarize_samples(
    samples: list[dict[str, dict[str, float]]], pose_name: str
) -> dict:
    if pose_name not in {"open", "closed"}:
        raise ValueError("pose_name must be open or closed")
    if not samples:
        raise ValueError("at least one valid MANUS sample is required")
    output = {
        "schema": CAPTURE_SCHEMA,
        "pose": pose_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(samples),
        "sides": {},
    }
    for side in ("left", "right"):
        names = set(samples[0][side])
        if any(set(sample[side]) != names for sample in samples):
            raise ValueError(f"{side} feature sets differ between samples")
        output["sides"][side] = {}
        for name in sorted(names):
            values = np.asarray(
                [sample[side][name] for sample in samples], dtype=np.float64
            )
            output["sides"][side][name] = {
                "median": float(np.median(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
    return output


def _capture_value(document: dict, side: str, feature: str) -> float:
    return float(document["sides"][side][feature]["median"])


def build_calibration(template: dict, opened: dict, closed: dict) -> dict:
    for name, document, expected_pose in (
        ("open", opened, "open"),
        ("closed", closed, "closed"),
    ):
        if (
            document.get("schema") != CAPTURE_SCHEMA
            or document.get("pose") != expected_pose
        ):
            raise ValueError(f"{name} capture has the wrong schema or pose")
    if (
        template.get("schema_version") != 1
        or template.get("source_format") != CALIBRATION_SOURCE
    ):
        raise ValueError("unsupported MANUS calibration template")
    result = copy.deepcopy(template)
    # YAML merge anchors in the template make left/right actuator mappings
    # share the same Python dictionaries after ``safe_load``.  A normal
    # deepcopy preserves that aliasing, so calibrating the right hand would
    # overwrite the endpoints already computed for the left hand.  Rebuild
    # every actuator mapping independently before inserting site endpoints.
    result["sides"] = {
        side: {
            actuator: copy.deepcopy(channel)
            for actuator, channel in template["sides"][side].items()
        }
        for side in ("left", "right")
    }
    # These are template-only helpers; retaining them would reintroduce YAML
    # anchors in the finalized runtime document.
    result.pop("channel_defaults", None)
    result.pop("side_channels", None)
    result["calibrated"] = True
    result["calibration"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "open_samples": int(opened["sample_count"]),
        "closed_samples": int(closed["sample_count"]),
    }
    for side in ("left", "right"):
        for actuator, channel in result["sides"][side].items():
            sources = [str(value) for value in channel["sources"]]
            weights = [float(value) for value in channel["weights"]]
            bias = float(channel.get("bias", 0.0))
            open_value = bias + sum(
                weight * _capture_value(opened, side, source)
                for source, weight in zip(sources, weights, strict=True)
            )
            closed_value = bias + sum(
                weight * _capture_value(closed, side, source)
                for source, weight in zip(sources, weights, strict=True)
            )
            if abs(closed_value - open_value) < 0.03:
                raise ValueError(
                    f"{side}.{actuator} open/closed separation is too small"
                )
            channel["source_open"] = float(open_value)
            channel["source_closed"] = float(closed_value)
    return result


def ergonomics_message_values(message: JointState) -> dict[str, float]:
    if len(message.name) != len(message.position):
        raise ValueError("Ergonomics JointState names/positions differ")
    if not message.name or len(set(message.name)) != len(message.name):
        raise ValueError("Ergonomics JointState names are empty or duplicated")
    values = {
        str(name): float(value)
        for name, value in zip(message.name, message.position, strict=True)
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Ergonomics JointState contains NaN or Inf")
    return values


def summarize_ergonomics_samples(
    samples: dict[str, list[dict[str, float]]], pose_name: str
) -> dict:
    if pose_name not in {"open", "closed"}:
        raise ValueError("pose_name must be open or closed")
    if any(not samples.get(side) for side in ("left", "right")):
        raise ValueError("both MANUS gloves require valid Ergonomics samples")
    output = {
        "schema": ERGONOMICS_CAPTURE_SCHEMA,
        "source_format": ERGONOMICS_CALIBRATION_SOURCE,
        "pose": pose_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sides": {},
    }
    for side in ("left", "right"):
        side_samples = samples[side]
        names = set(side_samples[0])
        if any(set(sample) != names for sample in side_samples):
            raise ValueError(f"{side} Ergonomics fields differ between samples")
        output["sides"][side] = {
            "sample_count": len(side_samples),
            "fields": {},
        }
        for name in sorted(names):
            values = np.asarray(
                [sample[name] for sample in side_samples], dtype=np.float64
            )
            output["sides"][side]["fields"][name] = {
                "median": float(np.median(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
    return output


def _ergonomics_capture_value(document: dict, side: str, source: str) -> float:
    try:
        return float(document["sides"][side]["fields"][source]["median"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"capture is missing {side}.{source}") from exc


def _ergonomics_channel_sources(
    channel: dict, side: str, actuator: str
) -> tuple[list[str], list[float], bool]:
    """Return sources, semantic weights, and whether per-source normalization is used."""
    legacy_source = str(channel.get("source", "")).strip()
    raw_sources = channel.get("sources")
    if legacy_source and raw_sources is not None:
        raise ValueError(f"{side}.{actuator} cannot define source and sources")
    if legacy_source:
        return [legacy_source], [1.0], False
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"invalid {side}.{actuator} Ergonomics sources")
    sources = [str(value).strip() for value in raw_sources]
    if any(not source for source in sources) or len(set(sources)) != len(sources):
        raise ValueError(f"invalid {side}.{actuator} Ergonomics sources")
    try:
        weights = [float(value) for value in channel.get("weights", [])]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {side}.{actuator} Ergonomics weights") from exc
    if (
        len(weights) != len(sources)
        or not all(np.isfinite(weight) and weight >= 0.0 for weight in weights)
        or sum(weights) <= 0.0
    ):
        raise ValueError(f"invalid {side}.{actuator} Ergonomics weights")
    return sources, weights, True


def build_ergonomics_calibration(template: dict, opened: dict, closed: dict) -> dict:
    for name, document, expected_pose in (
        ("open", opened, "open"),
        ("closed", closed, "closed"),
    ):
        if (
            document.get("schema") != ERGONOMICS_CAPTURE_SCHEMA
            or document.get("source_format") != ERGONOMICS_CALIBRATION_SOURCE
            or document.get("pose") != expected_pose
        ):
            raise ValueError(f"{name} Ergonomics capture has wrong schema/pose")
    if (
        template.get("schema_version") != 2
        or template.get("source_format") != ERGONOMICS_CALIBRATION_SOURCE
    ):
        raise ValueError("unsupported MANUS Ergonomics calibration template")
    result = copy.deepcopy(template)
    result["sides"] = {
        side: {
            actuator: copy.deepcopy(channel)
            for actuator, channel in template["sides"][side].items()
        }
        for side in ("left", "right")
    }
    result.pop("side_channels", None)
    result["calibrated"] = True
    result["calibration"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "operator_open_closed_ergonomics",
        "open_samples": {
            side: int(opened["sides"][side]["sample_count"])
            for side in ("left", "right")
        },
        "closed_samples": {
            side: int(closed["sides"][side]["sample_count"])
            for side in ("left", "right")
        },
    }
    for side in ("left", "right"):
        for actuator, channel in result["sides"][side].items():
            sources, _weights, normalized = _ergonomics_channel_sources(
                channel, side, actuator
            )
            open_values = [
                _ergonomics_capture_value(opened, side, source)
                for source in sources
            ]
            closed_values = [
                _ergonomics_capture_value(closed, side, source)
                for source in sources
            ]
            for source, open_value, closed_value in zip(
                sources, open_values, closed_values, strict=True
            ):
                if abs(closed_value - open_value) < 0.03:
                    raise ValueError(
                        f"{side}.{actuator}.{source} open/closed separation "
                        "is too small"
                    )
            output_open = float(channel.pop("output_open", 1000.0))
            output_closed = float(channel.pop("output_closed", 0.0))
            if normalized:
                full_close_progress = float(
                    channel.pop("full_close_progress", 1.0)
                )
                if not 0.0 < full_close_progress <= 1.0:
                    raise ValueError(
                        f"invalid {side}.{actuator} full_close_progress"
                    )
                channel["source_open"] = [float(value) for value in open_values]
                channel["source_closed"] = [
                    float(value) for value in closed_values
                ]
                channel["points"] = [
                    [0.0, output_open],
                    [full_close_progress, output_closed],
                ]
                if full_close_progress < 1.0:
                    channel["points"].append([1.0, output_closed])
            else:
                channel["points"] = [
                    [float(open_values[0]), output_open],
                    [float(closed_values[0]), output_closed],
                ]
    return result


class _CaptureNode(Node):
    def __init__(self, topic: str, frame_count: int) -> None:
        super().__init__("manus_calibration_capture")
        self.samples: list[dict[str, dict[str, float]]] = []
        self.frame_count = frame_count
        self.create_subscription(
            PoseArray, topic, self._on_pose, qos_profile_sensor_data
        )

    def _on_pose(self, message: PoseArray) -> None:
        if len(self.samples) >= self.frame_count:
            return
        try:
            self.samples.append(
                split_bimanual_pose_array(pose_message_values(message.poses))
            )
        except Exception as exc:
            self.get_logger().warning(
                f"discarding invalid MANUS frame: {exc}",
                throttle_duration_sec=1.0,
            )


class _ErgonomicsCaptureNode(Node):
    def __init__(self, left_topic: str, right_topic: str, frame_count: int) -> None:
        super().__init__("manus_ergonomics_calibration_capture")
        self.samples: dict[str, list[dict[str, float]]] = {
            "left": [],
            "right": [],
        }
        self.frame_count = frame_count
        for side, topic in (("left", left_topic), ("right", right_topic)):
            self.create_subscription(
                JointState,
                topic,
                lambda message, selected=side: self._on_message(selected, message),
                qos_profile_sensor_data,
            )

    def _on_message(self, side: str, message: JointState) -> None:
        if len(self.samples[side]) >= self.frame_count:
            return
        try:
            self.samples[side].append(ergonomics_message_values(message))
        except (TypeError, ValueError) as exc:
            self.get_logger().warning(
                f"discarding invalid {side} Ergonomics frame: {exc}",
                throttle_duration_sec=1.0,
            )


def capture(*, topic: str, pose_name: str, frames: int, timeout_s: float) -> dict:
    if frames < 10:
        raise ValueError("frames must be at least 10")
    if timeout_s <= 0.0:
        raise ValueError("timeout must be positive")
    rclpy.init(args=[])
    node = _CaptureNode(topic, frames)
    deadline = time.monotonic() + timeout_s
    try:
        while len(node.samples) < frames and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if len(node.samples) < frames:
            raise TimeoutError(
                f"received {len(node.samples)}/{frames} valid MANUS frames"
            )
        return summarize_samples(node.samples, pose_name)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def capture_ergonomics(
    *,
    left_topic: str,
    right_topic: str,
    pose_name: str,
    frames: int,
    timeout_s: float,
) -> dict:
    if frames < 10:
        raise ValueError("frames must be at least 10")
    if timeout_s <= 0.0:
        raise ValueError("timeout must be positive")
    rclpy.init(args=[])
    node = _ErgonomicsCaptureNode(left_topic, right_topic, frames)
    deadline = time.monotonic() + timeout_s
    try:
        while (
            any(len(node.samples[side]) < frames for side in ("left", "right"))
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
        missing = {
            side: f"{len(node.samples[side])}/{frames}"
            for side in ("left", "right")
            if len(node.samples[side]) < frames
        }
        if missing:
            raise TimeoutError(f"insufficient MANUS Ergonomics frames: {missing}")
        return summarize_ergonomics_samples(node.samples, pose_name)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate raw /xr_teleop/hand poses for Inspire actuators"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--pose", choices=("open", "closed"), required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--topic", default="/xr_teleop/hand")
    capture_parser.add_argument("--frames", type=int, default=90)
    capture_parser.add_argument("--timeout", type=float, default=20.0)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--open", type=Path, required=True)
    finalize_parser.add_argument("--closed", type=Path, required=True)
    finalize_parser.add_argument("--template", type=Path, required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)
    ergonomics_capture_parser = subparsers.add_parser("capture-ergonomics")
    ergonomics_capture_parser.add_argument(
        "--pose", choices=("open", "closed"), required=True
    )
    ergonomics_capture_parser.add_argument("--output", type=Path, required=True)
    ergonomics_capture_parser.add_argument(
        "--left-topic", default="/manus/left/ergonomics"
    )
    ergonomics_capture_parser.add_argument(
        "--right-topic", default="/manus/right/ergonomics"
    )
    ergonomics_capture_parser.add_argument("--frames", type=int, default=90)
    ergonomics_capture_parser.add_argument("--timeout", type=float, default=20.0)
    ergonomics_finalize_parser = subparsers.add_parser("finalize-ergonomics")
    ergonomics_finalize_parser.add_argument("--open", type=Path, required=True)
    ergonomics_finalize_parser.add_argument("--closed", type=Path, required=True)
    ergonomics_finalize_parser.add_argument("--template", type=Path, required=True)
    ergonomics_finalize_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "capture":
        document = capture(
            topic=args.topic,
            pose_name=args.pose,
            frames=args.frames,
            timeout_s=args.timeout,
        )
        _write_new_yaml(args.output, document)
    elif args.command == "finalize":
        template = yaml.safe_load(args.template.expanduser().read_text())
        opened = yaml.safe_load(args.open.expanduser().read_text())
        closed = yaml.safe_load(args.closed.expanduser().read_text())
        _write_new_yaml(args.output, build_calibration(template, opened, closed))
    elif args.command == "capture-ergonomics":
        document = capture_ergonomics(
            left_topic=args.left_topic,
            right_topic=args.right_topic,
            pose_name=args.pose,
            frames=args.frames,
            timeout_s=args.timeout,
        )
        _write_new_yaml(args.output, document)
    else:
        template = yaml.safe_load(args.template.expanduser().read_text())
        opened = yaml.safe_load(args.open.expanduser().read_text())
        closed = yaml.safe_load(args.closed.expanduser().read_text())
        _write_new_yaml(
            args.output,
            build_ergonomics_calibration(template, opened, closed),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
