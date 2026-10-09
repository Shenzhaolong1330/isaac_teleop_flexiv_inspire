"""Read-only capture and explicit fixed mapping asset creation; never commands hardware."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import yaml
from .world_teleop import FixedMapping, installation, calibrate, identity


def capture(output, duration):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from flexiv_inspire_interfaces.msg import BimanualCommand, ArmState

    rclpy.init()
    node = Node("world_reference_capture_read_only")
    cache, rows = {}, []

    def human(msg):
        metadata = dict(zip(msg.metadata_keys, msg.metadata_values))
        if "world_tracking" in metadata:
            cache["human"] = json.loads(metadata["world_tracking"])

    def robot(side, msg):
        if (
            not msg.connected
            or msg.fault
            or not msg.acquisition.valid
            or not msg.acquisition.timing_valid
            or msg.acquisition.host_clock_domain != "host_monotonic"
        ):
            cache.pop(side, None)
            return
        pose = msg.tcp_pose
        cache[side] = {
            "pose": [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            "time": int(msg.acquisition.mapped_host_time.sec) * 1_000_000_000
            + int(msg.acquisition.mapped_host_time.nanosec),
        }

    node.create_subscription(BimanualCommand, "/teleop/command", human, 10)
    for side in ("left", "right"):
        node.create_subscription(
            ArmState,
            f"/robot/{side}_arm/state",
            lambda m, s=side: robot(s, m),
            qos_profile_sensor_data,
        )
    deadline = time.monotonic() + duration
    last = -1
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            if not all(k in cache for k in ("human", "left", "right")):
                continue
            h = cache["human"]
            if h["sequence"] == last:
                continue
            last = h["sequence"]
            now = time.monotonic_ns()
            rows.append(
                {
                    "capture_ns": now,
                    "identity": h["identity"],
                    "human_source_ns": h["source_ns"],
                    "human": {
                        "left": h["left"],
                        "right": h["right"],
                        "left_time_ns": h["receive_ns"],
                        "right_time_ns": h["receive_ns"],
                    },
                    "robot": {
                        "left": cache["left"]["pose"],
                        "right": cache["right"]["pose"],
                        "left_time_ns": cache["left"]["time"],
                        "right_time_ns": cache["right"]["time"],
                    },
                }
            )
        with Path(output).open("x") as handle:
            handle.write(json.dumps(rows, indent=2))
        print(
            json.dumps({"samples": len(rows), "read_only": True, "output": str(output)})
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("output")
    cap.add_argument("--duration-s", type=float, default=3.0)
    cal = sub.add_parser("calibrate")
    cal.add_argument("installation")
    cal.add_argument("samples")
    cal.add_argument("output_dir")
    for name in ("validate", "preview"):
        p = sub.add_parser(name)
        p.add_argument("installation")
        p.add_argument("calibration")
        if name == "preview":
            p.add_argument(
                "tracking", help="JSON list of live-format tracking snapshots"
            )
    args = parser.parse_args()
    if args.operation == "capture":
        if not 0.2 <= args.duration_s <= 60:
            parser.error("capture duration must be [0.2,60] s")
        capture(args.output, args.duration_s)
        return
    if args.operation == "calibrate":
        raw = installation(yaml.safe_load(Path(args.installation).read_text()))
        rows = json.loads(Path(args.samples).read_text())
        # Boot identity can only be bound by this explicit calibration operation.
        raw["input_session_id"] = rows[0]["identity"]["input_session_id"]
        result = calibrate(raw, rows)
        result["reference_samples"] = "reference_samples.json"
        dest = Path(args.output_dir)
        dest.mkdir(parents=True, exist_ok=False)
        (dest / "reference_samples.json").write_text(json.dumps(rows, indent=2))
        (dest / "installation.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
        (dest / "calibration.yaml").write_text(yaml.safe_dump(result, sort_keys=False))
        print(
            json.dumps(
                {
                    "mapping_epoch": result["mapping_epoch"],
                    "identity": identity(raw),
                    "output": str(dest),
                }
            )
        )
        return
    mapper = FixedMapping.load(args.installation, args.calibration)
    if args.operation == "preview":
        for row in json.loads(Path(args.tracking).read_text()):
            print(json.dumps({s: p.wire() for s, p in mapper.targets(row).items()}))
    else:
        print(
            json.dumps(
                {
                    "valid": True,
                    "mapping_epoch": mapper.calibration["mapping_epoch"],
                    "motion": False,
                }
            )
        )


if __name__ == "__main__":
    main()
