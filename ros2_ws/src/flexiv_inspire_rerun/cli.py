"""Command line entry point for live ROS visualization and offline smoke."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m flexiv_inspire_isaac.rerun_viz.cli",
        description="Read-only, latest-only Rerun visualization",
    )
    sink = parser.add_mutually_exclusive_group()
    sink.add_argument("--save", metavar="FILE", help="write an .rrd recording")
    sink.add_argument("--connect", metavar="URL", help="connect to a Rerun gRPC viewer")
    sink.add_argument(
        "--spawn",
        action="store_true",
        help="spawn and connect to a local native viewer (default for live ROS)",
    )
    parser.add_argument("--viewer-port", type=int, default=9876)
    parser.add_argument(
        "--legacy-camera-topics",
        action="store_true",
        help=(
            "also subscribe to unpaired legacy CompressedImage topics; "
            "their timing is explicitly marked invalid"
        ),
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="generate deterministic data without importing ROS or touching hardware",
    )
    parser.add_argument("--frames", type=int, default=12, help="synthetic frame count")
    parser.add_argument(
        "--offline-config",
        metavar="YAML",
        help="play one DeviceIO episode from the shared playback YAML; never uses ROS",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments, ros_args = parser.parse_known_args(argv)
    if arguments.offline_config:
        if arguments.synthetic or arguments.save or arguments.connect or arguments.spawn:
            parser.error("--offline-config selects its sink from YAML")
        from .offline import run_offline

        return run_offline(arguments.offline_config)
    if arguments.synthetic:
        if arguments.connect or arguments.spawn:
            parser.error("--synthetic requires --save and never starts a viewer")
        if not arguments.save:
            parser.error("--synthetic requires --save FILE")
        from .synthetic import run_synthetic

        output = run_synthetic(arguments.save, frames=arguments.frames)
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"Rerun did not create a non-empty recording: {output}")
        print(f"wrote {output} ({output.stat().st_size} bytes)")
        return 0

    from .ros_node import run_ros

    save_path = str(Path(arguments.save).expanduser()) if arguments.save else None
    # Live ROS defaults to a viewer if no explicit sink was selected.
    spawn = bool(arguments.spawn or (not arguments.save and not arguments.connect))
    return run_ros(
        save_path=save_path,
        connect_url=arguments.connect,
        spawn=spawn,
        viewer_port=arguments.viewer_port,
        legacy_camera_topics=arguments.legacy_camera_topics,
        ros_args=ros_args or None,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
