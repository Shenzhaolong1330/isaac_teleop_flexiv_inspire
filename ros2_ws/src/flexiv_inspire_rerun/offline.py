"""Offline DeviceIO MCAP -> Rerun playback; this module never imports ROS."""

from __future__ import annotations

from pathlib import Path
import argparse
from itertools import chain
import sys
import time

from flexiv_inspire_isaac.data_pipeline.playback import (
    PlaybackConfigError,
    iter_deviceio_records,
    load_playback_config,
    override_playback_selection,
    resolve_episode,
)

from .runtime import RerunVisualizer


def _visualization_rate(record, visualize) -> float:
    topic = record.topic.lower()
    if "point" in topic or (
        isinstance(record.payload, dict)
        and "pointcloud_xyz_f32_b64" in record.payload
    ):
        return visualize.pointcloud_hz
    if "camera/" in topic or "image" in topic or "depth" in topic:
        return visualize.image_hz
    if "tactile" in topic:
        return visualize.tactile_hz
    return visualize.telemetry_hz


def run_offline(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    episode_selector: str | None = None,
) -> int:
    spec = load_playback_config(config_path)
    spec = override_playback_selection(
        spec, dataset_root=dataset_root, episode=episode_selector
    )
    episode = resolve_episode(spec, for_hardware=False)
    records = iter_deviceio_records(episode.deviceio_mcap)
    first = next(records, None)
    if first is None:
        raise PlaybackConfigError("deviceio MCAP contains no JSON records")
    source_start = first.timestamp_ns
    timeline_start = source_start
    wall_start = time.monotonic_ns()
    visualize = spec.visualize
    visualizer = RerunVisualizer(
        save_path=visualize.save_path if visualize.sink == "save" else None,
        spawn=visualize.sink == "spawn",
        viewer_port=visualize.viewer_port,
        recording_id=str(episode.manifest.get("episode_uuid", "")) or None,
    )
    logged = 0
    seen = 0
    last_logged_ns: dict[str, int] = {}
    try:
        for record in chain((first,), records):
            seen += 1
            rate = _visualization_rate(record, visualize)
            previous = last_logged_ns.get(record.topic)
            if previous is not None and (
                record.timestamp_ns >= previous
                and record.timestamp_ns - previous < int(1e9 / rate)
            ):
                continue
            last_logged_ns[record.topic] = record.timestamp_ns
            offset = record.timestamp_ns - source_start
            scaled_offset = int(offset / visualize.speed)
            if visualize.realtime:
                deadline = wall_start + scaled_offset
                remaining = deadline - time.monotonic_ns()
                if remaining > 0:
                    time.sleep(remaining / 1e9)
            visualizer.log_deviceio(
                topic=record.topic,
                payload=record.payload,
                playback_time_ns=timeline_start + scaled_offset,
                original_time_ns=record.timestamp_ns,
                sequence=record.sequence,
                valid=record.valid,
                timing_valid=record.timing_valid,
                invalid_reason=str(record.envelope.get("invalid_reason", "")),
            )
            logged += 1
    finally:
        visualizer.close()
    print(
        f"visualized {logged}/{seen} records from {episode.directory} "
        f"at {visualize.speed:g}x; hardware_writes=false"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    selected = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        prog="flexiv-inspire-visualize",
        description=(
            "Visualize the episode selected by config/playback.yaml; "
            "this command never writes to hardware."
        ),
    )
    parser.add_argument("--playback-config", default="")
    parser.add_argument("--dataset", default="", help="dataset directory")
    parser.add_argument(
        "--episode", default="", help="latest, numeric index, or directory name"
    )
    args = parser.parse_args(selected)
    project_root = Path(__file__).resolve().parents[3]
    config_path = (
        Path(args.playback_config)
        if args.playback_config
        else project_root / "config" / "playback.yaml"
    )
    try:
        return run_offline(
            config_path,
            dataset_root=args.dataset or None,
            episode_selector=args.episode or None,
        )
    except (OSError, ValueError, PlaybackConfigError, RuntimeError) as exc:
        raise SystemExit(f"visualize failed: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
