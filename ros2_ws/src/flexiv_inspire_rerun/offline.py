"""Offline DeviceIO MCAP -> Rerun playback; this module never imports ROS."""

from __future__ import annotations

from pathlib import Path
import sys
import time

from flexiv_inspire_isaac.data_pipeline.playback import (
    PlaybackConfigError,
    load_playback_config,
    read_deviceio_records,
    resolve_episode,
)

from .runtime import RerunVisualizer


def run_offline(config_path: str | Path) -> int:
    spec = load_playback_config(config_path)
    episode = resolve_episode(spec, for_hardware=False)
    records = read_deviceio_records(episode.deviceio_mcap)
    source_start = records[0].timestamp_ns
    timeline_start = source_start
    wall_start = time.monotonic_ns()
    visualize = spec.visualize
    visualizer = RerunVisualizer(
        save_path=visualize.save_path if visualize.sink == "save" else None,
        spawn=visualize.sink == "spawn",
        viewer_port=visualize.viewer_port,
        recording_id=str(episode.manifest.get("episode_uuid", "")) or None,
    )
    try:
        for record in records:
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
    finally:
        visualizer.close()
    print(
        f"visualized {len(records)} records from {episode.directory} "
        f"at {visualize.speed:g}x; hardware_writes=false"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    selected = sys.argv[1:] if argv is None else argv
    if selected:
        raise SystemExit(
            "flexiv-inspire-visualize takes no arguments; edit config/playback.yaml"
        )
    project_root = Path(__file__).resolve().parents[3]
    try:
        return run_offline(project_root / "config" / "playback.yaml")
    except (OSError, ValueError, PlaybackConfigError, RuntimeError) as exc:
        raise SystemExit(f"visualize failed: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
