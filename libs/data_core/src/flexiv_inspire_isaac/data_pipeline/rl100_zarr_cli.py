"""Convert completed raw MCAP episodes into one RL-100-compatible Zarr."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .lerobot_cli import _manifest_paths
from .lerobot_export import EpisodeAligner
from .mcap_input import (
    FULL_VALID_MASK,
    RIGHT_VALID_MASK,
    load_json_mcap_streams,
    load_ros2_camera_mcap_streams,
)
from .rl100_zarr import (
    IMAGE_SOURCES,
    RIGHT_IMAGE_SOURCES,
    RIGHT_PROFILE_ID,
    RL100SourceEpisode,
    export_rl100_zarr,
    load_rl100_zarr_spec,
)


def _load_source_episode(manifest_path: Path, spec) -> tuple[RL100SourceEpisode, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(manifest.get("completed", False)):
        raise ValueError(
            f"refusing incomplete episode {manifest_path}: "
            f"{manifest.get('completion_reason', '')}"
        )
    deviceio_value = str(manifest.get("deviceio_mcap", "")).strip()
    deviceio_paths = _manifest_paths(manifest_path, deviceio_value) if deviceio_value else []
    if not deviceio_paths:
        raise FileNotFoundError(
            f"manifest deviceio_mcap cannot be resolved: {manifest_path}"
        )
    image_sources = (
        RIGHT_IMAGE_SOURCES if spec.profile == RIGHT_PROFILE_ID else IMAGE_SOURCES
    )
    required_cameras = {
        source.replace("observation.images.", "camera/") + "/jpeg"
        for source in image_sources.values()
    }
    streams = load_json_mcap_streams(
        deviceio_paths,
        accepted_command_masks=(
            RIGHT_VALID_MASK if spec.profile == RIGHT_PROFILE_ID else FULL_VALID_MASK,
        ),
    )
    ros_paths: list[Path] = []
    recovered_cameras: list[str] = []
    if not required_cameras.issubset(streams):
        ros_value = str(manifest.get("ros_mcap", "")).strip()
        ros_paths = _manifest_paths(manifest_path, ros_value) if ros_value else []
        if not ros_paths:
            missing = sorted(required_cameras.difference(streams))
            raise FileNotFoundError(
                "native MCAP is missing camera streams and ros_mcap cannot be "
                f"resolved for recovery: {missing}"
            )
        recovered = load_ros2_camera_mcap_streams(ros_paths)
        for name in required_cameras.difference(streams):
            if name in recovered:
                streams[name] = recovered[name]
                recovered_cameras.append(name)
    for canonical, source in spec.channels.items():
        if source not in streams:
            raise ValueError(f"configured source stream is absent: {source}")
        streams[canonical] = streams[source]

    rows = EpisodeAligner(
        streams,
        timeline_source=spec.timeline_source,
        timeline_hz=spec.fps if spec.resample_timeline else None,
        action=spec.action,
        segment_gap_threshold_s=spec.gap_threshold_s,
        allow_future_camera_matches=False,
        compose_action_deltas=spec.resample_timeline,
    ).rows()
    return (
        RL100SourceEpisode(
            rows=rows,
            source_manifest=str(manifest_path),
            task=str(manifest.get("task_description", "")).strip(),
        ),
        {
            "manifest": str(manifest_path),
            "deviceio_mcaps": [str(path.resolve()) for path in deviceio_paths],
            "ros_mcaps": [str(path.resolve()) for path in ros_paths],
            "recovered_camera_streams": sorted(recovered_cameras),
        },
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge completed Flexiv/Inspire episodes into RL-100 Zarr"
    )
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--export-config", required=True)
    args = parser.parse_args(argv)

    spec = load_rl100_zarr_spec(args.export_config)
    manifests = [Path(value).expanduser().resolve() for value in args.manifest]
    if len(manifests) != len(set(manifests)):
        raise ValueError("--manifest entries must not contain duplicates")
    episodes: list[RL100SourceEpisode] = []
    provenance: list[dict] = []
    for manifest in manifests:
        if not manifest.is_file():
            raise FileNotFoundError(f"manifest does not exist: {manifest}")
        episode, source = _load_source_episode(manifest, spec)
        episodes.append(episode)
        provenance.append(source)

    result = export_rl100_zarr(
        episodes,
        output_root=args.output_root,
        spec=spec,
    )
    print(json.dumps({**asdict(result), "source": provenance}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
