"""Merge completed raw episodes into one strict policy-profile dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .export_spec import load_export_spec, remap_streams
from .lerobot_cli import _manifest_paths
from .lerobot_export import EpisodeAligner
from .mcap_input import load_json_mcap_streams, load_ros2_camera_mcap_streams
from .profile_export import PolicyEpisode, export_policy_episodes, profile_features


REQUIRED_CAMERAS = {
    "camera/head/jpeg",
    "camera/left_wrist/jpeg",
    "camera/right_wrist/jpeg",
}


def _load_episode(
    manifest_path: Path,
    *,
    task_override: str,
    spec,
) -> tuple[PolicyEpisode, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(manifest.get("completed", False)):
        raise ValueError(
            f"refusing incomplete episode {manifest_path}: "
            f"{manifest.get('completion_reason', '')}"
        )
    task = str(task_override or manifest.get("task_description", "")).strip()
    if not task:
        raise ValueError(f"episode manifest has no task_description: {manifest_path}")
    deviceio_value = str(manifest.get("deviceio_mcap", "")).strip()
    deviceio_paths = (
        _manifest_paths(manifest_path, deviceio_value) if deviceio_value else []
    )
    if not deviceio_paths:
        raise FileNotFoundError(
            f"manifest deviceio_mcap cannot be resolved: {manifest_path}"
        )
    streams = load_json_mcap_streams(deviceio_paths)
    ros_paths: list[Path] = []
    recovered_cameras: list[str] = []
    if not REQUIRED_CAMERAS.issubset(streams):
        ros_value = str(manifest.get("ros_mcap", "")).strip()
        ros_paths = _manifest_paths(manifest_path, ros_value) if ros_value else []
        if not ros_paths:
            missing = sorted(REQUIRED_CAMERAS.difference(streams))
            raise FileNotFoundError(
                "native MCAP is missing camera streams and ros_mcap cannot be "
                f"resolved for recovery: {missing}"
            )
        recovered = load_ros2_camera_mcap_streams(ros_paths)
        for name in REQUIRED_CAMERAS.difference(streams):
            if name in recovered:
                streams[name] = recovered[name]
                recovered_cameras.append(name)
    streams = remap_streams(streams, spec)
    rows = EpisodeAligner(
        streams,
        timeline_source=spec.timeline_source,
        timeline_hz=spec.fps if spec.resample_timeline else None,
        high_rate_arm_samples_per_frame=0,
        action=spec.action,
        depth_cameras=(),
        segment_gap_threshold_s=spec.segments.gap_threshold_s,
    ).rows()
    return (
        PolicyEpisode(
            rows=rows,
            task=task,
            source_manifest=str(manifest_path),
        ),
        {
            "manifest": str(manifest_path),
            "deviceio_mcaps": [str(path.resolve()) for path in deviceio_paths],
            "ros_mcaps": [str(path.resolve()) for path in ros_paths],
            "recovered_camera_streams": sorted(recovered_cameras),
            "task_description": task,
        },
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge completed episodes into one strict LeRobot profile"
    )
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--export-config", required=True)
    parser.add_argument(
        "--task",
        default="",
        help="optional VLA prompt override applied to every source episode",
    )
    args = parser.parse_args(argv)

    spec = load_export_spec(args.export_config)
    if spec.profile is None:
        raise ValueError("export config must select lerobot_export.profile")
    manifests = [Path(value).expanduser().resolve() for value in args.manifest]
    if len(manifests) != len(set(manifests)):
        raise ValueError("--manifest entries must not contain duplicates")
    episodes: list[PolicyEpisode] = []
    provenance: list[dict] = []
    for manifest in manifests:
        if not manifest.is_file():
            raise FileNotFoundError(f"manifest does not exist: {manifest}")
        episode, episode_provenance = _load_episode(
            manifest,
            task_override=args.task,
            spec=spec,
        )
        episodes.append(episode)
        provenance.append(episode_provenance)

    result = export_policy_episodes(
        episodes,
        output_root=args.output_root,
        repo_id=args.repo_id,
        profile_id=spec.profile,
        fps=spec.fps,
    )
    validation_path = Path(args.output_root) / "export_validation.json"
    validation_path.write_text(
        json.dumps(
            {
                **asdict(result),
                "repo_id": args.repo_id,
                "profile": spec.profile,
                "source": provenance,
                "timeline_source": spec.timeline_source,
                "timeline_fps": spec.fps,
                "timeline_resampled": spec.resample_timeline,
                "channel_remaps": dict(spec.channels),
                "feature_keys": list(
                    profile_features(
                        spec.profile,
                        {
                            key: (1, 1, 3)
                            for key in (
                                "observation.images.left_wrist_image",
                                "observation.images.right_wrist_image",
                                "observation.images.head_image",
                            )
                        },
                    )
                ),
                "rotation_representation": "XYZ_PLUS_ROTATION_VECTOR",
                "hand_domain": "normalized_0_to_1",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(validation_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
