"""MCAP + manifest -> aligned LeRobot 0.6.0 Dataset v3 CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .export_spec import load_export_spec, remap_streams
from .lerobot_export import EpisodeAligner
from .lerobot_v3 import export_rows
from .mcap_input import load_json_mcap_streams


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--task",
        default="",
        help="VLA prompt override; defaults to manifest.task_description",
    )
    parser.add_argument("--mcap", action="append", default=[])
    parser.add_argument("--export-config", default=None, help="schema-v1 YAML selecting timeline, action view and channel remaps")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if not bool(manifest.get("completed", False)):
        raise ValueError(
            f"refusing incomplete episode: {manifest.get('completion_reason', '')}"
        )
    task = str(args.task or manifest.get("task_description", "")).strip()
    if not task:
        raise ValueError("episode manifest has no task_description/VLA prompt")
    paths = [Path(value) for value in args.mcap]
    if not paths:
        value = manifest.get("deviceio_mcap")
        if value:
            path = Path(value)
            paths.append(
                path if path.is_absolute() else manifest_path.parent / path
            )
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("manifest/--mcap does not resolve to existing MCAP files")

    spec = load_export_spec(args.export_config)
    streams = remap_streams(load_json_mcap_streams(paths), spec)
    rows = EpisodeAligner(streams, timeline_source=spec.timeline_source, action=spec.action).rows()
    result = export_rows(
        rows,
        output_root=args.output_root,
        repo_id=args.repo_id,
        task=task,
        action=spec.action,
        fps=spec.fps,
    )
    validation_path = Path(args.output_root) / "export_validation.json"
    validation_path.write_text(
        json.dumps(
            {
                **result.__dict__,
                "source_manifest": str(manifest_path),
                "source_mcaps": [str(path.resolve()) for path in paths],
                "ros_mcap_provenance": str(
                    (manifest_path.parent / manifest["ros_mcap"]).resolve()
                ) if manifest.get("ros_mcap") else "",
                "ros_mcap_decoded": False,
                "timeline_source": spec.timeline_source,
                "timeline_fps": spec.fps,
                "action_view": spec.action.name,
                "action_shape": spec.action.shape,
                "channel_remaps": dict(spec.channels),
                "rotation_representation": "ROT6D_FIRST_TWO_COLUMNS",
                "task_description": task,
            },
            indent=2,
        )
    )
    print(validation_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
