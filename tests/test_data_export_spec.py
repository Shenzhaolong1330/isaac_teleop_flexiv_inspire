from pathlib import Path
from types import SimpleNamespace

import pytest

from flexiv_inspire_isaac.data_pipeline.export_spec import (
    ExportSpecError,
    load_export_spec,
)
from flexiv_inspire_isaac.data_pipeline import lerobot_cli


def test_export_spec_can_be_loaded_from_composed_site_config(tmp_path: Path):
    (tmp_path / "hardware.yaml").write_text(
        "hardware: {name: test}\n", encoding="utf-8"
    )
    (tmp_path / "recording.yaml").write_text(
        """
lerobot_export:
  schema_version: 1
  timeline: {source: camera/head/jpeg, fps: 30.0}
  action: {view: absolute_joint_position}
  channels: {}
""",
        encoding="utf-8",
    )
    site = tmp_path / "site.yaml"
    site.write_text(
        "schema_version: 1\nincludes: [hardware.yaml, recording.yaml]\n",
        encoding="utf-8",
    )

    spec = load_export_spec(site)

    assert spec.timeline_source == "camera/head/jpeg"
    assert spec.action.name == "absolute_joint_position"


def test_export_spec_rejects_fractional_fps_that_writer_cannot_preserve(
    tmp_path: Path,
):
    config = tmp_path / "export.yaml"
    config.write_text(
        """
schema_version: 1
timeline: {source: camera/head/jpeg, fps: 29.97}
action: {view: sent_command}
channels: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ExportSpecError, match="fps"):
        load_export_spec(config)


def test_export_spec_can_keep_recorded_camera_timeline(tmp_path: Path):
    config = tmp_path / "export.yaml"
    config.write_text(
        """
schema_version: 1
timeline: {source: camera/head/jpeg, fps: 15.0, resample: false}
action: {view: sent_command}
channels: {}
""",
        encoding="utf-8",
    )

    spec = load_export_spec(config)

    assert spec.resample_timeline is False


def test_lerobot_cli_action_view_overrides_export_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"completed": true, "task_description": "test", '
        '"deviceio_mcap": "deviceio.mcap"}',
        encoding="utf-8",
    )
    (tmp_path / "deviceio.mcap").touch()
    config = tmp_path / "export.yaml"
    config.write_text(
        """
schema_version: 1
timeline: {source: camera/head/jpeg, fps: 30.0}
action: {view: sent_command}
channels: {}
""",
        encoding="utf-8",
    )
    captured = {}

    class FakeAligner:
        def __init__(
            self,
            streams,
            *,
            timeline_source,
            timeline_hz,
            high_rate_arm_samples_per_frame,
            action,
            depth_cameras,
            segment_gap_threshold_s,
        ):
            captured["action_view"] = action.name
            captured["timeline_hz"] = timeline_hz
            captured["high_rate_samples"] = high_rate_arm_samples_per_frame
            captured["depth_cameras"] = depth_cameras
            captured["segment_gap_threshold_s"] = segment_gap_threshold_s

        def rows(self):
            return []

    def fake_export_rows(rows, *, output_root, **kwargs):
        Path(output_root).mkdir(parents=True)
        return SimpleNamespace(
            output_root=str(output_root),
            frames_written=0,
            frames_dropped_invalid_action=0,
            frames_dropped_invalid_image=0,
            frames_dropped_invalid_depth=0,
            episodes_written=0,
            reload_length=0,
        )

    monkeypatch.setattr(
        lerobot_cli,
        "load_json_mcap_streams",
        lambda paths: {
            "camera/head/jpeg": [],
            "camera/left_wrist/jpeg": [],
            "camera/right_wrist/jpeg": [],
        },
    )
    monkeypatch.setattr(lerobot_cli, "EpisodeAligner", FakeAligner)
    monkeypatch.setattr(lerobot_cli, "export_rows", fake_export_rows)

    assert lerobot_cli.main(
        [
            "--manifest",
            str(manifest),
            "--output-root",
            str(tmp_path / "output"),
            "--repo-id",
            "local/test",
            "--export-config",
            str(config),
            "--action-view",
            "absolute_joint_position",
        ]
    ) == 0
    assert captured["action_view"] == "absolute_joint_position"
    assert captured["timeline_hz"] == 30.0
    assert captured["high_rate_samples"] == 0


def test_export_spec_selects_exact_fields_and_head_z16_depth(tmp_path: Path):
    config = tmp_path / "export.yaml"
    config.write_text(
        """
schema_version: 1
timeline: {source: camera/head/jpeg, fps: 15.0, resample: false}
high_rate_arm_samples_per_frame: 20
action: {view: sent_command}
fields:
  - observation.images.head
  - observation.depth.head
  - observation.depth_scale_m.head
  - observation.depth_intrinsics.head
  - observation.source_timestamp_ns
  - action
depth: {enabled: true, cameras: [head], representation: z16, storage: parquet}
segments: {gap_threshold_s: 0.25, split_episodes: true}
channels: {}
""",
        encoding="utf-8",
    )

    spec = load_export_spec(config)

    assert spec.depth.enabled
    assert spec.depth.cameras == ("head",)
    assert spec.segments.split_episodes
    assert spec.fields == (
        "observation.images.head",
        "observation.depth.head",
        "observation.depth_scale_m.head",
        "observation.depth_intrinsics.head",
        "observation.source_timestamp_ns",
        "action",
    )


def test_export_spec_allows_omitting_optional_depth_intrinsics(tmp_path: Path):
    config = tmp_path / "export.yaml"
    config.write_text(
        """
schema_version: 1
timeline: {source: camera/head/jpeg, fps: 15.0}
action: {view: sent_command}
fields:
  - observation.depth.head
  - observation.depth_scale_m.head
  - action
depth: {enabled: true, cameras: [head], representation: z16, storage: parquet}
""",
        encoding="utf-8",
    )

    spec = load_export_spec(config)

    assert spec.fields == (
        "observation.depth.head",
        "observation.depth_scale_m.head",
        "action",
    )


def test_export_spec_rejects_depth_field_without_depth_metadata(tmp_path: Path):
    config = tmp_path / "export.yaml"
    config.write_text(
        """
schema_version: 1
timeline: {source: camera/head/jpeg, fps: 15.0}
action: {view: sent_command}
fields: [observation.depth.head, action]
depth: {enabled: false}
""",
        encoding="utf-8",
    )

    with pytest.raises(ExportSpecError, match="depth fields"):
        load_export_spec(config)


def test_export_spec_selects_strict_policy_profile(tmp_path: Path):
    config = tmp_path / "export.yaml"
    config.write_text(
        """
schema_version: 1
profile: joint_proprio_cartesian_v1
timeline: {source: camera/head/jpeg, fps: 15.0, resample: false}
action: {view: sent_command}
depth: {enabled: false, cameras: []}
channels: {}
""",
        encoding="utf-8",
    )

    spec = load_export_spec(config)

    assert spec.profile == "joint_proprio_cartesian_v1"
    assert spec.action.name == "sent_command"
    assert not spec.depth.enabled


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("fields: [observation.arm_q, action]", "owns its exact fields"),
        ("depth: {enabled: true, cameras: [head]}", "do not include depth"),
        ("high_rate_arm_samples_per_frame: 20", "high-rate"),
    ],
)
def test_policy_profile_rejects_redundant_fields(
    tmp_path: Path, extra: str, message: str
):
    config = tmp_path / "export.yaml"
    config.write_text(
        f"""
schema_version: 1
profile: joint_proprio_cartesian_v1
timeline: {{source: camera/head/jpeg, fps: 15.0}}
action: {{view: sent_command}}
{extra}
""",
        encoding="utf-8",
    )

    with pytest.raises(ExportSpecError, match=message):
        load_export_spec(config)
