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
        def __init__(self, streams, *, timeline_source, action):
            captured["action_view"] = action.name

        def rows(self):
            return []

    def fake_export_rows(rows, *, output_root, **kwargs):
        Path(output_root).mkdir(parents=True)
        return SimpleNamespace(
            output_root=str(output_root),
            frames_written=0,
            frames_dropped_invalid_action=0,
            frames_dropped_invalid_image=0,
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
