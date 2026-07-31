from pathlib import Path

import yaml

from flexiv_inspire_control.manus_calibration import (
    build_calibration,
    summarize_samples,
)


ROOT = Path(__file__).resolve().parents[2]


def _features(offset: float) -> dict[str, dict[str, float]]:
    names = {
        "little_mcp_flexion",
        "little_pip",
        "little_dip",
        "ring_mcp_flexion",
        "ring_pip",
        "ring_dip",
        "middle_mcp_flexion",
        "middle_pip",
        "middle_dip",
        "index_mcp_flexion",
        "index_pip",
        "index_dip",
        "thumb_cmc_flexion",
        "thumb_mcp",
        "thumb_ip",
        "thumb_cmc_abduction",
    }
    return {
        side: {name: offset + index * 0.01 for index, name in enumerate(names)}
        for side in ("left", "right")
    }


def test_endpoint_captures_finalize_a_calibrated_file():
    opened = summarize_samples([_features(0.1)] * 10, "open")
    closed = summarize_samples([_features(1.1)] * 10, "closed")
    template = yaml.safe_load(
        (
            ROOT
            / "ros2_ws/src/flexiv_inspire_control/config"
            / "manus_calibration_template.yaml"
        ).read_text()
    )
    result = build_calibration(template, opened, closed)
    assert result["calibrated"] is True
    assert result["calibration"]["open_samples"] == 10
    for side in ("left", "right"):
        for channel in result["sides"][side].values():
            assert channel["source_closed"] > channel["source_open"]
