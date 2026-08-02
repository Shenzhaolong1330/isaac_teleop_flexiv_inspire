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
    open_features = _features(0.1)
    closed_features = _features(1.1)
    for name in open_features["right"]:
        open_features["right"][name] += 0.25
        closed_features["right"][name] += 0.50
    opened = summarize_samples([open_features] * 10, "open")
    closed = summarize_samples([closed_features] * 10, "closed")
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
    assert "channel_defaults" not in result
    assert "side_channels" not in result
    assert (
        result["sides"]["left"]["index"]
        is not result["sides"]["right"]["index"]
    )
    assert (
        result["sides"]["left"]["index"]["source_open"]
        != result["sides"]["right"]["index"]["source_open"]
    )
    for side in ("left", "right"):
        for channel in result["sides"][side].values():
            assert channel["source_closed"] > channel["source_open"]
