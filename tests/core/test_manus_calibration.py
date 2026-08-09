from pathlib import Path

import numpy as np
import yaml
from flexiv_inspire_control.manus_calibration import (
    build_calibration,
    build_ergonomics_calibration,
    summarize_ergonomics_samples,
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
    assert result["sides"]["left"]["index"] is not result["sides"]["right"]["index"]
    assert (
        result["sides"]["left"]["index"]["source_open"]
        != result["sides"]["right"]["index"]["source_open"]
    )
    for side in ("left", "right"):
        for channel in result["sides"][side].values():
            assert channel["source_closed"] > channel["source_open"]


def test_ergonomics_captures_finalize_a_schema_v2_mapping():
    fields = {"ThumbMCPSpread": 0.6}
    for finger_index, finger in enumerate(
        ("Pinky", "Ring", "Middle", "Index", "Thumb"), start=1
    ):
        for joint_index, joint in enumerate(("MCP", "PIP", "DIP"), start=1):
            fields[f"{finger}{joint}Stretch"] = (
                0.1 * finger_index + 0.01 * joint_index
            )
    open_samples = {
        side: [fields.copy() for _ in range(10)] for side in ("left", "right")
    }
    closed_fields = {name: value + 0.5 for name, value in fields.items()}
    closed_samples = {
        side: [closed_fields.copy() for _ in range(10)] for side in ("left", "right")
    }
    opened = summarize_ergonomics_samples(open_samples, "open")
    closed = summarize_ergonomics_samples(closed_samples, "closed")
    template = yaml.safe_load(
        (
            ROOT
            / "ros2_ws/src/flexiv_inspire_control/config"
            / "manus_ergonomics_calibration_template.yaml"
        ).read_text()
    )

    result = build_ergonomics_calibration(template, opened, closed)

    assert result["schema_version"] == 2
    assert result["calibrated"] is True
    assert result["source_format"] == "MANUS_SDK_ERGONOMICS_RADIANS"
    assert result["calibration"]["open_samples"] == {
        "left": 10,
        "right": 10,
    }
    assert "side_channels" not in result
    index = result["sides"]["left"]["index"]
    assert index["sources"] == [
        "IndexMCPStretch",
        "IndexPIPStretch",
        "IndexDIPStretch",
    ]
    assert index["weights"] == [0.5, 0.3, 0.2]
    assert index["fusion"] == "max_primary_weighted"
    assert np.allclose(index["source_open"], [0.41, 0.42, 0.43])
    assert np.allclose(index["source_closed"], [0.91, 0.92, 0.93])
    assert index["points"] == [[0.0, 1000.0], [0.9, 0.0], [1.0, 0.0]]
    assert result["sides"]["left"]["thumb_bend"]["points"] == [
        [0.0, 1000.0],
        [0.6, 0.0],
        [1.0, 0.0],
    ]
