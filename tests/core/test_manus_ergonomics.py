from pathlib import Path

import numpy as np
import yaml
from flexiv_inspire_control.manus_ergonomics_source import (
    FIELD_NAMES,
    PROTOCOL,
    parse_packet,
)
from flexiv_inspire_control.teleop_input_node import (
    _HandCommandFilter,
    _Retarget,
)

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = (
    ROOT / "ros2_ws/src/flexiv_inspire_control/config/manus_ergonomics_bootstrap.yaml"
)


def test_plugin_packet_is_validated_and_converted_to_radians() -> None:
    values_deg = [float(index) for index in range(len(FIELD_NAMES))]
    payload = (f"{PROTOCOL},left,123,456," + ",".join(map(str, values_deg))).encode(
        "ascii"
    )

    packet = parse_packet(payload)

    assert packet.side == "left"
    assert packet.glove_id == 123
    assert packet.source_time == 456
    assert np.allclose(packet.values_rad, np.radians(values_deg))


def test_bootstrap_ergonomics_maps_open_and_closed_to_inspire_endpoints() -> None:
    opened = {
        "PinkyMCPStretch": -0.135873882268,
        "RingMCPStretch": -0.126361837844,
        "MiddleMCPStretch": -0.093270395227,
        "IndexMCPStretch": -0.073775067482,
        "ThumbMCPStretch": 0.474136144597,
        "ThumbMCPSpread": 0.157655591333,
    }
    closed = {
        "PinkyMCPStretch": 1.133888055143,
        "RingMCPStretch": 1.231713759717,
        "MiddleMCPStretch": 1.323779877760,
        "IndexMCPStretch": 1.255659677055,
        "ThumbMCPStretch": 0.226735723127,
        "ThumbMCPSpread": 0.478499467727,
    }

    open_retarget = _Retarget(str(BOOTSTRAP))
    closed_retarget = _Retarget(str(BOOTSTRAP))

    assert open_retarget.uses_ergonomics
    assert np.allclose(
        open_retarget.apply_ergonomics("left", opened, now_ns=1),
        np.full(6, 1000.0),
    )
    assert np.allclose(
        closed_retarget.apply_ergonomics("right", closed, now_ns=1),
        np.zeros(6),
    )


def test_hand_filter_applies_deadband_and_time_based_rate_limit() -> None:
    filter_ = _HandCommandFilter(
        alpha=1.0,
        deadband=2.0,
        max_rate_per_s=100.0,
    )
    initial = filter_.apply("left", np.full(6, 500.0), 1_000_000_000)
    within_deadband = filter_.apply("left", np.full(6, 501.0), 1_100_000_000)
    rate_limited = filter_.apply("left", np.full(6, 800.0), 1_200_000_000)

    assert np.allclose(initial, 500.0)
    assert np.allclose(within_deadband, 500.0)
    assert np.allclose(rate_limited, 510.0)


def test_multi_joint_ergonomics_normalizes_then_fuses_sources(tmp_path) -> None:
    channel = {
        "sources": ["IndexMCPStretch", "IndexPIPStretch", "IndexDIPStretch"],
        "weights": [0.5, 0.3, 0.2],
        # Include one reversed source to prove that signs/ranges cannot cancel.
        "source_open": [0.0, 2.0, -1.0],
        "source_closed": [1.0, 0.0, 3.0],
        "fusion": "weighted",
        "points": [[0.0, 1000.0], [1.0, 0.0]],
    }
    document = {
        "schema_version": 2,
        "calibrated": True,
        "source_format": "MANUS_SDK_ERGONOMICS_RADIANS",
        "filter": {
            "low_pass_alpha": 1.0,
            "output_deadband": 0.0,
            "max_output_rate_per_s": 1.0e9,
        },
        "sides": {
            side: {
                actuator: (
                    channel
                    if actuator != "thumb_rotate"
                    else {
                        "source": "ThumbMCPSpread",
                        "points": [[0.0, 1000.0], [1.0, 0.0]],
                    }
                )
                for actuator in (
                    "little",
                    "ring",
                    "middle",
                    "index",
                    "thumb_bend",
                    "thumb_rotate",
                )
            }
            for side in ("left", "right")
        },
    }
    calibration = tmp_path / "multi_joint.yaml"
    calibration.write_text(yaml.safe_dump(document), encoding="utf-8")
    values = {
        "IndexMCPStretch": 0.5,
        "IndexPIPStretch": 1.0,
        "IndexDIPStretch": 1.0,
        "ThumbMCPSpread": 0.5,
    }

    target = _Retarget(str(calibration)).apply_ergonomics(
        "left", values, now_ns=1
    )

    assert np.allclose(target, np.full(6, 500.0))


def test_primary_mcp_closure_cannot_be_diluted_by_distal_joints(tmp_path) -> None:
    channel = {
        "sources": ["IndexMCPStretch", "IndexPIPStretch", "IndexDIPStretch"],
        "weights": [0.5, 0.3, 0.2],
        "source_open": [0.0, 0.0, 0.0],
        "source_closed": [1.0, 1.0, 1.0],
        "fusion": "max_primary_weighted",
        "points": [[0.0, 1000.0], [0.9, 0.0], [1.0, 0.0]],
    }
    document = {
        "schema_version": 2,
        "calibrated": True,
        "source_format": "MANUS_SDK_ERGONOMICS_RADIANS",
        "filter": {
            "low_pass_alpha": 1.0,
            "output_deadband": 0.0,
            "max_output_rate_per_s": 1.0e9,
        },
        "sides": {
            side: {
                actuator: channel
                for actuator in (
                    "little",
                    "ring",
                    "middle",
                    "index",
                    "thumb_bend",
                    "thumb_rotate",
                )
            }
            for side in ("left", "right")
        },
    }
    calibration = tmp_path / "mcp_primary.yaml"
    calibration.write_text(yaml.safe_dump(document), encoding="utf-8")
    values = {
        "IndexMCPStretch": 1.0,
        "IndexPIPStretch": 0.0,
        "IndexDIPStretch": 0.0,
    }

    target = _Retarget(str(calibration)).apply_ergonomics(
        "left", values, now_ns=1
    )

    assert np.allclose(target, np.zeros(6))
