from pathlib import Path

import numpy as np
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
