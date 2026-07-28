import struct

import pytest

from flexiv_inspire_isaac.dftp.models import Acquisition, HandState
from flexiv_inspire_isaac.dftp.protocol import (
    TACTILE_LAYOUT,
    TOTAL_TAXELS,
    decode_modbus_i16,
    decode_packed_u8,
    decode_tactile,
)


def test_official_tactile_layout_is_complete_and_contiguous():
    assert len(TACTILE_LAYOUT) == 17
    assert TOTAL_TAXELS == 1062
    assert TACTILE_LAYOUT[0].start_address == 3000
    assert TACTILE_LAYOUT[-1].end_address == 5123
    for left, right in zip(TACTILE_LAYOUT, TACTILE_LAYOUT[1:]):
        assert left.end_address + 1 == right.start_address


def test_tactile_modbus_response_is_big_endian_raw_uint16():
    spec = TACTILE_LAYOUT[0]
    payload = bytes.fromhex("001b") + struct.pack(">8H", *range(1, 9))
    assert decode_tactile(payload, spec) == (27, 1, 2, 3, 4, 5, 6, 7, 8)


def test_actuator_words_are_standard_modbus_big_endian():
    payload = struct.pack(">6h", -1, 0, 1, 1000, -4000, 4000)
    assert decode_modbus_i16(payload) == (-1, 0, 1, 1000, -4000, 4000)
    assert decode_packed_u8(bytes((0, 1, 2, 3, 4, 5))) == (0, 1, 2, 3, 4, 5)


def test_hand_state_requires_six_values():
    with pytest.raises(ValueError, match="exactly 6"):
        HandState(
            side="left",
            actuator_position=(1,),
            actuator_angle=(1,) * 6,
            actual_force_g=(1,) * 6,
            current_ma=(1,) * 6,
            temperature_c=(1,) * 6,
            error_code=(0,) * 6,
            status_code=(0,) * 6,
            acquisition=Acquisition(1, 2, 3),
            field_times_ns={},
        )
