import struct

import pytest

from flexiv_inspire_isaac.dftp.force_calibration import _assert_safe_to_calibrate
from flexiv_inspire_isaac.dftp.modbus import (
    CommandCapableModbusTcpClient,
    LocalWritePermit,
)
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


def test_temperature_sentinel_does_not_disconnect_valid_actuator_state():
    state = HandState(
        side="left",
        actuator_position=(1,) * 6,
        actuator_angle=(1,) * 6,
        actual_force_g=(1,) * 6,
        current_ma=(1,) * 6,
        temperature_c=(255,) * 6,
        error_code=(0,) * 6,
        status_code=(0,) * 6,
        acquisition=Acquisition(1, 2, 3),
        field_times_ns={},
    )

    assert state.acquisition.valid is True
    assert state.temperature_c == (255,) * 6


def test_raw_actuator_position_outside_normalized_range_is_observational():
    state = HandState(
        side="right",
        actuator_position=(-12, 1001, 32767, -32768, 500, 0),
        actuator_angle=(500,) * 6,
        actual_force_g=(0,) * 6,
        current_ma=(0,) * 6,
        temperature_c=(20,) * 6,
        error_code=(0,) * 6,
        status_code=(2,) * 6,
        acquisition=Acquisition(1, 2, 3),
        field_times_ns={},
    )

    assert state.acquisition.valid is True
    assert state.actuator_position[:2] == (-12, 1001)


def test_actuator_angle_outside_command_range_remains_invalid():
    state = HandState(
        side="left",
        actuator_position=(0,) * 6,
        actuator_angle=(0, 1, 500, 999, 1000, 1001),
        actual_force_g=(0,) * 6,
        current_ma=(0,) * 6,
        temperature_c=(20,) * 6,
        error_code=(0,) * 6,
        status_code=(2,) * 6,
        acquisition=Acquisition(1, 2, 3),
        field_times_ns={},
    )

    assert state.acquisition.valid is False
    assert state.acquisition.invalid_reason == "actuator_angle-outside-0..1000"


def test_force_calibration_uses_single_register_echo_protocol():
    class Client(CommandCapableModbusTcpClient):
        def _request(self, function, payload, expected_bytes):
            self.seen = (function, payload, expected_bytes)
            return payload

    permit = LocalWritePermit.issue_after_local_authorization(
        "test", "DFTP-LOCAL-CONTROL-AUTHORIZED"
    )
    client = Client("127.0.0.1", permit=permit)
    client.write_single_u16(1009, 1)
    assert client.seen == (0x06, struct.pack(">HH", 1009, 1), 0)


def test_force_calibration_precheck_rejects_loaded_hand():
    state = {
        "angles": [1000] * 6,
        "forces_g": [0] * 6,
        "currents_ma": [0, 0, 101, 0, 0, 0],
        "errors": [0] * 6,
    }
    with pytest.raises(RuntimeError, match="not idle"):
        _assert_safe_to_calibrate(state)
