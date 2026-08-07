from __future__ import annotations

import math

import msgpack
import numpy as np
import pytest

from flexiv_inspire_isaac.oculus_reader_ros_source import (
    _button_scalar,
    _extract_oculus_payload,
    _matrix_pose,
    _select_adb_device,
)
from flexiv_inspire_isaac.quest_input_contract import (
    controller_payload,
    encode_controller_payload,
)


def _octets(values: list[bytes]) -> bytes:
    return b"".join(values)


def test_adb_device_selection_requires_one_authorized_device():
    output = "List of devices attached\n530bd095 device product:eureka\n"
    assert _select_adb_device(output) == "530bd095"
    assert _select_adb_device(output, "530bd095") == "530bd095"

    with pytest.raises(RuntimeError, match="exactly one"):
        _select_adb_device("List of devices attached\n")
    with pytest.raises(RuntimeError, match="not connected"):
        _select_adb_device(output, "another-device")


def test_oculus_logcat_payload_and_button_values_are_normalized():
    assert _extract_oculus_payload("08-07 I/wE9ryARX: l:data&r:data") == (
        "l:data&r:data"
    )
    assert _extract_oculus_payload("unrelated") == ""
    assert _button_scalar({"leftGrip": (0.25,)}, "leftGrip") == 0.25
    assert _button_scalar({"rightGrip": 2.0}, "rightGrip") == 1.0
    assert _button_scalar({}, "rightGrip") == 0.0


def test_oculus_matrix_maps_directly_to_ros_pose():
    angle = math.pi / 2.0
    matrix = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0, 0.1],
            [math.sin(angle), math.cos(angle), 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    position, quaternion = _matrix_pose(matrix)
    np.testing.assert_allclose(position, [0.1, -0.2, 0.3])
    np.testing.assert_allclose(np.abs(quaternion), [0.0, 0.0, 2**-0.5, 2**-0.5])

    invalid = matrix.copy()
    invalid[0, 0] = 3.0
    with pytest.raises(ValueError, match="orthonormal"):
        _matrix_pose(invalid)


def test_controller_contract_is_identical_for_all_providers():
    payload = controller_payload(
        timestamp_ns=123,
        left_squeeze_value=0.2,
        right_squeeze_value=0.8,
        left_primary_click=False,
        right_primary_click=True,
        left_is_active=True,
        right_is_active=True,
    )
    decoded = msgpack.unpackb(_octets(encode_controller_payload(payload)), raw=False)
    assert decoded == {
        "timestamp": 123,
        "left_squeeze_value": 0.2,
        "right_squeeze_value": 0.8,
        "left_primary_click": False,
        "right_primary_click": True,
        "left_is_active": True,
        "right_is_active": True,
    }
