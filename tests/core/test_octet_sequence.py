from __future__ import annotations

import pytest

from isaac_teleop_core.octet_sequence import (
    decode_octet_sequence,
    encode_octet_sequence,
)


def test_ros_octet_sequence_round_trip() -> None:
    payload = bytes((0, 1, 127, 128, 255))
    encoded = encode_octet_sequence(payload)
    assert encoded == [b"\x00", b"\x01", b"\x7f", b"\x80", b"\xff"]
    assert decode_octet_sequence(encoded) == payload


def test_decoder_accepts_legacy_integer_sequences() -> None:
    assert decode_octet_sequence([0, 1, 255]) == b"\x00\x01\xff"


@pytest.mark.parametrize("values", ([b""], [b"ab"], [-1], [256]))
def test_decoder_rejects_invalid_octets(values) -> None:
    with pytest.raises(ValueError):
        decode_octet_sequence(values)
