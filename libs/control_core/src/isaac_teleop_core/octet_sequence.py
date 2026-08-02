"""Lossless conversion at ROS ``sequence<octet>`` message boundaries."""

from __future__ import annotations

from collections.abc import Iterable


def encode_octet_sequence(payload: bytes) -> list[bytes]:
    """Encode bytes for rosidl's Python representation of ``octet[]``."""

    return [bytes((value,)) for value in payload]


def decode_octet_sequence(values: Iterable[bytes | int]) -> bytes:
    """Decode rosidl octets, accepting integer values for old recordings."""

    if isinstance(values, (bytes, bytearray, memoryview)):
        return bytes(values)
    decoded = bytearray()
    for value in values:
        if isinstance(value, bytes):
            if len(value) != 1:
                raise ValueError("ROS octet values must contain exactly one byte")
            decoded.extend(value)
            continue
        integer = int(value)
        if not 0 <= integer <= 255:
            raise ValueError("integer octet values must be in [0,255]")
        decoded.append(integer)
    return bytes(decoded)
