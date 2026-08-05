"""Vendor protocol constants and binary decoding for RH56DFTP-2.

Register addresses and tactile layout are transcribed from the public
RH56DFTP V1.0.0 manual (PRJ-02-TS-U-010, December 2024).  Numeric actuator
state is transported as Modbus words. Although the manual describes tactile
memory byte order as little-endian, a Modbus/TCP response has already encoded
each register in network order. Read-only captures from both installed hands
confirm that response bytes ``00 1b`` represent raw taxel value 27. The driver
decodes the Modbus payload as big-endian uint16 and preserves the raw value.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable


ACTUATOR_NAMES = (
    "little",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotate",
)


class Register:
    # Writing 1 with Modbus function 0x06 starts the vendor force-sensor
    # calibration routine. The hand must be open and unloaded.
    FORCE_CALIBRATION = 1009

    POSITION_ACTUAL = 1534
    ANGLE_ACTUAL = 1546
    FORCE_ACTUAL = 1582
    CURRENT = 1594
    ERROR = 1606
    STATUS = 1612
    TEMPERATURE = 1618

    # Write addresses are kept here for the command-capable transport.  The
    # read-only validation utility never imports or invokes that transport.
    ANGLE_TARGET = 1486
    FORCE_LIMIT = 1498


# All non-tactile observations live in one contiguous Modbus address window.
# Reading the complete window in one function-0x03 transaction removes six
# request/response round trips from every hand-state frame.  The unused bytes
# between ANGLE_ACTUAL and FORCE_ACTUAL are intentionally retained.
HAND_STATE_BLOCK_START = Register.POSITION_ACTUAL
HAND_STATE_BLOCK_END = Register.TEMPERATURE + 6
HAND_STATE_BLOCK_BYTES = HAND_STATE_BLOCK_END - HAND_STATE_BLOCK_START
HAND_STATE_BLOCK_WORDS = HAND_STATE_BLOCK_BYTES // 2

assert HAND_STATE_BLOCK_BYTES == 90
assert HAND_STATE_BLOCK_WORDS == 45


@dataclass(frozen=True)
class SurfaceSpec:
    name: str
    start_address: int
    rows: int
    cols: int

    @property
    def taxels(self) -> int:
        return self.rows * self.cols

    @property
    def byte_length(self) -> int:
        return self.taxels * 2

    @property
    def end_address(self) -> int:
        return self.start_address + self.byte_length - 1


TACTILE_LAYOUT = (
    SurfaceSpec("little_end", 3000, 3, 3),
    SurfaceSpec("little_tip", 3018, 12, 8),
    SurfaceSpec("little_pad", 3210, 10, 8),
    SurfaceSpec("ring_end", 3370, 3, 3),
    SurfaceSpec("ring_tip", 3388, 12, 8),
    SurfaceSpec("ring_pad", 3580, 10, 8),
    SurfaceSpec("middle_end", 3740, 3, 3),
    SurfaceSpec("middle_tip", 3758, 12, 8),
    SurfaceSpec("middle_pad", 3950, 10, 8),
    SurfaceSpec("index_end", 4110, 3, 3),
    SurfaceSpec("index_tip", 4128, 12, 8),
    SurfaceSpec("index_pad", 4320, 10, 8),
    SurfaceSpec("thumb_end", 4480, 3, 3),
    SurfaceSpec("thumb_tip", 4498, 12, 8),
    SurfaceSpec("thumb_middle", 4690, 3, 3),
    SurfaceSpec("thumb_pad", 4708, 12, 8),
    SurfaceSpec("palm", 4900, 8, 14),
)
TOTAL_TAXELS = sum(surface.taxels for surface in TACTILE_LAYOUT)

assert len(TACTILE_LAYOUT) == 17
assert TOTAL_TAXELS == 1062
assert TACTILE_LAYOUT[-1].end_address == 5123


def decode_modbus_i16(payload: bytes, expected_count: int = 6) -> tuple[int, ...]:
    """Decode standard big-endian signed Modbus words."""

    if len(payload) != expected_count * 2:
        raise ValueError(f"expected {expected_count * 2} bytes, got {len(payload)}")
    return tuple(struct.unpack(f">{expected_count}h", payload))


def decode_modbus_u16(payload: bytes, expected_count: int = 6) -> tuple[int, ...]:
    if len(payload) != expected_count * 2:
        raise ValueError(f"expected {expected_count * 2} bytes, got {len(payload)}")
    return tuple(struct.unpack(f">{expected_count}H", payload))


def decode_packed_u8(payload: bytes, expected_count: int = 6) -> tuple[int, ...]:
    """Decode the six one-byte ERROR/STATUS/TEMP registers."""

    if len(payload) != expected_count:
        raise ValueError(f"expected {expected_count} bytes, got {len(payload)}")
    return tuple(payload)


def decode_hand_state_block(payload: bytes) -> dict[str, tuple[int, ...]]:
    """Decode one contiguous non-tactile state read.

    Register addresses in the RH56DFTP manual are byte addresses, while the
    Modbus request length is expressed in 16-bit words.  Offsets below are
    therefore calculated in bytes relative to POSITION_ACTUAL.
    """

    if len(payload) != HAND_STATE_BLOCK_BYTES:
        raise ValueError(
            f"expected {HAND_STATE_BLOCK_BYTES} hand-state bytes, got {len(payload)}"
        )

    def field(address: int, size: int) -> bytes:
        start = address - HAND_STATE_BLOCK_START
        return payload[start : start + size]

    return {
        "position": decode_modbus_i16(field(Register.POSITION_ACTUAL, 12)),
        "angle": decode_modbus_i16(field(Register.ANGLE_ACTUAL, 12)),
        "force": decode_modbus_i16(field(Register.FORCE_ACTUAL, 12)),
        "current": decode_modbus_i16(field(Register.CURRENT, 12)),
        "error": decode_packed_u8(field(Register.ERROR, 6)),
        "status": decode_packed_u8(field(Register.STATUS, 6)),
        "temperature": decode_packed_u8(field(Register.TEMPERATURE, 6)),
    }


def decode_tactile(payload: bytes, spec: SurfaceSpec) -> tuple[int, ...]:
    if len(payload) != spec.byte_length:
        raise ValueError(
            f"{spec.name}: expected {spec.byte_length} bytes, got {len(payload)}"
        )
    return tuple(struct.unpack(f">{spec.taxels}H", payload))


def encode_modbus_i16(values: Iterable[int]) -> bytes:
    words = tuple(int(value) for value in values)
    if not words:
        raise ValueError("at least one value is required")
    if any(value < -32768 or value > 32767 for value in words):
        raise ValueError("value does not fit int16")
    return struct.pack(f">{len(words)}h", *words)
