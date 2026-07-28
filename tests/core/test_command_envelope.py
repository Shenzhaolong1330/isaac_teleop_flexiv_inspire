from dataclasses import replace

import numpy as np
import pytest

from isaac_teleop_core.command import (
    BimanualCommand,
    CommandSource,
    ControlRepresentation,
    CommandPoint,
    ROTATION_ORDER,
    SCHEMA_VERSION,
    ValidMask,
)


def command(*, ttl_ns=1_000_000_000, offset=0.0, frame="world", sequence=1):
    return BimanualCommand(
        schema_version=SCHEMA_VERSION,
        session_id="session",
        source=CommandSource.TELEOP,
        sequence=sequence,
        issued_monotonic_ns=1,
        ttl_ns=ttl_ns,
        representation=ControlRepresentation.CARTESIAN_ROT6D,
        frame_id=frame,
        points=(CommandPoint.identity(),)
        if offset == 0.0
        else (replace(CommandPoint.identity(), execute_after_s=offset),),
        valid_mask=ValidMask.LEFT_ARM,
        deadman=True,
        rotation_order=ROTATION_ORDER,
    )


def test_ttl_is_at_most_one_second():
    command(ttl_ns=1_000_000_000)
    with pytest.raises(ValueError, match="TTL|ttl_ns"):
        command(ttl_ns=1_000_000_001)


def test_execute_offset_must_be_strictly_before_ttl():
    command(ttl_ns=500_000_001, offset=0.5)
    with pytest.raises(ValueError, match="strictly before TTL"):
        command(ttl_ns=500_000_000, offset=0.5)


def test_sequence_must_fit_uint64():
    command(sequence=(1 << 64) - 1)
    with pytest.raises(ValueError, match="uint64"):
        command(sequence=1 << 64)


def test_v1_cartesian_frame_is_explicit_world():
    with pytest.raises(ValueError, match="frame_id='world'"):
        command(frame="base")
