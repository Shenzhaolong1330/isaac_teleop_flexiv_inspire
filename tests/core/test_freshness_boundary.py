from isaac_teleop_core.command import (
    BimanualCommand,
    CommandSource,
    ControlRepresentation,
    CommandPoint,
    ROTATION_ORDER,
    SCHEMA_VERSION,
    ValidMask,
)


def test_ttl_expiry_boundary_is_exclusive():
    command = BimanualCommand(
        schema_version=SCHEMA_VERSION,
        session_id="session",
        source=CommandSource.TELEOP,
        sequence=1,
        issued_monotonic_ns=100,
        ttl_ns=10,
        representation=ControlRepresentation.CARTESIAN_ROT6D,
        frame_id="world",
        points=(CommandPoint.identity(),),
        valid_mask=ValidMask.LEFT_ARM,
        deadman=True,
        rotation_order=ROTATION_ORDER,
    )
    assert command.is_fresh(100)
    assert command.is_fresh(109)
    assert not command.is_fresh(110)
