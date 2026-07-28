import numpy as np

from flexiv_inspire_control.teleop_mapping import (
    QuestSE3Mapper,
    TrackedPose,
    TrackingSample,
)
from isaac_teleop_core.rotation6d import rotation6d_to_matrix


def sample(sequence, x=0.0, quaternion=(0.0, 0.0, 0.0, 1.0), now=1_000_000_000):
    return TrackingSample(
        left=TrackedPose.make([x, 0, 0], quaternion),
        right=TrackedPose.make([x, 0, 0], quaternion),
        sequence=sequence,
        source_time_ns=sequence * 1_000_000,
        receive_monotonic_ns=now,
        frame_id="world",
    )


def test_first_clutched_frame_is_identity_then_incremental_world_delta():
    mapper = QuestSE3Mapper()
    first = mapper.update(sample(1), deadman=True, now_ns=1_000_000_000)
    assert first.active
    np.testing.assert_allclose(rotation6d_to_matrix(first.left_rotation6d), np.eye(3))
    second = mapper.update(
        sample(2, x=0.01, now=1_010_000_000),
        deadman=True,
        now_ns=1_010_000_000,
    )
    np.testing.assert_allclose(second.left_xyz, [0.01, 0, 0])


def test_q_and_minus_q_do_not_create_rotation_jump():
    mapper = QuestSE3Mapper()
    mapper.update(sample(1), deadman=True, now_ns=1_000_000_000)
    result = mapper.update(
        sample(2, quaternion=(0.0, 0.0, 0.0, -1.0), now=1_010_000_000),
        deadman=True,
        now_ns=1_010_000_000,
    )
    assert result.active and not result.hold_latched
    np.testing.assert_allclose(rotation6d_to_matrix(result.left_rotation6d), np.eye(3))


def test_stale_or_tracking_jump_latches_hold():
    stale_mapper = QuestSE3Mapper()
    result = stale_mapper.update(
        sample(1, now=1), deadman=True, now_ns=1_000_000_000
    )
    assert result.hold_latched
    jump_mapper = QuestSE3Mapper()
    jump_mapper.update(sample(1), deadman=True, now_ns=1_000_000_000)
    result = jump_mapper.update(
        sample(2, x=0.2, now=1_010_000_000),
        deadman=True,
        now_ns=1_010_000_000,
    )
    assert result.hold_latched


def test_clutch_release_clears_local_mapping_latch_and_reanchors():
    mapper = QuestSE3Mapper()
    mapper.update(sample(1), deadman=True, now_ns=1_000_000_000)
    mapper.update(
        sample(2, x=0.2, now=1_010_000_000),
        deadman=True,
        now_ns=1_010_000_000,
    )
    released = mapper.update(
        sample(3, x=0.2, now=1_020_000_000),
        deadman=False,
        now_ns=1_020_000_000,
    )
    assert not released.active
    reanchor = mapper.update(
        sample(4, x=0.2, now=1_030_000_000),
        deadman=True,
        now_ns=1_030_000_000,
    )
    assert reanchor.active and not reanchor.hold_latched
    np.testing.assert_allclose(reanchor.left_xyz, np.zeros(3))
