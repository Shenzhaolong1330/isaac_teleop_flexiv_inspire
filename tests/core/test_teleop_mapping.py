import numpy as np
import pytest

from flexiv_inspire_control.teleop_mapping import (
    QuestSE3Mapper,
    TrackedPose,
    TrackingSample,
)
from isaac_teleop_core.rotation6d import (
    geodesic_distance_rad,
    rotation6d_to_matrix,
)


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


def test_site_axis_mapping_and_gains_match_proven_flexiv_teleop():
    axis = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    mapper = QuestSE3Mapper(
        axis_rotation=axis,
        translation_gain=0.5,
        rotation_gain=0.5,
    )
    mapper.update(sample(1), deadman=True, now_ns=1_000_000_000)
    angle = 0.10
    result = mapper.update(
        sample(
            2,
            x=0.02,
            quaternion=(np.sin(angle / 2), 0.0, 0.0, np.cos(angle / 2)),
            now=1_010_000_000,
        ),
        deadman=True,
        now_ns=1_010_000_000,
    )

    # Quest +X is Flexiv -Y. Quest +roll becomes Flexiv -pitch.
    np.testing.assert_allclose(result.left_xyz, [0.0, -0.01, 0.0])
    rotation = rotation6d_to_matrix(result.left_rotation6d)
    expected = np.array(
        [
            [np.cos(angle / 2), 0.0, -np.sin(angle / 2)],
            [0.0, 1.0, 0.0],
            [np.sin(angle / 2), 0.0, np.cos(angle / 2)],
        ]
    )
    np.testing.assert_allclose(rotation, expected, atol=1.0e-7)


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


def test_valid_tracking_step_is_limited_without_dropping_clutch():
    mapper = QuestSE3Mapper(
        max_translation_jump_m=0.12,
        max_translation_step_m=0.01,
        max_rotation_jump_rad=0.6,
        max_rotation_step_rad=0.10,
    )
    mapper.update(sample(1), deadman=True, now_ns=1_000_000_000)
    angle = 0.20
    result = mapper.update(
        sample(
            2,
            x=0.02,
            quaternion=(0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)),
            now=1_010_000_000,
        ),
        deadman=True,
        now_ns=1_010_000_000,
    )

    assert result.active and not result.hold_latched
    assert result.reason == "mapped_step_limited"
    assert np.linalg.norm(result.left_xyz) == pytest.approx(0.01)
    assert geodesic_distance_rad(
        rotation6d_to_matrix(result.left_rotation6d), np.eye(3)
    ) == pytest.approx(0.10)
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


def test_right_only_mapper_ignores_left_jump_and_emits_identity_left_delta():
    mapper = QuestSE3Mapper(controlled_sides=("right",))
    first = TrackingSample(
        left=TrackedPose.make([0, 0, 0], [0, 0, 0, 1]),
        right=TrackedPose.make([0, 0, 0], [0, 0, 0, 1]),
        sequence=1,
        source_time_ns=1_000_000,
        receive_monotonic_ns=1_000_000_000,
        frame_id="world",
    )
    second = TrackingSample(
        left=TrackedPose.make([1, 0, 0], [0, 0, 0, 1]),
        right=TrackedPose.make([0.01, 0, 0], [0, 0, 0, 1]),
        sequence=2,
        source_time_ns=2_000_000,
        receive_monotonic_ns=1_010_000_000,
        frame_id="world",
    )

    mapper.update(first, deadman=True, now_ns=1_000_000_000)
    result = mapper.update(second, deadman=True, now_ns=1_010_000_000)

    assert result.active and not result.hold_latched
    np.testing.assert_allclose(result.left_xyz, np.zeros(3))
    np.testing.assert_allclose(result.right_xyz, [0.01, 0, 0])
