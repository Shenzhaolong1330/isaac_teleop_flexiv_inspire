import numpy as np

from flexiv_inspire_isaac.data_pipeline.alignment import TimedSample
from flexiv_inspire_isaac.data_pipeline.lerobot_export import EpisodeAligner
from isaac_teleop_core.rotation6d import (
    matrix_to_rotation6d, rotation6d_to_matrix, rotvec_to_matrix,
)


def command(time, dx=0.01, rotation=(0, 0, 0), hand=500, valid=True):
    value = np.zeros(30)
    value[3:9] = matrix_to_rotation6d(np.eye(3))
    value[9] = dx
    value[12:18] = matrix_to_rotation6d(rotvec_to_matrix(rotation))
    value[18:30] = hand
    return TimedSample(value, time, time, 0, valid=valid)


def align(samples):
    return EpisodeAligner(
        {"control/sent_command": samples},
        timeline_source="control/sent_command", timeline_hz=30,
        compose_action_deltas=True, segment_gap_threshold_s=0.05,
    ).rows()


def test_60_to_30_composes_world_rotation_and_keeps_last_hand():
    rows = align([command(0, rotation=(0.2, 0, 0), hand=100),
                  command(16_666_667, rotation=(0, 0.3, 0), hand=800),
                  command(33_333_334, hand=200)])
    assert len(rows) == 2
    np.testing.assert_allclose(rows[0]["action"][9:12], [0.02, 0, 0])
    np.testing.assert_allclose(rotation6d_to_matrix(rows[0]["action"][12:18]),
                               rotvec_to_matrix((0, 0.3, 0)) @ rotvec_to_matrix((0.2, 0, 0)))
    np.testing.assert_equal(rows[0]["action"][24:30], [800] * 6)


def test_30hz_unchanged_and_exact_boundary_owned_once():
    samples = [command(i * 33_333_333, dx=0.01 * (i + 1)) for i in range(5)]
    rows = align(samples)
    assert len(rows) == len(samples)
    for row, sample in zip(rows, samples):
        np.testing.assert_allclose(row["action"], sample.value)


def test_jitter_and_pause_preserve_total_motion_without_stale_repeats():
    samples = [command(t, rotation=(i * .01, .02, 0), hand=i)
               for i, t in enumerate([0, 16_000_000, 34_000_000, 49_000_000,
                                      99_500_000, 116_000_000, 134_000_000])]
    rows = align(samples)
    assert [r["observation.capture_segment"] for r in rows] == [0, 0, 1, 1]
    assert all(r["action.valid"] for r in rows)
    np.testing.assert_allclose(sum(r["action"][9] for r in rows), .07)
    raw_rotation = np.eye(3)
    output_rotation = np.eye(3)
    for sample in samples:
        raw_rotation = rotation6d_to_matrix(sample.value[12:18]) @ raw_rotation
    for row in rows:
        output_rotation = rotation6d_to_matrix(row["action"][12:18]) @ output_rotation
    np.testing.assert_allclose(output_rotation, raw_rotation, atol=1e-12)


def test_invalid_command_invalidates_whole_window():
    rows = align([command(0), command(16_000_000, valid=False), command(34_000_000)])
    assert not rows[0]["action.valid"]
    assert rows[1]["action.valid"]


def test_empty_window_does_not_repeat_previous_motion():
    rows = align([command(0), command(49_000_000), command(98_000_000), command(147_000_000)])
    assert [row["action.valid"] for row in rows] == [True, True, True, False, True]
    assert sum(row["action"][9] for row in rows if row["action.valid"]) == .04


def test_observation_never_uses_future_image():
    sample = command(0)
    rows = EpisodeAligner(
        {"control/sent_command": [sample, command(16_000_000)],
         "camera/head/jpeg": [TimedSample(b"future", 10, 10, 0)]},
        timeline_source="control/sent_command", timeline_hz=30,
        compose_action_deltas=True, allow_future_camera_matches=False,
    ).rows()
    assert not rows[0]["observation.images.head.valid"]
