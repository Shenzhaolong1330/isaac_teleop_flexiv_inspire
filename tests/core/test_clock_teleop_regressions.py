from __future__ import annotations

import numpy as np

from flexiv_inspire_control.clock_mapper import OnlineClockMapper
from flexiv_inspire_control.teleop_mapping import (
    QuestSE3Mapper,
    TrackedPose,
    TrackingSample,
)


def tracking(sequence: int, source_time: int, receive: int) -> TrackingSample:
    pose = TrackedPose.make([0, 0, 0], [0, 0, 0, 1])
    return TrackingSample(
        left=pose,
        right=pose,
        sequence=sequence,
        source_time_ns=source_time,
        receive_monotonic_ns=receive,
        frame_id="world",
    )


def test_repeated_fresh_quest_sample_is_active_identity_not_hold() -> None:
    mapper = QuestSE3Mapper()
    sample = tracking(1, 10_000_000, 1_000_000_000)
    mapper.update(sample, deadman=True, now_ns=1_000_000_000)
    repeated = mapper.update(sample, deadman=True, now_ns=1_010_000_000)
    assert repeated.active
    assert not repeated.hold_latched
    assert repeated.reason == "tracking_repeat"
    np.testing.assert_array_equal(repeated.left_xyz, np.zeros(3))


def test_actual_quest_sequence_reversal_still_latches() -> None:
    mapper = QuestSE3Mapper()
    mapper.update(
        tracking(2, 20_000_000, 1_000_000_000),
        deadman=True,
        now_ns=1_000_000_000,
    )
    reversed_sample = mapper.update(
        tracking(1, 10_000_000, 1_010_000_000),
        deadman=True,
        now_ns=1_010_000_000,
    )
    assert reversed_sample.hold_latched


def test_repeated_rdk_timestamp_preserves_converged_clock_window() -> None:
    mapper = OnlineClockMapper(min_samples=5, min_span_ns=3_000_000)
    result = None
    for index in range(6):
        result = mapper.update(
            1_000_000_000 + index * 1_000_000,
            10_000_000_000 + index * 1_000_000,
            1_800_000_000_000_000_000 + index * 1_000_000,
        )
    assert result is not None and result.timing_valid
    count = result.sample_count
    repeated = mapper.update(
        1_005_000_000,
        10_006_000_000,
        1_800_000_000_006_000_000,
    )
    assert repeated.timing_valid
    assert repeated.sample_count == count


def test_future_clock_fit_is_invalid_instead_of_age_clamped() -> None:
    mapper = OnlineClockMapper(
        min_samples=5,
        min_span_ns=3_000_000,
        max_rate_error=2e-3,
        max_residual_rms_ns=5_000_000,
    )
    host_offsets = [0, 1_000_500, 2_000_500, 3_000_500, 4_000_000]
    result = None
    for index, offset in enumerate(host_offsets):
        result = mapper.update(
            1_000_000_000 + index * 1_000_000,
            10_000_000_000 + offset,
            1_800_000_000_000_000_000 + offset,
        )
    assert result is not None
    assert result.slope > 1.0
    assert not result.timing_valid
    assert result.mapped_host_monotonic_ns == 0
