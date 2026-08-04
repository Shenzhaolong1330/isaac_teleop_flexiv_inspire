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


def test_quantized_rdk_clock_stays_valid_after_convergence() -> None:
    mapper = OnlineClockMapper(
        min_samples=5,
        min_span_ns=10_000_000,
        max_rate_error=2e-3,
        max_residual_rms_ns=5_000_000,
    )
    results = []
    for index in range(80):
        # The controller timestamp has a 1 ms resolution while the bridge
        # polls at 300 Hz.  Real hardware consequently advances in a repeating
        # 3/4/3 ms cadence even though host receives are evenly spaced.
        device_offset = round(index * 10_000_000 / 3)
        host_offset = round(index * 10_000_000 / 3) + 350_000
        results.append(mapper.update(
            1_000_000_000 + device_offset,
            10_000_000_000 + host_offset,
            1_800_000_000_000_000_000 + host_offset,
        ))

    assert all(result.timing_valid for result in results[20:])
    assert all(
        result.mapped_host_monotonic_ns
        <= 10_000_000_000 + round(index * 10_000_000 / 3) + 350_000
        for index, result in enumerate(results)
        if result.timing_valid
    )


def test_mapping_an_unobserved_future_device_time_is_invalid() -> None:
    mapper = OnlineClockMapper(min_samples=5, min_span_ns=3_000_000)
    for index in range(5):
        mapper.update(
            1_000_000_000 + index * 1_000_000,
            10_000_000_000 + index * 1_000_000,
            1_800_000_000_000_000_000 + index * 1_000_000,
        )

    result = mapper.map(1_010_000_000)
    assert not result.timing_valid
    assert result.mapped_host_monotonic_ns == 0
