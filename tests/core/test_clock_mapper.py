from flexiv_inspire_control.clock_mapper import OnlineClockMapper


def test_clock_mapper_converges_and_maps_epoch_or_boot_clock():
    mapper = OnlineClockMapper(min_samples=5, min_span_ns=3_000_000)
    result = None
    for index in range(10):
        device = 1_700_000_000_000_000_000 + index * 1_000_000
        host_mono = 20_000_000_000 + index * 1_000_000
        host_unix = 1_800_000_000_000_000_000 + index * 1_000_000
        result = mapper.update(device, host_mono, host_unix)
    assert result is not None and result.timing_valid
    assert result.mapped_host_monotonic_ns == host_mono
    assert result.mapped_host_unix_ns == host_unix


def test_clock_mapper_resets_on_device_time_reversal():
    mapper = OnlineClockMapper(min_samples=2, min_span_ns=1)
    mapper.update(100, 1000, 10_000)
    mapper.update(200, 1100, 10_100)
    result = mapper.update(150, 1200, 10_200)
    assert not result.timing_valid
    assert result.sample_count == 1
