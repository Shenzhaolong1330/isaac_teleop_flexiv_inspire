from flexiv_inspire_isaac.data_pipeline.recorder import (
    AsyncMcapRecorder,
    RecordEnvelope,
)


class Sink:
    def __init__(self):
        self.values = []
        self.closed = False

    def write(self, envelope):
        self.values.append(envelope)

    def close(self):
        self.closed = True


def record(topic, sequence):
    return RecordEnvelope(topic, sequence, sequence + 1, sequence, True, {"v": sequence})


def test_critical_events_never_drop_and_sensor_queue_drops_oldest():
    sink = Sink()
    recorder = AsyncMcapRecorder(sink, sensor_capacity=2)
    recorder.submit(record("sensor", 1))
    recorder.submit(record("sensor", 2))
    recorder.submit(record("sensor", 3))
    recorder.submit(record("event", 4), critical=True)
    recorder.start()
    recorder.close()
    assert [item.sequence for item in sink.values] == [4, 2, 3]
    assert recorder.dropped_by_topic["sensor"] == 1
    assert recorder.written_by_topic["event"] == 1
    assert sink.closed


def test_invalid_numeric_payload_is_reported_not_silently_written():
    sink = Sink()
    recorder = AsyncMcapRecorder(sink)
    bad = RecordEnvelope("bad", 1, 2, 1, False, {"value": float("nan")})
    recorder.start()
    recorder.submit(bad, critical=True)
    try:
        recorder.close()
    except RuntimeError as exc:
        assert "Out of range float values" in str(exc)
    else:
        raise AssertionError("NaN payload should fail serialization")
