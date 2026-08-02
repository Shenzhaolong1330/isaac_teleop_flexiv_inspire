from types import SimpleNamespace

from flexiv_inspire_isaac.data_pipeline.episode_manager import EpisodeSession
from flexiv_inspire_isaac.data_pipeline.manifest import StreamStats


def test_completed_demonstration_requires_training_critical_streams():
    session = EpisodeSession.__new__(EpisodeSession)
    session.manifest = SimpleNamespace(streams={})

    missing = session._required_stream_errors()

    assert "camera/head/color/image_raw/compressed" in missing
    assert "control/sent_command" in missing


def test_required_stream_check_counts_only_valid_samples():
    session = EpisodeSession.__new__(EpisodeSession)
    session.manifest = SimpleNamespace(streams={})
    missing = session._required_stream_errors()
    session.manifest = SimpleNamespace(
        streams={name: StreamStats(1.0, samples=1) for name in missing}
    )
    session.manifest.streams["control/sent_command"].invalid = 1

    assert session._required_stream_errors() == ["control/sent_command"]
