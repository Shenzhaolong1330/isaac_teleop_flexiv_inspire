from flexiv_inspire_isaac.data_pipeline.alignment import Pose, TimedSample
from flexiv_inspire_isaac.data_pipeline.lerobot_export import EpisodeAligner


def s(value, t, seq=0, valid=True):
    return TimedSample(value, t, t + 1, seq, valid)


def test_lerobot_uses_head_clock_sent_command_and_explicit_invalids():
    streams = {
        "camera/head/jpeg": [s(b"head", 100)],
        "camera/left_wrist/jpeg": [s(b"left", 90)],
        "camera/right_wrist/jpeg": [],
        "control/sent_command": [s(tuple(range(30)), 99)],
        "control/requested_command": [s(("must", "not", "use"), 99)],
        "robot/left_arm/tcp_pose": [
            s(Pose((0, 0, 0), (0, 0, 0, 1)), 95, 1),
            s(Pose((1, 0, 0), (0, 0, 0, 1)), 105, 2),
        ],
        "robot/right_arm/tcp_pose": [
            s(Pose((0, 0, 0), (0, 0, 0, 1)), 95, 1),
            s(Pose((1, 0, 0), (0, 0, 0, 1)), 105, 2),
        ],
    }
    row = EpisodeAligner(streams).rows()[0]
    assert row["timestamp_ns"] == 100
    assert row["action"] == tuple(range(30))
    assert row["observation.images.right_wrist"] is None
    assert not row["observation.images.right_wrist.valid"]
    assert len(row["observation.left_arm.pose"].rotation6d) == 6
