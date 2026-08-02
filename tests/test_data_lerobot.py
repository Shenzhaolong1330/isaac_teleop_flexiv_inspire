from flexiv_inspire_isaac.data_pipeline.alignment import Pose, TimedSample
from flexiv_inspire_isaac.data_pipeline.export_spec import ActionView
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


def test_absolute_joint_action_view_uses_timestamp_aligned_robot_q():
    streams = {
        "camera/head/jpeg": [s(b"head", 100)],
        "robot/left_arm/state": [s({"q": tuple(range(7))}, 99)],
        "robot/right_arm/state": [s({"q": tuple(range(10, 17))}, 98)],
    }
    row = EpisodeAligner(streams, action=ActionView("absolute_joint_position")).rows()[0]
    assert row["action"] == tuple(range(7)) + tuple(range(10, 17))
    assert row["action.valid"]


def test_absolute_cartesian_action_view_uses_interpolated_pose():
    poses = [s(Pose((0, 0, 0), (0, 0, 0, 1)), 95), s(Pose((2, 0, 0), (0, 0, 0, 1)), 105)]
    row = EpisodeAligner({"camera/head/jpeg": [s(b"head", 100)], "robot/left_arm/tcp_pose": poses, "robot/right_arm/tcp_pose": poses}, action=ActionView("absolute_cartesian_pose")).rows()[0]
    assert len(row["action"]) == 18
    assert row["action"][:3] == (1.0, 0.0, 0.0)
    assert row["action.valid"]


def test_episode_aligner_indexes_high_rate_stream_only_once_per_key():
    class CountingList(list):
        iterations = 0

        def __iter__(self):
            self.iterations += 1
            return super().__iter__()

    states = CountingList(
        s({"q": tuple(range(7))}, timestamp, timestamp)
        for timestamp in range(1, 10_001)
    )
    streams = {
        "camera/head/jpeg": [
            s(b"head", timestamp) for timestamp in range(100, 10_001, 100)
        ],
        "robot/left_arm/state": states,
        "robot/right_arm/state": states,
    }
    rows = EpisodeAligner(
        streams, action=ActionView("absolute_joint_position")
    ).rows()
    assert len(rows) == 100
    assert states.iterations == 2
