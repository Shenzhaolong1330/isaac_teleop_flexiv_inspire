from flexiv_inspire_isaac.data_pipeline.alignment import Pose, TimedSample
from flexiv_inspire_isaac.data_pipeline.export_spec import ActionView
from flexiv_inspire_isaac.data_pipeline.lerobot_export import EpisodeAligner
from flexiv_inspire_isaac.data_pipeline.lerobot_v3 import _arm_high_rate_frame
from flexiv_inspire_isaac.data_pipeline.mcap_input import _derive_compact_arm_streams


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


def test_wrist_images_accept_nearly_simultaneous_later_camera_frames():
    row = EpisodeAligner(
        {
            "camera/head/jpeg": [s(b"head", 100)],
            "camera/left_wrist/jpeg": [s(b"left", 95)],
            "camera/right_wrist/jpeg": [s(b"right", 110)],
        }
    ).rows()[0]

    assert row["observation.images.left_wrist"] == b"left"
    assert row["observation.images.right_wrist"] == b"right"
    assert row["observation.images.right_wrist.valid"]
    assert row["observation.images.right_wrist.age_ns"] == -10


def test_low_rate_hand_state_uses_its_own_bounded_causal_tolerance():
    row = EpisodeAligner(
        {
            "camera/head/jpeg": [s(b"head", 100_000_000)],
            "robot/left_hand/state": [
                s({"angle": [500.0] * 6}, 40_000_000)
            ],
            "robot/right_hand/state": [
                s({"angle": [500.0] * 6}, 30_000_000)
            ],
        }
    ).rows()[0]

    assert row["observation.left_hand.state.valid"]
    assert row["observation.right_hand.state.valid"]
    assert row["observation.left_hand.state.age_ns"] == 60_000_000
    assert row["observation.right_hand.state.age_ns"] == 70_000_000


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


def test_uniform_training_timeline_really_downsamples_camera_frames():
    streams = {
        "camera/head/jpeg": [
            s(b"head", timestamp)
            for timestamp in (0, 33_333_333, 66_666_667, 100_000_000)
        ]
    }

    rows = EpisodeAligner(streams, timeline_hz=20.0).rows()

    assert [row["timestamp_ns"] for row in rows] == [
        0,
        50_000_000,
        100_000_000,
    ]
    assert all(row["observation.images.head.valid"] for row in rows)


def test_source_gaps_are_marked_as_distinct_capture_segments_with_depth():
    depth = {
        "z16": b"\x01\x00\x02\x00",
        "width": 2,
        "height": 1,
        "scale_m": 0.001,
        "intrinsics": (100.0, 101.0, 1.0, 0.5),
    }
    streams = {
        "camera/head/jpeg": [s(b"head-a", 100), s(b"head-b", 400_000_100)],
        "camera/head/depth_z16": [s(depth, 100), s(depth, 400_000_100)],
    }

    rows = EpisodeAligner(
        streams,
        depth_cameras=("head",),
        segment_gap_threshold_s=0.25,
    ).rows()

    assert [row["observation.capture_segment"] for row in rows] == [0, 1]
    assert [row["observation.frame_in_segment"] for row in rows] == [0, 0]
    assert rows[1]["observation.source_gap_s"] == 0.4
    assert rows[0]["observation.depth.head"] == depth
    assert rows[0]["observation.depth.head.valid"]


def _native_arm_state(value: float) -> dict:
    return {
        **{
            field: [value] * 7
            for field in (
                "q",
                "dq",
                "tau",
                "tau_des",
                "tau_ext",
                "tau_interact",
                "temperature",
            )
        },
        "tcp_pose_rdk_xyz_wxyz": [value] * 7,
        "tcp_velocity": [value] * 6,
        "raw_ft": [value] * 6,
        "external_wrench": [value] * 6,
    }


def test_each_training_frame_embeds_twenty_dual_arm_samples():
    states = [
        s(_native_arm_state(float(index)), 1_000_000 * index, index)
        for index in range(1, 26)
    ]
    streams = {
        "camera/head/jpeg": [s(b"head", 25_000_000)],
        "robot/left_arm/state": states,
        "robot/right_arm/state": states,
    }

    row = EpisodeAligner(
        streams, high_rate_arm_samples_per_frame=20
    ).rows()[0]
    packed, valid, ages = _arm_high_rate_frame(row, 20)

    assert packed.shape == (20 * 2 * 74,)
    assert valid.shape == (40,)
    assert valid.all()
    assert ages.shape == (40,)
    # The window contains the most recent 20 samples: indices 6..25.
    assert packed[0] == 6.0
    assert packed[-1] == 25.0


def test_compact_arm_state_recreates_pose_force_and_twist_views():
    streams = {
        "robot/left_arm/state": [s(_native_arm_state(3.0), 10, 1)]
    }

    _derive_compact_arm_streams(streams)

    assert streams["robot/left_arm/tcp_pose"][0].value.xyz == (3.0, 3.0, 3.0)
    assert streams["robot/left_arm/tcp_twist"][0].value["values"] == [3.0] * 6
    assert streams["robot/left_arm/raw_ft"][0].value["values"] == [3.0] * 6
    assert streams["robot/left_arm/tcp_wrench"][0].value["values"] == [3.0] * 6
