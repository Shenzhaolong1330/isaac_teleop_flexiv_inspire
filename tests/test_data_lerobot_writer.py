import numpy as np

from flexiv_inspire_isaac.data_pipeline.alignment import InterpolatedPose, Pose
from flexiv_inspire_isaac.data_pipeline.lerobot_v3 import VALIDITY_FIELDS, aligned_row_to_frame, export_rows


class FakeDataset:
    instances = []

    @classmethod
    def create(cls, **kwargs):
        instance = cls(**kwargs)
        cls.instances.append(instance)
        return instance

    def __init__(self, **kwargs):
        self.create_kwargs = kwargs
        self.frames = []
        self.saved = False
        self.finalized = False
        if FakeDataset.instances:
            self.frames = list(FakeDataset.instances[-1].frames)

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved = True

    def finalize(self):
        self.finalized = True

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]


def valid_row():
    action = np.zeros(30, dtype=np.float32)
    action[3:9] = (1, 0, 0, 0, 1, 0)
    action[12:18] = (1, 0, 0, 0, 1, 0)
    row = {
        "action": action,
        "action.valid": True,
        "observation.left_arm.pose": InterpolatedPose(
            Pose((0, 0, 0), (0, 0, 0, 1)), (1, 0, 0, 0, 1, 0), 1, 2
        ),
        "observation.right_arm.pose": InterpolatedPose(
            Pose((0, 0, 0), (0, 0, 0, -1)), (1, 0, 0, 0, 1, 0), 1, 2
        ),
    }
    for side in ("left", "right"):
        row[f"observation.{side}_arm.state"] = {
            "q": (1,) * 7,
            "dq": (2,) * 7,
            "tau": (3,) * 7,
            "tau_des": (4,) * 7,
            "tau_ext": (5,) * 7,
            "tau_interact": (6,) * 7,
            "temperature": (31,) * 7,
        }
        row[f"observation.{side}_arm.tcp_twist"] = (7,) * 6
        row[f"observation.{side}_hand.state"] = {
            "angle": (10,) * 6,
            "position": (11,) * 6,
            "actual_force": (12,) * 6,
            "current": (13,) * 6,
            "temperature": (32,) * 6,
            "error": (0,) * 6,
            "status": (1,) * 6,
        }
        row[f"observation.{side}_hand.tactile"] = (0,) * 1062
        for sensor in ("raw_ft", "tcp_wrench"):
            row[f"observation.{side}_arm.{sensor}"] = (0,) * 6
    for name in VALIDITY_FIELDS:
        key = "action" if name == "action" else f"observation.{name}"
        row[f"{key}.valid"] = True
        row[f"{key}.age_ns"] = 0
    for camera in ("head", "left_wrist", "right_wrist"):
        row[f"observation.images.{camera}"] = np.zeros((240, 424, 3), dtype=np.uint8)
        row[f"observation.images.{camera}.valid"] = True
    return row


def test_writer_calls_create_add_save_finalize_and_reload(tmp_path):
    FakeDataset.instances.clear()
    result = export_rows(
        [valid_row()],
        output_root=tmp_path / "dataset",
        repo_id="local/test",
        task="test task",
        dataset_class=FakeDataset,
    )
    created = FakeDataset.instances[0]
    assert created.saved and created.finalized
    assert created.frames[0]["action"].shape == (30,)
    assert created.frames[0]["observation.tactile"].shape == (2124,)
    frame = created.frames[0]
    assert np.all(frame["observation.arm_q"] == 1)
    assert np.all(frame["observation.arm_tau_interact"] == 6)
    assert np.all(frame["observation.tcp_twist"] == 7)
    assert tuple(frame["observation.arm_quaternion_xyzw"][[3, 7]]) == (1, -1)
    assert frame["observation.tactile"].dtype == np.uint16
    assert frame["observation.hand_error"].dtype == np.uint16
    assert np.all(frame["observation.hand_actual_force"] == 12)
    assert frame["observation.hand_field_valid"].shape == (14,)
    assert np.all(frame["observation.hand_field_valid"])
    assert np.all(frame["observation.hand_field_age_s"] == 0)
    assert result.frames_written == result.reload_length == 1


def test_writer_accepts_an_existing_empty_output_directory(tmp_path):
    FakeDataset.instances.clear()
    output_root = tmp_path / "empty-dataset-root"
    output_root.mkdir()

    result = export_rows(
        [valid_row()],
        output_root=output_root,
        repo_id="local/test",
        task="test task",
        dataset_class=FakeDataset,
    )

    assert result.frames_written == 1


def test_writer_infers_configured_camera_resolution(tmp_path):
    FakeDataset.instances.clear()
    row = valid_row()
    for camera in ("head", "left_wrist", "right_wrist"):
        row[f"observation.images.{camera}"] = np.zeros(
            (120, 160, 3), dtype=np.uint8
        )

    result = export_rows(
        [row],
        output_root=tmp_path / "dataset",
        repo_id="local/test",
        task="test task",
        dataset_class=FakeDataset,
    )

    assert result.frames_written == 1
    assert FakeDataset.instances[0].frames[0][
        "observation.images.head"
    ].shape == (120, 160, 3)
    assert FakeDataset.instances[0].create_kwargs["features"][
        "observation.images.head"
    ]["shape"] == (120, 160, 3)


def test_writer_exports_selected_head_depth_and_segment_fields(tmp_path):
    FakeDataset.instances.clear()
    row = valid_row()
    row.update(
        {
            "observation.depth.head": {
                "z16": np.asarray([[1, 2], [3, 4]], dtype="<u2").tobytes(),
                "width": 2,
                "height": 2,
                "scale_m": 0.001,
                "intrinsics": (100.0, 101.0, 1.0, 1.0),
            },
            "observation.depth.head.valid": True,
            "observation.source_timestamp_ns": 123456789,
            "observation.source_gap_s": 0.3,
            "observation.capture_segment": 1,
            "observation.frame_in_segment": 0,
            "observation.capture_segment_start": True,
        }
    )
    fields = (
        "observation.images.head",
        "observation.depth.head",
        "observation.depth_scale_m.head",
        "observation.depth_intrinsics.head",
        "observation.source_timestamp_ns",
        "observation.source_gap_s",
        "observation.capture_segment",
        "observation.frame_in_segment",
        "observation.capture_segment_start",
        "action",
    )

    result = export_rows(
        [row],
        output_root=tmp_path / "depth-dataset",
        repo_id="local/test-depth",
        task="test task",
        dataset_class=FakeDataset,
        fields=fields,
        depth_cameras=("head",),
    )

    created = FakeDataset.instances[0]
    assert set(created.frames[0]).difference({"task"}) == set(fields)
    assert created.frames[0]["observation.depth.head"].dtype == np.uint16
    assert created.frames[0]["observation.depth.head"].shape == (2, 2)
    assert created.frames[0]["observation.depth.head"].tolist() == [[1, 2], [3, 4]]
    assert created.create_kwargs["features"]["observation.depth.head"] == {
        "dtype": "uint16",
        "shape": (2, 2),
        "names": None,
    }
    assert result.frames_dropped_invalid_depth == 0
    assert result.episodes_written == 1


def test_hand_field_timing_uses_each_modbus_read_timestamp():
    row = valid_row()
    row["timestamp_ns"] = 2_000_000_000
    row["observation.left_hand.state"]["angle_timing"] = {
        "valid": True,
        "timing_valid": True,
        "mapped_host_time": {"sec": 1, "nanosec": 900_000_000},
    }
    row["observation.left_hand.state"]["position_timing"] = {
        "valid": True,
        "timing_valid": True,
        "mapped_host_time": {"sec": 2, "nanosec": 100_000_000},
    }
    frame = aligned_row_to_frame(row, "test task")
    assert frame["observation.hand_field_valid"][0]
    assert np.isclose(frame["observation.hand_field_age_s"][0], 0.1)
    assert not frame["observation.hand_field_valid"][1]
    assert np.isnan(frame["observation.hand_field_age_s"][1])


def test_invalid_image_row_is_dropped_not_zero_filled(tmp_path):
    FakeDataset.instances.clear()
    row = valid_row()
    row["observation.images.left_wrist.valid"] = False
    try:
        export_rows(
            [row],
            output_root=tmp_path / "dataset",
            repo_id="local/test",
            task="test task",
            dataset_class=FakeDataset,
        )
    except ValueError as exc:
        assert "no fully valid" in str(exc)
    else:
        raise AssertionError("invalid image row must not be silently filled")


def test_export_report_separates_invalid_actions_from_invalid_images(tmp_path):
    FakeDataset.instances.clear()
    invalid_action = valid_row()
    invalid_action["valid"] = {"action": False}

    result = export_rows(
        [invalid_action, valid_row()],
        output_root=tmp_path / "dataset",
        repo_id="local/test",
        task="test task",
        dataset_class=FakeDataset,
    )

    assert result.frames_dropped_invalid_action == 1
    assert result.frames_dropped_invalid_image == 0
