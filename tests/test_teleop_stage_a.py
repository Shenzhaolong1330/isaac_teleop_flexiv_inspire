"""Stage A acceptance exercises production collection, export, actuator and scheduler paths."""

from __future__ import annotations
import ast
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from flexiv_inspire_isaac.data_pipeline.action_contract import (
    STEP_NS,
    export_supervision,
    reconstruct,
    windows,
)
from flexiv_inspire_isaac.data_pipeline.stage_a_smoke import (
    command_payload,
    envelope,
    run,
)
from flexiv_inspire_isaac.data_pipeline.provenance import (
    archive_episode,
    verify_archive,
    resolve_asset,
)
from flexiv_inspire_isaac.data_pipeline.quality import written_quality
from isaac_teleop_core.command import (
    BimanualCommand,
    CommandSource,
    CommandPoint,
    ValidMask,
)
from isaac_teleop_core.control import ControlState, HoldReason
from isaac_teleop_core.scheduling import PointSchedule
from isaac_teleop_core.evidence import (
    receipt,
    summarize_receipts,
    emit_receipt,
    command_id,
)
from flexiv_inspire_isaac.dftp.models import HandCommand
from flexiv_inspire_isaac.dftp.worker import DftpHandWorker

ROOT = Path(__file__).resolve().parents[1]


def records(count=8):
    result = [
        asdict(
            envelope(
                "episode/events", 900_000_000, 0, {"event_type": "episode_started"}
            )
        )
    ]
    for i in range(count):
        t = 1_000_000_000 + i * STEP_NS
        result.append(
            asdict(envelope("control/sent_command", t, i, command_payload(i)))
        )
    return result


def test_multirate_production_closure_and_move(tmp_path):
    result = run(tmp_path / "evidence")
    assert result["commands"] == 60 and result["camera_frames_per_stream"] == 15
    original = Path(result["manifest"]).parent
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    shutil.rmtree(tmp_path / "evidence" / "inputs")
    shutil.rmtree(original)
    verify_archive(moved)
    config = resolve_asset(moved, "effective_config").read_text()
    assert "test-secret-must-not-archive" not in config
    assert "<redacted>" in config
    index = verify_archive(moved)
    entry = index["entries"]["tool_payload"]
    (moved / entry["path"]).write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_archive(moved)


def test_required_asset_missing_detected(tmp_path):
    manifest = {
        "provenance": archive_episode(
            tmp_path, config={"sampling": {}}, assets={"required": tmp_path / "absent"}
        )
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="required archive asset missing"):
        verify_archive(tmp_path)


def test_noncommuting_rotation_order_hands_absolute_and_state_distinct():
    raw = records(2)
    rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    for record, rot in zip(raw[1:], (rx, rz)):
        record["payload"] = command_payload(record["sequence"], rotation=rot)
    for record in raw[1:].copy():
        raw.append(
            asdict(
                envelope(
                    "robot/left_arm/state",
                    record["source_time_ns"] - 1,
                    record["sequence"],
                    {"q": [999] * 7},
                )
            )
        )
    document = export_supervision(
        raw,
        episode_id="parent",
        required={"robot/left_arm/state": 20_000_000},
        optional={},
    )
    xyz, rot, hand = reconstruct(document["rows"])
    np.testing.assert_allclose(rot[0], rz @ rx)
    assert not np.allclose(rot[0], rx @ rz)
    np.testing.assert_allclose(hand[0], [101] * 6)
    assert (
        document["rows"][0]["observations"]["robot/left_arm/state"]["value"]["q"]
        == [999] * 7
    )
    assert document["rows"][0]["action"][0] == 0.001


def test_missing_commands_are_invalid_and_legacy_opt_in():
    raw = records()
    del raw[4]
    document = export_supervision(raw, episode_id="parent", required={}, optional={})
    assert document["quality"]["rejections"]["missing-command"] == 1
    assert any(
        row["command_id"] is None and row["action"] is None for row in document["rows"]
    )
    assert all(
        all(r["command_id"] is not None for r in w) for w in windows(document, 3)
    )
    with pytest.raises(ValueError, match="allow-legacy"):
        export_supervision(raw[1:], episode_id="parent", required={})
    legacy = export_supervision(
        raw[1:], episode_id="parent", required={}, allow_legacy=True
    )
    assert legacy["rows"][0]["boundary_basis"] == "legacy-gap-inference-uncertain"


@pytest.mark.parametrize(
    "event",
    [
        "pedal_released",
        "episode_paused",
        "home_started",
        "home_completed",
        "recovered",
        "fault",
        "control_changed",
    ],
)
def test_event_boundaries_source_time_delay_static_and_parent(event):
    raw = records()
    time = raw[5]["source_time_ns"]
    # Arrives AFTER subsequent commands, but effective time is the boundary.
    raw.append(
        asdict(
            envelope(
                "episode/events",
                time,
                10,
                {"event_type": event},
                receive=time + 10 * STEP_NS,
            )
        )
    )
    document = export_supervision(
        raw, episode_id="task-parent", required={}, optional={}
    )
    assert len({r["segment_id"] for r in document["rows"]}) == 2
    assert all(r["episode_id"] == "task-parent" for r in document["rows"])
    assert all(len({r["segment_id"] for r in w}) == 1 for w in windows(document, 3))
    # Exact zero deltas and holding the same hand target do not split segments.
    static = records()
    for record in static[1:]:
        record["payload"]["trajectory"][0]["left_delta_xyz"] = [0, 0, 0]
    assert (
        len(
            {
                r["segment_id"]
                for r in export_supervision(
                    static, episode_id="task", required={}, optional={}
                )["rows"]
            }
        )
        == 1
    )


def test_required_missing_stale_optional_masks_zero_saturation_and_future():
    raw = records(1)
    t = raw[1]["source_time_ns"]
    raw.append(asdict(envelope("required", t + 1, 1, {"value": 123})))
    doc = export_supervision(
        raw, episode_id="e", required={"required": 10}, optional={"tactile": 10}
    )
    assert not doc["rows"][0]["valid"]
    assert doc["rows"][0]["observations"]["tactile"]["value"] is None
    assert not doc["rows"][0]["observations"]["tactile"]["valid"]
    raw.append(asdict(envelope("required", t - 20, 0, {"value": 123})))
    assert not export_supervision(
        raw, episode_id="e", required={"required": 10}, optional={}
    )["rows"][0]["valid"]
    for valid, values in ((True, [0, 65535]), (False, [0, 65535])):
        data = records(1) + [
            asdict(envelope("tactile", t, 0, {"values": values}, valid=valid))
        ]
        row = export_supervision(
            data, episode_id="e", required={}, optional={"tactile": 10}
        )["rows"][0]
        assert row["valid"]
        assert row["observations"]["tactile"]["valid"] == valid
        assert row["observations"]["tactile"]["value"] == (
            {"values": values} if valid else None
        )


def test_written_quality_rates_pause_number_rules_queue_and_reset():
    raw = []
    for start, seqs in ((1_000_000_000, [100, 102, 104]), (10_000_000_000, [0, 2, 4])):
        raw.append(
            asdict(
                envelope(
                    "episode/events",
                    start - 1,
                    start,
                    {"event_type": "episode_resumed"},
                )
            )
        )
        for i, sequence in enumerate(seqs):
            raw.append(
                asdict(envelope("sensor", start + i * 100_000_000, sequence, {}))
            )
    result = written_quality(
        raw,
        config={},
        enqueued={"sensor": 8},
        queue_drops={"/sensor": 2},
        suppressed={"sensor": 90},
        sequence_rules={"sensor": {"kind": "continuous_counter", "step": 2}},
    )
    stream = result["streams"]["sensor"]
    assert stream["observed_hz"] == pytest.approx(10)
    assert (
        stream["written"] == 6
        and stream["queue_dropped"] == 2
        and stream["gated_suppressed"] == 90
    )
    assert stream["source_transport_loss"] is None
    assert all(s["observed_sequence_gaps"] == 0 for s in stream["segments"])
    unknown = written_quality(raw, config={})["streams"]["sensor"]
    assert all(s["observed_sequence_gaps"] is None for s in unknown["segments"])


def test_mailbox_receipts_replacement_failure_expiry_and_unknown():
    evidence = []

    class Sink:
        fail = False

        def apply(self, command):
            if self.fail:
                raise OSError("second modbus write failed")

    sink = Sink()
    worker = DftpHandWorker(
        "left",
        SimpleNamespace(),
        command_sink=sink,
        on_receipt=evidence.append,
        clock_ns=lambda: 100,
    )

    def cmd(i, deadline=1000):
        return HandCommand(
            i, (300,) * 6, (500,) * 6, deadline, "teleop", f"s:teleop:{i}"
        )

    worker.submit(cmd(1))
    worker.submit(cmd(2))
    worker._service_command(100)
    worker.submit(cmd(3, deadline=50))
    worker._service_command(100)
    sink.fail = True
    worker.submit(cmd(4))
    with pytest.raises(OSError):
        worker._service_command(100)
    statuses = {(r["command_id"], r["status"]) for r in evidence}
    assert ("s:teleop:1", "replaced") in statuses
    assert ("s:teleop:2", "sdk_returned") in statuses
    assert ("s:teleop:3", "timeout") in statuses
    assert ("s:teleop:4", "send_failed") in statuses
    summary = summarize_receipts("s:teleop:2", evidence, deadline_ns=1000, now_ns=2000)
    assert summary["actuators"]["left_hand"] == "sdk_returned"
    assert summary["actuators"]["right_hand"] == "timeout"
    assert summary["physical_arrival"] == "unknown"


def production_execute_method():
    # Load the actual method body, isolate ROS imports only; control path is not reimplemented.
    path = ROOT / "ros2_ws/src/flexiv_inspire_control/flexiv_inspire_control/node.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ControlBridge"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_execute_chunk"
    )
    namespace = {
        "PointSchedule": PointSchedule,
        "BimanualCommandMsg": object,
        "ControlState": ControlState,
        "HoldReason": HoldReason,
        "CommandSource": CommandSource,
        "ValidMask": ValidMask,
        "command_id": command_id,
        "receipt": receipt,
        "emit_receipt": emit_receipt,
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


@pytest.mark.parametrize(
    "delay,replace_at,expected", [(0, None, 4), (2 * STEP_NS, None, 2), (0, STEP_NS, 1)]
)
def test_production_scheduler_no_repeat_burst_replacement_exhaustion(
    delay, replace_at, expected
):
    clock = [1_000_000_000 + delay]
    sent = []
    bridge = SimpleNamespace(
        _chunk_lock=threading.Lock(),
        _chunk_generation=1,
        _arbiter=SimpleNamespace(
            snapshot=SimpleNamespace(state=ControlState.ACTIVE),
            tick=lambda **k: SimpleNamespace(state=ControlState.ACTIVE),
            mark_sent=lambda c: None,
        ),
        _update_gates=lambda now: None,
        _targets_for_point=lambda c, p: ({}, {}, {}),
        _single_point_message=lambda msg, index, seq, **k: SimpleNamespace(index=index),
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: None)
        ),
        _safe_pub=SimpleNamespace(publish=lambda msg: None),
        _sent_pub=SimpleNamespace(
            publish=lambda msg: sent.append((msg.index, clock[0]))
        ),
        _emit_deviceio=lambda *a, **k: True,
        _publish_trace=lambda *a: None,
    )

    def sleep(seconds):
        clock[0] += round(seconds * 1e9)
        if replace_at is not None and clock[0] >= 1_000_000_000 + replace_at:
            bridge._chunk_generation = 2

    namespace = production_execute_method()
    namespace["time"] = SimpleNamespace(monotonic_ns=lambda: clock[0], sleep=sleep)
    command = BimanualCommand.from_policy_vectors(
        [CommandPoint.identity().to_policy_vector()] * 4,
        session_id="s",
        source=CommandSource.POLICY,
        sequence=1,
        ttl_s=0.5,
        frame_id="world",
        deadman=True,
        valid_mask=ValidMask.LEFT_HAND | ValidMask.RIGHT_HAND,
        issued_monotonic_ns=1_000_000_000,
    )
    namespace["_execute_chunk"](bridge, 1, command, object(), 1_000_000_000)
    assert len(sent) == expected
    assert len({i for i, t in sent}) == len(sent)
    assert all(
        right[1] - left[1] >= STEP_NS // 2 for left, right in zip(sent, sent[1:])
    )
    # Finished iterator cannot repeat the final increment; standalone admission rejects duplicate polls.
    schedule = PointSchedule(command, 1_000_000_000)
    assert schedule.admit(0, 1_000_000_000) == "execute"
    assert schedule.admit(0, 1_000_000_000) == "already-consumed"


def test_actual_daemon_partial_send_receipts_do_not_claim_all_success():
    import time
    from flexiv_rdk_daemon.mock_backend import MockBackend
    from flexiv_rdk_daemon.server import (
        DaemonInterlock,
        HandObservationCache,
        RDKRequestDispatcher,
    )

    class Backend(MockBackend):
        def send_cartesian_target(self, side, pose_rdk, **kwargs):
            if side == "right":
                raise RuntimeError("right failed")
            return super().send_cartesian_target(side, pose_rdk, **kwargs)

    class Emitter:
        producer = "rdk"
        records = []

        def emit(self, document, **kwargs):
            self.records.append(document["payload"])

    backend = Backend()
    interlock = DaemonInterlock()
    interlock.ready_generation = backend.connection_generation
    emitter = Emitter()
    dispatcher = RDKRequestDispatcher(
        backend,
        SimpleNamespace(is_zeroed_for=lambda s: True),
        HandObservationCache(),
        interlock,
        deviceio_emitter=emitter,
        control_authorizations=SimpleNamespace(consume=lambda *a, **k: None),
    )
    target = {
        "tcp_pose_rdk": [0, 0, 0, 1, 0, 0, 0],
        "max_linear_velocity": 0.01,
        "max_angular_velocity": 0.05,
        "max_linear_acceleration": 0.05,
        "max_angular_acceleration": 0.1,
        "cartesian_stiffness": [1200] * 3 + [80] * 3,
        "cartesian_damping_ratio": [0.7] * 6,
    }
    payload = {
        "session_id": "s",
        "source": "teleop",
        "source_sequence": "64",
        "expires_monotonic_ns": str(time.monotonic_ns() + 100_000_000),
        "left": dict(target),
        "right": dict(target),
        "valid_mask": 3,
        "safety_validated": True,
        "local_permission": True,
        "physical_pedal": True,
        "local_arm_token": "t",
    }
    try:
        with pytest.raises(RuntimeError, match="right failed"):
            dispatcher("cartesian_command", 1, payload, (222, 0, 0))
        statuses = {(r["actuator"], r["status"]) for r in emitter.records}
        assert ("left_arm", "sdk_returned") in statuses
        assert ("right_arm", "send_failed") in statuses
        accepted = next(r for r in emitter.records if r["status"] == "sdk_returned")
        assert accepted["accepted_target"]["tcp_pose_rdk"] == target["tcp_pose_rdk"]
        assert accepted["physical_arrival"] == "unknown"
        summary = summarize_receipts(
            "s:teleop:64", emitter.records, deadline_ns=1, now_ns=2
        )
        assert summary["actuators"]["left_arm"] == "sdk_returned"
        assert summary["actuators"]["right_arm"] == "send_failed"
        assert summary["actuators"]["left_hand"] == "timeout"
    finally:
        dispatcher.close()


def test_both_writers_break_invalid_and_event_segments_preserve_parent(tmp_path):
    from test_data_policy_profile_export import CountingDataset, _row
    from test_data_lerobot_writer import valid_row
    from flexiv_inspire_isaac.data_pipeline.profile_export import (
        PolicyEpisode,
        export_policy_episodes,
    )
    from flexiv_inspire_isaac.data_pipeline.lerobot_v3 import export_rows

    for kind in ("generic", "profile"):
        CountingDataset.instances.clear()
        factory = valid_row if kind == "generic" else _row
        rows = [factory() for _ in range(4)]
        for i, row in enumerate(rows):
            row["parent_episode_id"] = "parent-task"
            row["observation.capture_segment"] = 0 if i < 3 else 1
            row["segment.boundary_basis"] = "explicit-event"
        rows[1]["training.valid"] = False
        output = tmp_path / kind
        if kind == "generic":
            result = export_rows(
                rows,
                output_root=output,
                repo_id="local/t",
                task="test",
                dataset_class=CountingDataset,
            )
        else:
            result = export_policy_episodes(
                [PolicyEpisode(rows, "test", "original-manifest")],
                output_root=output,
                repo_id="local/t",
                profile_id="joint_proprio_cartesian_v1",
                dataset_class=CountingDataset,
                fps=15,
            )
        assert result.frames_written == 3
        lineage = json.loads((output / "episode_lineage.json").read_text())
        assert len(lineage) == 3
        assert all(item["parent_episode_id"] == "parent-task" for item in lineage)
        assert [item["capture_segment"] for item in lineage] == [0, 0, 1]


def test_late_message_inside_paused_interval_rejected():
    raw = records(5)
    raw.append(
        asdict(
            envelope(
                "episode/events",
                raw[2]["source_time_ns"],
                10,
                {"event_type": "episode_paused"},
            )
        )
    )
    raw.append(
        asdict(
            envelope(
                "episode/events",
                raw[4]["source_time_ns"],
                11,
                {"event_type": "episode_resumed"},
            )
        )
    )
    # Writer arrival order deliberately places late commands after resume.
    late = raw.pop(3)
    raw.append(late)
    document = export_supervision(raw, episode_id="parent", required={}, optional={})
    assert (
        sum("inactive-recording-interval" in row["reasons"] for row in document["rows"])
        == 2
    )


def test_source_summary_counters_do_not_leak_between_episodes(tmp_path):
    run(tmp_path / "first", count=8)
    # The same production ingress counter logic is reused on a second fresh session.
    from flexiv_inspire_isaac.data_pipeline.episode_manager import EpisodeSession

    inputs = tmp_path / "first" / "inputs"
    session = EpisodeSession(
        tmp_path / "second",
        "synthetic",
        {"frames": inputs / "frames.yaml"},
        inputs / "tool.yaml",
        inputs / "ft.json",
        record_ros_mcap=False,
    )

    def stats(n):
        session.submit_native(
            {
                "topic": "/_deviceio/source_stats/dftp",
                "producer": "dftp",
                "payload": {"sent": n, "pending_sensor": 3},
            }
        )

    stats(1000)
    stats(1004)
    assert session.manifest.native_source_stats["dftp"]["sent"] == 4
    stats(2)
    stats(5)
    assert session.manifest.native_source_stats["dftp"]["sent"] == 9
    assert session.manifest.native_source_stats["dftp"]["pending_sensor"] == 3
    session.abort(reason="offline-test")


def test_actual_generic_and_profile_cli_emit_full_command_supervision(
    tmp_path, monkeypatch
):
    from test_data_policy_profile_export import CountingDataset
    from flexiv_inspire_isaac.data_pipeline import lerobot_cli, profile_cli
    from flexiv_inspire_isaac.data_pipeline.lerobot_v3 import export_rows
    from flexiv_inspire_isaac.data_pipeline.profile_export import export_policy_episodes

    result = run(tmp_path / "capture")
    manifest = result["manifest"]
    generic_config = tmp_path / "generic.yaml"
    generic_config.write_text(
        "schema_version: 1\ntimeline: {source: camera/head/jpeg, fps: 15, resample: false}\naction: {view: sent_command}\nfields: [observation.images.head, observation.images.left_wrist, observation.images.right_wrist, observation.arm_q, observation.hand_state, action]\n"
    )
    CountingDataset.instances.clear()
    monkeypatch.setattr(
        lerobot_cli,
        "export_rows",
        lambda *a, **k: export_rows(*a, **k, dataset_class=CountingDataset),
    )
    generic_root = tmp_path / "generic"
    assert (
        lerobot_cli.main(
            [
                "--manifest",
                manifest,
                "--output-root",
                str(generic_root),
                "--repo-id",
                "local/g",
                "--export-config",
                str(generic_config),
            ]
        )
        == 0
    )
    supervision = json.loads((generic_root / "command_supervision_v2.json").read_text())
    assert len(supervision["rows"]) == 60
    assert len(CountingDataset.instances[0].frames) == 15
    assert all(not row["missing_receipts"] for row in supervision["rows"])
    profile_config = tmp_path / "profile.yaml"
    profile_config.write_text(
        "schema_version: 1\nprofile: joint_proprio_cartesian_v1\ntimeline: {source: camera/head/jpeg, fps: 15, resample: false}\naction: {view: sent_command}\n"
    )
    CountingDataset.instances.clear()
    monkeypatch.setattr(
        profile_cli,
        "export_policy_episodes",
        lambda *a, **k: export_policy_episodes(*a, **k, dataset_class=CountingDataset),
    )
    profile_root = tmp_path / "profile"
    assert (
        profile_cli.main(
            [
                "--manifest",
                manifest,
                "--output-root",
                str(profile_root),
                "--repo-id",
                "local/p",
                "--export-config",
                str(profile_config),
            ]
        )
        == 0
    )
    documents = list((profile_root / "command_supervision").glob("*.json"))
    assert (
        len(documents) == 1 and len(json.loads(documents[0].read_text())["rows"]) == 60
    )
    assert len(CountingDataset.instances[0].frames) == 15


def test_effective_site_rendered_configs_archive_offline(tmp_path):
    from flexiv_inspire_isaac.system_config import (
        load_system_config,
        render_runtime_configs,
    )

    config = load_system_config(ROOT / "config/site.yaml")
    rendered = tmp_path / "rendered"
    render_runtime_configs(config, rendered)
    episode = tmp_path / "episode"
    episode.mkdir()
    ref = archive_episode(episode, config=config.document, rendered_dir=rendered)
    (episode / "manifest.json").write_text(json.dumps({"provenance": ref}))
    # Repository names a deployment calibration absent here; do not invent it.
    with pytest.raises(ValueError, match="required archive asset missing"):
        verify_archive(episode)
    assert not ref["complete"]
    index = json.loads((episode / "provenance/index.json").read_text())
    assert "rendered/control_bridge.yaml" in index["entries"]
    layouts = json.loads(
        (episode / index["entries"]["hand_models_and_layout"]["path"]).read_text()
    )
    assert layouts["left"]["taxel_count"] == 1062
    assert layouts["left"]["hardware_identity_probe"] == "unknown"
    archived_config = episode / index["entries"]["effective_config"]["path"]
    shutil.rmtree(rendered)
    assert "sampling:" in archived_config.read_text()
    # Integrity of complete, explicitly supplied assets is tested independently above.


def test_native_profile_gate_preserves_maintenance_receipts_and_pedal_edges(tmp_path):
    from flexiv_inspire_isaac.data_pipeline.episode_manager import EpisodeSession
    from flexiv_inspire_isaac.data_pipeline.action_contract import records_from_mcap

    run(tmp_path / "inputs_source", count=8)
    inputs = tmp_path / "inputs_source" / "inputs"
    now = [2_000_000_000]
    session = EpisodeSession(
        tmp_path / "episode",
        "synthetic",
        {"frames": inputs / "frames.yaml"},
        inputs / "tool.yaml",
        inputs / "ft.json",
        record_ros_mcap=False,
        deviceio_profile="training",
        record_only_while_pedal_pressed=True,
        clock_ns=lambda: now[0],
    )
    session.pause(reason="test")
    for topic, payload in (
        ("maintenance/events", {"event_type": "home_started"}),
        (
            "control/actuator_evidence",
            receipt("s:teleop:64", "left_hand", "send_failed", now_ns=now[0]),
        ),
        ("maintenance/ft_zero_event", {"event_type": "ft_zero_invalidated"}),
    ):
        document = asdict(envelope(topic, now[0], 1, payload))
        document["timing_valid"] = True
        session.submit_native(document)
    session.set_motion_recording(True)
    session.set_motion_recording(False)
    session.abort(reason="test")
    recorded = records_from_mcap(session.device_path)
    topics = {r["topic"] for r in recorded}
    assert "/control/actuator_evidence" in topics and "/maintenance/events" in topics
    assert "/maintenance/ft_zero_event" in topics
    events = {
        r["payload"].get("event_type")
        for r in recorded
        if r["topic"] == "/episode/events"
    }
    assert {"episode_paused", "pedal_pressed", "pedal_released"} <= events


def test_native_ros_collector_subscribes_to_events_before_mirror_return(
    tmp_path, monkeypatch
):
    import sys
    from flexiv_inspire_isaac.data_pipeline.episode_manager import build_node

    class Node:
        def __init__(self, *args):
            self.subscriptions = {}

        def create_subscription(self, typ, topic, callback, qos):
            self.subscriptions[topic] = callback

    names = (
        "ArmState",
        "BimanualCommand",
        "CameraFrame",
        "CommandTrace",
        "ControlState",
        "EpisodeEvent",
        "HandState",
        "TactileFrame",
    )
    monkeypatch.setitem(
        sys.modules,
        "flexiv_inspire_interfaces.msg",
        SimpleNamespace(**{n: object for n in names}),
    )
    monkeypatch.setitem(
        sys.modules, "geometry_msgs.msg", SimpleNamespace(PoseArray=object)
    )
    monkeypatch.setitem(
        sys.modules, "sensor_msgs.msg", SimpleNamespace(JointState=object)
    )
    monkeypatch.setitem(
        sys.modules, "std_msgs.msg", SimpleNamespace(Bool=object, ByteMultiArray=object)
    )
    monkeypatch.setitem(sys.modules, "rclpy.node", SimpleNamespace(Node=Node))
    monkeypatch.setitem(
        sys.modules, "rclpy.qos", SimpleNamespace(qos_profile_sensor_data=object())
    )
    session = SimpleNamespace()
    node = build_node(session, record_ros_mirror=False)
    assert {
        "/episode/events",
        "/maintenance/events",
        "/teleop/deadman",
    } <= node.subscriptions.keys()


def test_home_actual_status_complete_produces_completion_event():
    path = ROOT / "ros2_ws/src/flexiv_inspire_control/flexiv_inspire_control/node.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "ControlBridge"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_publish_home_status"
    )
    namespace = {"String": SimpleNamespace, "json": json}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    events = []
    bridge = SimpleNamespace(
        _home_status_pub=None,
        _emit_home_event=lambda kind, **kwargs: events.append(kind),
    )
    namespace["_publish_home_status"](bridge, "complete", "", request_id="home1")
    assert events == ["home_completed"]


def test_capture_required_streams_match_optional_tactile_contract():
    from flexiv_inspire_isaac.data_pipeline.episode_manager import EpisodeSession

    session = EpisodeSession.__new__(EpisodeSession)
    session.manifest = SimpleNamespace(streams={})
    required = session._required_stream_errors()
    assert "control/sent_command" in required
    assert "robot/left_hand/state" in required
    assert "robot/left_hand/tactile_raw" not in required
    assert "robot/right_hand/tactile_raw" not in required
