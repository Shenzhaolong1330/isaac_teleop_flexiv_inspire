from __future__ import annotations

import threading
import time

import numpy as np

from flexiv_inspire_isaac.rerun_viz.runtime import (
    LatestOnlyDispatcher,
    RerunVisualizer,
    ROT6D_IDENTITY,
    command_action_vector,
    tactile_atlas,
)


def _surfaces() -> list[dict]:
    layout = (
        *(
            (f"{finger}_{suffix}", rows, columns)
            for finger in ("little", "ring", "middle", "index")
            for suffix, rows, columns in (("end", 3, 3), ("tip", 12, 8), ("pad", 10, 8))
        ),
        ("thumb_end", 3, 3),
        ("thumb_tip", 12, 8),
        ("thumb_middle", 3, 3),
        ("thumb_pad", 12, 8),
        ("palm", 8, 14),
    )
    result = []
    value = 0
    for name, rows, columns in layout:
        taxels = np.arange(value, value + rows * columns, dtype=np.uint16)
        result.append(
            {
                "name": name,
                "rows": rows,
                "columns": columns,
                "taxels": taxels.tolist(),
            }
        )
        value += rows * columns
    result[-1]["taxels"][-1] = 65535
    return result


def test_tactile_atlas_preserves_raw_uint16() -> None:
    atlas = tactile_atlas(_surfaces())
    assert atlas.shape == (48, 48)
    assert atlas.dtype == np.uint16
    assert int(atlas.max()) == 65535


def test_command_action_is_exact_30d_rotation6d_layout() -> None:
    point = {
        "left_delta_xyz": [1.0, 2.0, 3.0],
        "left_delta_rotation6d": ROT6D_IDENTITY,
        "right_delta_xyz": [4.0, 5.0, 6.0],
        "right_delta_rotation6d": ROT6D_IDENTITY,
        "left_hand_targets": np.arange(6),
        "right_hand_targets": np.arange(6) + 10,
    }
    action = command_action_vector({"representation": 1, "trajectory": [point]})
    assert action.shape == (30,)
    np.testing.assert_array_equal(action[3:9], ROT6D_IDENTITY)
    np.testing.assert_array_equal(action[12:18], ROT6D_IDENTITY)


def test_latest_only_dispatcher_replaces_pending_same_stream() -> None:
    entered = threading.Event()
    release = threading.Event()
    output: list[int] = []

    def consume(value: int) -> None:
        output.append(value)
        if value == 1:
            entered.set()
            assert release.wait(2.0)

    dispatcher = LatestOnlyDispatcher()
    assert dispatcher.submit("arm:left", consume, 1)
    assert entered.wait(2.0)
    assert dispatcher.submit("arm:left", consume, 2)
    assert dispatcher.submit("arm:left", consume, 3)
    release.set()
    dispatcher.close()
    assert output == [1, 3]
    assert dispatcher.stats.dropped == 1
    assert dispatcher.stats.failed == 0


def test_dispatcher_contains_visualization_exception() -> None:
    dispatcher = LatestOnlyDispatcher()
    dispatcher.submit("bad", lambda _: (_ for _ in ()).throw(ValueError("bad frame")), None)
    dispatcher.close()
    assert dispatcher.stats.failed == 1
    assert "bad frame" in dispatcher.last_error


def _bare_visualizer():
    visualizer = object.__new__(RerunVisualizer)
    visualizer._set_time = lambda *args, **kwargs: None
    visualizer._scalar = lambda *args, **kwargs: None
    messages = []
    visualizer._text_if_changed = (
        lambda path, value, level="INFO": messages.append((path, value, level))
    )
    return visualizer, messages


def test_control_hold_and_rejection_text_are_cleared_on_recovery() -> None:
    visualizer, messages = _bare_visualizer()
    base = {
        "stamp_ns": 1,
        "state": 1,
        "generation": 1,
        "state_name": "HOLD_LATCHED",
        "active_source": "",
        "hold_reason": "heartbeat_timeout",
    }
    visualizer.log_control_state(base)
    visualizer.log_control_state({**base, "stamp_ns": 2, "state_name": "READY", "hold_reason": ""})
    assert ("control/state/hold_reason", "", "WARN") in messages

    visualizer.log_trace({"stamp_ns": 3, "trace_sequence": 1, "rejection_reason": "bad"})
    visualizer.log_trace({"stamp_ns": 4, "trace_sequence": 2, "rejection_reason": ""})
    assert ("control/trace/rejection_reason", "", "WARN") in messages


def test_visualizer_disconnects_even_when_flush_fails() -> None:
    class Stream:
        disconnected = False
        def flush(self, **_kwargs):
            raise RuntimeError("flush failed")
        def disconnect(self):
            self.disconnected = True

    visualizer = object.__new__(RerunVisualizer)
    visualizer._closed = False
    visualizer.stream = Stream()
    import pytest
    with pytest.raises(RuntimeError, match="flush failed"):
        visualizer.close()
    assert visualizer.stream.disconnected is True
    assert visualizer._closed is True
