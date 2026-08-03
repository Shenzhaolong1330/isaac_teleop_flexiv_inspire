from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

from isaacteleop.retargeting_engine.tensor_types.indices import (
    ControllerInputIndex,
    HandInputIndex,
    HandJointIndex,
    HeadPoseIndex,
)
from flexiv_inspire_isaac.openxr_errors import is_retryable_openxr_session_error
from flexiv_inspire_isaac.xr_raw_ros_source import (
    _CONTROLLER_TO_MANUS_WRIST,
    _compose_pose,
    _controller_rooted_manus_poses,
    _head_relative_controller_pose,
    _force_quest_browser_navigation,
    _invert_pose,
    _install_cloudxr_client_overrides,
    _maintain_oob_connection,
)


class _TensorGroup:
    is_none = False

    def __init__(self, values):
        self._values = values

    def __getitem__(self, index):
        return self._values[index]


def test_quest_browser_navigation_uses_existing_isaac_page(monkeypatch):
    sent = []
    removed = []

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def send(self, message):
            sent.append(json.loads(message))

        def recv(self, **_kwargs):
            return '{"id":1,"result":{"frameId":"ok"}}'

    class Oob:
        _CDP_LOCAL_PORT = 9223

        @staticmethod
        def _discover_devtools_socket():
            return "chrome_devtools_remote"

        @staticmethod
        def _adb_forward_cdp(_socket, _port):
            return None

        @staticmethod
        def _cdp_list_tabs(_port):
            return [
                {
                    "type": "page",
                    "url": "chrome://panel-app-nav/ntp",
                    "webSocketDebuggerUrl": "ws://new-tab",
                },
                {
                    "type": "page",
                    "title": "NVIDIA Isaac Teleop Web Client",
                    "url": "https://nvidia.github.io/IsaacTeleop/client/#/sim",
                    "webSocketDebuggerUrl": "ws://isaac-tab",
                },
            ]

        @staticmethod
        def _adb_forward_remove(port):
            removed.append(port)

    import json
    from websockets.sync import client
    import flexiv_inspire_isaac.xr_raw_ros_source as source

    monkeypatch.setattr(client, "connect", lambda url, **_kwargs: (sent.append(url) or Socket()))
    monkeypatch.setattr(source.time, "sleep", lambda _seconds: None)

    _force_quest_browser_navigation(Oob, "https://localhost:8080?oobEnable=1")

    assert sent[0] == "ws://isaac-tab"
    assert sent[1]["method"] == "Page.navigate"
    assert sent[1]["params"]["url"].endswith("oobEnable=1")
    assert removed == [9223]


def test_quest_browser_navigation_creates_page_on_cold_start(monkeypatch):
    sent = []
    removed = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        @staticmethod
        def read():
            return b'{"webSocketDebuggerUrl":"ws://browser"}'

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def send(self, message):
            sent.append(json.loads(message))

        @staticmethod
        def recv(**_kwargs):
            return '{"id":1,"result":{"targetId":"created-page"}}'

    class Oob:
        _CDP_LOCAL_PORT = 9223

        @staticmethod
        def _discover_devtools_socket():
            return "chrome_devtools_remote"

        @staticmethod
        def _adb_forward_cdp(_socket, _port):
            return None

        @staticmethod
        def _cdp_list_tabs(_port):
            return []

        @staticmethod
        def _adb_forward_remove(port):
            removed.append(port)

    import json
    from websockets.sync import client
    import flexiv_inspire_isaac.xr_raw_ros_source as source

    monkeypatch.setattr(
        source.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    monkeypatch.setattr(
        client,
        "connect",
        lambda url, **_kwargs: (sent.append(url) or Socket()),
    )
    monkeypatch.setattr(source.time, "sleep", lambda _seconds: None)

    _force_quest_browser_navigation(Oob, "https://localhost:8080?oobEnable=1")

    assert sent[0] == "ws://browser"
    assert sent[1]["method"] == "Target.createTarget"
    assert sent[1]["params"]["url"].endswith("oobEnable=1")
    assert removed == [9223]


def test_cloudxr_url_overrides_win_over_upstream_defaults(monkeypatch):
    from isaacteleop.cloudxr import oob_teleop_adb, oob_teleop_env

    monkeypatch.setattr(
        oob_teleop_env,
        "client_ui_fields_from_env",
        lambda: {"codec": "av1", "panelHiddenAtStart": False},
    )
    monkeypatch.setattr(
        oob_teleop_adb,
        "client_ui_fields_from_env",
        oob_teleop_env.client_ui_fields_from_env,
    )
    monkeypatch.setattr(
        oob_teleop_adb,
        "build_headset_bookmark_url",
        oob_teleop_env.build_headset_bookmark_url,
    )

    _install_cloudxr_client_overrides(
        {
            "perEyeWidth": 1792,
            "perEyeHeight": 1536,
            "deviceFrameRate": 72,
            "maxStreamingBitrateMbps": 80,
            "codec": "h264",
        }
    )

    expected = {
        "codec": "h264",
        "panelHiddenAtStart": False,
        "perEyeWidth": 1792,
        "perEyeHeight": 1536,
        "deviceFrameRate": 72,
        "maxStreamingBitrateMbps": 80,
    }
    assert oob_teleop_env.client_ui_fields_from_env() == expected
    assert oob_teleop_adb.client_ui_fields_from_env() == expected
    url = oob_teleop_adb.build_headset_bookmark_url(
        web_client_base="https://localhost:8080",
        stream_config={"serverIP": "127.0.0.1", "port": 48322},
    )
    assert "perEyeWidth=1792" in url
    assert "perEyeHeight=1536" in url
    assert "deviceFrameRate=72" in url
    assert "maxStreamingBitrateMbps=80" in url
    assert "codec=h264" in url


def test_oob_watchdog_reconnects_until_openxr_session_is_live():
    calls = 0
    live = threading.Event()

    async def connect(**_kwargs):
        nonlocal calls
        calls += 1
        return asyncio.create_task(asyncio.Event().wait())

    async def scenario():
        initial = await connect()
        watchdog = asyncio.create_task(
            _maintain_oob_connection(
                connect,
                {},
                live,
                initial,
                reconnect_grace_s=0.01,
            )
        )
        await asyncio.sleep(0.035)
        assert calls >= 2
        live.set()
        settled_calls = calls
        await asyncio.sleep(0.03)
        assert calls == settled_calls
        watchdog.cancel()
        with pytest.raises(asyncio.CancelledError):
            await watchdog

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "message",
    (
        "Failed to get OpenXR system: -35",
        "Failed to create OpenXR instance: -51",
    ),
)
def test_transient_openxr_startup_errors_are_retryable(message: str) -> None:
    assert is_retryable_openxr_session_error(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    (
        "Failed to create OpenXR instance: -9",
        "CloudXR runtime process exited unexpectedly",
        "invalid hand transport",
    ),
)
def test_non_transient_openxr_errors_remain_fatal(message: str) -> None:
    assert not is_retryable_openxr_session_error(RuntimeError(message))


def test_compose_pose_applies_parent_rotation_and_translation():
    half_sqrt = np.sqrt(0.5)
    position, orientation = _compose_pose(
        (1.0, 2.0, 3.0),
        (0.0, 0.0, half_sqrt, half_sqrt),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    np.testing.assert_allclose(position, (1.0, 3.0, 3.0), atol=1e-7)
    np.testing.assert_allclose(
        orientation,
        (0.0, 0.0, half_sqrt, half_sqrt),
        atol=1e-7,
    )


def test_compose_pose_composes_xyzw_orientations():
    half_sqrt = np.sqrt(0.5)
    _, orientation = _compose_pose(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, half_sqrt, half_sqrt),
        (0.0, 0.0, 0.0),
        (0.0, half_sqrt, 0.0, half_sqrt),
    )

    np.testing.assert_allclose(orientation, (-0.5, 0.5, 0.5, 0.5), atol=1e-7)


def test_invert_pose_cancels_source_wrist_transform():
    half_sqrt = np.sqrt(0.5)
    source = (
        np.asarray((1.0, 2.0, 3.0)),
        np.asarray((0.0, 0.0, half_sqrt, half_sqrt)),
    )
    identity_position, identity_orientation = _compose_pose(
        *source, *_invert_pose(*source)
    )

    np.testing.assert_allclose(identity_position, (0.0, 0.0, 0.0), atol=1e-7)
    np.testing.assert_allclose(
        identity_orientation, (0.0, 0.0, 0.0, 1.0), atol=1e-7
    )


def test_quest_controller_pose_is_head_relative_like_mature_stack():
    half_sqrt = np.sqrt(0.5)
    head_pose = (
        np.asarray((10.0, 20.0, 30.0)),
        np.asarray((0.0, 0.0, half_sqrt, half_sqrt)),
    )
    expected_relative = (
        np.asarray((1.0, 2.0, 3.0)),
        np.asarray((0.0, half_sqrt, 0.0, half_sqrt)),
    )
    controller_pose = _compose_pose(*head_pose, *expected_relative)
    head = _TensorGroup(
        {
            HeadPoseIndex.POSITION: head_pose[0],
            HeadPoseIndex.ORIENTATION: head_pose[1],
            HeadPoseIndex.IS_VALID: True,
        }
    )
    controller = _TensorGroup(
        {
            ControllerInputIndex.GRIP_POSITION: controller_pose[0],
            ControllerInputIndex.GRIP_ORIENTATION: controller_pose[1],
            ControllerInputIndex.GRIP_IS_VALID: True,
        }
    )

    relative = _head_relative_controller_pose(head, controller)

    np.testing.assert_allclose(relative[0], expected_relative[0], atol=1e-7)
    np.testing.assert_allclose(relative[1], expected_relative[1], atol=1e-7)


def test_manus_fingers_are_always_rooted_on_quest_wrist_pose():
    count = int(HandJointIndex.LITTLE_TIP) + 1
    positions = np.zeros((count, 3), dtype=np.float64)
    orientations = np.zeros((count, 4), dtype=np.float64)
    orientations[:, 3] = 1.0
    valid = np.ones(count, dtype=bool)
    wrist = int(HandJointIndex.WRIST)
    tip = int(HandJointIndex.LITTLE_TIP)
    # Simulate a MANUS/XDev source frame unrelated to the Quest controller.
    positions[wrist] = (10.0, 0.0, 0.0)
    positions[tip] = (11.0, 0.0, 0.0)
    hand = _TensorGroup(
        {
            HandInputIndex.JOINT_POSITIONS: positions,
            HandInputIndex.JOINT_ORIENTATIONS: orientations,
            HandInputIndex.JOINT_VALID: valid,
        }
    )
    controller = _TensorGroup(
        {
            ControllerInputIndex.GRIP_POSITION: np.asarray((1.0, 2.0, 3.0)),
            ControllerInputIndex.GRIP_ORIENTATION: np.asarray(
                (0.0, 0.0, 0.0, 1.0)
            ),
            ControllerInputIndex.GRIP_IS_VALID: True,
        }
    )

    rooted = _controller_rooted_manus_poses(hand, controller, "left")
    quest_wrist = _compose_pose(
        controller[ControllerInputIndex.GRIP_POSITION],
        controller[ControllerInputIndex.GRIP_ORIENTATION],
        *_CONTROLLER_TO_MANUS_WRIST["left"],
    )
    rooted_wrist = rooted[0]
    rooted_tip = rooted[tip - wrist]

    np.testing.assert_allclose(rooted_wrist[0], quest_wrist[0], atol=1e-7)
    np.testing.assert_allclose(rooted_wrist[1], quest_wrist[1], atol=1e-7)
    expected_tip = _compose_pose(
        *quest_wrist,
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    np.testing.assert_allclose(rooted_tip[0], expected_tip[0], atol=1e-7)
    np.testing.assert_allclose(rooted_tip[1], expected_tip[1], atol=1e-7)
