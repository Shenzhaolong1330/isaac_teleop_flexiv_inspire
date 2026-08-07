"""Publish Quest controllers and raw OpenXR/MANUS hands without robot retargeting.

This is the source node for the Flexiv + Inspire stack. It deliberately emits
the 25 OpenXR joints per hand and does not instantiate a Sharpa (or any other
robot-hand) retargeter.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import urllib.request
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray
from tf2_ros import TransformBroadcaster

from isaacteleop.cloudxr import CloudXRLauncher
from isaacteleop.retargeting_engine.deviceio_source_nodes import (
    ControllersSource,
    HeadSource,
    HandsSource,
)
from isaacteleop.retargeting_engine.interface import OutputCombiner
from isaacteleop.retargeting_engine.tensor_types.indices import (
    ControllerInputIndex,
    HandInputIndex,
    HandJointIndex,
    HeadPoseIndex,
)
from isaacteleop.teleop_session_manager import (
    SessionMode,
    TeleopSession,
    TeleopSessionConfig,
)
from .openxr_errors import is_retryable_openxr_session_error
from .quest_input_contract import controller_payload, encode_controller_payload


def _force_quest_browser_navigation(oob_teleop_adb, url: str) -> None:
    """Navigate an existing Quest Browser page when VIEW only raises its panel.

    Recent Quest Browser builds may report a successful ``am start`` while
    merely bringing the old browser task to the foreground.  The Intent keeps
    the requested URL in Activity state, but Chromium never navigates, so the
    upstream CDP helper cannot find the OOB tab and CloudXR has no HMD.  A CDP
    ``Page.navigate`` on the existing Isaac Teleop page is deterministic.  A
    cold Quest Browser can expose only its browser-level DevTools endpoint and
    no page at all; in that case create the page through CDP instead.
    """

    from websockets.sync.client import connect as ws_connect

    socket_name = oob_teleop_adb._discover_devtools_socket()
    if not socket_name:
        raise RuntimeError("Quest Browser DevTools socket is unavailable")
    port = int(oob_teleop_adb._CDP_LOCAL_PORT)
    oob_teleop_adb._adb_forward_cdp(socket_name, port)
    try:
        tabs = oob_teleop_adb._cdp_list_tabs(port)
        pages = [
            tab
            for tab in tabs
            if tab.get("type") == "page" and tab.get("webSocketDebuggerUrl")
        ]
        if not pages:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/version", timeout=3
            ) as response:
                browser = json.loads(response.read())
            websocket_url = browser.get("webSocketDebuggerUrl")
            if not websocket_url:
                raise RuntimeError(
                    "Quest Browser exposes neither a page nor a browser "
                    "DevTools websocket"
                )
            with ws_connect(str(websocket_url), open_timeout=3) as ws:
                ws.send(
                    json.dumps(
                        {
                            "id": 1,
                            "method": "Target.createTarget",
                            "params": {"url": url},
                        }
                    )
                )
                result = json.loads(ws.recv(timeout=5))
                target_id = result.get("result", {}).get("targetId")
                if not result.get("error") and target_id:
                    ws.send(
                        json.dumps(
                            {
                                "id": 2,
                                "method": "Target.activateTarget",
                                "params": {"targetId": target_id},
                            }
                        )
                    )
                    activated = json.loads(ws.recv(timeout=3))
                    if activated.get("error"):
                        raise RuntimeError(
                            "Quest Browser target activation failed: "
                            f"{activated['error']}"
                        )
            if result.get("error") or not target_id:
                raise RuntimeError(
                    "Quest Browser page creation failed: "
                    f"{result.get('error', result)}"
                )
            return
        page = next(
            (
                tab
                for tab in pages
                if "IsaacTeleop" in str(tab.get("url", ""))
                or "NVIDIA Isaac Teleop" in str(tab.get("title", ""))
            ),
            next(
                (
                    tab
                    for tab in pages
                    if not str(tab.get("url", "")).startswith("chrome://")
                ),
                pages[0],
            ),
        )
        with ws_connect(str(page["webSocketDebuggerUrl"]), open_timeout=3) as ws:
            # Quest Browser can report immersive-vr unsupported while a tab
            # is backgrounded. Activate it before navigation so the client's
            # first capability probe sees the real headset runtime instead of
            # installing the desktop-only IWER fallback.
            ws.send(json.dumps({"id": 1, "method": "Page.bringToFront"}))
            foreground = json.loads(ws.recv(timeout=3))
            if foreground.get("error"):
                raise RuntimeError(
                    "Quest Browser tab activation failed: "
                    f"{foreground['error']}"
                )
            # A previous failed desktop-IWER fallback stores this flag in the
            # tab's session storage. Clear it before the real Quest page is
            # reloaded so capability checks cannot inherit the failed mode.
            ws.send(
                json.dumps(
                    {
                        "id": 2,
                        "method": "Runtime.evaluate",
                        "params": {
                            "expression": (
                                "sessionStorage.removeItem('iwerWasLoaded')"
                            )
                        },
                    }
                )
            )
            cleared = json.loads(ws.recv(timeout=3))
            if cleared.get("error"):
                logging.getLogger("flexiv-inspire-oob-watchdog").debug(
                    "Could not clear stale Quest XR session flag: %s",
                    cleared["error"],
                )
            ws.send(
                json.dumps(
                    {
                        "id": 3,
                        "method": "Page.navigate",
                        "params": {"url": url},
                    }
                )
            )
            response = json.loads(ws.recv(timeout=3))
        if response.get("error"):
            raise RuntimeError(f"Quest Browser navigation failed: {response['error']}")
    finally:
        oob_teleop_adb._adb_forward_remove(port)


async def _cancel_background_task(task) -> None:
    if task is None:
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


async def _maintain_oob_connection(
    connect,
    connect_kwargs: dict[str, object],
    session_live: threading.Event,
    initial_monitor,
    *,
    reconnect_grace_s: float,
):
    """Repeat Quest OOB connection until OpenXR really has a live HMD."""

    log = logging.getLogger("flexiv-inspire-oob-watchdog")
    monitor = initial_monitor
    missing_since = asyncio.get_running_loop().time()
    try:
        while True:
            await asyncio.sleep(min(1.0, reconnect_grace_s))
            now = asyncio.get_running_loop().time()
            if session_live.is_set():
                missing_since = now
                continue
            if now - missing_since < reconnect_grace_s:
                continue

            await _cancel_background_task(monitor)
            monitor = None
            try:
                from isaacteleop.cloudxr.oob_teleop_env import oob_progress

                oob_progress(
                    "setup-oob",
                    "OpenXR still has no headset session — reopening the "
                    "Quest page and reconnecting ...",
                )
                monitor = await connect(**connect_kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # USB authorization, browser wake-up and CDP availability can
                # all recover. Keep retrying without terminating robot input.
                log.warning("Quest OOB reconnect failed; will retry: %s", exc)
            missing_since = asyncio.get_running_loop().time()
    finally:
        await _cancel_background_task(monitor)


def _install_cloudxr_client_overrides(
    fields: dict[str, object],
    *,
    session_live: threading.Event | None = None,
    reconnect_grace_s: float = 15.0,
) -> None:
    """Inject supported WebXR URL fields into Isaac Teleop OOB startup.

    Isaac Teleop 1.3 accepts these URL parameters in its WebXR client, but its
    Python OOB helper currently exposes only codec and panel visibility as
    environment overrides.  Keep the upstream package untouched and extend
    its field provider in this process before the WSS/OOB modules start.  URL
    values are deliberately used because the client applies them after
    browser localStorage, so an old 150 Mbps setting cannot silently return.
    """

    from isaacteleop.cloudxr import oob_teleop_env

    current = oob_teleop_env.client_ui_fields_from_env
    original = getattr(current, "_flexiv_original", current)

    def configured_fields() -> dict:
        result = dict(original())
        result.update(fields)
        return result

    configured_fields._flexiv_original = original  # type: ignore[attr-defined]
    oob_teleop_env.client_ui_fields_from_env = configured_fields

    # Isaac Teleop's URL builder currently forwards only a subset of the
    # WebXR client's supported parameters. Append the remaining, documented
    # form-backed query fields without changing the vendored/installed SDK.
    current_url_builder = oob_teleop_env.build_headset_bookmark_url
    original_url_builder = getattr(
        current_url_builder, "_flexiv_original", current_url_builder
    )

    def configured_url(**kwargs) -> str:
        url = original_url_builder(**kwargs)
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        for key, value in fields.items():
            if isinstance(value, bool):
                query[key] = str(value).lower()
            else:
                query[key] = str(value)
        return urlunsplit(
            (
                parts.scheme,
                parts.netloc,
                parts.path,
                urlencode(query),
                parts.fragment,
            )
        )

    configured_url._flexiv_original = original_url_builder  # type: ignore[attr-defined]
    oob_teleop_env.build_headset_bookmark_url = configured_url

    # oob_teleop_adb imports the provider into its own module namespace.
    # Import it only after replacing the source provider, then update the
    # reference explicitly in case another caller imported it earlier.
    from isaacteleop.cloudxr import oob_teleop_adb

    oob_teleop_adb.client_ui_fields_from_env = configured_fields
    oob_teleop_adb.build_headset_bookmark_url = configured_url

    current_bookmark = oob_teleop_adb.run_adb_headset_bookmark
    original_bookmark = getattr(
        current_bookmark, "_flexiv_original", current_bookmark
    )

    def reliable_bookmark(**kwargs):
        result = original_bookmark(**kwargs)
        if result[0] != 0:
            return result
        url = oob_teleop_adb.build_teleop_url(**kwargs)
        last_error = None
        for attempt in range(1, 4):
            try:
                _force_quest_browser_navigation(oob_teleop_adb, url)
                logging.getLogger("flexiv-inspire-oob-watchdog").info(
                    "Quest Browser navigated through CDP after Android VIEW intent"
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(1.0)
        else:
            # Keep upstream's normal tab discovery as a fallback for browser
            # versions where VIEW already navigates correctly.
            logging.getLogger("flexiv-inspire-oob-watchdog").warning(
                "Quest Browser CDP navigation failed after 3 attempts: %s",
                last_error,
            )
        return result

    reliable_bookmark._flexiv_original = original_bookmark  # type: ignore[attr-defined]
    oob_teleop_adb.run_adb_headset_bookmark = reliable_bookmark

    # The upstream helper closes the teleop tab before every reconnect. That
    # is useful once at process startup, but on Quest it turns a recoverable
    # headset-wake delay into an expensive close/open loop. Clean up once,
    # then reload and reuse the same foreground tab.
    current_cleanup = oob_teleop_adb._close_stale_teleop_tabs
    original_cleanup = getattr(
        current_cleanup, "_flexiv_original", current_cleanup
    )
    cleanup_complete = False

    def cleanup_stale_tabs_once() -> int:
        nonlocal cleanup_complete
        if cleanup_complete:
            return 0
        cleanup_complete = True
        return original_cleanup()

    cleanup_stale_tabs_once._flexiv_original = original_cleanup  # type: ignore[attr-defined]
    oob_teleop_adb._close_stale_teleop_tabs = cleanup_stale_tabs_once

    # Upstream can spend 30 seconds waiting for the button and another 30
    # seconds waiting for connection state. A healthy USB-local Quest is ready
    # in a few seconds; fail this attempt promptly and let the watchdog reload
    # the same page instead of making the operator wait a full minute.
    current_click = oob_teleop_adb._cdp_session_click_connect
    original_click = getattr(current_click, "_flexiv_original", current_click)

    async def fast_click_connect(ws_url: str) -> None:
        try:
            await asyncio.wait_for(original_click(ws_url), timeout=12.0)
        except TimeoutError as exc:
            raise RuntimeError(
                "Quest did not expose a usable native WebXR session within "
                "12 seconds. Wake the headset and wear it, or cover its "
                "proximity sensor while it is mounted facing the operator."
            ) from exc

    fast_click_connect._flexiv_original = original_click  # type: ignore[attr-defined]
    oob_teleop_adb._cdp_session_click_connect = fast_click_connect

    if session_live is not None:
        current_connect = oob_teleop_adb.run_oob_connect
        original_connect = getattr(
            current_connect, "_flexiv_original", current_connect
        )

        async def resilient_connect(**kwargs):
            initial_failed = False
            try:
                initial_monitor = await original_connect(**kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Do not let a cold browser, temporarily unauthorized ADB, or
                # a missing DevTools socket disable recovery for the rest of
                # the record run.
                logging.getLogger("flexiv-inspire-oob-watchdog").warning(
                    "initial Quest OOB connection failed; watchdog will retry: %s",
                    exc,
                )
                initial_monitor = None
                initial_failed = True
            # A first CONNECT failure means there is no useful XR session to
            # preserve. Retry promptly instead of applying the normal
            # mid-stream grace period: Quest Browser can occasionally leave
            # the first freshly opened page in a failed signalling state even
            # though the USB reverse ports and WSS proxy are already ready.
            # Once a session has existed, retain the longer grace so a short
            # headset/browser scheduling pause does not churn the connection.
            retry_grace_s = (
                min(2.0, reconnect_grace_s)
                if initial_failed
                else reconnect_grace_s
            )
            return asyncio.create_task(
                _maintain_oob_connection(
                    original_connect,
                    dict(kwargs),
                    session_live,
                    initial_monitor,
                    reconnect_grace_s=retry_grace_s,
                ),
                name="flexiv-quest-oob-reconnect-watchdog",
            )

        resilient_connect._flexiv_original = original_connect  # type: ignore[attr-defined]
        oob_teleop_adb.run_oob_connect = resilient_connect
        # wss imports the function into its own module namespace. Update that
        # reference as well in case the module was imported before this hook.
        from isaacteleop.cloudxr import wss

        wss.run_oob_connect = resilient_connect


# MANUS supplies finger articulation; this site deliberately takes wrist 6D
# pose from the Quest Touch Plus controllers.  These controller-to-wrist
# transforms match the physical convention used by the MANUS plugin.
_CONTROLLER_TO_MANUS_WRIST = {
    "left": (
        np.asarray((-0.1, 0.02, -0.02), dtype=np.float64),
        np.asarray((-0.70710678, -0.5, 0.0, 0.5), dtype=np.float64),
    ),
    "right": (
        np.asarray((0.1, 0.02, -0.02), dtype=np.float64),
        np.asarray((-0.70710678, 0.5, 0.0, 0.5), dtype=np.float64),
    ),
}


def _normalized_quaternion(value) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be positive")
    return quaternion / norm


def _compose_pose(
    parent_position,
    parent_orientation,
    child_position,
    child_orientation,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose two xyz + xyzw poses without an optional geometry dependency."""

    parent_xyz = np.asarray(parent_position, dtype=np.float64)
    child_xyz = np.asarray(child_position, dtype=np.float64)
    if (
        parent_xyz.shape != (3,)
        or child_xyz.shape != (3,)
        or not np.all(np.isfinite(parent_xyz))
        or not np.all(np.isfinite(child_xyz))
    ):
        raise ValueError("pose positions must contain three finite values")
    parent_xyzw = _normalized_quaternion(parent_orientation)
    child_xyzw = _normalized_quaternion(child_orientation)
    vector = parent_xyzw[:3]
    rotated = child_xyz + 2.0 * np.cross(
        vector,
        np.cross(vector, child_xyz) + parent_xyzw[3] * child_xyz,
    )
    left_xyz, left_w = parent_xyzw[:3], parent_xyzw[3]
    right_xyz, right_w = child_xyzw[:3], child_xyzw[3]
    orientation = np.concatenate(
        (
            left_w * right_xyz
            + right_w * left_xyz
            + np.cross(left_xyz, right_xyz),
            np.asarray((left_w * right_w - np.dot(left_xyz, right_xyz),)),
        )
    )
    return parent_xyz + rotated, _normalized_quaternion(orientation)


def _invert_pose(position, orientation) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(position, dtype=np.float64)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError("pose position must contain three finite values")
    xyzw = _normalized_quaternion(orientation)
    inverse_xyzw = np.concatenate((-xyzw[:3], xyzw[3:]))
    inverse_xyz, _ = _compose_pose(
        (0.0, 0.0, 0.0),
        inverse_xyzw,
        -xyz,
        (0.0, 0.0, 0.0, 1.0),
    )
    return inverse_xyz, inverse_xyzw


def _controller_rooted_manus_poses(hand, controller, side: str):
    """Root MANUS finger articulation on the Quest controller wrist pose.

    The incoming MANUS skeleton may be wrist-local, controller-rooted, or
    XDev-rooted depending on runtime capabilities.  First remove its incoming
    wrist transform, preserving only joint poses relative to the MANUS wrist;
    then apply the Quest controller root.  This makes the ownership invariant:
    MANUS controls finger articulation, Quest controls wrist translation and
    rotation.  ``None`` means no glove skeleton exists; an empty tuple means
    the Quest controller is not currently valid.
    """

    wrist = int(HandJointIndex.WRIST)
    if not _valid(hand, HandInputIndex.JOINT_VALID, wrist):
        return None
    wrist_position = np.asarray(
        hand[HandInputIndex.JOINT_POSITIONS][wrist], dtype=np.float64
    )
    wrist_orientation = np.asarray(
        hand[HandInputIndex.JOINT_ORIENTATIONS][wrist], dtype=np.float64
    )
    if not _valid(controller, ControllerInputIndex.GRIP_IS_VALID):
        return ()
    source_wrist_inverse = _invert_pose(wrist_position, wrist_orientation)
    controller_root = _compose_pose(
        controller[ControllerInputIndex.GRIP_POSITION],
        controller[ControllerInputIndex.GRIP_ORIENTATION],
        *_CONTROLLER_TO_MANUS_WRIST[side],
    )
    rooted = []
    for joint in range(HandJointIndex.WRIST, HandJointIndex.LITTLE_TIP + 1):
        if not _valid(hand, HandInputIndex.JOINT_VALID, joint):
            rooted.append(None)
            continue
        wrist_relative = _compose_pose(
            *source_wrist_inverse,
            hand[HandInputIndex.JOINT_POSITIONS][joint],
            hand[HandInputIndex.JOINT_ORIENTATIONS][joint],
        )
        rooted.append(
            _compose_pose(
                *controller_root,
                *wrist_relative,
            )
        )
    return tuple(rooted)


def _head_relative_controller_pose(head, controller):
    """Return ``head_T_controller`` to match the proven Quest teleop stack.

    The mature local project computes ``inverse(world_T_head) @
    world_T_controller`` before its XYZ/RPY channel mapping.  Raw OpenXR stage
    poses do not preserve that convention when the operator turns or recenters
    the headset, which makes the robot axes appear rotated.  Keep the same
    relative-pose boundary here and leave MANUS responsible only for fingers.
    """

    if not _valid(head, HeadPoseIndex.IS_VALID) or not _valid(
        controller, ControllerInputIndex.GRIP_IS_VALID
    ):
        return None
    return _compose_pose(
        *_invert_pose(
            head[HeadPoseIndex.POSITION],
            head[HeadPoseIndex.ORIENTATION],
        ),
        controller[ControllerInputIndex.GRIP_POSITION],
        controller[ControllerInputIndex.GRIP_ORIENTATION],
    )


def _pose(position, orientation=(0.0, 0.0, 0.0, 1.0)) -> Pose:
    result = Pose()
    result.position.x, result.position.y, result.position.z = (
        float(value) for value in position
    )
    (
        result.orientation.x,
        result.orientation.y,
        result.orientation.z,
        result.orientation.w,
    ) = (float(value) for value in orientation)
    return result


def _valid(group, index: int, nested: int | None = None) -> bool:
    if group.is_none:
        return False
    value = group[index]
    if nested is not None:
        value = value[nested]
    return bool(value)


def _controller_value(group, index: int, default):
    if group.is_none:
        return default
    value = group[index]
    array = np.asarray(value)
    if array.ndim:
        return [float(item) for item in array]
    return float(value)


def _controller_click(group, index: int) -> bool:
    """Normalize OpenXR's scalar click action to a transport boolean."""
    value = float(_controller_value(group, index, 0.0))
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("controller click value must be finite and in [0,1]")
    return value >= 0.5


class XrRawRosSource(Node):
    """Isaac DeviceIO to ROS adapter with no robot-hand model dependency."""

    def __init__(self) -> None:
        super().__init__("xr_raw_ros_source")
        defaults = {
            "rate_hz": 60.0,
            "world_frame": "world",
            "left_wrist_frame": "left_wrist",
            "right_wrist_frame": "right_wrist",
            "cloudxr_install_dir": "~/.cloudxr",
            "cloudxr_env_config": "",
            "cloudxr_accept_eula": False,
            "cloudxr_setup_oob": False,
            "cloudxr_usb_local": False,
            "cloudxr_client_per_eye_width": 1792,
            "cloudxr_client_per_eye_height": 1536,
            "cloudxr_client_frame_rate": 72,
            "cloudxr_client_max_bitrate_mbps": 80,
            "cloudxr_client_codec": "h264",
            "cloudxr_client_enable_tex_sub_image_2d": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        rate_hz = float(self.get_parameter("rate_hz").value)
        if not np.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("rate_hz must be finite and > 0")
        if bool(self.get_parameter("cloudxr_usb_local").value) and not bool(
            self.get_parameter("cloudxr_setup_oob").value
        ):
            raise ValueError("cloudxr_usb_local requires cloudxr_setup_oob")
        per_eye_width = int(
            self.get_parameter("cloudxr_client_per_eye_width").value
        )
        per_eye_height = int(
            self.get_parameter("cloudxr_client_per_eye_height").value
        )
        frame_rate = int(self.get_parameter("cloudxr_client_frame_rate").value)
        max_bitrate_mbps = int(
            self.get_parameter("cloudxr_client_max_bitrate_mbps").value
        )
        codec = str(self.get_parameter("cloudxr_client_codec").value).strip()
        if per_eye_width < 128 or per_eye_width % 16:
            raise ValueError(
                "cloudxr_client_per_eye_width must be >=128 and divisible by 16"
            )
        if per_eye_height < 128 or per_eye_height % 64:
            raise ValueError(
                "cloudxr_client_per_eye_height must be >=128 and divisible by 64"
            )
        if frame_rate not in {72, 90, 120}:
            raise ValueError("cloudxr_client_frame_rate must be 72, 90, or 120")
        if max_bitrate_mbps not in {80, 100, 120, 150, 180, 200}:
            raise ValueError("unsupported CloudXR client bitrate")
        if codec not in {"h264", "h265", "av1"}:
            raise ValueError("cloudxr_client_codec must be h264, h265, or av1")
        self._cloudxr_client_fields = {
            "perEyeWidth": per_eye_width,
            "perEyeHeight": per_eye_height,
            "deviceFrameRate": frame_rate,
            "maxStreamingBitrateMbps": max_bitrate_mbps,
            "codec": codec,
            "enableTexSubImage2D": str(
                bool(
                    self.get_parameter(
                        "cloudxr_client_enable_tex_sub_image_2d"
                    ).value
                )
            ).lower(),
        }
        self._sleep_s = 1.0 / rate_hz
        self._world = str(self.get_parameter("world_frame").value).strip()
        self._left_wrist = str(
            self.get_parameter("left_wrist_frame").value
        ).strip()
        self._right_wrist = str(
            self.get_parameter("right_wrist_frame").value
        ).strip()
        if (
            not self._world
            or not self._left_wrist
            or not self._right_wrist
            or len({self._world, self._left_wrist, self._right_wrist}) != 3
        ):
            raise ValueError("world and wrist frame names must be non-empty/distinct")

        controllers = ControllersSource(name="controllers")
        head = HeadSource(name="head")
        hands = HandsSource(name="hands")
        pipeline = OutputCombiner(
            {
                "controller_left": controllers.output(ControllersSource.LEFT),
                "controller_right": controllers.output(ControllersSource.RIGHT),
                "head": head.output("head"),
                "hand_left": hands.output(HandsSource.LEFT),
                "hand_right": hands.output(HandsSource.RIGHT),
            }
        )
        self._session_config = TeleopSessionConfig(
            app_name="FlexivInspireRawSource",
            pipeline=pipeline,
            mode=SessionMode.LIVE,
        )
        self._openxr_session_live = threading.Event()
        self._ee_pub = self.create_publisher(PoseArray, "xr_teleop/ee_poses", 10)
        self._hand_pub = self.create_publisher(PoseArray, "xr_teleop/hand", 10)
        self._controller_pub = self.create_publisher(
            ByteMultiArray, "xr_teleop/controller_data", 10
        )
        self._tf = TransformBroadcaster(self)

    def _publish_controllers(self, result: dict, now) -> None:
        message = PoseArray()
        message.header.stamp = now
        message.header.frame_id = self._world
        transforms = []
        head = result["head"]
        for side, frame in (
            ("left", self._left_wrist),
            ("right", self._right_wrist),
        ):
            controller = result[f"controller_{side}"]
            head_relative = _head_relative_controller_pose(head, controller)
            is_valid = head_relative is not None
            pose = (
                _pose(*head_relative)
                if is_valid
                else _pose((0.0, 0.0, 0.0))
            )
            message.poses.append(pose)
            if is_valid:
                transform = TransformStamped()
                transform.header = message.header
                transform.child_frame_id = frame
                transform.transform.translation.x = pose.position.x
                transform.transform.translation.y = pose.position.y
                transform.transform.translation.z = pose.position.z
                transform.transform.rotation = pose.orientation
                transforms.append(transform)
        self._ee_pub.publish(message)
        if transforms:
            self._tf.sendTransform(transforms)

        left = result["controller_left"]
        right = result["controller_right"]
        payload = controller_payload(
            timestamp_ns=time.time_ns(),
            left_squeeze_value=_controller_value(
                left, ControllerInputIndex.SQUEEZE_VALUE, 0.0
            ),
            right_squeeze_value=_controller_value(
                right, ControllerInputIndex.SQUEEZE_VALUE, 0.0
            ),
            left_primary_click=_controller_click(
                left, ControllerInputIndex.PRIMARY_CLICK
            ),
            right_primary_click=_controller_click(
                right, ControllerInputIndex.PRIMARY_CLICK
            ),
            left_is_active=not left.is_none,
            right_is_active=not right.is_none,
        )
        controller_message = ByteMultiArray()
        controller_message.data = encode_controller_payload(payload)
        self._controller_pub.publish(controller_message)

    def _publish_hands(self, result: dict, now) -> None:
        message = PoseArray()
        message.header.stamp = now
        message.header.frame_id = self._world
        for side in ("left", "right"):
            hand = result[f"hand_{side}"]
            rooted = _controller_rooted_manus_poses(
                hand, result[f"controller_{side}"], side
            )
            for joint in range(
                HandJointIndex.WRIST, HandJointIndex.LITTLE_TIP + 1
            ):
                rooted_index = joint - int(HandJointIndex.WRIST)
                if rooted == ():
                    # Wrist ownership belongs to Quest: never publish MANUS
                    # joints in another frame while its controller is absent.
                    message.poses.append(_pose((0.0, 0.0, 0.0)))
                elif rooted is not None and rooted[rooted_index] is not None:
                    message.poses.append(_pose(*rooted[rooted_index]))
                elif _valid(hand, HandInputIndex.JOINT_VALID, joint):
                    message.poses.append(
                        _pose(
                            hand[HandInputIndex.JOINT_POSITIONS][joint],
                            hand[HandInputIndex.JOINT_ORIENTATIONS][joint],
                        )
                    )
                else:
                    message.poses.append(_pose((0.0, 0.0, 0.0)))
        if len(message.poses) != 50:
            raise RuntimeError(
                f"raw hand transport must contain 50 poses, got {len(message.poses)}"
            )
        self._hand_pub.publish(message)

    def _run_sessions(self, launcher: CloudXRLauncher) -> int:
        while rclpy.ok():
            launcher.health_check()
            self._openxr_session_live.clear()
            try:
                with TeleopSession(self._session_config) as session:
                    self._openxr_session_live.set()
                    self.get_logger().info(
                        "raw Quest + MANUS session started (no hand URDF)"
                    )
                    while rclpy.ok():
                        launcher.health_check()
                        result = session.step()
                        rclpy.spin_once(self, timeout_sec=0.0)
                        now = self.get_clock().now().to_msg()
                        self._publish_controllers(result, now)
                        self._publish_hands(result, now)
                        time.sleep(self._sleep_s)
            except RuntimeError as exc:
                if not is_retryable_openxr_session_error(exc):
                    raise
                self.get_logger().warning(
                    f"OpenXR session not ready ({exc}); retrying in 2 seconds"
                )
                time.sleep(2.0)
            finally:
                self._openxr_session_live.clear()
        return 0

    def run(self) -> int:
        env_config = str(
            self.get_parameter("cloudxr_env_config").value
        ).strip() or None
        _install_cloudxr_client_overrides(
            self._cloudxr_client_fields,
            session_live=self._openxr_session_live,
        )
        self.get_logger().info(
            "CloudXR Quest low-latency client: "
            f"{self._cloudxr_client_fields['perEyeWidth']}x"
            f"{self._cloudxr_client_fields['perEyeHeight']} per eye, "
            f"{self._cloudxr_client_fields['deviceFrameRate']} FPS, "
            f"{self._cloudxr_client_fields['maxStreamingBitrateMbps']} Mbps, "
            f"{self._cloudxr_client_fields['codec']}"
        )
        with CloudXRLauncher(
            install_dir=str(self.get_parameter("cloudxr_install_dir").value),
            env_config=env_config,
            accept_eula=bool(
                self.get_parameter("cloudxr_accept_eula").value
            ),
            setup_oob=bool(self.get_parameter("cloudxr_setup_oob").value),
            usb_local=bool(self.get_parameter("cloudxr_usb_local").value),
        ) as launcher:
            self.get_logger().info(
                "CloudXR runtime/WSS started; launch the MANUS plugin using "
                "~/.cloudxr/run/cloudxr.env"
            )
            return self._run_sessions(launcher)


def main(args=None) -> int:
    rclpy.init(args=args)
    node = None
    try:
        node = XrRawRosSource()
        return node.run()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
