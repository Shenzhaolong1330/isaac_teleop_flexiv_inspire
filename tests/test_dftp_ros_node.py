from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from flexiv_inspire_isaac.dftp import ros_node
from flexiv_inspire_isaac.dftp.ros_node import (
    hand_reset_targets,
    hand_target_reached,
)


class _Publisher:
    def publish(self, _message) -> None:
        pass


class _FakeNode:
    instances: list["_FakeNode"] = []

    def __init__(self, _name: str) -> None:
        # Mirror the rclpy invariant that exposed the original regression.
        self._publishers: list[_Publisher] = []
        self._parameters = {}
        self.instances.append(self)

    def declare_parameter(self, name: str, default) -> None:
        self._parameters[name] = default

    def get_parameter(self, name: str):
        return SimpleNamespace(value=self._parameters[name])

    def create_publisher(self, *_args):
        publisher = _Publisher()
        self._publishers.append(publisher)
        return publisher

    def create_subscription(self, *_args):
        return object()

    def create_service(self, *_args):
        return object()

    def create_timer(self, *_args):
        return object()

    def get_logger(self):
        return SimpleNamespace(info=lambda *_args: None)

    def destroy_node(self):
        return True


class _Worker:
    def __init__(self, *_args, safe_force_limits, **_kwargs) -> None:
        self.safe_force_limits = safe_force_limits

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def test_dftp_topic_registry_does_not_shadow_rclpy_publishers(monkeypatch) -> None:
    _FakeNode.instances.clear()
    shutdown_calls = []

    def interrupt(_node) -> None:
        raise KeyboardInterrupt

    fake_rclpy = _module(
        "rclpy",
        init=lambda **_kwargs: None,
        spin=interrupt,
        ok=lambda: False,
        shutdown=lambda: shutdown_calls.append(True),
    )
    fake_rclpy_node = _module("rclpy.node", Node=_FakeNode)
    fake_rclpy_qos = _module(
        "rclpy.qos",
        HistoryPolicy=SimpleNamespace(KEEP_LAST=1),
        ReliabilityPolicy=SimpleNamespace(BEST_EFFORT=1, RELIABLE=2),
        QoSProfile=lambda **kwargs: kwargs,
    )
    fake_control_msgs = _module("control_msgs")
    fake_control_msgs_msg = _module(
        "control_msgs.msg", DynamicJointState=type("DynamicJointState", (), {}),
        InterfaceValue=type("InterfaceValue", (), {}),
    )
    fake_sensor_msgs = _module("sensor_msgs")
    fake_sensor_msgs_msg = _module(
        "sensor_msgs.msg", JointState=type("JointState", (), {})
    )
    fake_rosidl_runtime = _module("rosidl_runtime_py")
    fake_rosidl_convert = _module(
        "rosidl_runtime_py.convert", message_to_ordereddict=lambda message: message
    )
    fake_std_srvs = _module("std_srvs")
    fake_std_srvs_srv = _module(
        "std_srvs.srv", Trigger=type("Trigger", (), {})
    )
    fake_interfaces = _module("flexiv_inspire_interfaces")
    message_types = {
        name: type(name, (), {})
        for name in (
            "BimanualCommand",
            "ControlState",
            "HandState",
            "TactileFrame",
            "TactileSurface",
        )
    }
    fake_interfaces_msg = _module("flexiv_inspire_interfaces.msg", **message_types)

    for name, module in {
        "rclpy": fake_rclpy,
        "rclpy.node": fake_rclpy_node,
        "rclpy.qos": fake_rclpy_qos,
        "control_msgs": fake_control_msgs,
        "control_msgs.msg": fake_control_msgs_msg,
        "sensor_msgs": fake_sensor_msgs,
        "sensor_msgs.msg": fake_sensor_msgs_msg,
        "rosidl_runtime_py": fake_rosidl_runtime,
        "rosidl_runtime_py.convert": fake_rosidl_convert,
        "std_srvs": fake_std_srvs,
        "std_srvs.srv": fake_std_srvs_srv,
        "flexiv_inspire_interfaces": fake_interfaces,
        "flexiv_inspire_interfaces.msg": fake_interfaces_msg,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(ros_node, "ReadOnlyModbusTcpClient", lambda *_a, **_k: object())
    monkeypatch.setattr(ros_node, "DftpProtocolReader", lambda *_a, **_k: object())
    monkeypatch.setattr(ros_node, "DftpHandWorker", _Worker)
    monkeypatch.setattr(
        ros_node,
        "AsyncDeviceIOEmitter",
        lambda *_a, **_k: SimpleNamespace(close=lambda: None),
    )

    assert ros_node.main([]) == 0
    node = _FakeNode.instances[-1]
    assert isinstance(node._publishers, list)
    assert len(node._publishers) == 8
    assert len(node._topic_publishers) == 8
    assert shutdown_calls == []


def test_hand_reset_targets_match_installed_inspire_convention() -> None:
    assert hand_reset_targets() == (
        (1000,) * 6,
        (0,) * 6,
        (1000,) * 6,
    )


def test_hand_target_reached_checks_every_actuator() -> None:
    target = (1000,) * 6
    assert hand_target_reached((999, 998, 1000, 997, 988, 1000), target, 30)
    assert not hand_target_reached((999, 998, 1000, 997, 969, 1000), target, 30)
    assert not hand_target_reached(None, target, 30)
