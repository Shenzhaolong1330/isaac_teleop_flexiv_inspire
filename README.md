# Isaac Teleop × Flexiv 双臂 × Inspire 灵巧手（隔离适配）

这个目录是对现有系统的旁路适配，不修改：

- `/home/hb/flexiv_inspire_ws`
- `/home/hb/flexiv_inspire_ws/src/dual_arm_teleop`
- `/home/hb/flexiv_inspire_ws/src/Le-nero`

上游 NVIDIA Isaac Teleop 固定为 `v1.3.131`，实际 commit 为
`7002ed63d69454ae4f15c0ee19f803fd2846592b`，位于
`upstream/IsaacTeleop`。

## 当前完成范围

- Isaac ROS 2 `xr_teleop/ee_poses` 绝对位姿到现有 12 个
  `left/right_delta_ee_pose.{x,y,z,rx,ry,rz}` 字段的适配。
- 首帧归零、clutch/rebase、左右耦合失效保护、TF 有效性门控、陈旧数据
  watchdog、跳变检测、速度/加速度/单步限幅。
- requested 与 applied action 分开发布，只有硬件网关成功 ACK 后才推进内部
  command target。
- ROS 2 Jazzy（Python 3.12）与现有 LeRobot/Flexiv Conda
  `flexiv_teleop`（Python 3.10）通过本机 Unix datagram socket 隔离。
- 硬件侧复用现有 `FlexivDualArm` 与现有 YAML 的内存副本；不写回原文件。
- Inspire 初版继续复用当前稳定的 Manus UDP 6 通道控制。桥接动作不写任何
  `hand_cmd` 字段，因此不会抢占 Manus。
- 标准 ROS 2 观测话题和 rosbag2 MCAP 录制脚本。

当前默认 `command_enabled: false`。本次搭建没有连接、使能或移动机器人。

## 为什么是两个进程

当前环境已经确认：

- ROS 2 Jazzy 的 `rclpy` 属于 `/usr/bin/python3`（Python 3.12）。
- `flexivrdk`、LeRobot 和现有机器人类属于 Conda `flexiv_teleop`
  （Python 3.10）。

把两套 ABI 强行装进一个进程容易破坏现有环境。这里使用：

```text
Quest / Isaac Teleop
        │ ROS 2 PoseArray + TF + controller_data
        ▼
Python 3.12: isaac_flexiv_bridge
  absolute SE(3) → clutch anchor → bounded delta
  requested/applied ROS topics + MCAP
        │ /tmp Unix datagram, versioned JSON, ACK/watchdog
        ▼
Python 3.10: hardware gateway
  local pedal gate → existing FlexivDualArm.send_action()
        │
        ├── Flexiv 双臂（现有 RDK/200 Hz servo）
        └── Inspire（现有 Manus UDP fallback）
```

## 关键接口事实

Isaac `v1.3.131` 的 ROS 2 示例在 `controller_teleop` 模式发布：

- `/xr_teleop/ee_poses`：`geometry_msgs/PoseArray`，固定
  `poses[0]=left, poses[1]=right`；这是 controller **AIM** pose。
- 四元数顺序为 `xyzw`。
- 无效的一侧仍可能是零位姿占位，因此桥额外要求对应 wrist TF 新鲜。
- `/xr_teleop/controller_data` 是 msgpack 编码的 `ByteMultiArray`，默认用双侧
  squeeze 作为软件 deadman。

当前 `FlexivDualArm` 的 delta 语义是：

```text
p_new = p_internal_target + dp
R_new = Exp(rotvec_delta) · R_internal_target
```

所以本项目没有用相邻 XR 帧差直接驱动，而是从 clutch 时刻的绝对 anchor
持续计算绝对相对目标，再相对已 ACK 的 command target 生成有界步长。

## 安装桥接环境

系统 CUDA 和现有 Conda 环境均不修改：

```bash
cd /home/hb/isaac_teleop_flexiv_inspire
/home/hb/.local/bin/uv venv \
  --python /usr/bin/python3 \
  --system-site-packages \
  .venv
/home/hb/.local/bin/uv pip install \
  --python .venv/bin/python \
  -e '.[ros,test]'
```

验证：

```bash
./scripts/verify_environment.sh
.venv/bin/python -m pytest
```

## 先做 shadow mode

终端 1：

```bash
./scripts/run_shadow_mode.sh
```

终端 2 可先不用 Isaac/Quest，运行确定性的合成输入：

```bash
./scripts/run_synthetic_source.sh --deadman --duration 10
```

观察：

```bash
ros2 topic echo /isaac_flexiv/status
ros2 topic echo /isaac_flexiv/applied/left_delta
```

shadow mode 即使 squeeze/deadman 为真也不会创建硬件网关，更不会调用
`FlexivDualArm.send_action()`。

## 真机门控（当前不要直接执行）

真机至少需要依次完成：

1. shadow 中确认三轴正方向、左右顺序、旋转方向、TF 连续性和延迟。
2. 标定 `mapping.left/right.axis_rotation`；矩阵必须是正交且
   `det=+1`，禁止把反射矩阵当四元数坐标变换。
3. 物理急停、Flexiv Elements、工作区、脚踏和人员位置检查。
4. 在**新 YAML** 中显式设置 `command_enabled: true`。
5. 先启动硬件网关：

   ```bash
   ./scripts/run_hardware_gateway.sh \
     configs/flexiv_inspire.yaml \
     --enable-hardware-command \
     --confirm-robot-area-clear FLEXIV-AREA-CLEAR
   ```

6. 再启动命令桥：

   ```bash
   ./scripts/run_command_bridge.sh
   ```

硬件侧仍要求本机脚踏按下；Quest deadman、脚踏、配置开关和两个 CLI
确认缺一不可。松开或数据超时会调用固定上游版本的
`_refresh_cached_poses()`，把未完成的 200 Hz servo target 钉到新鲜实测 TCP；
若该能力缺失或状态读取失败，则停止 servo、调用 RDK `Stop()` 并锁存故障。

## MCAP 与 LeRobot

录 ROS 2 MCAP：

```bash
./scripts/record_ros_mcap.sh my_session_id
```

它包含：

- Isaac EE/手/控制器/TF；
- requested/applied 双臂 action；
- Flexiv 关节和 TCP；
- Inspire 6 个归一化 actuator state；
- bridge/gateway 状态与 diagnostics。

NVIDIA Isaac 内建 MCAP 只记录 DeviceIO 原始 tracker，不会自动包含机器人状态、
相机、requested/applied action 或 Inspire。后续建议同一个 `session_id` 保存：

```text
sessions/<id>/isaac_raw.mcap
sessions/<id>/ros2/*.mcap
sessions/<id>/clock_manifest.json
```

再离线对齐并导出 LeRobot episode。当前项目尚未伪造一个“通用
MCAP→LeRobot”转换器；真实图像 topic、力矩、六维力和触觉需要先在硬件驱动层
补出带时间戳的观测，才能可靠写入 MCAP/LeRobot。

## Inspire 手的边界

Isaac 默认 TriHand 是 7 关节/手，DexPilot Sharpa 是 22 关节/手；当前 Inspire
接口是 6 actuator/手，不能靠改 joint name 直接等价。因此第一版明确：

- Quest/Isaac 控双臂；
- Manus UDP 继续控 Inspire；
- `/xr_teleop/hand` 和 `/xr_teleop/finger_joints` 可录制，但不直接下发；
- 将来单独标定 26 个手部关键点到 6 actuator 的 retargeter 后，再用完整原子的
  12 个 `left/right_hand_cmd_0..5` 接管。

## 当前运行阻断项

机器 NVIDIA 驱动满足当前 Isaac 要求，但系统 `nvcc` 是 CUDA 12.0；Isaac Teleop
当前要求 CUDA 12.8 及以上。因此上游源码已经固定并完成适配，桥接和 ROS/MCAP
可以测试，但尚未安装/启动 CloudXR/Isaac 真正运行时。不要直接升级系统 CUDA；
应另建隔离 Isaac 环境并先确认 RTX 5070、CloudXR、Quest 网络链路。

