# 现场运行手册：双臂 + DFTP-2 + Quest/MANUS

本文档是 `/home/hb/isaac_teleop_flexiv_inspire` 的唯一推荐启动顺序。所有
命令均在机器人主机本地执行；不要 source 其他机器人工作空间。

## 0. 先读：默认状态与尚未完成的现场条件

系统默认是零动作：

- RDK daemon 只有显式同时提供 CLI 开关、环境确认和本地 `0600` permit
  才能执行任何写操作。
- 控制桥从 `MAINTENANCE` 启动；当前硬件会话完成双臂 F/T 清零前不能
  进入 `READY`。
- DFTP 驱动默认只读。超时、掉线和松脚踏不会自动回零、张手或回家。
- 遥操作输入默认以 `command_enabled:=false` 启动。
- gRPC 不能执行 `Enable`、F/T 清零或本地控制授权。

首次真机验收前仍必须由现场人员完成这些外部条件：

1. 核对控制器当前工具及 payload、默认七轴限位、相机序列号和 MANUS 标定；
2. 阅读并亲自接受 NVIDIA CloudXR EULA；不得脚本代替接受；
3. 连接 Quest，并确认 `adb devices -l` 能看到设备；
4. 按上游许可要求获取官方 Sharpa Wave 左右手 URDF。缺少它们时
   DexPilot/MANUS 手部链路不能验收；
5. 确认物理脚踏的稳定设备路径和安全人员/碰撞许可流程。

当前代码和 mock 测试不能替代上述现场验收。若任一条件不满足，只运行到
shadow/只读阶段。

## 1. 一次性环境、生成代码与构建

```bash
export ROOT=/home/hb/isaac_teleop_flexiv_inspire
cd "$ROOT"

./scripts/verify_host_cuda.sh
./scripts/env/create_envs.sh
./scripts/generate_protos.sh
./scripts/env/smoke_switch_isolation.sh
./scripts/verify_independence.sh

source "$ROOT/scripts/env/activate_ros.sh"
colcon --log-base "$ROOT/ros2_ws/log" build \
  --base-paths "$ROOT/ros2_ws/src" \
  --build-base "$ROOT/ros2_ws/build" \
  --install-base "$ROOT/ros2_ws/install" \
  --symlink-install \
  --cmake-args -DPython3_EXECUTABLE="$ROOT/envs/ros-py312/bin/python3"

source "$ROOT/ros2_ws/install/setup.bash"
ros2 pkg executables flexiv_inspire_control
```

不要运行无 `--base-paths` 的裸 `colcon build`。该命令构建
`ros2_ws/src` 下的全部八个 ROS 包；`envs/`、`third_party/IsaacTeleop/`
和 `vendor/` 不会进入工作空间扫描。

若 MANUS 插件尚未安装，按 [MANUS_SETUP.md](MANUS_SETUP.md) 构建。安装后
必须存在：

```bash
test -x "$ROOT/third_party/IsaacTeleop/install/manus-isaac/plugins/manus/manus_hand_plugin"
```

CUDA 12.8 只由 `activate_isaac.sh`/`activate_data.sh` 局部选择。不要改
`.bashrc`、`/usr/bin/nvcc` 或系统 CUDA 12.0。

## 2. 核对现场配置

不需要创建固定的 `artifacts/site/`。`config/site.yaml` 是统一入口；
RDK 工具审计、MANUS 标定等需要独立哈希的记录由它引用。`render` 会把
进程级配置生成到本会话的 runtime 目录：

```bash
export ROOT=/home/hb/isaac_teleop_flexiv_inspire
source "$ROOT/scripts/env/activate_ros.sh"
flexiv-inspire --config "$ROOT/config/site.yaml" validate
flexiv-inspire --config "$ROOT/config/site.yaml" render
```

真机前逐项复核：

- `apps/flexiv_daemon/config/robots.yaml`：机器人序列号、软件兼容前缀和
  期望活动工具名。当前两臂均应为 `inspire_rs`。
- `apps/flexiv_daemon/config/tool_payload.yaml`：这是控制器工具的本地审计
  快照，不负责选择或切换工具。质量、质心、惯量和 TCP 由 RDK 从当前
  `inspire_rs` 读取；物理工具序列号与安装修订仍需现场确认。确认后才可
  设置 `locally_verified: true`。
- `config/site.yaml`：当前已填写两台控制器 `RobotInfo` 返回的 Rizon4s
  默认七轴限位。生成的控制配置会携带这些值，控制时软件仍会逐帧检查。
- `teleop.manus_calibration`：未完成标定时保持空值，只允许手臂
  shadow/录制；手指命令保持 invalid。
- 相机序列号、DFTP IP 和脚踏 by-id。不要把凭据或私钥放入仓库。

只读验证工具/payload 记录：

```bash
source "$ROOT/scripts/env/activate_rdk.sh"
flexiv-rdk-daemon \
  --config "$ROOT/apps/flexiv_daemon/config/robots.yaml" \
  --print-tool-payload-hash
```

在 `locally_verified` 仍为 `false` 时，这条命令应拒绝，这是现场物理审计
尚未完成的证据，不是要求在 RDK 中再次选择工具。

## 3. 为一次硬件会话固定变量

在一个本地 shell 创建会话变量文件，随后每个终端都 source 同一文件：

```bash
export ROOT=/home/hb/isaac_teleop_flexiv_inspire
export SESSION_ID="hardware-$(date -u +%Y%m%dT%H%M%SZ)"
export RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/isaac_teleop"
mkdir -p "$ROOT/artifacts/runtime" "$RUNTIME_DIR"
source "$ROOT/scripts/env/activate_ros.sh"
export RENDER_DIR="$(
  flexiv-inspire --config "$ROOT/config/site.yaml" render |
    python -c 'import json,pathlib,sys; print(pathlib.Path(json.load(sys.stdin)["snapshot"]).parent)'
)"
test -d "$RENDER_DIR"
umask 077
printf '%s\n' \
  "export ROOT='$ROOT'" \
  "export RDK_CONFIG='$ROOT/apps/flexiv_daemon/config/robots.yaml'" \
  "export TOOL_CONFIG='$ROOT/apps/flexiv_daemon/config/tool_payload.yaml'" \
  "export CONTROL_CONFIG='$RENDER_DIR/control_bridge.yaml'" \
  "export TELEOP_CONFIG='$RENDER_DIR/teleop.yaml'" \
  "export DFTP_CONFIG='$RENDER_DIR/dftp.yaml'" \
  "export CAMERA_CONFIG='$RENDER_DIR/camera.yaml'" \
  "export MANUS_CONFIG=''" \
  "export SESSION_ID='$SESSION_ID'" \
  "export RUNTIME_DIR='$RUNTIME_DIR'" \
  "export RDK_SOCKET='$RUNTIME_DIR/rdk.sock'" \
  "export DEVICEIO_SOCKET='$RUNTIME_DIR/deviceio.sock'" \
  "export ISAAC_TELEOP_DEVICEIO_SOCKET='$RUNTIME_DIR/deviceio.sock'" \
  "export RDK_EVENTS='$ROOT/artifacts/runtime/${SESSION_ID}-ft-zero.jsonl'" \
  "export WRITE_PERMIT='$RUNTIME_DIR/${SESSION_ID}.write-permit'" \
  > "$ROOT/artifacts/runtime/current-session.env"
```

后文每个终端先执行：

```bash
source /home/hb/isaac_teleop_flexiv_inspire/artifacts/runtime/current-session.env
```

不要复用旧会话的 F/T 结果、write permit 或 `SESSION_ID`。

## 4. 阶段 A：全部只读/Shadow 启动

### 4.1 RDK 兼容性只读检查

终端 A：

```bash
source "$ROOT/scripts/env/activate_rdk.sh"
flexiv-rdk-daemon \
  --hardware \
  --verify-compatibility \
  --config "$RDK_CONFIG"
```

这一步只读 `RobotInfo` 和当前 `Tool.params()`；它不会升级控制器、选择
工具或修改 payload。必须确认 RDK 为 1.9.x、两台机器人分别为
`Rizon4s-063326`/`Rizon4s-063806`、软件前缀和许可兼容，活动工具均为
`inspire_rs`，并查看控制器返回的关节限位。
然后启动只读 daemon：

```bash
flexiv-rdk-daemon \
  --hardware \
  --config "$RDK_CONFIG" \
  --socket "$RDK_SOCKET" \
  --events "$RDK_EVENTS"
```

此命令没有 `--allow-hardware-writes`，所以不能 Enable、清零或运动。

### 4.2 三相机

终端 B：

```bash
source "$ROOT/scripts/env/activate_ros.sh"
source "$ROOT/ros2_ws/install/setup.bash"
flexiv-inspire-camera-verify --config "$CAMERA_CONFIG"
flexiv-inspire-camera-node --ros-args \
  -p config:="$CAMERA_CONFIG"
```

必须看到三台正确序列号、librealsense 2.57.7、`424x240@30`、JPEG90、
RGB-only。不要混入 2.58 或默认开启深度。

### 4.3 DFTP-2 只读

终端 C 先做一次只读协议检查：

```bash
source "$ROOT/scripts/env/activate_ros.sh"
source "$ROOT/ros2_ws/install/setup.bash"
flexiv-inspire-dftp-read-only --include-tactile
```

再启动只读 ROS 驱动：

```bash
flexiv-inspire-dftp-node --ros-args \
  --params-file "$DFTP_CONFIG"
```

确认配置仍是 `hardware_write_enabled: false`。此阶段只能读角度、位置、力、
电流、温度、错误、状态和 1062 taxels。

### 4.4 Quest、CloudXR 与 MANUS

先检查外部前置条件：

```bash
adb devices -l
test -f \
  "$ROOT/third_party/IsaacTeleop/examples/teleop_ros2/assets/urdf/sharpa_standalone/left_sharpa_wave.urdf"
test -f \
  "$ROOT/third_party/IsaacTeleop/examples/teleop_ros2/assets/urdf/sharpa_standalone/right_sharpa_wave.urdf"
```

若 URDF 缺失，按上游
`examples/teleop_ros2/scripts/fetch_sharpa_wave_urdfs.py` 的许可说明人工获取，
不要用任意其他 URDF 冒充。随后在本地交互终端 D 启动官方 ROS 参考节点：

```bash
source "$ROOT/scripts/env/activate_isaac.sh"
source "$ROOT/ros2_ws/install/setup.bash"
cd "$ROOT/third_party/IsaacTeleop/examples/teleop_ros2/python"
python teleop_ros2_node.py --ros-args \
  -p mode:=controller_teleop \
  -p hand_retargeter:=dexpilot \
  -p config_asset_root:="$ROOT/third_party/IsaacTeleop/examples/teleop_ros2" \
  -p cloudxr_accept_eula:=false \
  -p cloudxr_setup_oob:=true \
  -p cloudxr_usb_local:=true
```

`cloudxr_accept_eula:=false` 保留首次启动的人工 EULA 提示。该节点在进程内
拥有 CloudXR runtime/WSS proxy；不要同时运行独立
`python -m isaacteleop.cloudxr`。

CloudXR 运行并生成 `~/.cloudxr/run/cloudxr.env` 后，在终端 E 启动 MANUS：

```bash
source "$ROOT/scripts/env/activate_isaac.sh"
source "$ROOT/ros2_ws/install/setup.bash"
set -a
source "$HOME/.cloudxr/run/cloudxr.env"
set +a
"$ROOT/third_party/IsaacTeleop/install/manus-isaac/plugins/manus/manus_hand_plugin"
```

先验证 `/xr_teleop/ee_poses`、`/xr_teleop/controller_data`、`/tf` 和
50-pose `/xr_teleop/hand`。MANUS SDK 连接成功不等于手套/Quest/坐标映射
已经标定成功。

### 4.5 控制桥和遥操作映射保持 Shadow

控制桥终端 F（整个硬件会话持续运行，不要为切换 shadow 重启它）：

```bash
source "$ROOT/scripts/env/activate_ros.sh"
source "$ROOT/ros2_ws/install/setup.bash"
ros2 run flexiv_inspire_control control_bridge --ros-args \
  --params-file "$CONTROL_CONFIG" \
  -p session_id:="$SESSION_ID" \
  -p rdk_socket:="$RDK_SOCKET" \
  -p foot_pedal:=/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd
```

遥操作映射终端 G：

```bash
source "$ROOT/scripts/env/activate_ros.sh"
source "$ROOT/ros2_ws/install/setup.bash"
ros2 run flexiv_inspire_control teleop_input --ros-args \
  --params-file "$TELEOP_CONFIG" \
  -p session_id:="$SESSION_ID" \
  -p command_enabled:=false
```

此时允许查看映射，但不发布有效动作。`/control/state` 应为
`MAINTENANCE`，不是 `READY`。

### 4.6 Rerun 与只读频率检查

终端 H：

```bash
source "$ROOT/scripts/env/activate_ros.sh"
source "$ROOT/ros2_ws/install/setup.bash"
flexiv-inspire-rerun --spawn
```

检查状态但不要伪造脚踏：

```bash
ros2 topic echo /control/state --once
ros2 topic hz /robot/left_arm/state
ros2 topic hz /robot/left_hand/tactile_raw
ros2 topic hz /camera/head/color/frame
```

Rerun 是 latest-only 可视化，不是记录真源。空 fault/hold、invalid timing
和缺失流都应在界面中明确显示为已清除或无效，不能沿用旧值。

若清零前只想排障，可用：

```bash
"$ROOT/scripts/record_ros_mcap.sh" "${SESSION_ID}-preflight"
```

它只是 ROS-only 诊断 bag，不生成 episode manifest，也不替代正式录制。

## 5. 阶段 B：现场工具/payload 审计与强制 F/T 清零

只有人在机器人旁、双臂/双手/线缆完全静止且无接触时才能继续。

1. 在终端 A 用 `Ctrl-C` 停止只读 daemon。
2. 最后复核 `tool_payload.yaml` 与控制器配置一致。
3. 在同一台主机的交互 TTY 创建本会话一次性 write permit：

```bash
source "$ROOT/scripts/env/activate_rdk.sh"
flexiv-create-write-permit "$WRITE_PERMIT" \
  --confirm FLEXIV-RDK-WRITES-ENABLED
export ISAAC_TELEOP_ALLOW_HARDWARE_WRITES=FLEXIV-RDK-WRITES-ENABLED
flexiv-rdk-daemon \
  --hardware \
  --allow-hardware-writes \
  --local-permit-file "$WRITE_PERMIT" \
  --config "$RDK_CONFIG" \
  --socket "$RDK_SOCKET" \
  --events "$RDK_EVENTS"
```

三个条件只开放 daemon 写边界，不会自行 Enable、清零或运动。daemon
会在清零前再次将 `$TOOL_CONFIG` 的审计快照与控制器当前工具逐项比较；
任一名称、质量、质心、惯量或 TCP 不一致都会拒绝。重启会使旧 F/T 状态
失效，桥必须仍显示 `MAINTENANCE`。

先在另一个本地 ROS TTY 做不执行的预览：

```bash
source "$ROOT/scripts/env/activate_ros.sh"
source "$ROOT/ros2_ws/install/setup.bash"
ros2 run flexiv_inspire_control zero_ft_local \
  --rdk-socket "$RDK_SOCKET" \
  --tool-payload-config "$TOOL_CONFIG"
```

核对输出的 session、配置 SHA-256、双手姿态和两秒稳定性。确认无人接触、
机器人及线缆静止后，才重新执行：

```bash
ros2 run flexiv_inspire_control zero_ft_local \
  --rdk-socket "$RDK_SOCKET" \
  --tool-payload-config "$TOOL_CONFIG" \
  --confirm-ft-unloaded FLEXIV-FT-UNLOADED
```

该确认只能从机器人本机交互 TTY 发出。过程中 teleop/policy/replay 都无权
控制；任一臂运动、接触、超时、primitive 失败、残差超限或 daemon/RDK
重连都会失败并保持 `MAINTENANCE`/`FAULT`。成功后检查：

```bash
ros2 topic echo /control/state --once
test -s "$RDK_EVENTS"
```

只有同一 `daemon_instance_id` 和同一 `connection_generation` 的成功结果
才能原子地进入 `READY`。

## 6. 阶段 C：本地授权并按级别验收

正式动作前，保持双手松开 Quest deadman，并确认机械区域无人。只有真实
安全系统/现场人员确认后，才发布本地许可和碰撞许可：

```bash
ros2 topic pub --once /control/local_permission std_msgs/msg/Bool \
  '{data: true}'
ros2 topic pub --once /safety/collision_clear std_msgs/msg/Bool \
  '{data: true}'
```

物理脚踏必须由配置的 `/dev/input/by-id/...` 设备产生；不要用 ROS 话题或
软件脚本模拟脚踏。

若要控制灵巧手，先停止只读 DFTP 节点，再以本会话的显式写门禁重启：

```bash
test -n "$MANUS_CONFIG"
test -f "$MANUS_CONFIG"
flexiv-inspire-dftp-node --ros-args \
  --params-file "$DFTP_CONFIG" \
  -p hardware_write_enabled:=true \
  -p local_session_id:="$SESSION_ID" \
  -p local_write_confirmation:=DFTP-LOCAL-CONTROL-AUTHORIZED
```

它仍只接受 supervisor 发布的 `/control/sent_command`；超时不会自动张手。

在终端 G 用 `Ctrl-C` 只停止 shadow `teleop_input`，不要停止控制桥。然后
在本地交互 TTY 授权 teleop：

```bash
ros2 run flexiv_inspire_control authorize_control \
  --session-id "$SESSION_ID" \
  --source teleop \
  --confirm FLEXIV-CONTROL-ARM \
  --socket "$RDK_SOCKET"
```

以命令模式重新启动遥操作映射：

```bash
ros2 run flexiv_inspire_control teleop_input --ros-args \
  --params-file "$TELEOP_CONFIG" \
  -p manus_calibration:="$MANUS_CONFIG" \
  -p session_id:="$SESSION_ID" \
  -p command_enabled:=true
```

先保持 deadman 松开完成 rebase。严格按以下顺序逐级验收，每一级都记录
requested/safe/sent、wrench 和现场结果，失败立即松脚踏并停止：

1. 单臂，速度/加速度上限降到额定值的 10%；
2. 双臂；
3. 完成 MANUS 标定后单手；
4. 完成 MANUS 标定后双臂 + 双手；
5. 策略 shadow；
6. 策略真机。

`/control/safe_command` 只表示完整本地安全检查和授权已通过、准备送到
硬件边界；`/control/sent_command` 只在 RDK 对整条双臂事务返回 accepted
后发布。超步长/无本地 token 不产生 safe/sent；RDK 拒绝可有 safe，但
不能有 sent。

发生锁存 hold 后，先松开 Quest deadman 和脚踏、排除原因，并在本地确认。
重新授权时才可使用：

```bash
ros2 run flexiv_inspire_control authorize_control \
  --session-id "$SESSION_ID" \
  --source teleop \
  --confirm FLEXIV-CONTROL-ARM \
  --clear-hold-latched \
  --socket "$RDK_SOCKET"
```

## 7. 正式异步录制与数据语义

正式入口是 `isaac-flexiv-episode`，它会拒绝没有匹配 session、工具哈希和
成功 `ft_zero_completed` 事件的录制：

```bash
source "$ROOT/scripts/env/activate_ros.sh"
source "$ROOT/ros2_ws/install/setup.bash"
isaac-flexiv-episode \
  --root "$ROOT/sessions" \
  --session-id "$SESSION_ID" \
  --tool-config "$TOOL_CONFIG" \
  --ft-zero-record "$RDK_EVENTS" \
  --calibration cameras="$CAMERA_CONFIG" \
  --camera-recording-mode jpeg \
  --deviceio-mode native \
  --deviceio-socket "$DEVICEIO_SOCKET" \
  --duration-s 1800
```

完成 MANUS 标定后再额外传入
`--calibration manus="$MANUS_CONFIG"`；手臂-only 数据不伪造 MANUS 标定
记录。

正式运动前必须先确认该进程仍在运行，socket 权限为 `0600`。各硬件进程可
先于 recorder 启动；此时传感器记录会明确计为 transport drop，而控制关键
事件会保留并在队列耗尽时 fail-closed。推荐顺序是在取得控制授权前先启动
本 episode。

两个文件承担不同职责：

- `ros_mcap/`：rosbag2 MCAP，保存原生 DDS CDR ROS 消息，供 ROS 回放与诊断；
- `deviceio.mcap`：默认 `native-pre-dds`，由 RDK、DFTP、相机和控制监督器在
  DDS 发布前通过本机 `AF_UNIX/SOCK_DGRAM` 写入；保存 source/receive/mapped
  time、sequence、validity、age 和 producer/drop 统计，是训练的原始真源。

只有显式 `--deviceio-mode post-dds-mirror` 才使用兼容 typed mirror；manifest
会记录实际 capture layer，禁止混称。原生传感器队列 bounded/drop-oldest，
控制/维护/episode 事件使用独立关键队列。正式训练视图应从有 manifest、
清零证据和校准哈希的 episode 可重复导出；缺失或超容差数据标 invalid，
不得补零。`scripts/record_ros_mcap.sh` 仅供 ROS 排障。

LeRobot 默认 30 维动作为：

```text
left Δxyz(3), left Δrot6d(6),
right Δxyz(3), right Δrot6d(6),
left hand(6), right hand(6)
```

Rotation-6D 顺序固定为
`[R00,R10,R20,R01,R11,R21]`，单位旋转为
`[1,0,0,0,1,0]`。导出时先对合法旋转做 SO(3) 插值，再生成 Rotation-6D；
不得直接线性插值 Rotation-6D。

## 8. TLS gRPC 策略接入

先用 loopback 验证，证书/私钥必须位于 Git 忽略目录：

```bash
source "$ROOT/scripts/env/activate_ros.sh"
source "$ROOT/ros2_ws/install/setup.bash"
flexiv-inspire-policy-server \
  --bind 127.0.0.1 \
  --port 50051 \
  --server-cert "$ROOT/certs/server.crt" \
  --server-key "$ROOT/certs/private/server.key"
```

远程策略显式绑定机器人地址时必须使用 mTLS：

```bash
flexiv-inspire-policy-server \
  --bind 192.168.110.221 \
  --port 50051 \
  --server-cert "$ROOT/certs/server.crt" \
  --server-key "$ROOT/certs/private/server.key" \
  --client-ca "$ROOT/certs/client-ca.crt"
```

先保持策略 shadow，仅调用 `GetCapabilities`/观察流并比较动作。进入策略
真机前停止 teleop 控制、松开 deadman/脚踏、清除 hold，再由本地 TTY：

```bash
ros2 run flexiv_inspire_control authorize_control \
  --session-id "$SESSION_ID" \
  --source policy \
  --confirm FLEXIV-CONTROL-ARM \
  --clear-hold-latched \
  --socket "$RDK_SOCKET"
```

远程客户端随后才能 `AcquireControlLease`。lease/heartbeat/TTL 任一失效都
锁存 hold；观测拥塞丢旧保新，动作不积压。teleop、policy、replay 永远
互斥。

## 9. 关机

1. 松开 Quest deadman 和物理脚踏；
2. 发布一次本地 stop：`ros2 topic pub --once /control/stop std_msgs/msg/Empty '{}'`；
3. 确认 `/control/state` 已进入锁存 hold/非 ACTIVE；
4. 先结束 episode 并等待 MCAP/manifest flush，再结束策略、Rerun、
   teleop、MANUS/CloudXR、DFTP、相机、控制桥，最后结束 RDK daemon；
5. 删除本会话 write permit：

```bash
rm -- "$WRITE_PERMIT"
```

任何 daemon/RDK 重连、工具配置变化或新硬件会话都必须重新做工具审计和
双臂 F/T 清零，不能复用上一次 `READY`。
