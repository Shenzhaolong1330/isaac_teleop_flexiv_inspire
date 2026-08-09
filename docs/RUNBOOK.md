# 现场运行手册：双臂 + Inspire RH56 + Quest/MANUS

日常使用只执行下面三个顶层命令；daemon、内部许可及各 ROS 服务由启动器
自动管理。后面的分阶段流程保留给首次验收和故障排查。

```bash
cd /home/hb/isaac_teleop_flexiv_inspire
source scripts/env/activate_ros.sh
robot reset
robot record
# robot record 达到目标条数或按 Ctrl-C 后会自行结束
```

`robot stop` 只用于终端/主机异常退出后回收遗留进程，不是正常录制流程的一步。

命令均在机器人主机本地执行；不要 source 其他机器人工作空间。

## 0. 先读：默认状态与尚未完成的现场条件

系统默认是零动作：

- RDK daemon 只有显式同时提供 CLI 开关、环境确认和本地 `0600` permit
  才能执行任何写操作。
- 控制桥从 `MAINTENANCE` 启动；当前硬件会话完成双臂 F/T 清零前不能
  进入 `READY`。
- Inspire 手部驱动默认只读；本站 `config/hardware.yaml` 已显式开启同一驱动的
  写通道供 MANUS 和 Reset 使用。超时、掉线和松脚踏不会自动回零或回家。
- 遥操作输入默认以 `command_enabled:=false` 启动。
- gRPC 不能执行 `Enable`、F/T 清零或本地控制授权。

首次真机验收前仍必须由现场人员完成这些外部条件：

1. 核对控制器当前工具及 payload、默认七轴限位、相机序列号和 MANUS 标定；
2. 阅读并亲自接受 NVIDIA CloudXR EULA；不得脚本代替接受；
3. 连接 Quest，并确认 `adb devices -l` 能看到设备；
4. 确认物理脚踏的稳定设备路径和安全人员/碰撞许可流程。

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
python -c 'import flexiv_inspire_control.node, flexiv_inspire_isaac.dftp.ros_node'
```

不要运行无 `--base-paths` 的裸 `colcon build`。该命令构建
`ros2_ws/src` 下的全部八个 ROS 包；`envs/`、`third_party/IsaacTeleop/`
和 `vendor/` 不会进入工作空间扫描。

真机 Python 节点使用上述 `flexiv-inspire-*` 入口，它们固定在
`envs/ros-py312`，可同时看到 ROS 绑定和项目内部库；不要用系统 Python
直接执行 ROS 包脚本。

若 MANUS 插件尚未安装，按 [MANUS_SETUP.md](MANUS_SETUP.md) 构建。安装后
必须存在：

```bash
test -x "$ROOT/third_party/IsaacTeleop/install/manus-isaac/plugins/manus/manus_hand_plugin"
```

CUDA 12.8 只由 `activate_isaac.sh`/`activate_data.sh` 局部选择。不要改
`.bashrc`、`/usr/bin/nvcc` 或系统 CUDA 12.0。

## 2. 核对现场配置

不需要创建固定的 `artifacts/site/`。`config/site.yaml` 是组合入口，它依次
加载 `hardware.yaml`、`sensors.yaml`、`recording.yaml` 和 `runtime.yaml`；
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
  `inspire_rs` 读取；物理工具序列号与安装修订只是可选追溯信息，不阻塞
  操作。确认两只实际安装的手与上位机活动工具均为 `inspire_rs` 后设置
  `locally_verified: true`。
- `config/hardware.yaml`：当前已填写两台控制器 `RobotInfo` 返回的 Rizon4s
  默认七轴限位。生成的控制配置会携带这些值，控制时软件仍会逐帧检查。
- `flexiv.home`：这是项目定义的“双臂复位/遥操作准备位姿”，不是控制器
  标定动作，也不是 Flexiv 内置 `PLAN-Home`。启动、重连和超时都不会自动
  执行它；目标必须在上述七轴限位内，速度不得超过软件关节速度上限。
- `flexiv.cartesian_control`：`position` 和 `impedance` 是同一
  `NRT_CARTESIAN_MOTION_FORCE` 控制器的两套刚度预设。运行时会实际调用
  `SetCartesianImpedance(K_x, Z_x)`；`damping_ratio` 必须在
  `[0.3,0.8]`，刚度还会与每台真机的 `RobotInfo.K_x_nom` 比较。
- `config/sensors.yaml` 的 `teleop.manus_calibration`：未完成标定时保持空值，只允许手臂
  shadow/录制；手指命令保持 invalid。
- 相机序列号、左右 Inspire 型号/IP 和脚踏 by-id。RH56DFTP-2 与 RH56E2
  可按侧混用，配置键为 `inspire.left_model/right_model`；两只手的 IP 必须唯一。
  不要把凭据或私钥放入仓库。

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
source "$ROOT/scripts/env/activate_ros.sh"
export SESSION_ID="$(python -c 'from flexiv_inspire_isaac.system_config import load_system_config; print(load_system_config("config/site.yaml").document["session"]["id"])')"
export RUNTIME_DIR="$(python -c 'from flexiv_inspire_isaac.system_config import load_system_config; print(load_system_config("config/site.yaml").document["session"]["runtime_root"])')"
mkdir -p "$ROOT/artifacts/runtime" "$RUNTIME_DIR"
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
  "export RDK_EVENTS='$RUNTIME_DIR/ft_zero_events.jsonl'" \
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
scripts/generate_loopback_policy_cert.sh  # 只需一次；已有证书时会拒绝覆盖
source "$ROOT/scripts/env/activate_ros.sh"
source "$ROOT/ros2_ws/install/setup.bash"
flexiv-inspire-camera-verify --config "$CAMERA_CONFIG"
flexiv-inspire-camera-node --ros-args \
  -p config:="$CAMERA_CONFIG"
```

必须看到三台正确序列号、librealsense 2.57.7、`424x240@30`、JPEG90，
并确认报告中的模态与配置一致：头部 RGB + 深度 + 点云，两个腕部仅 RGB。
不要混入 2.58，也不要在腕部相机上误开不需要的深度/点云。

### 4.3 Inspire RH56 手只读

终端 C 先做一次只读协议检查：

```bash
source "$ROOT/scripts/env/activate_ros.sh"
source "$ROOT/ros2_ws/install/setup.bash"
flexiv-inspire-dftp-read-only \
  --left-model rh56e2_2l_t1 \
  --right-model rh56e2_2r_t1 \
  --left 192.168.5.11 \
  --right 192.168.5.12 \
  --include-tactile
```

再启动只读 ROS 驱动：

```bash
flexiv-inspire-dftp-node --ros-args \
  --params-file "$DFTP_CONFIG"
```

确认配置仍是 `hardware_write_enabled: false`。此阶段只能读角度、位置、力、
电流、温度、错误、状态和 1062 taxels。RH56DFTP-2 与已验收的 RH56E2-T1
复用六轴寄存器和 17 面触觉布局，但型号、左右手和力上限由独立 profile 校验；
不要把尚未验收的 T2 电容触觉手伪装成 T1。

两只出厂地址相同的新手必须逐只接入并改成唯一地址。IP 修改只写网络寄存器，
保存后需要重新上电；例如把当前单独连接的左手改为 `.5.11`：

```bash
python -m flexiv_inspire_isaac.dftp.network_config \
  --host 192.168.11.210 \
  --expected-current-ip 192.168.11.210 \
  --new-ip 192.168.5.11 \
  --side left --model rh56e2_2l_t1 \
  --confirm INSPIRE-NETWORK-CONFIG
```

右手同理使用 `.5.12`。不要在两只仍使用相同出厂 IP 时同时连接并执行修改。

力传感器零点明显异常时，保持指定手完全张开、无接触、线缆无拉扯，
再单独执行官方寄存器 1009 校准。命令会先检查张开角度、静止电流和
错误码，只操作明确选择的一只手，并打印校准前后值：

```bash
flexiv-inspire-dftp-calibrate-force \
  --side left \
  --confirm INSPIRE-FORCE-CALIBRATION
```

这不是恢复出厂，也不会写目标角度；不要把它加入自动启动流程。

### 4.4 Quest、CloudXR 与 MANUS

先检查 Quest：

```bash
adb devices -l
```

本系统不使用 NVIDIA 示例的 Sharpa retargeter，也不需要 Sharpa URDF。
`flexiv-inspire-xr-raw-source` 发布 Quest 控制器腕部位姿；MANUS
Integrated SDK 的 Ergonomics 回调发布手指关节角。Inspire 六路映射
由下一层左右手独立现场标定完成。
在本地交互终端 D 启动：

```bash
source "$ROOT/scripts/env/activate_isaac.sh"
source "$ROOT/ros2_ws/install/setup.bash"
flexiv-inspire-xr-raw-source --ros-args \
  -p cloudxr_accept_eula:=false \
  -p cloudxr_setup_oob:=true \
  -p cloudxr_usb_local:=true
```

`cloudxr_accept_eula:=false` 保留首次启动的人工 EULA 提示。该节点在进程内
拥有 CloudXR runtime/WSS proxy；不要同时运行独立
`python -m isaacteleop.cloudxr`。

`robot record` 会自动启动 MANUS 插件和 Ergonomics ROS 源。单独调试时
可在终端 E 启动：

```bash
source "$ROOT/scripts/env/activate_isaac.sh"
source "$ROOT/ros2_ws/install/setup.bash"
"$ROOT/orchestration/run_manus_plugin.sh" \
  --ergonomics-udp 127.0.0.1:15053
```

该启动脚本显式跳过插件内的 OpenXR 会话，直接输出 Ergonomics；
不会因 CloudXR/视频失败禁用手指。验证 `/xr_teleop/ee_poses`、
`/xr_teleop/controller_data`、`/manus/left/ergonomics` 和
`/manus/right/ergonomics`。MANUS SDK 连接成功不等于手指标定成功。

保持遥操作为 shadow，用同一操作者和同一副手套分别采集自然完全张开和
自然握拳，每个姿态保持约两秒：

```bash
mkdir -p "$ROOT/artifacts/calibration"
flexiv-inspire-manus-calibrate capture-ergonomics \
  --pose open \
  --output "$ROOT/artifacts/calibration/manus_ergonomics_open.yaml"
flexiv-inspire-manus-calibrate capture-ergonomics \
  --pose closed \
  --output "$ROOT/artifacts/calibration/manus_ergonomics_closed.yaml"
flexiv-inspire-manus-calibrate finalize-ergonomics \
  --open "$ROOT/artifacts/calibration/manus_ergonomics_open.yaml" \
  --closed "$ROOT/artifacts/calibration/manus_ergonomics_closed.yaml" \
  --template "$ROOT/ros2_ws/src/flexiv_inspire_control/config/manus_ergonomics_calibration_template.yaml" \
  --output "$ROOT/artifacts/calibration/manus_ergonomics_site.yaml"
```

检查生成文件后，把 `config/sensors.yaml` 的 `teleop.manus_calibration` 指向
`artifacts/calibration/manus_ergonomics_site.yaml`，重新 `validate`/`render`。
采集工具拒绝覆盖已有文件，重做时先保留旧文件并换新文件名。

### 4.5 控制桥和遥操作映射保持 Shadow

控制桥终端 F（整个硬件会话持续运行，不要为切换 shadow 重启它）：

```bash
source "$ROOT/scripts/env/activate_ros.sh"
source "$ROOT/ros2_ws/install/setup.bash"
flexiv-inspire-control-bridge --ros-args \
  --params-file "$CONTROL_CONFIG" \
  -p session_id:="$SESSION_ID" \
  -p rdk_socket:="$RDK_SOCKET" \
  -p foot_pedal:=/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd
```

遥操作映射终端 G：

```bash
source "$ROOT/scripts/env/activate_ros.sh"
source "$ROOT/ros2_ws/install/setup.bash"
flexiv-inspire-teleop-input --ros-args \
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

这里是在线状态查看。完整 episode 的离线 Rerun 播放以及受控真机 Replay
统一读取 `config/playback.yaml`，分别运行 `./scripts/visualize.sh` 和
`./scripts/replay.sh`；真机条件和拒绝规则见
[`PLAYBACK.md`](PLAYBACK.md)。禁止用 `ros2 bag play` 回放 episode，因为
ROS bag 包含带历史授权上下文的控制 topic，而不是一个安全的执行接口。

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

正常使用不再分别执行清零和 Home。在另一个本地 ROS TTY 直接运行：

```bash
source "$ROOT/scripts/env/activate_ros.sh"
source "$ROOT/ros2_ws/install/setup.bash"
robot reset
```

该命令自动从 `config/site.yaml` 读取 daemon socket、工具/负载配置、session
和 Home/Inspire Reset 参数，一次完成“两秒静止检查 -> 双臂 F/T 清零 ->
双臂 Home -> 双手张开/闭合/张开（最终张开）”。手部动作沿用旧工程的
`1000 -> 0 -> 1000` 和 0.35 秒间隔，且发生在 F/T 清零之后。无需设置
`RDK_SOCKET`/`TOOL_CONFIG`，也无需输入确认字符串。也可使用等价入口
`flexiv-inspire-reset`。

该命令可在同一会话内重复执行。处于 `MAINTENANCE` 时会执行 F/T 清零；若
已经处于 `READY` 且 `/control/state.ft_zeroed_for_session=true`，则复用本次
清零结果，直接执行 Home 和双手开合。其他状态仍会拒绝动作。

命令只能从机器人本机交互 TTY 发出。过程中 teleop/policy/replay 都无权
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

`robot reset` 会把本机执行该命令视为本次 Reset/Home 的许可，
并自动发布这两个门控；下面的手工命令只用于后续单独恢复门控。

```bash
ros2 topic pub --once /control/local_permission std_msgs/msg/Bool \
  '{data: true}'
ros2 topic pub --once /safety/collision_clear std_msgs/msg/Bool \
  '{data: true}'
```

物理脚踏必须由配置的 `/dev/input/by-id/...` 设备产生；不要用 ROS 话题或
软件脚本模拟脚踏。

### 6.1 单独执行配置的双臂 Home

通常不需要本节：`robot reset` 已在清零成功后执行 Home。下面的
命令仅用于同一硬件会话中不重新清零、只回 Home。

Home 只允许从 `READY` 执行，且遥操作、策略和回放均未占用控制权。它使用
`NRT_JOINT_POSITION + SendJointPosition` 移动到 `flexiv.home` 的左右
七轴目标；到位后切回笛卡尔模式并保持实测 TCP。保持现场许可和碰撞许可
为 true，然后在本机交互 TTY 执行（Home 不要求 down-arrow 脚踏）：

```bash
flexiv-inspire-home \
  --session-id "$SESSION_ID" \
  --confirm FLEXIV-HOME-MOVE \
  --socket "$RDK_SOCKET"
```

该命令生成 30 秒有效的一次性 bootstrap token，并向控制桥发出一次 Home
请求。首次执行被 daemon 接受后，会建立绑定当前控制桥进程、session 和
RDK connection generation 的 Home lease；同一硬件会话内可以直接用 Quest
右手 A 键重复触发。也可以只运行 `flexiv-inspire-authorize-home`，随后在
30 秒内按一次 A 键建立 lease。桥/daemon 断线、桥重启或重新 F/T zero 后
必须重新做本机授权。监视：

```bash
ros2 topic echo /control/home_status
ros2 topic echo /robot/left_arm/joint_states
ros2 topic echo /robot/right_arm/joint_states
```

桥/daemon 断线、安全检查失败、150 ms keepalive 超时或配置改变，都会停止
关节运动、切回实测位姿保持并锁存 daemon hold。排除原因后重新授权；若要
清除该 hold，显式增加 `--clear-hold-latched`。直接发布
`/control/home_request` 仍无法绕过专用 token 和所有现场门禁。

一键数采时，右键为“保存当前条 → Home → 下一条”，左键为“丢弃当前条 →
Home → 重录同一编号”，Quest A 为“暂停当前条 → Home → 续录同一条”。三种
操作都会先等待 ROS bag 和 DeviceIO 同时确认暂停，Home 数据不会写入
episode。完整流程见 [DATA_COLLECTION.md](DATA_COLLECTION.md)。

Home 首次真机验收应先把 `home_max_velocity_rad_s` 和
`home_max_acceleration_rad_s2` 再降到当前值的 10%，分别检查左右目标方向
和双臂空间是否干涉，确认后再逐级提高。本仓库测试只验证命令链和看门狗，
不会证明这组位姿在现场无碰撞。

### 6.2 遥操作与笛卡尔阻抗

本站配置已经启用灵巧手写通道，`robot reset` 和 `robot record` 会自动生成并
启动带本会话确认的 DFTP 参数，无需手动停止或重启节点。可检查运行时快照：

```bash
robot render
sed -n '1,80p' /run/user/1000/isaac_teleop/launcher/<config-hash>/dftp.yaml
```

正常遥操作仍只接受 supervisor 发布的 `/control/sent_command`；另有本机
`/maintenance/cycle_hands` 服务，只允许 `READY` 状态下执行 Reset 的
张开/闭合/张开动作。

在终端 G 用 `Ctrl-C` 只停止 shadow `teleop_input`，不要停止控制桥。然后
在本地交互 TTY 授权 teleop：

```bash
flexiv-inspire-authorize-control \
  --session-id "$SESSION_ID" \
  --source teleop \
  --confirm FLEXIV-CONTROL-ARM \
  --socket "$RDK_SOCKET"
```

以命令模式重新启动遥操作映射：

```bash
flexiv-inspire-teleop-input --ros-args \
  --params-file "$TELEOP_CONFIG" \
  -p manus_calibration:="$MANUS_CONFIG" \
  -p session_id:="$SESSION_ID" \
  -p command_enabled:=true
```

每个送往 RDK 的笛卡尔目标都携带当前 profile 的 `K_x` 和
`damping_ratio`。daemon 在该 profile 首次使用时调用阻塞式
`SetCartesianImpedance`，后续相同值不重复下发。切换
`cartesian_control.mode` 必须修改 `config/hardware.yaml` 后重新
`validate`/`render` 并重启控制桥；不要运行中临时发布未审计刚度。

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
flexiv-inspire-authorize-control \
  --session-id "$SESSION_ID" \
  --source teleop \
  --confirm FLEXIV-CONTROL-ARM \
  --clear-hold-latched \
  --socket "$RDK_SOCKET"
```

## 7. 正式异步录制与数据语义

日常数采只修改 `config/recording.yaml`，然后在已经 `READY` 的硬件会话本机
终端执行：

```bash
cd "$ROOT"
robot record
```

它会立即开始第一条，自动管理脚踏路由、episode 生命周期以及 Home/控制
授权；达到配置的成功条数或按 `Ctrl-C` 后退出。无需再手工启动脚踏路由、
episode controller 或逐次建立 Home 授权。启动前的 daemon/控制桥/传感器、
F/T 清零和现场许可仍属于一次硬件会话准备，不能由数采命令伪造。

`isaac-flexiv-episode` 保留为调试单条 recorder 的底层入口；它会拒绝没有
匹配 session、工具哈希和成功 `ft_zero_completed` 事件的录制：

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

旧 `dual_arm_teleop` 的 Python 3.10 环境只安装两个可移植小包，不安装本仓库
Python 3.12 主包，也不替换它现有的 LeRobot：

```bash
/home/hb/miniconda3/envs/flexiv_teleop/bin/pip install \
  -e "$ROOT/libs/policy_contracts" \
  -e "$ROOT/libs/policy_client"
```

`isaac_flexiv_rpc` 默认 `shadow_only: true`：它用 `GetSnapshot` 读取与 checkpoint
metadata 完全一致的观测，策略动作只校验和打印，绝不下发真机。P4 动作代码必须
在完成长时间 shadow 后由操作者显式设置 `shadow_only: false` 才会启用；本地
Home/Enable/F/T 清零始终由本仓库负责。

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
flexiv-inspire-authorize-control \
  --session-id "$SESSION_ID" \
  --source policy \
  --confirm FLEXIV-CONTROL-ARM \
  --clear-hold-latched \
  --socket "$RDK_SOCKET"
```

将旧 LeRobot 配置中的 `shadow_only` 显式改为 `false` 后，客户端会在第一个 action
到来时申请 lease；连接和加载 checkpoint 本身不会申请运动权限。lease、脚踏、
heartbeat、TTL 任一失效都
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
