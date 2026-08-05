# Independent Isaac Teleop for Flexiv + Inspire

This repository is a self-contained teleoperation, acquisition, visualization,
and policy-control workspace for two Flexiv arms and two Inspire DFTP-2 hands.
It owns its Python environments, ROS 2 packages, RDK daemon, hand driver,
interfaces, and data pipeline. No external robot workspace is sourced,
imported, linked, or launched.

## Daily commands

日常使用只需要 ROS 环境和顶层命令，不需要分别启动 daemon 或导出 socket、
permit、工具配置等变量：

```bash
cd /home/hb/isaac_teleop_flexiv_inspire
source scripts/env/activate_ros.sh

robot reset   # F/T 清零 -> 双臂 Home -> 双手张开/闭合/张开
robot record  # 前台启动数采；完成设定条数或 Ctrl-C 后自动结束
robot stop    # 仅用于异常退出后清理残留的托管进程
```

`robot reset` 可重复执行：`MAINTENANCE` 会进行 F/T 清零；若当前会话已经是
`READY` 且清零仍有效，则复用该结果，直接 Home 并执行双手开合。

`robot record` 启动后，双臂仅在 Quest 跟踪有效且持续踩住中间运动脚踏时响应；
松开脚踏立即停止增量控制。Inspire 手指通过同一安全命令链随 MANUS 控制。
命令会持续占用当前终端；`Ctrl-C` 先保存并关闭当前 episode，再停止本次
录制启动的支持服务，不需要随后再执行 `robot stop`。

旧命令 `flexiv-inspire` 和 `flexiv-inspire-reset` 继续保留兼容。

The hardware path is fail-closed. The DFTP executable defaults to read-only; this
installed site's hardware configuration explicitly enables its command path.
Starting services alone does not zero force/torque sensors, return home, or send
a motion target; those actions require the corresponding Reset/control flow.

## Data and control path

```text
Quest wrist position/orientation + MANUS SDK Ergonomics
                    |
                    v
teleop_input (SE(3) wrist mapping + calibrated finger retargeting)
              |
              v
teleop / policy / replay arbitration
              |
              v
Rotation-6D validation + safety supervisor
       |                         |
       | Unix SOCK_SEQPACKET     | ROS 2 safe command
       v                         v
Flexiv RDK 1.9 daemon       Inspire DFTP-2 driver
       |                         |
       +------ observations -----+
                    |
          MCAP / Rerun / gRPC
                    |
             LeRobot v3 export
```

Policy actions default to 30 values: two arms each use
`delta_xyz(3) + delta_rot6d(6)`, followed by twelve hand actuator targets.
Rotation-6D is always the first two columns of a rotation matrix in column
order:

```text
[R00, R10, R20, R01, R11, R21]
```

The identity rotation is `[1, 0, 0, 0, 1, 0]`; an all-zero vector is invalid.
ROS poses remain quaternion `xyzw`, while the RDK boundary explicitly converts
to and from `wxyz`.

## Repository layout

- `libs/control_core/`: Rotation-6D, command schema, safety state machine and clock mapping.
- `ros2_ws/src/flexiv_inspire_control/`: ROS 2 control, Quest/MANUS mapping, pedal and F/T action.
- `apps/flexiv_daemon/`: isolated Python 3.10 Flexiv RDK 1.9 daemon and typed local IPC.
- `ros2_ws/src/flexiv_inspire_dftp/`: project-owned Modbus TCP driver for DFTP-2.
- `ros2_ws/src/flexiv_inspire_cameras/`: three-camera RGB acquisition.
- `apps/policy_server/src/flexiv_inspire_isaac/policy_api/`: TLS gRPC PolicyService v1.
- `apps/policy_client/`: standalone copyable multi-rate remote policy client.
- `libs/data_core/src/flexiv_inspire_isaac/data_pipeline/`: asynchronous MCAP recording and
  deterministic LeRobot v3 export.
- `ros2_ws/src/flexiv_inspire_rerun/`: live/offline Rerun visualization.
- `ros2_ws/src/flexiv_inspire_interfaces/`: ROS 2 messages/actions.
- `libs/rpc_interfaces/proto/`: protobuf schemas.
- `scripts/env/`: four pinned environments.

Start with the exact, fail-closed startup sequence in
[`docs/RUNBOOK.md`](docs/RUNBOOK.md). Also read `docs/ENVIRONMENTS.md`,
`docs/SAFETY_INVARIANTS.md`, `docs/MANUS_SETUP.md`, and
`docs/MANUS_POSE_ADAPTER.md` before integration. The staged design for
multi-rate policy RPC and LeRobot interoperability is in
[`docs/POLICY_INTEROPERABILITY_PLAN.md`](docs/POLICY_INTEROPERABILITY_PLAN.md).

## Environments

The machine-wide CUDA selection is intentionally untouched. Isaac commands
select `/usr/local/cuda-12.8` only inside their launcher environment. ROS 2 uses
CycloneDDS with a project-local configuration and remains localhost-only by
default.

```bash
cd /home/hb/isaac_teleop_flexiv_inspire
./scripts/verify_host_cuda.sh
./scripts/env/create_envs.sh
./scripts/env/smoke_switch_isolation.sh
./scripts/verify_independence.sh
```

The four environments are:

- `envs/isaac-py312`: Isaac Teleop 1.3.131, CloudXR/MANUS, CUDA 12.8.
- `envs/ros-py312`: ROS bridge, gRPC, Rerun, and acquisition management.
- `envs/rdk-py310`: exactly `flexivrdk==1.9.0`, with no ROS import.
- `envs/data-py312`: LeRobot 0.6.0 and offline MCAP conversion.

Canonical protobuf sources are under `libs/rpc_interfaces/proto`; regenerate checked-in
Python stubs with:

```bash
./scripts/generate_protos.sh
```

## Build and software-only validation

```bash
export PROJECT_ROOT=/home/hb/isaac_teleop_flexiv_inspire
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/env/activate_ros.sh"
colcon --log-base "$PROJECT_ROOT/ros2_ws/log" build \
  --base-paths "$PROJECT_ROOT/ros2_ws/src" \
  --build-base "$PROJECT_ROOT/ros2_ws/build" \
  --install-base "$PROJECT_ROOT/ros2_ws/install" \
  --symlink-install \
  --cmake-args -DPython3_EXECUTABLE="$PROJECT_ROOT/envs/ros-py312/bin/python3"

source "$PROJECT_ROOT/ros2_ws/install/setup.bash"
"$PROJECT_ROOT/envs/ros-py312/bin/python" -m pytest
"$PROJECT_ROOT/scripts/verify_independence.sh"
```

Mock and static tests never issue hardware commands. The control bridge starts
the hardware session in `MAINTENANCE`; the DFTP driver starts read-only; the RDK daemon requires a
short-lived local write permit before accepting a command session.

## Hardware-session gate

Every new hardware session must complete the protected local
`/maintenance/zero_ft_sensors` action before `READY`. The operator must verify
the configured tool/payload, keep both arms, hands, and cables unloaded and
still, and enter the literal local confirmation
`FLEXIV-FT-UNLOADED`. The action records pre/post statistics and fails the
whole session on motion, contact, residual error, timeout, primitive failure,
or reconnect. Remote gRPC clients cannot enable hardware or initiate/confirm
this maintenance action.

After successful zeroing, actual motion still requires local authorization, a
physical pedal, a current source heartbeat/deadman, valid TTL and timestamps,
online arms and hands, and all safety checks. Any failure enters a latched hold.
Follow the staged acceptance sequence in `docs/SAFETY_INVARIANTS.md`; do not
skip directly to dual-arm motion.

## One-command collection

After the persistent hardware/control stack is connected, F/T-zeroed and
`READY`, daily collection is configured only in `config/recording.yaml` and
started without command-line parameters:

```bash
robot record
```

Recording starts automatically. Right pedal commits, Homes and advances; left
pedal discards, Homes and retries the same index; Quest A pauses both MCAP
paths during Home and resumes the same episode. The process exits at the
configured successful-episode count or on `Ctrl-C`. See
[`docs/DATA_COLLECTION.md`](docs/DATA_COLLECTION.md) for the exact workflow and
output layout.

`config/site.yaml` is a small composed entry point over hardware, sensors,
recording and runtime fragments. Generated runtime snapshots are not operator
configuration and no `artifacts/site/` directory is required.

Dataset playback is configured separately in `config/playback.yaml`:

```bash
./scripts/visualize.sh  # offline DeviceIO MCAP -> Rerun; robot never moves
./scripts/replay.sh     # guarded, timing-faithful execution on the robot
```

Visualization has no ROS publishers or hardware connections. Hardware replay
is disabled by default, accepts only recorded `sent_command` samples, Homes
first, and remains behind the existing local authorization, physical pedal,
F/T-zero, collision, health, limit, freshness, and source-exclusivity gates.
See [`docs/PLAYBACK.md`](docs/PLAYBACK.md).

## Recording, visualization, and policy

The low-level recording entry point is `isaac-flexiv-episode`; it requires
a successful F/T-zero event matching the session and tool hash, then writes an
atomic manifest, ROS MCAP, and native asynchronous DeviceIO MCAP. The exact
command is in `docs/RUNBOOK.md`. `scripts/record_ros_mcap.sh` is explicitly a
ROS-only diagnostic capture and does not create a valid training episode.

Storage semantics are explicit: `ros_mcap/` contains DDS CDR messages recorded
by rosbag2, while `deviceio.mcap` defaults to `native-pre-dds`. RDK observations,
DFTP state/tactile, camera frames, and requested/safe/sent/control traces enter a
private mode-0600 Unix datagram ingress before DDS publication. The manifest
records the capture layer and producer/drop statistics. A post-DDS typed mirror
remains available only as an explicit compatibility mode.

Each typed sample retains source time, host receive time, sequence, validity,
and age. Images, force/torque, and tactile samples remain asynchronous and are
aligned only during export. Rotation-6D is recomputed from the interpolated
rotation matrix rather than linearly interpolated.

The policy server binds to loopback by default. A non-loopback endpoint must be
enabled explicitly with TLS credentials and still cannot bypass the local
hardware gate. Rerun consumes the same ROS observations and shows camera
images, poses, joint/torque/wrench data, tactile heatmaps, and
requested-versus-safe-versus-sent commands.
