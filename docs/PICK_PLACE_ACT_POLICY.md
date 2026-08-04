# Pick/place 数据与策略 RPC 边界

两个仓库保持明确的 server/client 分工：

- `isaac_teleop_flexiv_inspire`：采集、转换后的 LeRobot dataset、Flexiv RDK、
  Inspire、相机、脚踏、安全监督器和 Policy RPC server；它是唯一硬件 owner。
- `dual_arm_teleop`：LeRobot train/reason 配置、checkpoint、RPC Robot client 和
  policy rollout；它不直接打开 RDK、Modbus 或 RealSense。

## 1. 数据（server 仓库）

数据集保持在：

```text
/home/hb/isaac_teleop_flexiv_inspire/sessions/pick_place_demo/lerobot_merged/dual_arm_lerobot_v1
```

重新转换新版本时仍在本仓库运行：

```bash
source scripts/env/activate_ros.sh
robot convert --conversion-config config/conversion_dual_arm.yaml
```

## 2. 训练与 checkpoint（client 仓库）

专属配置位于：

```text
/home/hb/flexiv_inspire_ws/src/dual_arm_teleop/scripts/config/experiments/pick_place_act_v1/
```

从 client 仓库训练：

```bash
cd /home/hb/flexiv_inspire_ws/src/dual_arm_teleop
conda activate flexiv_teleop
robot-train --config scripts/config/experiments/pick_place_act_v1/train_cfg.yaml
```

checkpoint 只写入 client 仓库：

```text
/home/hb/flexiv_inspire_ws/src/dual_arm_teleop/outputs/train/pick_place_demo_act_v1/
```

## 3. RPC server（本仓库）

```bash
cd /home/hb/isaac_teleop_flexiv_inspire
source scripts/env/activate_ros.sh
robot policy-serve
```

该命令以前台方式运行，默认先执行本机 Reset，然后启动 RDK、控制桥、Inspire、
三路相机、脚踏和 loopback TLS Policy RPC。它不会启动 Quest、MANUS、Episode
Controller、LeRobot 或训练/推理程序。`Ctrl-C` 关闭 server 及其硬件服务。

只做不允许运动的 shadow 诊断且不希望自动 Home 时，可以使用：

```bash
robot policy-serve --no-reset
```

RPC 地址、证书和频率在 `config/policy_server.yaml`。非 loopback 监听必须配置
client CA 并使用双向 TLS。

## 4. 推理（client 仓库）

server 就绪后，在另一个终端运行：

```bash
cd /home/hb/flexiv_inspire_ws/src/dual_arm_teleop
conda activate flexiv_teleop
robot-record --config scripts/config/experiments/pick_place_act_v1/run_policy_shadow.yaml
```

默认配置为 `shadow_only: true`：client 读取 RPC observation、加载本仓库数据训练出的
checkpoint、执行 ACT 并验证 24D action，但不申请 lease、不向真机发送动作。

真机模式必须在完成 shadow 验收后使用单独的 guarded 配置；server 端还必须由本机
TTY 授权 policy source，并保持中踏板按下：

```bash
flexiv-inspire-authorize-control \
  --session-id flexiv-inspire-site \
  --source policy \
  --confirm FLEXIV-CONTROL-ARM \
  --clear-hold-latched \
  --socket /run/user/1000/isaac_teleop/rdk.sock
```

然后启动 `run_policy_guarded.yaml`。动作继续经过 lease、TTL、heartbeat、脚踏、
软限位和 source-exclusive 控制。
