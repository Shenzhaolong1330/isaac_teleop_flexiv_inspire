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
Controller、LeRobot 或训练/推理程序。本机配置为 policy 直连模式：前台进程自动完成
policy 授权，不需要中踏板或单独运行 authorize-control。Quest 遥操的踏板逻辑不受影响。
`Ctrl-C` 关闭 server 及其硬件服务。

只做不允许运动的 shadow 诊断且不希望自动 Home 时，可以使用：

```bash
robot policy-serve --no-reset
```

RPC 地址、证书和频率在 `config/policy_server.yaml`。非 loopback 监听必须配置
client CA 并使用双向 TLS。

## 4. 推理 client（client 仓库）

server 就绪后，在另一个终端运行：

```bash
cd /home/hb/flexiv_inspire_ws/src/dual_arm_teleop
conda activate flexiv_teleop
robot-record --config scripts/config/experiments/pick_place_act_v1/run_policy_shadow.yaml
```

默认配置为 `shadow_only: true`：client 读取 RPC observation、加载本仓库数据训练出的
checkpoint、执行 ACT 并验证 24D action，但不申请 lease、不向真机发送动作。

真机模式使用单独的 `run_policy_guarded.yaml`。Server 完成自动授权后，Client 每发送
一帧 action，Server 就立即映射和下发，不需要踏板或逐帧授权。动作仍经过
lease、TTL、heartbeat、软限位和
source-exclusive 控制。

遥操与策略使用不同的来源 topic，随后进入完全相同的真机执行链：

```text
/command_sources/teleop/command ─┐
                                 ├─> requested -> safe -> sent -> RDK
/command_sources/policy/command ─┘
```

真机推理只由 `policy-serve` 创建一个 Rerun Viewer，显示映射后的 30D
requested/safe/sent action；guarded Client 禁止自行 spawn 第二个 Viewer。控制状态中的
`active_source` 标识 `teleop` 或 `policy`。Shadow Client 不控制真机，可以单独显示
checkpoint 的 24D action。
