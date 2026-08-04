# Pick/place ACT checkpoint 与推理

这套配置只服务于 `pick_place_demo` 的 10 条、15 Hz 数据，不修改
`dual_arm_teleop` 原有训练、直连推理或 checkpoint。状态/动作契约固定为
`dual_arm_lerobot_v1`：38D state、24D action 和三路 RGB。

## 1. 训练

```bash
cd /home/hb/isaac_teleop_flexiv_inspire
./scripts/policy/train_pick_place_act.sh
```

训练配置为 `config/policies/pick_place_act_train.yaml`。RTX 5070 使用 batch 8，
训练 10,000 step，每 1,000 step 保存一次。输出目录：

```text
artifacts/policies/pick_place_demo_act_v1/
  checkpoints/
    001000/ ... 010000/
    last -> 010000
```

推理始终引用：

```text
artifacts/policies/pick_place_demo_act_v1/checkpoints/last/pretrained_model
```

脚本不会覆盖已有输出。如果检测到完整的 `checkpoints/last/training_state`，再次运行
同一条命令会自动 resume；若目录存在却没有可恢复 checkpoint，则停止并要求人工
检查，不能删除一个仍需使用的 checkpoint 后从头覆盖。

## 2. Shadow 推理

先让本仓库作为唯一硬件 owner 运行，使 RDK、Inspire 和三路相机持续发布。当前
数采流程可直接使用：

```bash
cd /home/hb/isaac_teleop_flexiv_inspire
source scripts/env/activate_ros.sh
robot record
```

另开终端启动只绑定 loopback 的 TLS 策略服务：

```bash
cd /home/hb/isaac_teleop_flexiv_inspire
./scripts/policy/start_pick_place_policy_server.sh
```

第三个终端运行 checkpoint：

```bash
cd /home/hb/isaac_teleop_flexiv_inspire
./scripts/policy/run_pick_place_act_shadow.sh
```

`pick_place_rpc_robot.yaml` 默认 `shadow_only: true`。策略会以 15 Hz 获取与训练完全
同名、同顺序的观测，完成 ACT 推理、反归一化和 24D action 契约校验；终端会打印
`[ISAAC RPC SHADOW]`，但不会申请 lease 或发送动作。每次 rollout 仍会生成一份
LeRobot 记录，便于离线检查预测。

## 3. 真机前验收

至少先完成：

1. 连续 shadow 运行 10 条 episode，无 stale/schema/NaN 错误；
2. 可视化预测的左右臂 XYZ/旋转向量及双手 0..1 命令，确认方向、尺度和手序一致；
3. 从 001000、002000 等 checkpoint 中根据离线 loss 和 shadow 行为选择，而不是
   默认认为最后一步最好；
4. 再复制 RPC robot 配置，将副本的 `shadow_only` 显式改成 `false`，并在本机完成
   policy source 授权、F/T zero、Home 和中踏板测试。

真机动作仍经过本仓库的 lease、TTL、脚踏、软限位和 source-exclusive 控制。不要
把 `shadow_only: false` 写进默认 shadow 配置。
