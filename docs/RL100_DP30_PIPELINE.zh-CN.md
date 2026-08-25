# RL-100 双臂三视角 30 Hz MVP

本文描述 Flexiv 双臂、两只 Inspire 六维手、三路 RGB 与 RL-100 之间已经实现的
最小闭环。目标是先可靠跑通：Quest/MANUS 遥操数采 -> 原始 MCAP -> DP Zarr ->
RL-100 行为克隆（BC）-> 离线回放检查 -> 影子推理。真实策略下发保留了接口和
安全门，但在真机验收前不应开启。

## 1. 已实现与未实现

已经实现：

- 使用现有 `robot record` 采集双臂、双手、三路 RGB、头部深度/点云和触觉；
- 可选的 `site_rl100_dp30.yaml` 将训练主时间轴、三路 RGB 和遥操目标设为 30 Hz；
- Quest 原始追踪、机械臂/手状态和底层伺服仍保持各自高频率；
- 以真实通过安全链并下发的 `/control/sent_command` 为动作时间轴；
- 从多个完整 episode 的 MCAP 合并导出 RL-100 Zarr；
- 26D 状态、24D 动作、三路 CHW RGB 的严格维度和语义检查；
- 按松开中踏板产生的时间空档切分训练 episode，不在空档中插值造帧；
- 通过显式人工成功/失败标签生成离线 RL 所需 reward/done/return；
- Policy RPC 的只读观测、影子推理和带 lease、TTL、踏板、安全监督的动作接口。

当前明确不包含：

- 触觉作为首版 DP 输入；触觉仍完整保留在原始 MCAP；
- 深度或融合点云作为首版 DP 输入；头部深度/点云仍保留在 MCAP；
- 三相机点云融合；三路相机在 MVP 中作为三个独立 RGB 视角；
- 自动生成任务奖励。遥操作数据没有天然 reward，必须人工或由后续任务判定器标注；
- 已验收的真机策略运动。代码默认影子模式，真实执行必须单独完成分阶段验收；
- 可直接运行的在线 RL 环境。在线 RL 还需要任务 reset、成功检测、超时和故障恢复。

原始 MCAP 是不可替代的数据源，Zarr 是按某个训练契约生成的派生数据。以后加入
触觉、深度、DP3 或新的状态定义时，应从 MCAP 重新转换，不应覆盖原始数据。

## 2. 进程与环境边界

本仓库是唯一硬件 owner。它负责 Flexiv RDK、Inspire、RealSense、Quest/MANUS、
脚踏、安全监督、MCAP 和 Policy RPC。RL-100 只读取 Zarr 或通过 RPC 读取观测，
不能直接打开 RDK、Modbus 或相机。

数采环境不是 Conda，而是仓库自带的虚拟环境：

```bash
cd /home/hb/isaac_teleop_flexiv_inspire
source scripts/env/activate_ros.sh
```

数据转换命令由 `robot convert` 自动调用 `envs/data-py312`。RL-100 训练使用独立的
Conda 环境 `RL100`，避免 ROS 与训练依赖互相污染。

## 3. 频率设计

选择 `config/site_rl100_dp30.yaml` 后：

| 信号 | 频率 | 用途 |
|---|---:|---|
| Quest 原始追踪 | 60 Hz | 保留设备原始高频输入 |
| Quest/MANUS 目标命令 | 30 Hz | 与 DP 动作时间步一致 |
| 三路 RGB | 30 Hz | DP 的三个独立视觉输入 |
| DP/Zarr 时间轴 | 30 Hz | 使用实际 sent_command，不重采样 |
| 双臂状态 | 300 Hz | MCAP 原始观测，转换时取最新因果值 |
| 双手非触觉状态 | 200 Hz | MCAP 原始观测，转换时取最新因果值 |
| 触觉 | 15 Hz | 原始保留，首版 DP 不使用 |
| 机器人底层伺服 | 原有高频 | 不由 DP 数据集频率替代 |

这里的 30 Hz 是策略数据和目标生成频率，不是把机器人伺服降为 30 Hz。高频状态
继续记录，转换器在每个动作时刻只选择时间不晚于该动作的最新观测和图像，防止
把未来信息泄漏给训练样本。

## 4. 状态与动作契约

Profile 固定为 `joint_proprio_cartesian_v1`。

### 4.1 26D 状态

```text
[left_arm_q(7), right_arm_q(7), left_hand_measured(6), right_hand_measured(6)]
```

- 机械臂部分是实际测量关节位置，单位为 rad；
- 手部部分来自 Inspire 的六个 actuator angle，并从原生 `[0,1000]` 归一化到
  `[0,1]`；
- 状态不重复加入末端位姿。末端位姿可以由 7 个关节和运动学模型计算，首版用
  关节本体状态可避免冗余且与现有 RPC profile 一致。

### 4.2 24D 策略动作

```text
[left_delta_xyz(3), left_delta_rotvec(3),
 right_delta_xyz(3), right_delta_rotvec(3),
 left_hand_target(6), right_hand_target(6)]
```

- 双臂动作是 world 坐标系中的末端相对位移和旋转向量；
- 双手动作是归一化 `[0,1]` 的绝对目标，不是手指增量；
- 训练标签只取实际被安全链接受并发送给硬件的 `sent_command`，不使用尚未通过
  安全检查的 requested/safe 中间值。

当前控制桥的原生命令是 30D：每只机械臂为 `xyz(3)+Rotation-6D(6)`，双手为
原生目标 12D，即 `9+9+12=30`。转换关系为：

```text
MCAP 原生 30D -> 每臂 Rotation-6D 转 rotvec -> 策略 24D
策略 24D -> 每臂 rotvec 转 Rotation-6D、手目标乘 1000 -> 真机原生 30D
```

30D 是当前 Flexiv/Inspire 控制系统的 canonical 原生格式，不是 RL-100 上游仓库的
固定格式。两边共用同一份 `policy_contracts` 映射，非法旋转、NaN/Inf 或越界手目标
会拒绝整条双臂命令。

## 5. 现场数采

先只做配置检查和启动预览，不连接或移动机器人：

```bash
cd /home/hb/isaac_teleop_flexiv_inspire
source scripts/env/activate_ros.sh
robot --config config/site_rl100_dp30.yaml validate
robot --config config/site_rl100_dp30.yaml record --dry-run
```

第一次现场验收建议使用独立的一条轨迹 smoke 配置。它继承相同硬件、标定、
30 Hz 与安全设置，但固定写入 `sessions/rl100_dp30_smoke`，保存一条 episode 后
自动退出，不会混入正式任务数据：

```bash
robot --config config/site_rl100_dp30_smoke.yaml validate
robot --config config/site_rl100_dp30_smoke.yaml record --dry-run --no-xr
# 下面这条会执行 F/T 清零和配置的 Home，只能在现场确认安全后运行。
robot --config config/site_rl100_dp30_smoke.yaml record --no-xr
```

至少持续踩住中踏板并缓慢操作双臂和双手 5 秒，再松开中踏板并踩右踏板提交。
对应的 smoke 数据转换命令为：

```bash
robot --config config/site_rl100_dp30_smoke.yaml convert \
  --conversion-config config/conversion_rl100_dp30_smoke.yaml
```

转换通过后再编辑 `config/recording.yaml` 中的正式任务名、描述和 episode 数量，
使用下面的正式 profile 采集训练示教。

真机现场确认安全后才运行：

```bash
robot --config config/site_rl100_dp30.yaml record
```

录制期间脚踏语义保持现有实现：

| 输入 | 行为 |
|---|---|
| 中踏板按住 | 允许 Quest 遥操并写训练模态 |
| 中踏板松开 | 立即 hold、暂停训练模态；再次踩下从当前姿态接管 |
| 左踏板 | 丢弃当前 attempt，受控 Home 后重录同一编号 |
| 右踏板 | 提交当前 episode，受控 Home 后进入下一条 |
| Quest A | 暂停/恢复当前 episode，并执行受控 Home |
| `Ctrl-C` | 完成本条写入并退出，等待 MCAP 落盘 |

`record_only_while_pedal_pressed: true` 会在松开中踏板时留下真实时间空档。转换器把
超过 0.05 s 的空档或 capture segment 变化作为 episode 边界；不足 9 帧的短片段被
丢弃并计入验证报告。这样不会让模型跨过 Home、暂停或重新接管学习虚假连续动作。

## 6. 转为 RL-100 Zarr

示例配置默认读取 `sessions/open_boxes_first_try` 下所有已完成 episode：

```bash
robot --config config/site_rl100_dp30.yaml convert \
  --conversion-config config/conversion_rl100_dp30.yaml
```

也可用 `--manifest` 只转换一条，或用 `--output-root` 指定新的空目录。转换器拒绝：

- 未完成的 manifest；
- 缺失任一路 RGB、状态或 sent_command；
- 未来图像匹配、30 Hz 人工插值、非法动作或图像尺寸中途变化；
- sent_command 原生中位周期偏离 30 Hz 超过 25%；
- 覆盖已有非空 Zarr；
- 低于最小长度的连续片段。

默认输出：

```text
sessions/open_boxes_first_try/rl100/joint_proprio_cartesian_v1.zarr
```

Zarr v2 schema：

| 路径 | shape / dtype | 含义 |
|---|---|---|
| `data/state` | `[N,26] float32` | 26D 本体状态 |
| `data/action` | `[N,24] float32` | 24D 已执行动作 |
| `data/rgb_head` | `[N,3,H,W] uint8` | 头部 RGB |
| `data/rgb_left_wrist` | `[N,3,H,W] uint8` | 左腕 RGB |
| `data/rgb_right_wrist` | `[N,3,H,W] uint8` | 右腕 RGB |
| `data/timestamp_ns` | `[N] int64` | sent_command 单调时钟时间戳 |
| `data/source_episode_index` | `[N] int32` | 原始 manifest 索引 |
| `data/capture_segment` | `[N] int32` | 原始采集片段编号 |
| `meta/episode_ends` | `[E] int64` | RL-100 episode 累积结束下标 |
| `meta/episode_source_index` | `[E] int32` | 每个导出 episode 的来源 |

根属性保存 schema/profile、字段名、频率、对齐语义和来源 manifest。目录中的
`export_validation.json` 记录写入帧数及每类丢弃原因。转换完成后必须先检查该报告，
再交给训练。

## 7. 奖励与离线 RL

刚转换出的示教 Zarr 只能用于 BC：`reward_available=false`、
`offline_rl_ready=false`。不要把所有示教默认标成成功后直接声称已经具备离线 RL
数据；失败样本、任务结束和奖励定义必须真实可靠。

复制并填写 `config/rl100_reward_labels.example.json`，每个导出 episode 必须恰好有
一个布尔 success 标签，然后运行：

```bash
source scripts/env/activate_data.sh
flexiv-inspire-rl100-reward-label \
  --zarr sessions/open_boxes_first_try/rl100/joint_proprio_cartesian_v1.zarr \
  --labels /absolute/path/to/reward_labels.json
```

该命令生成稀疏终止奖励的 `data/reward`、`data/done`、`data/return`，并将
`offline_rl_ready` 设为 true。标签不完整、重复、越界或已有奖励数组时会失败；只有
明确需要替换并重新审核时才使用 `--overwrite`。

## 8. 与 RL-100 的交接

RL-100 仓库内的 `FLEXIV_DP30.zh-CN.md` 给出环境、训练和推理命令。BC 训练只需
上述核心数组，不要求 reward。离线 RL 数据加载必须显式启用 transitions，并会检查
奖励数组和 `offline_rl_ready`。

策略部署仍由本仓库启动唯一硬件 server：

```bash
robot --config config/site_rl100_dp30.yaml policy-serve --no-reset
```

`--no-reset` 只适合不允许运动的影子诊断。正式运动前需去掉它并在现场完成 Reset、
Home、方向、单臂小增量、双臂、双手、踏板、TTL、断连和急停验收。

30 Hz profile 强制 `policy_control.require_pedal: true`。即使 RL-100 客户端使用真实执行
参数，server 仍要求当前 session、本地授权、物理中踏板、lease、deadman、新鲜 TTL、
健康硬件和有效动作。客户端退出、断网、动作过期或踏板松开都会进入 hold。

## 9. 验收顺序

1. `validate` 与 `record --dry-run` 通过，确认渲染频率和设备序列号。
2. 现场录一条短 episode，只做 `visualize`，检查三路 RGB、双臂/双手和动作。
3. 转换 Zarr，核对 `export_validation.json`、数组维度和 episode 边界。
4. 用 RL-100 数据集测试和少量 batch 跑通 BC，不启动 RPC。
5. 使用已训练 checkpoint 对 Zarr 离线回放，检查输出有限值、手目标范围和误差。
6. 启动 `policy-serve --no-reset` 与 RL-100 shadow，确认 30 Hz 稳态、无动作下发。
7. 现场隔离工作区后，按单臂平移、单臂旋转、双臂、双手逐级验收真实执行。
8. 奖励判定与自动 reset 单独验收后，才进入离线 RL/在线 RL。

任何一步失败都应保留原始 MCAP、日志和验证报告，在该层修复后重试，不应绕过
schema、踏板、lease、TTL 或本地安全门。
