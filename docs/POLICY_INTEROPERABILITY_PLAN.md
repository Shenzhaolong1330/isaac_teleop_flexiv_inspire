# 通用策略 RPC 与 LeRobot 互操作实施计划

## 1. 目标和当前结论

本计划解决两个问题：

1. 为不同训练/推理框架提供带时间戳、可同步或异步读取的通用数据接口；
2. 让本仓库采集的数据可由
   `/home/hb/flexiv_inspire_ws/src/dual_arm_teleop` 的 LeRobot 流程直接训练，
   并让该流程在推理时通过本仓库唯一控制真机。

现有 `PolicyService v1` 保留。它提供安全 lease、聚合观测和固定 30 维动作，
适合作为已有策略控制接口，但不适合表达相机 15 Hz、双臂状态 200 Hz、动作
30 Hz 等独立频率，也不能描述不同机器人或策略的动态 feature schema。

`dual_arm_teleop` 已经能从 LeRobot `meta/info.json` 自动获得策略的输入输出
feature；真正缺失的是两层适配：

- 数据层：把本仓库的带时间戳原生通道导出成策略数据集声明的 feature；
- 运行时层：实现一个只通过 RPC 读写的 LeRobot `Robot`，禁止再次直连 RDK、
  Inspire 或 RealSense。

### 1.1 实施硬约束

后续实现必须同时满足以下三条，任一不满足都不能合并：

1. **默认路径零回归**：现有 `robot record/reset/replay/visualize/convert`、
   `PolicyService v1`，以及 `dual_arm_teleop` 里直连 Flexiv 的老版数采、训练和
   推理继续可用；新功能只通过新 profile、新 robot type 或新 v2 endpoint 启用。
2. **模块之间只依赖契约**：schema/映射是无 ROS、无 gRPC、无 LeRobot 依赖的
   纯模块；transport、离线 exporter、LeRobot adapter 和硬件驱动分层，不能互相
   import 具体实现。
3. **策略张量不携带重复表达**：设备原始通道可以完整保留，但一个具体 policy
   profile 只能选择一种关节/笛卡尔状态和一种旋转表达；同一个 state/action 中
   不同时放 quaternion、Rotation-6D、rotation vector，也不默认同时放可由关节
   正运动学得到的 TCP pose。

兼容旧 checkpoint 是第 3 条唯一允许的例外：legacy profile 必须原样生成它训练时
使用的 38 维 state 和 24 维 action，但这个例外不能成为新 profile 的默认设计。

## 2. 目标结构

```text
Flexiv / Inspire / Cameras
            |
            v
本仓库 ROS + 本地安全监督器（唯一硬件所有者）
      |                         |
      | 多频率原生通道           | canonical action + lease/TTL
      v                         v
Data/Policy RPC v2        现有安全控制路径
      |
      +---- async Subscribe: vision 15 / arm 200 / hand+tactile 15 Hz
      +---- sync GetSnapshot: 按策略时间线对齐
      +---- schema negotiation: 名称、形状、单位、坐标系、频率、哈希
      |
      +---- offline LeRobot profile exporter
      +---- online IsaacRpcRobot adapter
                         |
                         v
                 dual_arm_teleop / LeRobot
```

原则是“一个硬件所有者，多个策略客户端”。`dual_arm_teleop` 的现有
`FlexivDualArm` 仍可单独使用，但与本系统联机时必须改用 RPC Robot，不能同时
占用机械臂、手和相机。

模块边界固定为：

| 模块 | 职责 | 禁止依赖 |
|---|---|---|
| `policy_contracts` | descriptor、schema hash、FeatureContract、纯数学映射 | ROS、gRPC、LeRobot、硬件 SDK |
| `rpc_interfaces` | protobuf wire contract | ROS、硬件 SDK |
| `policy_server` | ROS channel 与 RPC transport 适配 | LeRobot |
| `data_core` profile | MCAP 到 FeatureContract 的离线转换 | 真机硬件 SDK |
| `IsaacFlexivRpcRobot` | LeRobot Robot API 到 RPC client 的适配 | RDK、Inspire、RealSense SDK |

新机器人或灵巧手只新增 device descriptor 和 mapping profile，不修改 RPC
transport、安全监督器或 LeRobot 训练核心。

## 3. RPC v2 设计

新增 `libs/rpc_interfaces/proto/policy_data_v2.proto`，不修改 v1 wire contract。
第一版只实现以下接口：

| RPC | 用途 |
|---|---|
| `DescribeSystem` | 返回 channel/action schema、设备、单位、frame、原生频率和 schema hash |
| `SubscribeSamples` | 异步 server stream；客户端可按 channel 选择频率和 drop policy |
| `GetSnapshot` | 同步 unary；按目标时间返回一组已对齐的 channel |
| `StreamActions` | 带 action schema hash、序列、TTL 和未来执行 offset 的动作流 |

`AcquireControlLease`、`ReleaseControlLease` 和 `Stop` 的安全语义沿用 v1。v2
动作最终仍转换为本仓库的 canonical command，并经过本地授权、脚踏、互斥、
软限位、TTL 和 hold；RPC 不增加远程 Enable、F/T 清零或 Home 权限。

### 3.1 Channel schema

每个 channel descriptor 至少包含：

- 稳定 ID 和语义名，例如 `arm.left.state`、`camera.head.rgb`；
- dtype、shape、轴/元素名、单位和坐标系；
- 原生频率、时钟域、允许的编码；
- schema version 和内容哈希。

RPC 不定义一个包含“所有东西”的大 state。`arm.q`、`arm.tcp_pose`、`hand.angle`、
`force_torque`、`tactile` 和图像都是可独立订阅的原子 channel；FeatureContract
只请求当前策略真正使用的集合。时间戳、validity、age 和 sequence 位于 envelope
metadata，不重复拼进 policy state tensor。

每个 sample envelope 包含 source time、mapped host monotonic time、receive time、
sequence、validity、age 和 payload。Tensor 使用 little-endian packed bytes；图像
首版支持 JPEG/RGB8，避免 `repeated double` 和图像阻塞高频状态流。

客户端应分别打开状态、图像和触觉 subscription。每个 subscription 使用独立
bounded queue：状态可选 reliable，图像/触觉默认 drop-oldest。这样慢图像消费者
不会阻塞 200 Hz 本体状态。

### 3.2 同步语义

服务端为每个 channel 保留短时 ring buffer。`GetSnapshot` 必须显式指定：

- 目标 host-monotonic 时间；
- channel 列表；
- `LATEST_CAUSAL`、`NEAREST` 或 `INTERPOLATE`；
- 各 channel 的最大 age/tolerance。

关节/位置可线性插值，姿态必须在 SO(3) 上做 SLERP，图像和触觉不插值。
在线推理默认 `LATEST_CAUSAL`，防止使用目标时刻之后的数据；离线相机对齐可以
显式使用 `NEAREST`。超时或过旧返回 `valid=false`，禁止无标记地复制陈旧数据。

### 3.3 动作频率

动作频率属于 action schema，不与视觉频率绑定。例如策略可以每 15 Hz 产生一个
包含两个未来点的 chunk，由控制端按 30 Hz 执行。

相对位姿动作不能通过重复同一个非零 delta 来“补到 30 Hz”，否则位移会翻倍。
需要升频时，应在 SE(3) 上拆分：平移除以点数，旋转使用
`exp(log(dR) / N)`，确保 N 个增量复合后等于原动作；双手目标只做带时间戳的
插值。每个动作序列只消费一次。

## 4. LeRobot schema 适配

### 4.1 策略 state/action 的最小化规则

原始 MCAP 和 RPC channel 保留可复现所需的完整数据；policy dataset/profile 则采用
白名单，未声明的字段不会进入 `observation.state` 或 `action`。首批 profile 为：

| Profile | State | Action | 用途 |
|---|---|---|---|
| `dual_arm_lerobot_v1` | legacy 38D | legacy 24D | 兼容既有数据和 checkpoint，不改变任何字段 |
| `joint_proprio_cartesian_v1` | `q(14) + hand_angle(12)` = 26D | `delta_xyz + delta_rotvec` 双臂 + hand = 24D | 推荐新通用 profile |
| `cartesian_proprio_v1` | `tcp_xyz + tcp_rotvec` 双臂 + hand = 24D | 同上 24D | 明确需要笛卡尔 proprio 的策略 |

新 profile 不同时包含 `q` 和 TCP pose。力、触觉、深度或高频 history 只有在策略
明确声明时才加入专用 profile；加入 history 时不能再把同一时刻状态复制成另一组
等价 feature。控制内部现有 30D Rotation-6D command 保持不变，它是兼容且安全的
wire representation；新策略的最小 24D action 只在边界转换一次。

### 4.2 已确认的两套 schema

本仓库当前 `sent_command` 数据：

- `observation.arm_pose`: 18 维，双臂 XYZ + Rotation-6D；
- `observation.hand_state`: 12 维，Inspire 原始角度 0..1000；
- `action`: 30 维，双臂 XYZ + Rotation-6D 增量 + 双手 0..1000；
- 图像键为 `observation.images.head/left_wrist/right_wrist`；
- 还可提供关节、力矩、力传感、触觉、深度和 200 Hz arm history。

`dual_arm_teleop` 的既有 `flexiv_dual_arm` 数据：

- `observation.state`: 38 维，14 关节 + 双臂 XYZ/旋转向量 + 双手 0..1；
- `action`: 24 维，双臂 XYZ/旋转向量增量 + 双手 0..1；
- 图像键为 `observation.images.*_image`。

因此不能按向量长度猜映射。新增版本化 profile
`dual_arm_lerobot_v1`，转换关系固定如下：

| 本仓库语义 | LeRobot 目标 | 变换 |
|---|---|---|
| 双臂 `q` | `observation.state` 的 14 个 joint 字段 | 按 left 1..7、right 1..7 重排 |
| world TCP XYZ + quaternion/rot6d | 两组 `ee_pose.{x..rz}` | Rotation-6D 解码为 SO(3)，再取 rotation vector |
| Inspire angle，顺序 little/ring/middle/index/thumb_bend/thumb_rotate | `hand_state_0..5` | 保持通道顺序并除以 1000 |
| world `delta_xyz + delta_rot6d` | `delta_ee_pose.{x..rz}` | `rotvec = log(dR)`；保持 world-frame 左乘语义 |
| hand target 0..1000 | `hand_cmd_0..5` | 保持通道顺序并除以 1000 |
| `images.head/left_wrist/right_wrist` | `images.head_image/left_wrist_image/right_wrist_image` | 只重命名，保持 RGB HWC 424x240 |

旋转转换和 `dual_arm_teleop` 的 `_pose7_to_pose6`、
`_apply_delta_to_pose7` 做 golden parity test。手部转换必须验证 1000=张开、0=闭合，
以及每个单独通道的真机方向，不能只验证全开/全闭。

### 4.3 FeatureContract

新增 `FeatureContract`：从目标数据集 `meta/info.json` 或 checkpoint 保存的 dataset
metadata 读取有序 feature、dtype、shape、元素名和 fps，然后与 RPC
`DescribeSystem` 返回的语义 schema 编译映射。

自动映射的含义是“根据名字和已注册的语义转换规则自动生成”，不是根据 24、30、
38 这些维度猜测。缺字段、单位不一致、frame 不一致或映射有歧义时直接拒绝启动。
编译结果、源/目标 schema hash、标定 hash 和 tool payload hash写入 inference run
manifest，保证可复现。

FeatureContract 输出必须与目标 metadata **精确相等**：不缺字段，也不附带额外
state/action 字段。训练、离线推理和在线推理共用同一个已编译 mapping，禁止三处
各写一套通道顺序。

## 5. 离线训练适配

在本仓库 exporter 中增加 `profile: dual_arm_lerobot_v1` 和批量合并输出：

```yaml
lerobot_export:
  profile: dual_arm_lerobot_v1
  output_dataset: sessions/pick_place_demo/lerobot_merged
  timeline:
    fps: 15
```

实现后，`robot convert` 应把所有有效 raw episode 写成一个标准 LeRobot v3
dataset，而不是每个 episode 一个独立 dataset。任务 description、episode 边界、
validity 和时间戳继续保留在 metadata/manifest。`dual_arm_lerobot_v1` 只生成 legacy
38 维状态、24 维动作和三路图像，不附加力、触觉、高频 history 等 policy feature；
需要丰富模态时必须选择另一个显式 profile，避免旧策略把额外字段误当输入。

`dual_arm_teleop/scripts/core/run_train.py` 已通过 `dataset.meta` 推导 policy
features，因此训练配置只需指向这个 dataset root，不再手写 observation/action：

```yaml
train:
  dataset:
    root: /home/hb/isaac_teleop_flexiv_inspire/sessions/pick_place_demo/lerobot_merged
```

数据集 fps 必须是真实策略时间线。当前新数据是 15 Hz；旧 30 Hz 数据不能在未显式
重采样和记录 provenance 的情况下直接混合。200 Hz 本体状态默认只用于同步时选择
最新有效样本，不自动复制为 history；需要时序 proprio 的策略必须选择专用 history
profile。需要原生异步训练的策略可直接读取 RPC/MCAP channel，不强行塞进标准
LeRobot 单时间线。

## 6. 在线推理适配

在 `dual_arm_teleop` 新增 `IsaacFlexivRpcRobot(Robot)`：

- `connect()`：连接 RPC、读取 system schema、加载 checkpoint/dataset
  `FeatureContract` 并编译映射；
- `observation_features` / `action_features`：由 contract 动态生成；
- `get_observation()`：按 dataset fps 调用 `GetSnapshot`，生成与训练数据完全同名、
  同顺序的字段；
- `send_action()`：将策略动作转换到 canonical 30 维动作并发送 lease/TTL chunk；
- `disconnect()`：释放 lease 并请求 stop，但不关闭 RDK、相机或手驱动。

策略运行频率从训练数据 metadata 读取。15 Hz 模型以 15 Hz 消费动作；30 Hz 模型
以 30 Hz 获取快照，15 Hz 图像会以 causal hold 提供并携带真实 age。若模型输出
action chunk，则保留 chunk 的 30 Hz offset；若要对单步 delta 升频，必须使用
第 3.3 节的 SE(3) 拆分器。

Home、Enable 和 F/T 清零仍由本机 `robot reset/record` 流程负责。RPC Robot 不复制
`FlexivDualArm` 里的直连 reset，也不允许策略进程越过本地安全监督器。

## 7. 分阶段交付与验收

### P-1：兼容基线

- 固化当前 v1 proto、默认 conversion schema、ROS topics 和 CLI 行为；
- 保存现有本仓库数据集和 legacy LeRobot 数据集的 `meta/info.json` golden fixture；
- 给 `dual_arm_teleop` 直连 `FlexivDualArm` 的 feature schema、旧训练加载和旧
  checkpoint shadow inference 建回归测试。

验收：不启用任何 v2/profile 配置时，命令、topic、feature 名称/顺序和输出目录均
与实现前一致；老数据可加载，旧 checkpoint 可完成 shadow inference。

### P0：契约与转换纯函数

- 建立独立 `policy_contracts`、v2 proto、channel/action descriptor 和 schema hash；
- 实现 Rotation-6D/rotvec、hand normalization、feature reorder；
- 用现有两套 `meta/info.json` 建 golden fixtures。

验收：往返旋转误差 `<1e-6 rad`，手部 12 通道逐项一致，所有缺失/歧义 schema
均 fail closed。

### P1：只读多频率 RPC

- ROS adapter 为每个原生 channel 建 ring buffer；
- 实现 `DescribeSystem`、`SubscribeSamples` 和 `GetSnapshot`；
- 提供 Python client 和 60 秒统计工具。

验收：arm 达到约 200 Hz、RGB 达到约 15 Hz、无跨流 head-of-line blocking；
每路 sequence 单调，age/validity 正确，断开设备后在阈值内变 invalid。

### P2：LeRobot 离线 profile

- 实现严格兼容的 `dual_arm_lerobot_v1` 和一个无冗余的新 minimal profile；
- 合并多 episode 为单个 LeRobot v3 dataset；
- 在 `dual_arm_teleop` 直接运行 dataset load、1 个 batch 和短训练 smoke test。

验收：feature 名称/顺序与 legacy 38/24 schema 完全一致，视频可解码，统计量有限，
训练不需要手写 observation/action；minimal profile 不含重复姿态表达或未声明字段。

### P3：RPC Robot shadow inference

- 实现 `IsaacFlexivRpcRobot` 和 config 入口；
- 加载 ACT/Diffusion/SmolVLA checkpoint，仅记录预测动作，不控制真机；
- 对同一录制片段比较离线输入与在线 snapshot 的 feature。

验收：所有输入 shape/dtype/name 与 checkpoint 一致；连续 30 分钟无设备争用、
无陈旧图像伪装成新帧、无动作队列积压。

### P4：受控真机动作

- 实现 v2 action stream，复用现有 lease/TTL/safety supervisor；
- 先单臂小增量、再双臂、最后双手；
- 校验动作频率、过期动作、断网和进程崩溃的 stop/hold 行为。

验收：delta 每序列只执行一次，坐标/旋转与现有 teleop 方向一致；lease、脚踏、
TTL 或客户端任一丢失都会停止，不发生重复非零 delta。

### P5：通用策略插件

- 将 `FeatureContract + MappingRegistry` 独立成 Python 包；
- 增加新机器人/灵巧手时只注册 device schema 和 mapping profile；
- 增加 NumPy/PyTorch client 示例及性能基准。

验收：至少用两个不同 feature contract 通过同一 RPC 读取数据并进行 shadow
inference，控制核心无需修改。

## 8. 推荐实施顺序

先完成 P-1，再完成 P0-P2，立即打通“现有数据直接训练”；之后做 P3 的不动机器人
在线推理，最后做 P4 真机控制。不要先把固定 30 维 v1 改成任意向量，也不要让
`dual_arm_teleop` 同时直连硬件，这两种做法都会把 schema 错误或设备争用带到真机。

每个阶段都运行以下兼容矩阵，失败则停止推进：

| 路径 | 必须保持的结果 |
|---|---|
| 本仓库旧 CLI | record/reset/replay/visualize/convert 默认配置行为不变 |
| PolicyService v1 | wire schema、30D action 和安全语义不变 |
| 本仓库当前 LeRobot export | 未选择新 profile 时 schema/output 不变 |
| dual_arm 老版直连数采 | `FlexivDualArm` 仍直接可用，不 import RPC adapter |
| dual_arm 老版训练/推理 | 旧 dataset/checkpoint 无需转换即可继续运行 |
| 新 RPC 路径 | 只有显式选择新 robot type/profile 后才启动 |
