# 阶段 A：数据语义与离线消费契约

阶段 A 的完成标准是离线工程实现。机械臂物理安全、持续实测 60 Hz、ROS/RDK/Modbus 部署联调、标定准确度及外部模型训练均需独立验收。阶段 A 提交时保持原有踏板运动许可、Home 路线和控制算法；现阶段显式 world 模式的许可与接管扩展见 [阶段 B 契约](TELEOP_STAGE_B_CONTRACT.md)，incremental 保持兼容。

## 多速率命令监督

规范产物是 `command-supervision-v2`，入口：

```bash
flexiv-inspire-command-export /path/to/episode/manifest.json /path/to/command_supervision_v2.json
# 同一入口也可作为 Python 模块调用
python -m flexiv_inspire_isaac.data_pipeline.action_contract MANIFEST OUTPUT
```

`robot convert` 的通用路径保存 `command_supervision_v2.json`；profile 合并路径保存 `command_supervision/<原episode_uuid>.json`。两条路径都读原始 MCAP 中每一条 `sent_command`，与相机帧数量无关。归档完整性在导出前检查。

| 时间量 | 契约 |
|---|---|
| 传感器采样 | 读取最终配置；相机可为 15 Hz，臂和手反馈独立采样 |
| 策略调用 | 独立于动作步长，允许低频推理后返回多个未来动作 |
| 动作步长 | 60 Hz，1/60 秒；每个样本含 `start_ns/end_ns` |
| 命令调度 | 每个增量点一次；超过一个步长的迟到点丢弃，不压缩补发 |
| 控制器内部伺服 | 仓库未观测的控制器参数，保持 unknown |

动作为 30D：左右臂各 3D 世界坐标平移增量（米）、6D 旋转增量（旋转矩阵前两列），再接左右手各 6D 绝对目标（驱动原始 0–1000 目标域）。组合顺序为 `R_next = R_delta @ R_previous`。手目标赋值，不相加；增量不乘相机/命令频率比。

每行保留原 episode ID、segment ID、边界事件 ID、命令 ID、源时间戳、有效性及拒绝原因。命令 ID 为 `session:source:sequence`；桥展开 chunk 的 sequence 是 `(source_sequence << 6) | point_index`，不是连续帧计数。原始 envelope/命令流继续保存在 MCAP。

按实际命令源映射时间生成行，不伪造精确均匀的实测节拍。间隔超过 1.5 个规范步长会插入 `missing-command` 无效槽；小于半步的间隔标为异常。没有命令时没有有效标签；重复 ID 被拒绝。动作区间跨事件边界时无效。缺失不是零动作。命令时钟无法映射时拒绝生成整条任务的有效监督，原始记录保留诊断。

观测从同一连续片段内、命令起点之前最近的源样本因果选取；保存真实源时间、映射时间及 age。未来帧和未来反馈不会作为训练观测。默认 required 是双臂状态、双手角度和三路 RGB；默认 age 上限分别为 20/150/100 ms。触觉为 optional，episode 收尾不再要求触觉至少一帧，age 上限 100 ms；缺测为 `value=null, valid=false`，合法零和真实饱和值在采集 valid 时保留。手角度独立 timing 存在时还检查字段 timing；表面采集无效则触觉降级。接口 `export_supervision(required={topic:age_ns}, optional={...})` 可明确采用任务自己的模态策略，禁止把缺测改为完整模态。

`windows(document, horizon)` 只返回有效、同片段窗口；无效行也切断连续窗口。`reconstruct(rows, initial_xyz=..., initial_rotation=...)` 是可执行参考消费者，按契约积分，拒绝无效、重复、跨片段或有时间缺口的输入。每个片段需要自己的已知控制目标锚点。缺少历史锚点时只能做带明确起点的代数重建，不能宣称恢复真实硬件轨迹。默认零/单位旋转仅用于合成验证。

## 调度、替换与兼容

默认 chunk offsets 按 `index/60` 生成。明确 offsets 仍受现有 TTL 校验。ROS 适配器对已经过期一个步长的点拒绝整个 chunk，保留明确原因；不把多个过期点都压成 offset=0。桥内生产 `PointSchedule` 防重复和突发：迟到超过一步的点跳过，耗尽后不再次积分。已有 TTL/看门狗仍按原控制规则处理 HOLD。新 chunk generation 取消旧 chunk 未发送部分，已穿过 SDK 边界的点仍记录其证据；取消不是撤销已经发送的硬件目标。

普通单频 LeRobot 文件继续标为 `legacy-camera-sampled-v1`，只作为显式兼容视图，**不自动满足多速率训练**。相机行中的 action 仍是最近命令；完整的未来动作监督必须从 v2 命令文件消费。本阶段不定义新的低频聚合动作、不修改 π0.5/ACT/DP 的外部训练器。旧 profile 布局保持兼容；使用者须显式适配 v2 时间轴，不能把 15 Hz 最近增量当成完整 1/15 秒动作。

`absolute_joint_position` 和 `absolute_cartesian_pose` 导出明确拒绝：旧实现把同期实测 q/TCP 当成 action。state 保留实测；动作只能来自命令。当前绝对接受 TCP 目标仅为审计证据，尚未提供绝对动作训练视图；缺少内部关节目标的事实保持 unknown。

## episode、事件与训练边界

一个任务 episode 可以包含多个连续片段。暂停、踏板释放/重按、Home 开始/完成/失败、故障/恢复和控制权变化记录源时间上的显式事件。`episode-event-v2`、命令审计与执行器证据不受 training 白名单、暂停或踏板录制门控制。control state 的状态/控制源/保持原因/踏板变化生成显式边界；静止、零增量、同一目标保持不切段。

native 入口与 ROS mirror 入口均接入 episode/维护事件；Home 在实际调用点发出事件。按事件 effective time 分段，异步写盘顺序不作为边界依据。落在暂停/关闭录制区间内的延迟命令明确拒绝。通用和 profile 写入器遇到边界或无效行都结束当前训练子 episode；`episode_lineage.json` 保存导出索引、父任务身份、源 segment 和帧数。原任务身份不变。

旧数据无事件时，独立命令导出需 `--allow-legacy`，并标记 `legacy-gap-inference-uncertain`；兼容转换也显式保留该依据。不能据 gap 反向恢复未记录的 Home/踏板事件或观测。

## 执行器证据

`sent_command` 的事实是桥侧在机械臂事务 ACK 后发布，手还未通过 mailbox/Modbus。它不代表双臂双手都成功。

`actuator-evidence-v2` 按命令 ID 与每个执行器保存 submitted、accepted、replaced、rejected、timeout、sdk_returned 或 send_failed。daemon 在已有 `send_cartesian_target` 调用点记录两臂分别的结果；第二臂失败时保留第一臂返回证据及原来的双臂 HOLD 行为。接受目标来自 validated target，RDK base 坐标及 wxyz 四元数明示。桥 ACK 接受目标与实测 TCP 分开，safe/requested/trace 和原始命令按原字段语义保留，并在 training 模式保持审计记录，确保请求到接受目标的关联。

手通过原 latest-only mailbox 接入，同一 ID 跟踪替换、过期、拒绝、发送异常和成功返回；成功表示 force-limit 与 angle write 调用均返回，部分异常为 send_failed，不宣称寄存器全部成功或物理到位。采集使用原非阻塞 DeviceIO，不等待所有执行器或磁盘。

传输确认、控制器确认和物理到位在没有独立接口证据时均为 unknown。v2 行包含每个执行器的证据列表和 missing_receipts；缺少记录不推定成功。`summarize_receipts` 可按显式 deadline 和检查时刻区分尚缺记录/超时，超时是证据汇集结果，不是物理失败实验。

## 写盘质量与永久归档

manifest v2 的 `quality` 按最终配置频率统计真正写入的记录。区分 received、enqueued、written、valid_written、queue_dropped、gated_suppressed；训练有效数量由导出门禁统计。源 produced、主动降采样与传感器/传输丢失在证据不足时为 null。`StreamStats.samples` 现在是成功写盘数，入队数独立保存；写失败令收尾失败。

频率汇总为各片段 `(written-1)` 总和 / 各片段首尾源时间差总和，暂停不进入分母。明确编号规则时可用 continuous_counter+step 计算观测序号空洞，并单独计 reset；不把空洞直接宣称源端丢失。默认未知编号规则不估计丢帧。生产者进程累计统计从每个 episode 第一份 summary 建立基线；此前区间未知，重置后增量重新核算，pending 是 gauge。

`provenance/index.json` 和 episode 内的 SHA256 objects 保存完整最终配置（秘密字段脱敏）、实际生成子配置、tool/payload、F/T 记录、已知标定/变换和手布局配置，以及独立的软件来源信息。手型号与 17 面/1062 taxel 布局按实际配置选择已有驱动 profile，单独保存其定义，并标明是软件声明、不是硬件身份探测。索引保存归档哈希和资产源哈希；规范化/脱敏文件明示，不把其字节哈希当成原文件哈希。必需配置/显式资产缺失时 complete=false，未知可选标定不补造。SDK/固件等未探测版本为 unknown；声明版本与 recorder 环境探测版本分列。

```python
from flexiv_inspire_isaac.data_pipeline.provenance import verify_archive, resolve_asset
index = verify_archive('/moved/episode')
config = resolve_asset('/moved/episode', 'effective_config')
# YAML 中旧路径通过索引 key 离线解析，不重新访问 /run/user/ 或部署目录。
```

校验拒绝缺失必需资产、篡改索引/对象或逃逸 episode 的引用。临时配置和源标定消失后仍可解析已归档内容。秘密、凭据、私钥不归档；归档失败不得声称完全可复现。历史缺失回执、被门控压掉的观测、丢失标定以及未记录目标不能恢复。

manifest v2 传入最终配置时永久归档是必需行为；旧 `snapshot_config` 声明保留，不能以关闭声明代替可复现性证据。

启动/录制摘要显示任务、profile、各模态配置、配置频率、输入、安全开关、门控、目录、配置标识和动作契约；实测频率仅在收尾质量统计中给出。

## 离线验证与部署清单

```bash
python -m flexiv_inspire_isaac.data_pipeline.stage_a_smoke /new/output/path
python -m pytest tests/test_teleop_stage_a.py
```

合成闭环运行生产 EpisodeSession/异步 MCAP sink/归档/导出/消费者。它是软件契约证据，不是真机轨迹或实时性能证据。测试另覆盖生产 bridge 调度方法（隔离 ROS 导入）、daemon 部分失败、手 mailbox/发送、归档搬迁/校验、两个写入器边界和默认门禁。

部署端尚需验证：SDK/Modbus 实际回执含义，DeviceIO 压力与磁盘性能，实际频率/时钟映射、边界时刻、标定与各片段控制锚点。ROS 消息生成/现场节点启动和硬件行为 NOT_RUN。阶段 A 提交时没有修改 protobuf/ROS message。阶段 B 在 RDK typed IPC 增加 world_session_authority/mapping_epoch 与 sdk_returned_sides，已同步客户端、服务端和 Python 生成物；部署需按阶段 B 指南增量更新 protobuf 并生成一致协议。ROS message 未改。阶段 C、真机连接、远程部署、push/PR 均未执行。

## 阶段 B 扩展后的边界

内部 world 绝对目标仍导出原 30D 增量：相邻同 execution epoch 的已接受 TCP
目标之差/world 左乘旋转，手是绝对目标；measured 永远是 observation。
world 的 accepted 唯一由桥内管理器按每侧 sdk_returned_sides 更新；部分失败保留
各侧证据并 HOLD，下一次显式 start 从实测重建执行基准，不跨 epoch 积分。
world_control_changed、Home、ALIGNING/FOLLOW、authority 丢失/恢复形成边界；
world_state 非 FOLLOW 的 sent_command 同时在 v2、通用/profile 输入被拒绝。
world 的真实踏板边缘不切分无踏板 FOLLOW，旧 incremental 踏板门保持原语义。
保存/丢弃不改变 mapping epoch；Home 改变 execution epoch，恢复保留父任务身份。

原始稳定参考 reference_samples 是标定的必需归档资产；缺失不能标为 complete。
零空间与 world 审计都是关键 native 记录，不受普通训练录制门抑制。SDK 返回仍只
表示调用返回，控制器确认/物理到位没有独立证据时保持 unknown。
