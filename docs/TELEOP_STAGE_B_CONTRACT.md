# 阶段 B：固定遥操作映射与软零空间契约

默认仍为 incremental。选择 `teleop.mapping_mode: world_absolute`、有效标定、本地硬件许可与显式 `start` 都是必要条件；录制进程和启动输入节点均不授予 world 会话运动权。离线完成不代表部署、物理安全、关节连续性或实时性能通过。现场执行见 [硬件验收指南](TELEOP_STAGE_B_HARDWARE_ACCEPTANCE.md)。

## 固定空间与标定

`config/world_absolute.example.yaml` 是安装参数模板，`config/world_execution.example.yaml` 是时间管理与路径批准模板。二者的数值只用于诊断，现场在对比前测量、确定并冻结阈值。

全部位姿表示为 `[x,y,z,qx,qy,qz,qw]`，米、xyzw；`world_from_base` 把两侧 RDK base 位姿变换到公共 robot world，与既有 `frame_config` 必须一致。人端输入也叫 world，但它是追踪 world；`axis_rotation = A` 把人端向量转到公共 robot world，`translation_scale = s` 是共同平移比例。`controller_from_reference = C_i` 与 `tcp_from_reference = E_i` 分别是人控制器、机器人 TCP 到掌心/任务参考点的刚体变换。偏置必须人工填写或额外测量，静止参考不能估计未知轴、比例和全部偏置。

记当前人端 `H_i`，初始人端 `H_i0`、机器人实测 TCP `R_i0`，定义：

```
h_i = H_i C_i                         # 人端任务参考
h_i0 = H_i0 C_i
r_i0 = R_i0 E_i                       # 机器人任务参考
b = robust paired mean(p(r_i0) - s A p(h_i0))  # 左右共用一个 b
B_i = R(h_i0)^T A^T R(r_i0)
p(reference_desired_i) = b + s A p(h_i)
R(reference_desired_i) = A R(h_i) B_i
TCP_desired_i = reference_desired_i E_i^-1
```

左右参考必须与同一个 `b` 相容，参考姿态偏置 `B_left/B_right` 也须相容；不相容时拒绝标定。左右任务参考坐标约定必须一致（掌面法向等几何由 `C/E` 表达），不能用两个任意零点承诺双手合拢。掌面接近与手指握拳分别由公共空间/工具几何和既有 MANUS→Inspire 重定向承担，本阶段没有重写手指算法。

`python -m isaac_teleop_core.world_cli` 提供真实入口：

- `capture OUTPUT --duration-s 3`：仅订阅 `/teleop/command` 和左右实测 ArmState，无运动发布；保存带身份/时间的左右配对参考。
- `calibrate INSTALLATION SAMPLES OUTPUT_DIR`：至少十对、至少 0.2 s；位置用中位数，姿态在 SO(3) 上用 SVD 投影估计，逐样本检查旋转、稳定性、新鲜度、源时间推进和配对时差；绑定实际输入节点 boot UUID，写入新的 installation/calibration 文件和 mapping epoch，不覆盖已有目录。
- `validate INSTALLATION CALIBRATION`：检查版本、安装参数 SHA256、身份、共同空间和参考几何。
- `preview INSTALLATION CALIBRATION TRACKING_JSON`：离线计算目标，不下发。桥侧 `/teleop/world_state` 同时提供实时只读 desired 预览。

工具身份是现场已审计 tool/payload YAML 的规范化 SHA256，桥启动与文件变化时检查。换 TCP、工具、控制器握持方式、显式 recenter、输入节点重启、可检测追踪跳变/源时钟倒退/输入长断流必须显式失效、重新验证参数并标定新 mapping epoch。

现有 PoseArray/Quest 接口不能报告追踪原点或 XR 后端进程身份。`origin_reported: false` 必须如实填写；当前实现记录操作者声明的 origin、输入节点 boot UUID，检测跳变、时序和断流。**小幅且没有身份/时间异常的 recenter、XR 后端无缝重启不能保证自动识别**；操作者执行之前必须先 invalidate，再重新标定。不得把能力限制理解为允许静默沿用。

## 执行状态与权威

唯一 world 执行管理器在 ControlBridge 内。输入节点只传递追踪快照与原有手绝对目标；world 不使用输入增量。`desired` 是固定映射，`limited` 是候选轨迹，`accepted` 只在各臂实际 SDK 返回证据后更新，`measured` 是独立反馈。旧 `_safe_pose_rdk` 缓存不再决定 world 模式的下发基准。手执行证据仍由 DFTP mailbox/驱动维护，软限速手目标不冒充手接受回执。

状态为 `PREVIEW/INVALID → ALIGNING → FOLLOW`；Home 独占 `HOMING`，异常 `HOLD`。所有恢复需要显式操作，不自动恢复跟随。会话操作使用 `/teleop/world_operation` 的 JSON String，含当前 `session_id`、`confirm: WORLD-TELEOP-LOCAL`：

| operation | 行为 |
|---|---|
| start | 已有本地 TTY 生成的 daemon token、有效 F/T、local permission、实测与追踪后重建 execution epoch，从实测捕获基准，捕获当前人端固定目标，批准路径后进入 ALIGNING |
| stop | 停止会话、锁存 HOLD，保持固定映射 |
| invalidate | 失效固定映射、停止；需重新标定并加载桥 |
| fingers | 仅 FOLLOW；显式提供 targets.left/right，与两侧新鲜实测手角度最大误差不超过配置 finger_takeover_tolerance（示例 10 原始单位），才接管 MANUS；不强制张开持物手 |

ROS 本地操作表达意图，不能替代 daemon 的 CLI/env/permit、TTY token、PID/session/source lease、F/T、TTL/watchdog 和控制器故障/急停边界。daemon 另需 `control.world_absolute_enabled: true`。core 的 `teleop_session_authority` 只适用于 TELEOP；policy/replay 既有许可规则保持。物理踏板状态真实发布和记录。world 控制强制使用实测软限位/力/速度/温度与 collision_clear 门，不能用旧 practical/software_safety_limits_enabled=false 绕过；policy/replay 继续沿用既有模式规则。

Home、重连和执行基准重建改变 execution epoch，保存/丢弃、静止、短 HOLD 不改变 mapping epoch。不能跨 execution epoch 求增量。每条成功 point 的审计动作遵循阶段 A：`Δp = accepted_next - accepted_previous`，`ΔR = R_next R_previous^T`，world 左乘；手仍赋值绝对目标。失败 ACK 新增 `sdk_returned_sides` 保留已成功侧，未成功侧不提交；其他异常/未知回执锁存 HOLD，下一次显式 start 从新鲜实测重建。SDK 返回不证明物理到位。

## 时间、路径和 Home 接管

轨迹在真实单调 dt 下积分，平移/旋转分别限制速度、加速度；端点采用受限速度的阻尼接近，不按输入帧裁剪，不跳到最终端点。转动使用 world 左乘的 SO(3) rotation vector。ALIGNING/FOLLOW 共用速度状态，正常切换不重置。安全 HOLD/测量领先保护会清零软件速度并请求硬件 Stop，不能将该紧急停止过程宣称为普通轨迹加速度保证。jerk 明确 unsupported，不提供硬关节 Δq 上限。

重复输入仅在原始追踪仍新鲜时有效；源时间冻结、倒退、变身份、超龄、控制 dt 非正/过长进入 HOLD 或 INVALID。长 dt 不补偿丢失时间。accepted/limited 超过实测 TCP 的配置领先量时阻止推进，持续不满足条件锁存 HOLD；不积分 desired-measured 到旧目标，受阻速度不积累。模型/机器人可达性未由本地 IK 证明，拒绝工作区外路径或收敛超时，不把有限目标当作可达性证书。

第一版路径接口 `ApprovedRegion.approve(start,end,mapping)` 接受现场明确批准的公共 TCP 凸区域，绑定 mapping epoch、工具身份和批准记录。为空、身份不符或端点不在区域内即拒绝。区域必须由现场验证完整连杆、双臂、工具、环境和可能的零空间姿态变化；TCP 区域检查不具有全臂碰撞规划能力。代码没有路点规划、在线避障或硬约束 IK/QP。需要其他路径时可实现同一批准接口并独立验收，当前未实现的路点 CLI 不可使用。

Home 完成后 accepted 基准清除；操作者无需回初始人端位置，只要短暂稳定在有效且经批准的映射位置，重新授权/start 捕获固定接近目标。途中人端超出捕获位置/姿态变化限额即 HOLD；显式重新 start 重新捕获，不追逐旧快照。左右实测对最新目标误差、accepted 对最新目标误差、追踪新鲜、目标变化率、驻留和超时同时通过才 FOLLOW。手指在新 FOLLOW 另行接管。

## 录制与永久证据

world 下 recorder 以 episode 生命周期和显式 pause_recording/resume_recording 控制，配置中旧踏板录制门被禁用；静止有效 FOLLOW 可录制。右踏板保存并进入下一条，左踏板丢弃并重录，两者不 Home、不重标定、不授予或撤回会话运动权；Quest Home 按键仍执行原独占 Home。停止录制不等于停止机器人；`/control/stop` 才是明确停止入口。

`/maintenance/world_state`、`world_control_changed`、`/maintenance/nullspace` 及阶段 A 命令、逐执行器回执保留关键审计，不受普通录制门控。world_state 保存 desired/limited/accepted/measured、两个 epoch、状态/原因及真实踏板。Home/ALIGNING/HOLD 审计不进入普通 FOLLOW 监督；原 episode 保留父身份，恢复形成新的连续片段。v2 命令导出以及通用/profile MCAP 输入都拒绝 world_state 非 FOLLOW。world 中物理踏板边缘仅为真实状态记录，不切断有效无踏板片段。

永久归档递归包含 world_installation、world_calibration、world_execution_config、原始 reference_samples、工具/payload、实际生成进程配置。无现场 collector 的只读预览不积压关键队列；有 collector 时关键审计入队失败使会话 HOLD。正式 world 模式使用 native DeviceIO；配置入口已要求 native。底层 post-dds-mirror 调试录制不提供本阶段 world/零空间证据闭环，不用于 world 验收。

## RDK 1.9.x 软零空间

仓库 RDK adapter 的实际版本检查为 1.9.x。官方一手依据是 [Flexiv RDK 1.9 Robot API](https://www.flexiv.com/software/rdk/api/classflexiv_1_1rdk_1_1_robot.html)，SetNullSpacePosture/SetNullSpaceObjectives 文档说明笛卡尔模式适用、阻塞、模式重入重置。

`apps/flexiv_daemon/config/robots.yaml` 的 `control.nullspace` 为版本 1、每侧默认 disabled。显式启用配置：

```yaml
nullspace:
  schema_version: 1
  left:
    enabled: true
    rdk_version: 1.9.x
    reference_strategy: takeover_measured
    weights: [0.0, 0.0, 0.5]
    failure_policy: hold
    online_update: false
  right:
    enabled: false
```

可选 manual 策略需七个有限 rad `q_ref`，并用实际 RobotInfo q_min/q_max 检查。权重分别 linear_manipulability、angular_manipulability（各 [0,1]）和 ref_positions_tracking（[0.1,1]）。示例权重不代表现场最佳值。接管时实测 q 固定，模式重入恢复原 q_ref；新显式接管可重新采集。不是每周期追踪当前 q，也不默认固定 Home q 最佳。

适配器探测实际绑定 `set_null_space_posture/SetNullSpacePosture`、`set_null_space_objectives/SetNullSpaceObjectives`，按 1.9 两种命名调用：姿态一个 7D vector，权重三个标量。调用仅在模式进入/本地接管准备时发生，失败和缺接口拒绝启用，不静默成功。禁用模式不调用新接口，兼容旧部署。在线更新不支持。归档中记录 q_ref、权重、实测 q、策略/模式、实际版本、耗时、sdk_returned/failed；模拟调用不证明改善真机关节轨迹。

## 离线验收入口

使用已有 ROS 测试环境（含本仓库 typed IPC `.pth`、MCAP/Pillow），或同等隔离离线环境；不导入真实 RDK SDK或启动 ROS 节点。纯 data 环境默认没有 daemon 源码，不应把诊断入口的导入失败当作现场硬件故障。

```bash
python -m pytest tests/test_teleop_stage_b.py tests/test_teleop_stage_a.py
python -m flexiv_inspire_isaac.data_pipeline.stage_b_smoke /new/evidence/path
python -m flexiv_inspire_isaac.data_pipeline.stage_a_smoke /new/stage-a/path
```

真实 ROS 节点启动、SDK 能力、全臂物理碰撞安全、关节连续性和实测实时性能为 NOT_RUN，必须在现场独立验收。
