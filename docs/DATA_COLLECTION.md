# 一键数采工作流

日常录制参数只在 [`config/recording.yaml`](../config/recording.yaml) 修改：

```yaml
recording:
  dataset_name: pick_place_demo
  episode_count: 20
  output_root: sessions
  task_description: "Pick up the target object and place it in the designated area."
```

- `dataset_name`：数据集目录名，也是本批 episode 的名字；
- `episode_count`：成功保存多少条后自动退出；左键丢弃的 attempt 不计数；
- `output_root`：输出根目录；相对路径相对于仓库根目录；
- `task_description`：写入每条 manifest，LeRobot 导出时默认作为 VLA prompt。

底层硬件进程已连接、当前 session 已完成 F/T 清零、控制桥为 `READY`，并且
现场的 local-permission/collision-clear 仍有效后，在机器人本机交互终端只需：

```bash
cd /home/hb/chp_ws/rl100_dp30/isaac_teleop_flexiv_inspire
robot record
```

该命令读取 `config/site.yaml`，自动生成子配置，启动所需服务、建立本机
Home/teleop 授权，并立即开始第一条录制。它以前台方式占用终端；完成配置的
episode 数量会自动退出，`Ctrl-C` 会保存当前条、等待 MCAP/DeviceIO 落盘并
停止本次录制，不需要再执行 `robot stop`。`./scripts/collect.sh` 仅保留为兼容
入口，用于支持服务已经单独运行时只启动脚踏和 episode 控制器。

录制期间的行为固定如下：

| 输入 | 当前 episode | Home 期间 | Home 完成后 |
|---|---|---|---|
| 右脚踏 | 结束并保存 | ROS bag 与 DeviceIO 都暂停，不写入 Home | 条数加一，未达目标则自动开始下一条 |
| 左脚踏 | 标记丢弃 | 两套记录都暂停 | 同一编号增加 attempt，重新录制；旧数据保留审计但不能导出训练 |
| Quest 右手 A | 暂停，不切 episode | 两套记录都暂停 | 恢复同一条 episode |
| `Ctrl-C` | 正常保存当前条 | 不自动 Home | flush manifest/MCAP 后退出 |

每次左键或右键都会执行一次配置的双臂 Home。Home 失败、被拒绝或超时会让
整次采集 fail-closed，不会偷偷开始下一条。达到 `episode_count` 时，最后
一次右键仍完成 Home，然后保存并自动退出。

输出结构如下：

```text
sessions/pick_place_demo/
  raw/
    episode_000001_attempt_01_20260731_2359/
      manifest.json
      deviceio.mcap
      ros_mcap/
    episode_000002_attempt_01_20260801_0004/
      ...
  rl100/
    joint_proprio_cartesian_v1.zarr/
  lerobot/
    episode_000001_attempt_01_20260731_2359/
      sent_command/
```

目录名中的 `YYYYMMDD_HHMM` 是该 episode 创建时的本机日期、小时和分钟；
同一值也写入 `manifest.json` 的 `collection_timestamp_local` 字段。目录内文件
保持固定名称，确保可视化、回放和 LeRobot 导出继续按 manifest 稳定读取。
如果同一分钟内再次创建完全相同的 episode/attempt，录制会拒绝覆盖已有目录。

转换选择由 `config/conversion.yaml` 的 `source.episode` 决定，并自动使用隔离的
data 环境。输入数据集和输出目录默认从 `recording.yaml` 推导；`all` 会把所有已完成
episode 合并为 RL-100 训练使用的 Zarr，已有非空输出不会被覆盖：

```bash
robot convert
```

默认输出写入 `sessions/<dataset>/rl100/joint_proprio_cartesian_v1.zarr`，包含
26D 本体状态、24D 策略动作和三路 RGB。时间轴来自实际下发的
`control/sent_command`，图像和状态按因果关系对齐，不跨踏板空档插值。

需要导出头部深度、触觉、高频关节历史或其他动作视图时，显式使用完整 LeRobot
配置：

```bash
robot convert --conversion-config config/conversion_lerobot_full.yaml
robot convert --conversion-config config/conversion_lerobot_full.yaml \
  --action-view absolute_joint_position
robot convert --conversion-config config/conversion_lerobot_full.yaml \
  --action-view absolute_cartesian_pose
```

`conversion_lerobot_full.yaml` 的 `lerobot_export.fields` 是字段白名单。三种动作视图
分别是 30D 实际下发命令、14D 双臂绝对关节位置和 18D world 下双臂绝对 EE
位姿。输出与原始 MCAP 的 `raw/` 平行，已有非空目录不会被覆盖。

需要把多条轨迹合并成一个可直接训练的 LeRobot dataset 时，使用显式 profile
配置：

```bash
# 旧 dual_arm_teleop 的 38D state / 24D action 兼容格式
robot convert --conversion-config config/conversion_dual_arm.yaml

# 推荐的新策略格式：q(14)+hand(12) state，24D 笛卡尔 action
robot convert --conversion-config config/conversion_policy_minimal.yaml
```

两个配置都读取所有已完成 raw episode，并在同一个 dataset 中保留各自的 episode
边界。profile 严格拥有字段白名单，因此不能再配置 `fields`、深度或高频 history。
兼容 profile 只写 `observation.state`、`action` 和三路旧名称图像；minimal profile
不会同时放 q 和 TCP pose，也不把时间戳、validity 等传输元数据拼进 state。
输出目录拒绝覆盖，重新导出时应更换 `output.root` 或先人工归档原结果。

中踏板松开时若启用了 `record_only_while_pedal_pressed`，源 MCAP 会保留真实
时间空档。转换不会复制旧图像或跨空档插值；每帧额外导出源时间戳、空档时长、
片段编号和片段内帧号。`segments.split_episodes: false` 保留原始任务 episode；
改为 `true` 时才把空档两侧写成不同的 LeRobot episode。

`config/site.yaml` 现在只是组合入口，不再堆全部参数：

- `config/hardware.yaml`：session、Flexiv、Inspire、限位、Home、阻抗；
- `config/sensors.yaml`：相机、Quest/XR、MANUS；
- `config/recording.yaml`：每批数采需要改的参数；
- `config/runtime.yaml`：进程命令，仅用于开发/启动编排。

生成的进程配置位于 `/run/user/1000/isaac_teleop/launcher/<config-hash>/`；
它们是运行时快照，不需要手工维护，也不需要 `artifacts/site/`。

一键数采只合并重复的录制控制，不绕过一次性的真机安全准备：它不会自动
启动/Enable 机器人、代替工具/payload 审核、执行 F/T 清零、伪造现场许可，
也不会在 MANUS 未标定或 Inspire 写门禁未开启时发送手指命令。启动前检查
若发现脚踏、RDK socket、F/T 记录或标定文件缺失，会直接报错且不开始录制。
