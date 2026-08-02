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
cd /home/hb/isaac_teleop_flexiv_inspire
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
  episode_000001_attempt_01_20260731_2359/
    manifest.json
    deviceio.mcap
    ros_mcap/
  episode_000002_attempt_01_20260801_0004/
    ...
```

目录名中的 `YYYYMMDD_HHMM` 是该 episode 创建时的本机日期、小时和分钟；
同一值也写入 `manifest.json` 的 `collection_timestamp_local` 字段。目录内文件
保持固定名称，确保可视化、回放和 LeRobot 导出继续按 manifest 稳定读取。
如果同一分钟内再次创建完全相同的 episode/attempt，录制会拒绝覆盖已有目录。

转换默认选择当前数据集最新的已完成 episode，并自动使用隔离的 data 环境：

```bash
robot convert
robot convert --action-view absolute_joint_position
robot convert --action-view absolute_cartesian_pose
```

默认动作视图来自 `config/recording.yaml`；三种视图分别是 30 维实际下发命令、
14 维双臂绝对关节位置和 18 维 world 下双臂绝对 EE 位姿。输出写入
`artifacts/lerobot/<dataset>/<episode>/<action-view>/`，已有非空目录不会被覆盖。

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
