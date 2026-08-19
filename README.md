# Flexiv 双臂 + Inspire + Quest/MANUS：使用手册

这是双 Flexiv 机械臂、两只 Inspire 手、RealSense 相机、Quest 和 MANUS 的
遥操作、数采、可视化、回放和数据转换系统。日常操作只使用 `robot`；不要再
手工启动 RDK daemon、ROS 控制桥、手部驱动、socket 或脚踏服务。

面向 RL-100 的 30 Hz 三视角 RGB MVP、26D/24D 数据契约、MCAP 到 Zarr
转换和安全部署流程见 [docs/RL100_DP30_PIPELINE.zh-CN.md](docs/RL100_DP30_PIPELINE.zh-CN.md)。

> 真机操作只能在机器人主机本地终端执行。运行前确认工作区无人、双臂和双手
> 无接触/无外载、线缆不受拉扯，且中踏板处于松开状态。

## 最短日常流程

```bash
cd /home/hb/isaac_teleop_flexiv_inspire
source scripts/env/activate_ros.sh

# 修改本批任务后，直接开始遥操作数采
robot record
```

`robot record` 会自动启动所需服务，并执行 F/T 清零（或复用本会话有效结果）、
Home 和双手开合准备。录制结束时会自动保存、关闭本次启动的服务。

如果只想让机器人回到可操作初始状态：

```bash
robot reset
```

如果终端、电脑或某服务异常中断，才使用：

```bash
robot stop
robot reset
```

## 每次开始前检查

```bash
robot validate       # 配置文件是否有效；不连接机器人
robot xr-doctor      # 检查当前选择的 Quest 控制器输入
```

现场确认：

- 两台 Flexiv 控制器、两只 Inspire 手和三台相机已上电；
- PC 网口仍为手部网络，能访问 `config/hardware.yaml` 中两只手的地址；
- Quest 已解锁并允许 USB 调试；OculusReader 默认经 USB 读取双腕与按键；
- MANUS 服务及其标定文件正常；
- 中踏板没有被压住。中踏板是机械臂运动离合，不是“录制开关”。

## 配置：日常只改哪些文件

| 文件 | 何时改 | 主要内容 |
|---|---|---|
| `config/recording.yaml` | 每一批新任务 | 数据集名、任务名/描述、episode 数、录制模态 |
| `config/playback.yaml` | 可视化或回放前 | 选择数据集与 episode、回放速度 |
| `config/conversion.yaml` | 导出训练数据前 | 批量选择、LeRobot 输出、字段和动作视图 |
| `config/sensors.yaml` | 更换相机、Quest、MANUS 或相机模态 | 序列号、相机 RGB/深度/点云开关、外参 |
| `config/hardware.yaml` | 改 Home、速度、机械臂/手网络配置 | 硬件安装参数；非日常修改 |

`config/site.yaml` 是上述配置的组合入口。运行时生成的文件在
`/run/user/1000/isaac_teleop/launcher/`，不要手工编辑。

### 选择 Inspire 手型号

左右手可独立选择型号，ROS topic、六轴动作格式和数据格式不变：

```yaml
inspire:
  left_model: rh56e2_2l_t1
  left_host: 192.168.5.11
  right_model: rh56e2_2r_t1
  right_host: 192.168.5.12
```

当前支持 `rh56dftp_2`、`rh56e2_2l_t1`、`rh56e2_2r_t1`。老配置没有
`left_model/right_model` 时自动按 `rh56dftp_2` 运行。两只 RH56E2 的出厂地址
通常都为 `192.168.11.210`，不可直接同时使用；首次安装时需逐只连接并改成唯一
地址。访问出厂地址时，PC 有线网卡也必须有该网段地址，例如：

```bash
sudo ip address add 192.168.11.22/24 dev eno1
```

这条命令只在本次开机有效；本站两只手已分别设置为 `.5.11/.5.12`，日常运行
不需要添加出厂网段地址。

### 新建一批任务

编辑 `config/recording.yaml`：

```yaml
recording:
  dataset_name: open_boxes_first_try  # 输出目录名
  task_name: open_boxes               # 新 episode 的目录前缀
  episode_count: 20
  task_description: "Use both thumbs to open the latches..."
```

之后执行 `robot record`。不要在同一数据集目录中混用不相同的任务描述。

## 遥操作与数采

```bash
robot record                 # 默认不启动 Quest 视频
robot record --with-xr       # 同时在 Quest/桌面启动相机显示
robot record --no-xr         # 明确关闭 Quest 视频；控制器输入仍可用
```

录制期间：

| 输入 | 行为 |
|---|---|
| 中踏板按住 | Quest 双腕位姿有效时，允许双臂遥操作；数据在此期间写入 |
| 中踏板松开 | 立即保持；再次按下从当前机器人姿态重新接管 |
| 左踏板 | 丢弃当前 attempt，Home 后重录同一 episode 编号 |
| 右踏板 | 保存当前 episode，Home 后进入下一条 |
| Quest A | 暂停/恢复当前 episode，并执行受控 Home |
| `Ctrl-C` | 保存当前条并退出；等待 MCAP 写完 |

MANUS 控制手指；Quest 控制双臂末端。若中踏板踩下仍不动，先确认 Quest 双腕
追踪、MANUS、脚踏和控制器在线状态，而不要重复启动多个 `robot record`。

Quest 输入后端在 `config/sensors.yaml` 中选择。默认轻量 USB 后端不会启动
CloudXR/OpenXR；需要回退到原 Isaac Teleop 输入时只改一行并重启 `robot record`：

```yaml
teleop:
  quest_input:
    provider: oculus_reader   # 或 isaac_openxr
```

两个后端发布完全相同的双腕、TF 和控制器按键话题，不能同时运行。首次建环境会
自动安装固定版本；单独补装可执行 `bash scripts/env/install_oculus_reader.sh`。

录制的数据结构：

```text
sessions/<dataset_name>/
  raw/
    <task_name>_episode_001_YYYYMMDD_HHMM/
      manifest.json          # 任务、配置哈希、完成状态和数据来源
      deviceio.mcap          # 原始异步传感器、状态与 sent_command
      ros_mcap/              # 可选的 ROS bag
  lerobot/                   # robot convert 的平行输出
```

`deviceio.mcap` 是数采、可视化、回放和转换的主数据。不要移动或重命名
episode 内的 `manifest.json` / `deviceio.mcap`。

## 相机、深度和点云

每台相机在 `config/sensors.yaml` 中独立设置：

```yaml
recording: {rgb: true, depth: true, pointcloud: false}
```

- `rgb`：彩色图像，所有用于训练的相机建议保留；
- `depth`：原始 Z16 深度和尺度，通常头部相机保留；
- `pointcloud`：由深度和内参计算的 XYZ/RGB 表示，体积更大；腕部一般关闭。

没有外参时，深度和点云仍然有效，但位于**各自相机光学坐标系**，不能直接和
双臂 world 坐标拼接。固定头部相机建议记录 RGB + 深度；需要统一世界点云时再
打开点云并完成外参标定。

## 相机标定

先打印 ArUco 4x4/100 标定板，记录实际边长；采集程序不会自主移动机械臂，仍由
正常遥操作移动。

```bash
# 手眼（腕部相机）：标定板固定在工作区，移动同侧手臂
robot camera-calibrate capture \
  --camera left_wrist --arm left --mode eye_in_hand \
  --intrinsics left_wrist_intrinsics.yaml --marker-size-m 0.080 \
  --samples 15 --output artifacts/calibration/left_wrist_samples.yaml
robot camera-calibrate solve \
  --samples artifacts/calibration/left_wrist_samples.yaml \
  --output artifacts/calibration/left_wrist_extrinsics.yaml

# 眼在手外（头部固定相机）：标定板刚性固定到 TCP，再移动手臂
robot camera-calibrate capture --camera head --arm left --mode eye_to_hand \
  --intrinsics head_intrinsics.yaml --marker-size-m 0.080 \
  --samples 15 --output artifacts/calibration/head_samples.yaml
robot camera-calibrate solve --samples artifacts/calibration/head_samples.yaml \
  --output artifacts/calibration/head_extrinsics.yaml
```

将求解出的文件填到对应相机的 `extrinsics:`，然后 `robot validate`。完整说明见
[docs/CAMERA_CALIBRATION.md](docs/CAMERA_CALIBRATION.md)。

## 查看录制数据

编辑 `config/playback.yaml`：

```yaml
dataset:
  root: sessions/open_boxes_first_try
  episode: latest              # latest、数字 episode_index 或目录名
```

然后：

```bash
robot visualize
```

这只读取 MCAP 并打开 Rerun，机器人绝不会运动。Rerun 中可以查看 RGB、深度、
点云、双臂关节/TCP/力、手部状态/触觉，以及 requested/safe/sent command。
如果数据很多，把 `visualize.realtime` 设为 `false` 可以更快载入。

## Replay：让机器人复现示教

`robot replay` 会让真机运动。先在 `config/playback.yaml` 确认数据集和
episode，再执行：

```bash
robot replay
```

它会自动启动所需服务、F/T/Home、启动脚踏路由；第二次 Home 后按提示持续踩住
中踏板。它只重放录制时真正下发的 `/control/sent_command`，并在录制中的自然
空档发送零增量保持，因此不会重复上一帧的运动。松开踏板、`Ctrl-C`、硬件异常
或正常结束都会停止，并在退出后自动 Reset。

不要回放以下数据：任务对象/工作空间已改变、首帧不是从当前 Home 录制、工具或
payload 已改、episode 未完成，或你不愿意让机器人复现其动作。

## 转换为 LeRobot

在 `config/conversion.yaml` 设置输入和输出：

```yaml
source:
  dataset_root: sessions/open_boxes_first_try
  episode: all                 # all / latest / 编号 / 目录名
output:
  root: sessions/open_boxes_first_try/lerobot
  episode_subdirectory: true
```

执行：

```bash
robot convert
robot convert --action-view sent_command
robot convert --action-view absolute_joint_position
robot convert --action-view absolute_cartesian_pose
```

默认 `all` 批量转换所有已完成 raw episode；已有非空输出会跳过，不会覆盖。
三种动作视图分别是：实际下发的 30 维动作、14 维绝对关节、18 维 world 下双臂
末端位姿。导出字段、深度、触觉、高频关节历史及空档切片策略都在
`config/conversion.yaml` 配置。详见 [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md)。

## Quest 图像与 MANUS

```bash
robot xr-doctor       # 检查当前 Quest 输入后端和 USB 连接
robot xr-view         # 只启动 Quest/桌面相机显示
robot record --with-xr
```

`oculus_reader` 只负责双腕位置姿态和按键，MANUS 仍只负责手指。`xr-view` 是
独立的 Isaac Teleop 图像显示，不开始数采；Quest 图像失败不会影响默认
OculusReader 遥操作输入。MANUS 的安装与标定参见
[docs/MANUS_SETUP.md](docs/MANUS_SETUP.md)。

## Policy RPC（可选）

外部策略通过 TLS RPC 发送动作时，使用：

```bash
robot policy-serve
```

默认也会先执行 Reset。策略动作仍经过同一机械臂、手、触觉与控制门限；它不能
绕过本地硬件状态。调试启动命令可使用 `robot policy-serve --dry-run`，细节见
[docs/PICK_PLACE_ACT_POLICY.md](docs/PICK_PLACE_ACT_POLICY.md)。

## 常见问题

| 现象 | 先做什么 |
|---|---|
| `rclpy` 找不到 | 重新执行 `source scripts/env/activate_ros.sh`；`robot` 后台服务也会自动补 ROS 环境 |
| RDK socket 未运行 | 直接 `robot reset`、`robot record` 或 `robot replay`；无需手启 daemon |
| 双手 `No route to host` | 检查 `eno1`、`192.168.5.11/.12`、手部电源和网线 |
| 中踏板没反应 | 确认 `robot record`/`robot replay` 正在前台运行，脚踏设备未被其他程序独占 |
| Quest 有画面但不遥操 | 确认中踏板持续按住、Quest 双腕追踪有效；用 `robot xr-doctor` 检查 Quest |
| `record` 或 `replay` 残留服务 | `robot stop`，然后 `robot reset` |
| LeRobot 提示无有效图像 | 先 `robot visualize` 检查三路 RGB 是否录到；确认 conversion 的相机字段与数据集一致 |
| 点云与世界坐标不重合 | 完成相机外参标定并在 `config/sensors.yaml` 填写 `extrinsics` |

## 首次安装、开发与文件结构

首次创建环境和构建见 [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md)。不需要为日常
运行执行 `colcon build`；只有改动 ROS 源码或重建环境时才需要。

```text
apps/session_manager/      robot 命令与会话编排
apps/flexiv_daemon/        独立 Flexiv RDK daemon
ros2_ws/src/               控制桥、手部、相机、接口和 Rerun ROS 包
libs/control_core/         动作格式、坐标/旋转、控制状态机
libs/data_core/            MCAP、episode 管理、LeRobot 导出与回放
config/                    唯一的现场配置来源
sessions/                  原始数据与 LeRobot 平行输出
docs/                      标定、MANUS、数据与高级说明
```

更详细的专题文档：

- [数据采集与导出](docs/DATA_COLLECTION.md)
- [离线可视化与硬件回放](docs/PLAYBACK.md)
- [相机标定](docs/CAMERA_CALIBRATION.md)
- [MANUS 安装](docs/MANUS_SETUP.md)
- [环境与构建](docs/ENVIRONMENTS.md)
