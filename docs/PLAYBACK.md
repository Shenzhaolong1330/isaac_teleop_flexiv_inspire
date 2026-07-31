# Dataset playback

Replay 和 visualize 共用 [`config/playback.yaml`](../config/playback.yaml)，两个启动命令都不接收参数：

```bash
./scripts/visualize.sh
./scripts/replay.sh
```

## 数据集选择

```yaml
dataset:
  root: sessions/pick_place_demo
  episode: latest
```

`episode` 可写 `latest`、数值 `episode_index`，或 `root` 下的完整 episode 目录名。数据源固定为 episode manifest 指向的 `deviceio.mcap`。

## Visualize：机器人不动

`./scripts/visualize.sh` 不初始化 ROS、不创建 publisher，也不连接 Flexiv、Inspire 或相机。它把完整 DeviceIO MCAP 写入 Rerun 时间线，包括：

- 相机 JPEG/raw RGB、深度图和点云；
- 双臂关节、TCP、速度、力矩、外力和控制状态曲线；
- Inspire 角度、位置、力、电流、温度和触觉；
- requested/safe/sent 动作曲线及其他有效数值字段。

`visualize.sink: spawn` 打开 Rerun；`save` 写入 `save_path`。`speed` 缩放 Rerun 时间线。`realtime: false` 会尽快载入全部数据，然后在 Rerun 内按时间线播放；`true` 会在导入时按该速度实时等待。两种模式的 `hardware_writes` 都是 `false`。

## Replay：机器人会动

Replay 只读取 `/control/sent_command`，即录制时真正通过安全检查并被控制桥确认发送的动作。它不会执行 `ros2 bag play`，也不会把历史 session、时间戳或 TTL 原样发回系统；每一帧都会重建为当前 session 的新鲜 `replay` 指令，并按照原采样间隔执行。

默认配置不能运动：

```yaml
replay:
  enabled: false
  operator_confirmation: ""
```

现场确认目标数据、工作空间和人员安全后，必须同时改成：

```yaml
replay:
  enabled: true
  operator_confirmation: FLEXIV-REPLAY-EXECUTE
```

启动前还必须满足：

- 在机器人本机交互 TTY 运行；
- 常驻硬件/控制栈已经启动，当前 session 已完成 F/T 清零；
- 数采 Episode Controller 已停止，系统中只有一个控制桥和一个 Replay 发布端；
- 本地许可、碰撞许可、双臂和双手状态健康；
- 当前工具/payload 配置 hash 与数据集 manifest 一致；
- episode 完整，且没有中途 Home；
- 首条动作前记录到的左右关节位置与当前 Home 一致；
- 整个动作段时间戳连续，配置的速度和 TTL 不会造成过期，调度延迟不超过
  `max_schedule_lateness_s`。

执行顺序固定为：自动走当前配置的双臂 Home，获得 `replay` 独占授权，等待 `start_delay_s`，操作员持续踩住物理脚踏，然后按采样时序发送。进入 Replay 授权后，松开脚踏、心跳/动作超时、状态异常、`Ctrl-C` 或正常结束都会经现有控制桥进入 hold。Replay 不会伪造脚踏，也不能绕过关节限位、速度、外力、碰撞和在线状态检查。

`speed` 最大为 `1.0`，不会快于原始示教。`include_hands: false` 可只执行录制的双臂动作；双手仍须在线，因为这是当前控制桥的整机安全门。
