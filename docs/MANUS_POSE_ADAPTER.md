# Isaac `/xr_teleop/hand` → Inspire 手指适配

本项目直接订阅 Isaac Teleop ROS 2 的原始 `geometry_msgs/PoseArray`
`/xr_teleop/hand`，不依赖不存在的 `/manus/*/joint_states`，也不把
模式相关、已经重定向过的 `/xr_teleop/finger_joints` 当作通用 MANUS
原始数据。

消息布局固定为 50 个 Pose：

- `poses[0:25]`：左手 OpenXR `WRIST..LITTLE_TIP`
- `poses[25:50]`：右手 OpenXR `WRIST..LITTLE_TIP`
- OpenXR/MANUS 内部模型每手有 26 个关节，但 ROS transport 省略
  `PALM`（index 0），因此每手传输 25 个 Pose
- 每手 local index 0 是 `WRIST`，local index 1 是
  `THUMB_METACARPAL`
- Isaac 对无效关节填充“零位置 + 单位四元数”；适配器把任一必需
  关节的这种值视为整手无效

适配器使用父子骨段的相对旋转推导四指 MCP/PIP/DIP 和拇指
CMC/MCP/IP 屈曲特征；拇指外展使用腕坐标系中的有符号掌骨方向角。
这些特征只进入现场标定映射，不直接当作 Inspire 命令。

这里不需要 Sharpa URDF：Sharpa 只属于 NVIDIA 示例的目标手模型。本项目
没有把 MANUS 关节先变成 Sharpa 关节，而是把原始 OpenXR 特征直接标定为
Inspire 的六个执行器端点。当前映射也不做 Inspire 手部 IK，因此不需要
Inspire URDF；若将来改为模型 IK，目标模型才应换成经过核对的 Inspire
URDF。

`ros2_ws/src/flexiv_inspire_control/config/manus_calibration_template.yaml`
默认 `calibrated: false`。默认
空 `manus_calibration` 参数时系统仅允许手臂遥操作，手的 valid mask
不会置位。必须在真实操作者、手套和双手上分别采集开/闭端点，审核
权重与方向，然后复制模板、设为 `calibrated: true` 并把
`manus_calibration` 指向该文件。错误 pose 数、零 Pose、NaN/Inf、
非法四元数、缺失特征、未知标定字段或超时都会 fail closed，不会沿用
上一帧手命令。

现场工具 `flexiv-inspire-manus-calibrate` 分两次采集双手
张开/握拳各 90 帧，以中位数生成左右手独立端点。完整命令见
`docs/RUNBOOK.md` 4.4。
