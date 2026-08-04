# MANUS SDK Ergonomics → Inspire 手指适配

当前主链路不再从 OpenXR 手部骨架反推手指角度。Isaac Teleop
的 MANUS Integrated SDK 实例直接读取 Ergonomics 流，经本机 UDP
传给 `manus_ergonomics_source`；该 ROS 节点将度转为弧度，分别发布：

- `/manus/left/ergonomics`
- `/manus/right/ergonomics`

每只手包含 20 个 SDK Ergonomics 通道：拇指和四指的
MCP spread/stretch、PIP stretch 和 DIP stretch。`teleop_input` 按左右手
独立标定把它们映射到 Inspire RH56 的六个执行器，并对命令做
低通、死区、变化率限制和 `[0,1000]` 夹取。

Quest 只提供左右腕的位置和姿态，MANUS Ergonomics 只提供手指
动作。两条输入互相独立：XR 视频显示失败不会阻断 MANUS 手指流。

这里不需要 Sharpa URDF：Sharpa 只属于 NVIDIA 示例的目标手模型。
本项目把 SDK 的人手关节角直接标定到 Inspire 六个执行器端点，
不做手部模型 IK，因此也不需要 Inspire URDF。如果未来改成模型 IK，
目标模型才应换成经过核对的 Inspire URDF。

`manus_ergonomics_bootstrap.yaml` 使用本地已有的实测范围，仅用于首次
低速验证。正式数采前，用 `capture-ergonomics` 分别采集自然
完全张开和自然握拳各 90 帧，再用 `finalize-ergonomics` 生成这一
操作者/手套的左右手独立标定。完整命令见 `docs/MANUS_SETUP.md`。

旧的 `/xr_teleop/hand` 50-Pose 骨架标定仍保留向后兼容，但不再是
`config/sensors.yaml` 的默认路径。骨架流中的零 Pose、非法四元数、
NaN/Inf 和缺失特征仍会 fail closed。
