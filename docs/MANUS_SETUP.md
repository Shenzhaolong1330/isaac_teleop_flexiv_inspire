# MANUS SDK 3.1.1 独立构建与验证

## 来源与隔离

使用机器上已有的 SDK：

```text
/home/hb/ws/manus_sdk/ManusSDK_v3.1.1/SDKClient_Linux/ManusSDK
```

它已实体复制到：

```text
/home/hb/isaac_teleop_flexiv_inspire/vendor/ManusSDK
```

`vendor/` 已加入项目 `.gitignore`。目标目录不是软链接，目录内也没有
软链接；头文件和共享库与来源逐文件比较一致。没有加载外部工作空间，
也没有修改全局环境。

## 构建命令

构建使用 Isaac 环境、Python 3.12、项目局部 CUDA 12.8，以及上游
Isaac Teleop 1.3.131 的官方 MANUS CMake 目标：

```bash
cd /home/hb/isaac_teleop_flexiv_inspire
source scripts/env/activate_isaac.sh

cmake -S third_party/IsaacTeleop -B build/manus-isaac -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/home/hb/isaac_teleop_flexiv_inspire/third_party/IsaacTeleop/install/manus-isaac \
  -DISAAC_TELEOP_PYTHON_VERSION=3.12 \
  -DUV_EXECUTABLE=/home/hb/.local/bin/uv \
  -DMANUS_SDK_ROOT=/home/hb/isaac_teleop_flexiv_inspire/vendor/ManusSDK \
  -DMANUS_ADD_SDK_TO_BUILD_RPATH=ON \
  -DBUILD_PLUGINS=ON \
  -DBUILD_PLUGIN_OAK_CAMERA=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TESTING=OFF \
  -DBUILD_VIZ=OFF \
  -DENABLE_CLOUDXR_BUNDLE_CHECK=OFF \
  -DENABLE_CLANG_FORMAT_CHECK=OFF \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc

./scripts/apply_manus_controller_wrist_patch.sh
cmake --build build/manus-isaac \
  --target manus_hand_plugin manus_hand_tracker_printer --parallel 8
cmake --install build/manus-isaac --component manus
```

站点补丁使 `ISAAC_TELEOP_MANUS_WRIST_SOURCE=controllers` 可显式关闭 Quest
光学手腕根。运行脚本会设置该变量，因此腕部始终来自 Quest Touch 控制器，
MANUS 只提供手指关节；拿起控制器后不会因光学手跟踪消失而把手指流标成
无效。补丁脚本可重复执行。

安装产物：

```text
third_party/IsaacTeleop/install/manus-isaac/plugins/manus/manus_hand_plugin
third_party/IsaacTeleop/install/manus-isaac/bin/manus_hand_tracker_printer
third_party/IsaacTeleop/install/manus-isaac/lib/libIsaacTeleopPluginsManus.so
third_party/IsaacTeleop/install/manus-isaac/lib/libManusSDK_Integrated.so
```

`ldd` 对插件和诊断工具均没有 `not found`；安装后的 RUNPATH 分别指向
项目自己的 `third_party/IsaacTeleop/install/manus-isaac/lib`。

## 一次性只读 smoke 结果

只执行过一次 10 秒只读启动探测：

```bash
source scripts/env/activate_isaac.sh
timeout --signal=TERM --kill-after=2s 10s \
  third_party/IsaacTeleop/install/manus-isaac/plugins/manus/manus_hand_plugin
```

关键结果：

```text
[Manus] SDK initialized successfully
Successfully connected to Manus host after 1 attempts
Plugin running. Press Ctrl+C to stop.
0x318DCDA6 is connected as MetaglovePro Dongle
0x2D99C54 is connected as Prime1 Dongle
Prime1 license: Cust=Manus, Seat=1, EDate=9 July, 2505
MANUS_SMOKE_EXIT=124
```

超时退出码 124 是预期的外部 timeout 终止，不是初始化失败。SDK 输出中
还包含许可密钥字段；出于凭据安全考虑未写入仓库。

当前没有复现历史上的 `No compatible license found`。本次检测发现有效
许可，因此没有许可阻塞，也没有修改或重试许可。`lsusb` 同时能看到
MANUS Sensor Dongle。

Flexiv/Inspire 运行链路使用仓库提供的
`flexiv-inspire-xr-raw-source`，直接输出 `/xr_teleop/hand` 的
left25+right25 原始 OpenXR Pose。它不实例化 NVIDIA Sharpa
retargeter，所以不需要 Sharpa URDF；后级再用现场张开/握拳端点映射为
Inspire 六路命令。

## 当前 OpenXR/Quest 状态

smoke 没有启动 CloudXR/OpenXR runtime，所以插件明确降级为
`Manus-only mode`；这不影响 SDK、许可和 dongle 发现验证。当前
`NV_CXR_RUNTIME_DIR` 未设置，CloudXR EULA 标记不存在，且
`adb devices -l` 没有 Quest。完成 Quest 连接和由操作者接受 CloudXR
EULA 后，才能继续验证 OpenXR hand injection、Quest 位姿和完整遥操作
数据流。

本次 smoke 没有执行手套校准、固件升级、触觉输出或任何设备写操作。
