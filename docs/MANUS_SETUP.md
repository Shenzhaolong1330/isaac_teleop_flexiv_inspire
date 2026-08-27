# MANUS SDK 3.1.1 独立构建与验证

## 来源与隔离

使用机器上已有的 SDK：

```text
/home/hb/ws/manus_sdk/ManusSDK_v3.1.1/SDKClient_Linux/ManusSDK
```

它已实体复制到：

```text
/home/hb/chp_ws/rl100_dp30/isaac_teleop_flexiv_inspire/vendor/ManusSDK
```

`vendor/` 已加入项目 `.gitignore`。目标目录不是软链接，目录内也没有
软链接；头文件和共享库与来源逐文件比较一致。没有加载外部工作空间，
也没有修改全局环境。

## 构建命令

构建使用 Isaac 环境、Python 3.12、项目局部 CUDA 12.8，以及上游
Isaac Teleop 1.3.131 的官方 MANUS CMake 目标：

```bash
cd /home/hb/chp_ws/rl100_dp30/isaac_teleop_flexiv_inspire
source scripts/env/activate_isaac.sh

cmake -S third_party/IsaacTeleop -B build/manus-isaac -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/home/hb/chp_ws/rl100_dp30/isaac_teleop_flexiv_inspire/third_party/IsaacTeleop/install/manus-isaac \
  -DISAAC_TELEOP_PYTHON_VERSION=3.12 \
  -DUV_EXECUTABLE=/home/hb/.local/bin/uv \
  -DMANUS_SDK_ROOT=/home/hb/chp_ws/rl100_dp30/isaac_teleop_flexiv_inspire/vendor/ManusSDK \
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

站点补丁完成三件事：

- `ISAAC_TELEOP_MANUS_WRIST_SOURCE=controllers` 固定由 Quest Touch 提供腕部位姿；
- `ISAAC_TELEOP_MANUS_OPENXR_ENABLED=0` 让插件跳过 OpenXR 会话，手指数据
  不等待 CloudXR 或视频；
- `ISAAC_TELEOP_MANUS_ERGONOMICS_UDP=127.0.0.1:15053` 从同一个 Integrated
  SDK 实例输出 Ergonomics，避免第二个 MANUS Core 争抢手套。

`manus_ergonomics_source` 将 SDK 的度数转换为 ROS 标准弧度，发布
`/manus/left/ergonomics` 和 `/manus/right/ergonomics`。灵巧手命令使用
Ergonomics；OpenXR skeleton 仅作为旧标定/诊断兼容路径。补丁脚本可重复执行。

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

Flexiv/Inspire 运行链路不实例化 NVIDIA Sharpa retargeter，所以不需要
Sharpa URDF。Quest 的腕部位姿与 MANUS Ergonomics 是相互独立的输入；
Quest 视频/OpenXR 显示失败不会阻断手指 Ergonomics 数据。

## Ergonomics 现场标定

启动 `robot record` 后保持双手完全张开：

```bash
flexiv-inspire-manus-calibrate capture-ergonomics \
  --pose open --frames 90 \
  --output artifacts/calibration/manus_ergonomics_open.yaml
```

再自然握拳（拇指同时完成对掌）：

```bash
flexiv-inspire-manus-calibrate capture-ergonomics \
  --pose closed --frames 90 \
  --output artifacts/calibration/manus_ergonomics_closed.yaml
```

生成左右手独立标定：

```bash
flexiv-inspire-manus-calibrate finalize-ergonomics \
  --open artifacts/calibration/manus_ergonomics_open.yaml \
  --closed artifacts/calibration/manus_ergonomics_closed.yaml \
  --template ros2_ws/src/flexiv_inspire_control/config/manus_ergonomics_calibration_template.yaml \
  --output artifacts/calibration/manus_ergonomics_site.yaml
```

最后将 `config/sensors.yaml` 的 `teleop.manus_calibration` 改为生成文件。
仓库自带 bootstrap 范围可用于首次低速检查，正式数采前应完成操作者标定。

## OpenXR/Quest 边界

默认数采启动器把 MANUS 插件运行在显式 `MANUS-only` 模式，不创建
OpenXR 会话。Quest 位姿由独立 `xr_raw_ros_source` 输出，XR 视频由独立
视频链路处理；它们失败时 MANUS Ergonomics 仍然可用。

本次 smoke 没有执行手套校准、固件升级、触觉输出或任何设备写操作。
