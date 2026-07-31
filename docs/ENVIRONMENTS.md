# 独立运行环境

本项目只使用项目内虚拟环境、系统 ROS 2 Jazzy，以及显式选择的 CUDA
Toolkit。环境脚本不会修改全局 `.bashrc`，也不会加载任何外部工作空间
overlay。

## 已验证的主机状态

| 项目 | 已验证值 |
|---|---|
| NVIDIA 驱动 | `580.173.02` |
| 系统默认 CUDA | `/usr/bin/nvcc`，CUDA `12.0` |
| Isaac CUDA | `/usr/local/cuda-12.8/bin/nvcc`，CUDA `12.8` |
| CUDA 12.8 包 | `cuda-toolkit-12-8=12.8.2-1` |
| ROS 2 | Jazzy |
| DDS | `rmw_cyclonedds_cpp` |
| ROS Domain | `42` |
| ROS 发现范围 | `ROS_LOCALHOST_ONLY=1`、`LOCALHOST`、CycloneDDS 仅 `lo` |
| librealsense | `2.57.7`，未安装或混入 `2.58` |

`/usr/local/cuda` 和 `/usr/local/cuda-12` 通用软链接保持不存在，因此安装
CUDA 12.8 没有改变系统默认 CUDA 12.0。验证命令：

```bash
cd /home/hb/isaac_teleop_flexiv_inspire
./scripts/verify_host_cuda.sh
```

## 四个环境

| 环境 | Python | 主要锁定内容 |
|---|---:|---|
| `envs/isaac-py312` | 3.12.3 | Isaac Teleop 1.3.131、Torch 2.8.0+cu128、CloudXR |
| `envs/ros-py312` | 3.12.3 | rclpy/CycloneDDS、gRPC、MCAP、Rerun、OpenCV |
| `envs/rdk-py310` | 3.10.19 | flexivrdk 1.9.0、protobuf 5.29.5、PyYAML |
| `envs/data-py312` | 3.12.3 | LeRobot 0.6.0、MCAP、Rerun、Torch/TorchCodec cu128 |

直接依赖写在 `requirements/*.in`，完整解析结果和哈希写在
`requirements/*.lock`。重建方式：

```bash
cd /home/hb/isaac_teleop_flexiv_inspire
./scripts/env/create_envs.sh
```

`create_envs.sh` 在同步第三方锁文件后，以 `--no-deps -e` 安装本项目
Python 包。RDK 环境安装 `core` 和 `rdk_daemon`；ROS、Isaac、Data 环境
安装 `core` 和项目根包。ROS 环境另外以项目内 `.pth` 暴露 typed IPC
源码，并显式接入系统 ROS Python 依赖。

## 激活与隔离

每个新 shell 只 source 一个目标环境：

```bash
source /home/hb/isaac_teleop_flexiv_inspire/scripts/env/activate_isaac.sh
source /home/hb/isaac_teleop_flexiv_inspire/scripts/env/activate_ros.sh
source /home/hb/isaac_teleop_flexiv_inspire/scripts/env/activate_rdk.sh
source /home/hb/isaac_teleop_flexiv_inspire/scripts/env/activate_data.sh
```

激活脚本设置 `PYTHONNOUSERSITE=1`。切换环境时会先清理 Python、ROS、
CUDA 和虚拟环境路径；RDK 环境不会继承 Isaac CUDA。下列测试覆盖同一
shell 中 `isaac -> rdk -> ros -> data` 的连续切换：

```bash
./scripts/env/smoke_switch_isolation.sh
```

ROS 环境通过项目内 `.pth` 使用
`/usr/local/lib/python3.12/dist-packages/pyrealsense2`。已验证扩展文件和
`librealsense2.so` 均为 2.57.7，`pkg-config --modversion realsense2`
返回 `2.57.7`。

## 推荐 smoke

```bash
# RDK 核心/daemon 测试
source scripts/env/activate_rdk.sh
python -m pytest -q --confcutdir=tests/core tests/core

# ROS、DFTP、映射、数据、可视化与配置测试
source scripts/env/activate_ros.sh
python -m pytest -q

# 环境、CUDA 与项目导入
./scripts/env/smoke_switch_isolation.sh
```

RDK 的 `flexiv-rdk-daemon --help`、ROS 的 `flexiv-inspire --help`、
Isaac Teleop/CloudXR 模块导入，以及 Data 环境的 LeRobot、MCAP、
Rerun、TorchCodec 导入均已通过。所有 GPU smoke 都在 RTX 5070 上使用
Torch CUDA 12.8 完成。

## CloudXR 与 Quest 当前门禁

CloudXR Python/native 组件已经安装，动态库检查无缺失。当前没有
`eula_accepted` 标记；本次实施没有代替操作者接受 NVIDIA CloudXR
EULA。首次实际启动必须由操作者阅读并现场确认。

`adb devices -l` 当前没有列出 Quest，因此只完成了主机侧安装与
`--help`/动态库 smoke，尚未完成 Quest 串流验收。`coturn` 已安装但保持
`disabled`、`inactive`，只有在明确启动 CloudXR USB-local 流程时才应
按需启动。
