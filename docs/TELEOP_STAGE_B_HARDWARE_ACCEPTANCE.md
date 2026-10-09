# 阶段 B：真机电脑迁移与分级验收

本指南由阶段 B 离线工程交付。开发机没有硬件、不是部署机；下述现场命令未在开发机执行。当前结论仅为离线 PASS，部署集成、SDK 能力、物理碰撞安全、关节连续性与实测实时性能均 **NOT_RUN**。先读 [阶段 A 契约](TELEOP_STAGE_A_CONTRACT.md)、[阶段 B 契约](TELEOP_STAGE_B_CONTRACT.md)、[RUNBOOK](RUNBOOK.md) 和 [环境说明](ENVIRONMENTS.md)。不用重装已工作的全部环境。

所有路径、序列号、session、环境和数值须按现场实际填写。命令中使用 `$SITE_ROOT`（旧 main）、`$ACCEPT_ROOT`（新版独立验收目录）、`$EVIDENCE`（现场证据目录）。含硬件写入的步骤均有前置批准门；未通过前置门就停在观察阶段。不要复制开发机示例覆盖现场配置，不通过真实撞击、危险受阻或越限验证故障。

## 1. 清点和保护旧部署

在原仓库目录，先记录，不切换分支、不安装、不启动新控制：

```bash
export SITE_ROOT=/absolute/path/to/deployed/repo
export EVIDENCE=/absolute/path/to/private/stage-b-acceptance-YYYYMMDD
mkdir -p "$EVIDENCE"
chmod 700 "$EVIDENCE"
git -C "$SITE_ROOT" rev-parse --show-toplevel > "$EVIDENCE/site-root.txt"
git -C "$SITE_ROOT" rev-parse HEAD > "$EVIDENCE/site-head.txt"
git -C "$SITE_ROOT" branch --show-current > "$EVIDENCE/site-branch.txt"
git -C "$SITE_ROOT" status --porcelain=v1 > "$EVIDENCE/site-status.txt"
git -C "$SITE_ROOT" diff --binary > "$EVIDENCE/site-unstaged.patch"
git -C "$SITE_ROOT" diff --cached --binary > "$EVIDENCE/site-staged.patch"
git -C "$SITE_ROOT" ls-files --others --exclude-standard > "$EVIDENCE/site-untracked.txt"
ps -eo pid,ppid,args > "$EVIDENCE/processes-before.txt"
```

人工备份有效 config、tool/payload、相机/MANUS/基座标定、runtime 启动参数、原环境位置与快捷命令定义，逐项核对上面列出的未跟踪文件。数据、证书、私钥、凭据保留原目录及私有备份；不放入源码 bundle、公开日志或 Git。不要对整个家目录/数据目录做无差别复制。备份环境清单：

```bash
export ROS_ENV=/actual/path/to/working/ros-py312
export RDK_ENV=/actual/path/to/working/rdk-py310
export DATA_ENV=/actual/path/to/working/data-py312
export ISAAC_ENV=/actual/path/to/working/isaac-py312
"$ROS_ENV/bin/python" -m pip freeze > "$EVIDENCE/ros-freeze.txt"
"$RDK_ENV/bin/python" -m pip freeze > "$EVIDENCE/rdk-freeze.txt"
"$DATA_ENV/bin/python" -m pip freeze > "$EVIDENCE/data-freeze.txt"
"$ISAAC_ENV/bin/python" -m pip freeze > "$EVIDENCE/isaac-freeze.txt"
```

记录实际工具/payload/TCP、握持方式、软件与固件版本、F/T 流程、机器人/手/相机标识，保留旧版短正常 episode 作为对照。若旧 `main` 不是预期分支或有未知并发修改，先完成清点与人工归属确认。

切换环境或启动新版前，在旧部署按其原有方式停止控制并确认安全保持、旧进程已退出。已有 `robot stop`/旧启动器可按旧 RUNBOOK 使用；它会停止进程，执行前须确认持物与现场状态。随后保存 `ps`、`/control/state` 和硬件状态。**不能同时让两套桥/daemon/手驱动占用硬件**。

## 2. 无 push 的源码交付

交付端最终提交后生成 `teleop-stage-b.bundle` 和 `delivery.json`/SHA256 sidecar。bundle 只包含 `refs/heads/develop/teleop` 及其可达历史，包含阶段 A/B 与正式指南，不含 `.local`、凭据、数据和其他 refs。生成端模板（只由开发端执行）：

```bash
git bundle create .local/teleop-stage-b.bundle develop/teleop
git bundle verify .local/teleop-stage-b.bundle
git bundle list-heads .local/teleop-stage-b.bundle
sha256sum .local/teleop-stage-b.bundle
```

经可移动介质等方式转交这三个文件。目标端先对照交付端提供的 SHA256 和最终 SHA；不要只相信同介质上任意自报的值：

```bash
export BUNDLE=/absolute/path/to/teleop-stage-b.bundle
sha256sum "$BUNDLE" | tee "$EVIDENCE/bundle-sha256.txt"
git -C "$SITE_ROOT" bundle verify "$BUNDLE" |& tee "$EVIDENCE/bundle-verify.txt"
git bundle list-heads "$BUNDLE" | tee "$EVIDENCE/bundle-heads.txt"
# 应只有 refs/heads/develop/teleop；SHA 必须与交付记录一致。
git -C "$SITE_ROOT" fetch "$BUNDLE" \
  refs/heads/develop/teleop:refs/remotes/stage-b-delivery/develop/teleop
export DELIVERY_SHA=$(git -C "$SITE_ROOT" rev-parse refs/remotes/stage-b-delivery/develop/teleop)
git -C "$SITE_ROOT" merge-base HEAD "$DELIVERY_SHA"
git -C "$SITE_ROOT" log --oneline --left-right HEAD..."$DELIVERY_SHA" > "$EVIDENCE/history-comparison.txt"
git -C "$SITE_ROOT" merge-base --is-ancestor d36037c "$DELIVERY_SHA"
git -C "$SITE_ROOT" merge-base --is-ancestor 857e16e "$DELIVERY_SHA"
```

任何哈希/历史/refs 不符即停止，不 reset main。如果没有共同历史，保留旧 main，使用完整 bundle 在新目录克隆并验收；人工比对差异后另定整合方案，不能自动 cherry-pick 依赖或假定现场 main 等于开发基线。

推荐 detached worktree，保持旧 main 和工作区完整：

```bash
export ACCEPT_ROOT=/absolute/path/to/separate/stage-b-checkout
# ACCEPT_ROOT 必须尚不存在，且不是旧部署或数据目录。
git -C "$SITE_ROOT" worktree add --detach "$ACCEPT_ROOT" "$DELIVERY_SHA"
git -C "$ACCEPT_ROOT" status --short
```

无共同历史时替代为 `git clone --branch develop/teleop "$BUNDLE" "$ACCEPT_ROOT"`。正式通过后，可在独立整合分支审核合并到现场 main；只在旧 main 是交付 SHA 祖先且没有现场修改时考虑 fast-forward。现场配置和提交不同则先人工解决并重新验收，禁止无条件覆盖/强制重置。

## 3. 复用环境、增量安装与构建

记录将要修改的环境绑定并先停止旧进程。以下方案复用已工作的四个环境，只更新本仓库 editable 包；旧源码、第三方 SDK/驱动/Isaac/MANUS/CUDA 保留。它会暂时将原环境中的本仓库包指向验收目录，因此验收期间不能再用该环境启动旧源码；第 8 节给出恢复命令。需要同时保留两套运行环境时，应复制环境或使用现场既有环境管理方法，不能混用两个 editable 根。

```bash
mkdir -p "$ACCEPT_ROOT/envs"
# 只在这些路径不存在时建立链接；不加 -f，不覆盖已有环境。
ln -s "$ROS_ENV" "$ACCEPT_ROOT/envs/ros-py312"
ln -s "$RDK_ENV" "$ACCEPT_ROOT/envs/rdk-py310"
ln -s "$DATA_ENV" "$ACCEPT_ROOT/envs/data-py312"
ln -s "$ISAAC_ENV" "$ACCEPT_ROOT/envs/isaac-py312"
"$ROS_ENV/bin/python" -m pip install --no-deps -e "$ACCEPT_ROOT/libs/control_core" -e "$ACCEPT_ROOT"
"$RDK_ENV/bin/python" -m pip install --no-deps -e "$ACCEPT_ROOT/libs/control_core" -e "$ACCEPT_ROOT/apps/flexiv_daemon"
"$DATA_ENV/bin/python" -m pip install --no-deps -e "$ACCEPT_ROOT/libs/control_core" -e "$ACCEPT_ROOT"
"$ISAAC_ENV/bin/python" -m pip install --no-deps -e "$ACCEPT_ROOT/libs/control_core" -e "$ACCEPT_ROOT"
```

保留/显式关联现场已工作的 `third_party`、MANUS 插件与 vendor 路径；bundle 不携带未跟踪安装产物。不要运行全量 `create_envs.sh` 或换 SDK 来消除未知现场问题。依赖增量为 core 的 PyYAML（原 ROS/RDK 通常已有）和生成的 RDK IPC Python binding 所需 protobuf `>=5.29,<6`；RDK 仍 1.9.x、Python 3.10，ROS/data/Isaac Python 3.12。先检查，不满足时只安装缺项；留存安装前后 `pip freeze`。不要更改策略客户端环境和 CUDA。

IPC 新增 world_session_authority/mapping_epoch 和 sdk_returned_sides，daemon 与桥必须一起更新。本次没有改变 ROS message，仍需构建新的 Python 包/入口。已提交生成物可直接用；需重生成时在 ROS 环境安装已验证的生成器：

```bash
"$ROS_ENV/bin/python" -m pip install 'grpcio-tools==1.71.0' 'protobuf==5.29.6'
"$ROS_ENV/bin/python" -m grpc_tools.protoc \
  --proto_path="$ACCEPT_ROOT/libs/rpc_interfaces/proto" \
  --python_out="$ACCEPT_ROOT/apps/flexiv_daemon/src/flexiv_rdk_daemon/generated" \
  "$ACCEPT_ROOT/libs/rpc_interfaces/proto/rdk_ipc.proto"
```

RDK 环境若低于 protobuf 5.29，只增量更新其 protobuf 并保留回退版本；不在 RDK 环境安装 ROS。

ROS 环境已有 `.pth` 可能指向旧 daemon 源码。先列出、保存并人工更新本项目所属的那一项，不批量删 `.pth`：

```bash
"$ROS_ENV/bin/python" -c 'import site; print("\n".join(site.getsitepackages()))'
# 在上面目录查看 *.pth，备份旧文件，确认 typed IPC 路径为 ACCEPT_ROOT/apps/flexiv_daemon/src。
# 进入一个新 shell，只加载新版环境与新版 overlay。
export ROOT="$ACCEPT_ROOT"
source "$ROOT/scripts/env/activate_ros.sh"
colcon --log-base "$ROOT/ros2_ws/log" build \
  --base-paths "$ROOT/ros2_ws/src" --build-base "$ROOT/ros2_ws/build" \
  --install-base "$ROOT/ros2_ws/install" --symlink-install \
  --cmake-args -DPython3_EXECUTABLE="$ROS_ENV/bin/python3"
source "$ROOT/ros2_ws/install/setup.bash"
python -c 'import isaac_teleop_core.world_teleop as w, flexiv_inspire_control.world_bridge as b, flexiv_rdk_daemon.generated.rdk_ipc_pb2 as p; print(w.__file__); print(b.__file__); print(p.__file__)'
python -c 'import flexiv_inspire_isaac.cli as c; print(c.__file__)'
```

输出必须全部属于 ACCEPT_ROOT；`PYTHONPATH`、`AMENT_PREFIX_PATH`、`COLCON_PREFIX_PATH`、editable metadata 或 namespace 路径中出现旧本仓库即停止并修正。不要 source 两套 install/setup.bash。测试 RDK 环境也要确认 backend/nullspace 的 `__file__`。使用单独 runtime_root、session、数据目录；`commands.rdk_daemon` 的绝对可执行路径必须指向复用后的正确环境，不能继续使用开发机/hb 示例。

在私有现场副本修改组合 site 配置，不覆盖旧配置。相对资产路径依赖 site 文件位置，必须在 validate/render 后核对每一条 resolved 路径。先保持 incremental、nullspace disabled、原映射/阻抗；之后 world 模式配置见第 5 节：

```bash
export SITE_CONFIG=/absolute/path/to/private/stage-b-site.yaml
flexiv-inspire --config "$SITE_CONFIG" validate | tee "$EVIDENCE/config-validate.json"
flexiv-inspire --config "$SITE_CONFIG" render | tee "$EVIDENCE/rendered-paths.json"
flexiv-inspire --config "$SITE_CONFIG" record --dry-run > "$EVIDENCE/startup-commands.json"
```

render/dry-run 不接触硬件。`record` 没有 `--dry-run` 会启动设备，并可能按 auto_reset_before_record 执行 Reset/F/T/Home/开合手，因此不能在观察阶段直接运行。验收方案将 auto_reset_before_record 设 false，硬件准备按原 RUNBOOK 单独批准执行，现场工具/F/T 设置不随本文自动更改。

## 4. 先补验阶段 A

现场已批准原旧模式的小幅动作后，用新版代码、旧 incremental 与原有效现场参数短录一条；先验证真实SDK/手驱动回执的含义和时钟，不能把 sdk_returned 等同控制器确认或物理到位。保留：

- requested/safe/sent 命令 ID，逐臂/逐手 submitted、sdk_returned、accepted、失败/替换/过期证据；
- F/T/session/tool 哈希，实测反馈与命令分离，源/接收/映射时间、age/validity；
- 静止、暂停/恢复、保存后的 MCAP、manifest、quality、provenance；
- 实际频率、磁盘/队列、归档完整性与秘密脱敏、新旧节点协议兼容。

导出真实完整 episode（完整目录不可只拷 MCAP）：

```bash
export MANIFEST=/actual/episode/manifest.json
python -m flexiv_inspire_isaac.data_pipeline.action_contract "$MANIFEST" "$EVIDENCE/stage-a-supervision.json"
python -c 'import os; from pathlib import Path; from flexiv_inspire_isaac.data_pipeline.provenance import verify_archive; print(verify_archive(Path(os.environ["MANIFEST"]).parent)["complete"])'
```

不要用 `--allow-legacy` 来掩盖新数据缺事件。回执/时间/必需模态/归档链路失败即停止阶段 B；保持故障数据和日志，在旧模式修复基础链路。

## 5. world 只读配置、人工参考与预览

填写安装模板的实测双基座 world、共同轴 A/比例 s、controller→掌心/任务参考和 TCP→同义参考偏置。左/右任务参考坐标约定必须相同，不能只填写任意零点。`tool_identity` 填入已审计工具/payload 的 canonical SHA256；可用原 RUNBOOK 的 `flexiv-rdk-daemon --config "$RDK_CONFIG" --print-tool-payload-hash` 查询，它拒绝未完成现场审计的记录。记录实际输入 provider、操作者声明 tracking_origin_id，现有输入 origin_reported=false。

先保持以下门禁：daemon 没有写 permit/CLI 写授权或仅运行只读配置；`teleop.control_enabled: false`；未发 world start；手无接管。将 site 的 teleop.mapping_mode 设 world_absolute，并填写 installation、待生成 calibration 路径、execution config；旧 mapping 参数继续保留用于兼容。桥在未标定时报 INVALID、只提供观测，不运动。

从 render 结果取得真实 `$CONTROL_CONFIG/$TELEOP_CONFIG`，沿用现场已工作的只读 XR/MANUS 源和单一 daemon。手/相机按旧 RUNBOOK 的只读入口运行，不能同时占用旧进程。分别启动桥和输入，便于仅重启桥而保持输入 boot identity：

```bash
export SESSION_ID=actual-unique-stage-b-session
export RDK_SOCKET=/actual/stage-b/runtime/rdk.sock
# 分别在两个已加载新版 ROS overlay 的终端运行。
flexiv-inspire-control-bridge --ros-args --params-file "$CONTROL_CONFIG" \
  -p session_id:="$SESSION_ID" -p rdk_socket:="$RDK_SOCKET"
flexiv-inspire-teleop-input --ros-args --params-file "$TELEOP_CONFIG" \
  -p session_id:="$SESSION_ID" -p command_enabled:=false
```

此时初始化目标参考必须由现场安全地准备机器人（机械臂此刻不由遥操作运动）。只读采集：

```bash
python -m isaac_teleop_core.world_cli capture "$EVIDENCE/reference.json" --duration-s 3
python -m isaac_teleop_core.world_cli calibrate "$INSTALLATION_TEMPLATE_FILLED" \
  "$EVIDENCE/reference.json" "$EVIDENCE/mapping-epoch-01"
python -m isaac_teleop_core.world_cli validate \
  "$EVIDENCE/mapping-epoch-01/installation.yaml" "$EVIDENCE/mapping-epoch-01/calibration.yaml"
```

采集要求机器人 ArmState connected/无 fault、acquisition.valid/timing_valid，时间域 host_monotonic；human/robot 两侧新鲜、稳定、配对。失败不要强行放宽阈值，先检查输入/坐标/时间/摆放。静止数据没有估计未知轴/比例/工具偏置。

将 site 路径改成输出 installation/calibration，重新 render。**保持输入节点持续运行，只重启桥**并加载新 CONTROL_CONFIG；重启输入会产生新的 boot UUID，需重新 capture/calibrate，不能复用旧标定来消除报错。以下无运动目标预览先通过：

```bash
ros2 topic echo /teleop/world_state std_msgs/msg/String
ros2 topic echo /control/state flexiv_inspire_interfaces/msg/ControlState
```

应显示 PREVIEW 和 desired，物理踏板为实际状态；不应出现有效硬件发送或运动。逐轴小幅移动、非交换转腕、双手张开/合拢、闭合回访；对照目标方向、比例、姿态、掌心/任务参考点和双手相对位姿。保存截图/JSON目标、安装测量和误差。方向/左右/单位/姿态/工具偏置异常即停。

## 6. 分级运动、录制、Home 与软零空间

### 6.1 冻结批准和逐臂诊断

现场按任务冻结 world_execution limits（单位见模板），将 ApprovedRegion 填入现场批准的公共 TCP 凸区域、approved_by/证据 ID、mapping_epoch 和 tool_identity。为空的 checker 会拒绝 start；这不是待移除的检查。现场需验证整个双臂连杆、工具、环境、零空间变化和接近过程。批准 TCP 区域不能证明全臂安全，若无法批准本版直达区域，就停止 ALIGNING，不能用插值代替规划。

本版始终管理两臂。逐臂诊断时，另一侧人端保持静止、另一臂也在批准范围，并监测两臂；没有单臂绕过双侧门禁的命令。先左右分别小幅测试，再双臂、静态保持。

**只有前置只读/方向/路径验收通过，现场操作员与安全观察员批准后**，才按原 RUNBOOK 取得本地写 permit、准备工具/F/T/模式，保持 nullspace 关闭，开启 daemon `control.world_absolute_enabled: true` 并重启一次 daemon（重连后重新 F/T 与执行授权）。不更改现场工具、不运行未经批准的 Reset；`robot reset` 会开合手，持物时不可当作普通 Home 使用。

准备完成后将输入 command_enabled 显式开启（原参数在初始化读取，需重启输入并重新标定；建议现场批准后先以该值启动输入但始终不 start，完成一次标定/预览）。确认 effective config/bridge PREVIEW、本地权限：

```bash
# 本地 TTY；不通过 SSH/远程策略伪造授权。
flexiv-inspire-authorize-control --session-id "$SESSION_ID" --source teleop \
  --confirm FLEXIV-CONTROL-ARM --clear-hold-latched --socket "$RDK_SOCKET"
# 既有桥的本地权限门，经过现场批准才发布。
ros2 topic pub --once /control/local_permission std_msgs/msg/Bool '{data: true}'
# 仅在完整连杆/工具/环境检查已经批准时；并记录批准证据。
ros2 topic pub --once /control/collision_clear std_msgs/msg/Bool '{data: true}'
# 操作者短暂保持当前有效人端位置；不必回初始位置。
ros2 topic pub --once /teleop/world_operation std_msgs/msg/String \
  "{data: '{\"session_id\":\"$SESSION_ID\",\"confirm\":\"WORLD-TELEOP-LOCAL\",\"operation\":\"start\"}'}"
```

token 有期限；若准备耗时到期，重新本地授权，不能改 TTL。预期 ALIGNING 缓慢接近捕获的当前映射目标，双侧最新目标/实测/accepted/变化率/新鲜度/驻留均通过后 FOLLOW。中踏板不用持续踩，真实 false 仍可 FOLLOW；松/踩踏板不重新定义映射。HOLD 原因不清楚或目标不收敛即停止。

停止与显式失效：

```bash
ros2 topic pub --once /control/stop std_msgs/msg/Empty '{}'
# 在 recenter、握持方式改变或 XR 重启之前，先停止并显式失效。
ros2 topic pub --once /teleop/world_operation std_msgs/msg/String \
  "{data: '{\"session_id\":\"$SESSION_ID\",\"confirm\":\"WORLD-TELEOP-LOCAL\",\"operation\":\"invalidate\"}'}"
```

普通短 HOLD：固定映射不变，先修复原因，新鲜反馈/追踪、批准路径、重新本地授权/start。INVALID：重新验证人工参数、capture/calibrate 新 mapping epoch，重新载入桥。小幅无身份变化的 recenter 不能保证自动识别，必须遵守显式 invalidate 流程。

### 6.2 手指、录制与数据门

FOLLOW 后人工使 MANUS 目标与当前实测手角度一致，持物时保持握持，不自动张开。读取手反馈和 `/teleop/command` 手目标，填入真实 6D targets（0–1000原始驱动单位），示例数字不可直接执行：

```bash
export LEFT_TARGETS='[现场六个数值]'
export RIGHT_TARGETS='[现场六个数值]'
# 只在两侧实测新鲜、目标匹配且现场批准时执行。
ros2 topic pub --once /teleop/world_operation std_msgs/msg/String \
  "{data: '{\"session_id\":\"$SESSION_ID\",\"confirm\":\"WORLD-TELEOP-LOCAL\",\"operation\":\"fingers\",\"targets\":{\"left\":$LEFT_TARGETS,\"right\":$RIGHT_TARGETS}}'}"
```

应拒绝大幅不匹配目标；匹配后按已有 MANUS 算法跟随，受新鲜实测附近的手目标限制，手执行接受/异常以 DFTP 证据为准。失去手反馈撤回手接管，恢复需重新显式 fingers。

`flexiv-inspire --config "$SITE_CONFIG" collect` 仅启动现场配置的 recorder/踏板路由，与独立控制进程配合；先启动 collector 再授予运动，可确保审计写入。world 下右踏板保存/下一条、左踏板丢弃/重录，均不重锚、不 Home；Quest Home 保留。先测试静止有效 FOLLOW 仍录制，保存不改变 desired/epoch，丢弃只改变 episode。

```bash
ros2 topic pub --once /episode/control std_msgs/msg/String '{data: pause_recording}'
ros2 topic pub --once /episode/control std_msgs/msg/String '{data: resume_recording}'
```

停录不撤回运动，操作者须明确 `/control/stop` 停机器人。停止机器人不代表任务成功。`robot record` 是整套前台会话编排，退出时会关闭其控制服务；独立暂停测试使用上述 episode 操作。记录完成仍需检查 manifest/task_result，不能把保存按钮当作物理任务成功。

### 6.3 Home、追踪和故障

确认 Home 既有路线和当前持物状态经现场批准；使用保留的 Quest Home/episode `home`，不要为了验收运行会开合手的 Reset。Home 时遥操作不得同时发送，审计应有 home_started/completed 或 failed。完成后显示 HOLD（等待显式 start），mapping_epoch 不变、旧 accepted 清除。

操作者在任意有效且批准的映射位置短暂保持，重新 authorize/start；检查新 execution_epoch、从 Home 后实测建立基准、ALIGNING → FOLLOW 连续。先在低风险场景测试接近途中小幅移动人端导致 HOLD，显式重新捕获恢复；不应到旧快照后突然跳到实时目标。两臂中只有一侧到位不能 FOLLOW。手指仍无接管，需另行 fingers。

安全可复现的故障项：停止追踪发布/输入节点、显式 invalidate/recenter、让批准路径检查拒绝、按冻结 timeout 无接管完成、断开模拟执行器（离线）。真实断网/急停只能按现场既有安全规程和独立观察员批准；不通过物理撞击、危险受阻或越限验证。检查故障发生到最后命令/Stop 的时差、清楚 HOLD 原因、不能自动恢复、保持原始日志。源重启/长断流/identity 变化后 INVALID，需重新标定；RDK 重连需重新工具/F/T 验证和 execution 基准。

Home/ALIGNING 后同一父 episode 的训练 segment 应变化，训练窗口不得跨边界；静止 FOLLOW 不切段，物理中踏板边缘不切无踏板连续片段。保存 world_state 两个 epoch、四层目标、回执/实测分离和完整 provenance。

### 6.4 零空间对照

先关闭零空间完成上述验证。只在映射、阻抗、增益、批准区域和任务输入保持固定后，再每侧分别启用 `control.nullspace`（配置字段与范围见阶段 B 契约）。首选 takeover_measured，manual 需独立测量、审核 q_ref。权重、q_ref 策略、失败策略、RDK 版本在正式对比前冻结。

以相同人端输入/同一路径对照 disabled/enabled；启用后重新按现场规程进入笛卡尔模式并授权接管。检查 `/maintenance/nullspace` native MCAP 记录：实际版本/模式、q_ref、weights、实测 q、elapsed_ns、sdk_returned/failed。姿态设置只在模式进入/重入，60 Hz 每步不应重复阻塞配置调用。绑定接口不支持或调用失败即拒绝启用/HOLD，不能标记已经生效。

只有真机记录能判定关节改善。比较 TCP 误差、每关节 Δq、速度/加速度、总路程、限位裕量及接触/持物情况；软偏好不是硬关节上限，也不具备碰撞避障能力。零空间改变完整连杆轨迹，原批准区域必须覆盖此次变化，否则重新批准。

## 7. 冻结指标、完整任务和结论

正式对比前签署任务/输入集、候选配置 SHA、mapping epoch、工具、模式、阈值、重复次数、统计口径与失败停止条件；不能看结果后调整阈值。分别比较映射、ALIGNING 与零空间，不同时修改映射增益、刚度和零空间权重。

| 对象 | 记录指标 | 通过条件由现场冻结 |
|---|---|---|
| 映射 | 单臂 TCP p/R、工具参考点、双手相对 p/R、回访误差 | 精度与方向/比例/公共空间正确 |
| 接管 | 捕获→首目标连续性、accepted/measured 对最新目标、左右收敛、驻留/超时 | 无跳变、批准路径、双侧满足 |
| 关节 | 每轴 Δq、dq、ddq、总路程、到 q_min/q_max 裕量 | 按任务固定门槛，不用模拟 q 证明 |
| 时间 | 命令/反馈频率、源年龄、SDK/IPC/端到端延迟、失效响应 | 实测分位数和最大值，非配置频率 |
| 数据 | received/enqueued/written/valid、缺测/队列损失、有效比例、连续窗口 | 必需模态/归档与时间关联完整 |

完成实际任务采集，验证保存/丢弃/暂停/Home 和片段 lineage；用第 4 节 CLI 导出，再按真实 conversion 配置执行 `flexiv-inspire --config "$SITE_CONFIG" convert`，新输出目录不覆盖旧数据。普通 LeRobot 相机视图仍是兼容视图，多速率训练消费 command-supervision-v2。确认 observation 实测、action 已接受命令增量和绝对手目标；缺回执 unknown，未通过数据不进入训练。

结论分别填写：离线工程 PASS；部署集成 PASS/FAIL/NOT_RUN；映射物理验收 PASS/FAIL/NOT_RUN；ALIGNING 路径与接管 PASS/FAIL/NOT_RUN；零空间实际接口与关节效果 PASS/FAIL/NOT_RUN；数据交付 PASS/FAIL。不能合并为笼统“安全”。任一失败保留证据，停止该级与下游级，不调参后覆盖旧结果。

## 8. 回退

先明确 `/control/stop`、确认 HOLD/实测静止，再停止新版输入、桥、daemon、手驱动/collector 和新版会话；持物安全由现场规程处理。保留证据目录、失败原始日志和新数据。检查没有第二套进程，不能用 reset/clean/stash 强制抹除现场工作。

若复用了原环境，将本仓库 editable 绑定恢复旧源码，并恢复已备份的本项目 `.pth` 与仅本次改变的依赖版本：

```bash
"$ROS_ENV/bin/python" -m pip install --no-deps -e "$SITE_ROOT/libs/control_core" -e "$SITE_ROOT"
"$RDK_ENV/bin/python" -m pip install --no-deps -e "$SITE_ROOT/libs/control_core" -e "$SITE_ROOT/apps/flexiv_daemon"
"$DATA_ENV/bin/python" -m pip install --no-deps -e "$SITE_ROOT/libs/control_core" -e "$SITE_ROOT"
"$ISAAC_ENV/bin/python" -m pip install --no-deps -e "$SITE_ROOT/libs/control_core" -e "$SITE_ROOT"
```

新 shell 只 source 旧环境/旧 overlay，恢复原有效 site 配置、启动方式和 runtime，打印 `__file__` 确认只有旧源码。核对旧 main SHA/工作区与备份一致（验收 worktree 不应改变它们）；按旧部署已验证流程重新准备、只启动旧实例。验收 worktree、bundle 和失败证据保留用于诊断，不强制删除或覆盖旧环境/用户修改。
