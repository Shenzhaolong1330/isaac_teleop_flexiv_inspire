# Independent Isaac Teleop for Flexiv + Inspire

This repository is a self-contained teleoperation, acquisition, visualization,
and policy-control workspace for two Flexiv arms and two Inspire DFTP-2 hands.
It owns its Python environments, ROS 2 packages, RDK daemon, hand driver,
interfaces, and data pipeline. No external robot workspace is sourced,
imported, linked, or launched.

The hardware path is fail-closed. A normal start connects read-only and publishes
observations; it does not enable an arm, zero force/torque sensors, open a hand,
return home, or send a motion target.

## Data and control path

```text
Quest PoseArray + MANUS PoseArray
              |
              v
teleop_input (SE(3) mapping + calibrated hand retargeting)
              |
              v
teleop / policy / replay arbitration
              |
              v
Rotation-6D validation + safety supervisor
       |                         |
       | Unix SOCK_SEQPACKET     | ROS 2 safe command
       v                         v
Flexiv RDK 1.9 daemon       Inspire DFTP-2 driver
       |                         |
       +------ observations -----+
                    |
          MCAP / Rerun / gRPC
                    |
             LeRobot v3 export
```

Policy actions default to 30 values: two arms each use
`delta_xyz(3) + delta_rot6d(6)`, followed by twelve hand actuator targets.
Rotation-6D is always the first two columns of a rotation matrix in column
order:

```text
[R00, R10, R20, R01, R11, R21]
```

The identity rotation is `[1, 0, 0, 0, 1, 0]`; an all-zero vector is invalid.
ROS poses remain quaternion `xyzw`, while the RDK boundary explicitly converts
to and from `wxyz`.

## Repository layout

- `core/`: Rotation-6D, command schema, safety state machine, clock mapping, and
  the ROS 2 control/teleop bridge.
- `rdk_daemon/`: isolated Python 3.10 Flexiv RDK 1.9 daemon and typed local IPC.
- `src/flexiv_inspire_isaac/dftp/`: project-owned Modbus TCP driver for DFTP-2.
- `src/flexiv_inspire_isaac/cameras/`: three-camera RGB acquisition.
- `src/flexiv_inspire_isaac/policy_api/`: TLS gRPC PolicyService v1.
- `src/flexiv_inspire_isaac/data_pipeline/`: asynchronous MCAP recording and
  deterministic LeRobot v3 export.
- `src/flexiv_inspire_isaac/rerun_viz/`: live/offline Rerun visualization.
- `interfaces/`: ROS 2 messages/actions and protobuf schemas.
- `requirements/` and `scripts/env/`: four pinned environments.

Start with the exact, fail-closed startup sequence in
[`docs/RUNBOOK.md`](docs/RUNBOOK.md). Also read `docs/ENVIRONMENTS.md`,
`docs/SAFETY_INVARIANTS.md`, `docs/MANUS_SETUP.md`, and
`docs/MANUS_POSE_ADAPTER.md` before integration.

## Environments

The machine-wide CUDA selection is intentionally untouched. Isaac commands
select `/usr/local/cuda-12.8` only inside their launcher environment. ROS 2 uses
CycloneDDS with a project-local configuration and remains localhost-only by
default.

```bash
cd /home/hb/isaac_teleop_flexiv_inspire
./scripts/verify_host_cuda.sh
./scripts/env/create_envs.sh
./scripts/env/smoke_switch_isolation.sh
./scripts/verify_independence.sh
```

The four environments are:

- `envs/isaac-py312`: Isaac Teleop 1.3.131, CloudXR/MANUS, CUDA 12.8.
- `envs/ros-py312`: ROS bridge, gRPC, Rerun, and acquisition management.
- `envs/rdk-py310`: exactly `flexivrdk==1.9.0`, with no ROS import.
- `envs/data-py312`: LeRobot 0.6.0 and offline MCAP conversion.

Canonical protobuf sources are under `interfaces/proto`; regenerate checked-in
Python stubs with:

```bash
./scripts/generate_protos.sh
```

## Build and software-only validation

```bash
source scripts/env/activate_ros.sh
colcon --log-base log/ros build \
  --base-paths interfaces/flexiv_inspire_interfaces \
               core/ros2/flexiv_inspire_control \
  --build-base build/ros \
  --install-base install/ros \
  --symlink-install \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

source install/ros/setup.bash
envs/ros-py312/bin/python -m pytest
./scripts/verify_independence.sh
```

Mock and static tests never issue hardware commands. The control bridge starts
in `DISABLED`; the DFTP driver starts read-only; the RDK daemon requires a
short-lived local write permit before accepting a command session.

## Hardware-session gate

Every new hardware session must complete the protected local
`/maintenance/zero_ft_sensors` action before `READY`. The operator must verify
the configured tool/payload, keep both arms, hands, and cables unloaded and
still, and enter the literal local confirmation
`FLEXIV-FT-UNLOADED`. The action records pre/post statistics and fails the
whole session on motion, contact, residual error, timeout, primitive failure,
or reconnect. Remote gRPC clients cannot enable hardware or initiate/confirm
this maintenance action.

After successful zeroing, actual motion still requires local authorization, a
physical pedal, a current source heartbeat/deadman, valid TTL and timestamps,
online arms and hands, and all safety checks. Any failure enters a latched hold.
Follow the staged acceptance sequence in `docs/SAFETY_INVARIANTS.md`; do not
skip directly to dual-arm motion.

## Recording, visualization, and policy

The formal recording entry point is `isaac-flexiv-episode`; it requires
a successful F/T-zero event matching the session and tool hash, then writes an
atomic manifest, ROS MCAP, and native asynchronous DeviceIO MCAP. The exact
command is in `docs/RUNBOOK.md`. `scripts/record_ros_mcap.sh` is explicitly a
ROS-only diagnostic capture and does not create a valid training episode.

Storage semantics are explicit: `ros_mcap/` contains DDS CDR messages recorded
by rosbag2, while `deviceio.mcap` defaults to `native-pre-dds`. RDK observations,
DFTP state/tactile, camera frames, and requested/safe/sent/control traces enter a
private mode-0600 Unix datagram ingress before DDS publication. The manifest
records the capture layer and producer/drop statistics. A post-DDS typed mirror
remains available only as an explicit compatibility mode.

Each typed sample retains source time, host receive time, sequence, validity,
and age. Images, force/torque, and tactile samples remain asynchronous and are
aligned only during export. Rotation-6D is recomputed from the interpolated
rotation matrix rather than linearly interpolated.

The policy server binds to loopback by default. A non-loopback endpoint must be
enabled explicitly with TLS credentials and still cannot bypass the local
hardware gate. Rerun consumes the same ROS observations and shows camera
images, poses, joint/torque/wrench data, tactile heatmaps, and
requested-versus-safe-versus-sent commands.
