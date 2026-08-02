# Rerun visualization

This package is a read-only observer for the independent Flexiv + Inspire
teleoperation workspace. It creates subscriptions only. It does not publish a
robot command, request an RDK lease, enable a robot, zero an F/T sensor, or write
Modbus registers.

The ROS callbacks use `KEEP_LAST(depth=1)` and hand work to a latest-only
background dispatcher. If Rerun or disk output is slower than the observation
rate, old pending samples are replaced instead of blocking control or building
latency.

## Live viewer

From the project root:

```bash
source scripts/env/activate_ros.sh
source ros2_ws/install/setup.bash
flexiv-inspire-rerun --spawn
```

To connect to an already running viewer:

```bash
flexiv-inspire-rerun \
  --connect rerun+http://127.0.0.1:9876/proxy
```

To record instead of opening a viewer:

```bash
flexiv-inspire-rerun \
  --save /absolute/path/episode.rrd
```

The node treats the three atomic
`/camera/{head,left_wrist,right_wrist}/color/frame` (`CameraFrame`) streams as
authoritative, along with both complete `ArmState` streams, both hand state and
raw tactile streams, requested/safe/sent commands, the command trace, and control
state. Every event uses its ROS header timestamp; acquisition validity, timing
validity, age, source sequence, and mapped host time are preserved as
metadata/timelines.

`--legacy-camera-topics` optionally displays the old standalone
`CompressedImage` topics. It never joins images to `AcquisitionInfo` by nearest
timestamp; those legacy images are explicitly marked timing-unpaired and invalid.

Tactile panels contain each of the 17 surfaces and an anatomical palm-view
atlas. Finger end/tip/pad surfaces run from top to bottom, the palm is below
the four fingers, and the thumb runs along the outside edge. The right-hand
atlas mirrors the left. The default Rerun blueprint contains exactly two
equal-width plots: one composite atlas for the left hand and one for the right.
Individual surface entities remain available for diagnostics but are not
expanded into separate default views. Values remain raw `uint16`; there is
deliberately no division by 4096.

## Hardware-free smoke

This mode does not import ROS and cannot touch hardware:

```bash
source scripts/env/activate_ros.sh
flexiv-inspire-rerun \
  --synthetic --frames 12 --save /tmp/flexiv_inspire_rerun_smoke.rrd
rerun rrd verify /tmp/flexiv_inspire_rerun_smoke.rrd
rerun rrd stats /tmp/flexiv_inspire_rerun_smoke.rrd
```

The synthetic recording includes three JPEG RGB streams, both arms and hands,
all 1062 taxels per hand across 17 surfaces, and requested/safe/sent action
differences.

## Offline episode playback

The repository-level `./scripts/visualize.sh` reads `config/playback.yaml` and
loads the selected episode's complete `deviceio.mcap` into Rerun. This path
does not initialize ROS and has no publishers or hardware connections. It adds
recorded numeric curves, RGB/depth images and point clouds to the Rerun
timeline; see `docs/PLAYBACK.md` for dataset selection and speed controls.
