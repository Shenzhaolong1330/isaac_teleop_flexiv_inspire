# Camera extrinsic calibration

`robot camera-calibrate` uses an ArUco 4x4/100 marker that you print at a
known side length. It **never publishes robot commands**. Keep the normal
control bridge, local authorization, F/T safety state and pedal protections in
place; move the arm only through the usual teleoperation workflow.

Every camera has independent recording modes in `config/sensors.yaml`:

```yaml
recording: {rgb: true, depth: true, pointcloud: false}
```

RGB is the atomic camera timeline and must remain enabled. A point cloud
requires depth, but depth can be retained without retaining a point cloud.
The supplied site configuration enables point cloud only for `head`.

The capture tool needs the real camera intrinsics (`fx`, `fy`, `ppx`, `ppy`).
Copy them from the camera record's `depth_intrinsics` field into a YAML file;
the recorded depth is aligned to the color optical frame. For example:

```yaml
fx: 384.2
fy: 384.2
ppx: 211.5
ppy: 119.5
distortion: [0, 0, 0, 0, 0]
```

## Eye in hand

Use a wrist camera. Fix the printed marker rigidly in the workspace, then move
the corresponding arm through at least 12 varied positions and orientations
while keeping the marker visible:

```bash
robot camera-calibrate capture \
  --camera left_wrist --arm left --mode eye_in_hand \
  --intrinsics left_wrist_intrinsics.yaml --marker-size-m 0.080 \
  --samples 15 --output left_wrist_samples.yaml
robot camera-calibrate solve \
  --samples left_wrist_samples.yaml --output left_wrist_extrinsics.yaml
```

The result contains `tcp_T_camera`. It is the fixed transform needed with each
measured `world_T_tcp` to place wrist point clouds in world.

## Eye to hand

Use a fixed camera, normally `head`. Rigidly attach the printed marker to the
TCP, move the selected arm through varied poses, then run the same commands
with `--mode eye_to_hand`. The result contains `world_T_camera` and the fitted
`tcp_T_calibration_target`. This mode needs no prior measurement of the marker
mount transform.

Do not calibrate from nearly identical poses: include translation and rotation
in multiple axes. Visually inspect the result before using it for collision,
planning or point-cloud fusion.

## 接入录制

求解完成后，把结果文件填到 `config/sensors.yaml` 对应相机的
`extrinsics` 字段。`robot record` 会检查文件存在，把其 SHA-256 写入每个
episode manifest，并把标定身份附在原生相机记录中。

固定相机的 `world_T_camera` 会在采集端直接把点云变换到 `world`；消息的
`frame_id` 也会改成 `world`。腕部相机的 `tcp_T_camera` 必须再与同一时刻的
`world_T_tcp` 相乘，因此原始点云仍保留在腕部相机光学坐标系，供离线按时间
对齐后转换。未填写 `extrinsics` 时，所有点云明确保持在各自相机光学坐标系，
不能视为已与世界对齐。

标定采集使用相机的传感器 QoS，并按 ROS 时间选择最近的 TCP 位姿；默认只接受
相差不超过 50 ms 的配对。采样既接受足够的平移，也接受原地旋转，避免把有用
的旋转激励误删。
