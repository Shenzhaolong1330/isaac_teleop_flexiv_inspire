# RealSense recording contract

- One D455 head stream and two D435i wrist streams are selected by immutable
  serial number.
- RGB plus depth: `424×240 @ 30 Hz`; depth is aligned to the color optical
  frame and stored as native `z16` with depth scale and intrinsics.
- An organized `xyz_f32_le` point cloud is derived from that aligned depth and
  recorded atomically with the RGB/depth frame. Invalid depth pixels are NaN.
- Episode recording uses JPEG quality 90. Short calibration captures may use
  raw RGB.
- The host currently uses librealsense `2.57.7`. Build any ROS wrapper against
  that exact `/usr/local` SDK; do not install or link an apt `2.58` library into
  the same process.
- Firmware is observed and recorded, never upgraded automatically.

Each image record carries the device/source timestamp, host receive timestamp,
sequence, validity, and age. The head source timestamp defines the default
30 Hz LeRobot export timeline.

Depth and point cloud payloads are in the producer-side DeviceIO record under
the same camera envelope, so they cannot be mismatched with their RGB image.
The point cloud frame is `<camera>_color_optical_frame`; a world-frame cloud
requires an explicit camera extrinsic calibration, which is intentionally not
invented by this package.
