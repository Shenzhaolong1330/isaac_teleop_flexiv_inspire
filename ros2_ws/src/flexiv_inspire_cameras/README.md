# RealSense first-release contract

- Three D435i streams are selected by immutable serial number.
- RGB only: `424×240 @ 30 Hz`; depth is disabled.
- Episode recording uses JPEG quality 90. Short calibration captures may use
  raw RGB.
- The host currently uses librealsense `2.57.7`. Build any ROS wrapper against
  that exact `/usr/local` SDK; do not install or link an apt `2.58` library into
  the same process.
- Firmware is observed and recorded, never upgraded automatically.

Each image record carries the device/source timestamp, host receive timestamp,
sequence, validity, and age. The head source timestamp defines the default
30 Hz LeRobot export timeline.
