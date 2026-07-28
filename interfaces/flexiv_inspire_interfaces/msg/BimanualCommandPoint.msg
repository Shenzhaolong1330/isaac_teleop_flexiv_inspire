# Relative time from the command issue time.
builtin_interfaces/Duration execute_after

# Cartesian Rotation-6D mode. Rotation order is declared on BimanualCommand.
float64[3] left_delta_xyz
float64[6] left_delta_rotation6d
float64[3] right_delta_xyz
float64[6] right_delta_rotation6d

# Cartesian quaternion mode (ROS xyzw); ignored in Rotation-6D mode.
float64[4] left_delta_quaternion_xyzw
float64[4] right_delta_quaternion_xyzw

# Joint-position mode; ignored in Cartesian modes.
float64[] left_arm_joint_positions
float64[] right_arm_joint_positions

# DFTP-2 actuator order:
# little, ring, middle, index, thumb_bend, thumb_rotate.
float64[6] left_hand_targets
float64[6] right_hand_targets
