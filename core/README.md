# Isaac Teleop control core

This package is hardware-independent. It contains the canonical Rotation-6D
implementation, the 30-element policy action schema, and the fail-closed control
arbiter. It deliberately imports neither ROS nor Flexiv RDK.

All public quaternions are ROS `xyzw`. Rotation-6D is always the first two
**columns** of a rotation matrix:

`[R00, R10, R20, R01, R11, R21]`.

The hold representation is an invalid command (`valid_mask == 0`), not a vector
of zeros. In particular, the identity Rotation-6D value is
`[1, 0, 0, 0, 1, 0]`.
