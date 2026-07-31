# Flexiv RDK daemon

This is a Python 3.10-only process. It owns both Flexiv connections and must not
load ROS, Isaac, LeRobot, or any code from another workspace. RDK is pinned to
`flexivrdk==1.9.0`.

The default CLI mode is `--mock`. A live connection additionally requires
`--hardware`. Any call that changes robot state is blocked unless all local write
guards are deliberately enabled. Merely starting the daemon cannot enable, move,
zero, open, or home a robot.

The Flexiv upper computer remains authoritative for selecting the active tool.
The daemon only reads `Tool.name()`/`Tool.params()`: it rejects an unexpected
tool at connection time and compares the live mass, center of mass, inertia,
and TCP with the locally audited snapshot before F/T zero. It never calls a
Tool switch/add/update API.

IPC uses `AF_UNIX/SOCK_SEQPACKET`, a `0600` socket, protobuf serialization, a
64 KiB packet limit, monotonic sequence numbers, and schema version checks.

`ZeroFTSensor` is implemented as a guarded maintenance transaction. It is never
available through the remote policy service.

Cartesian motion commands include an explicit stiffness and damping-ratio
profile. The daemon checks stiffness against each connected robot's
`RobotInfo.K_x_nom`, checks damping ratio against RDK's `[0.3, 0.8]` range, and
calls `SetCartesianImpedance` before motion.

Configured Home is a separate local-only transaction. It requires successful
F/T zero for the current session, a local-TTY bootstrap token, local permission,
collision-clear, and healthy observations. The first accepted token establishes
a Home lease bound to the bridge PID, session, and RDK generation; Home does not
use the teleoperation pedal. The daemon sends the audited joint target with
`NRT_JOINT_POSITION`, requires a 150 ms keepalive, and stops into a measured
Cartesian hold on any failure.
