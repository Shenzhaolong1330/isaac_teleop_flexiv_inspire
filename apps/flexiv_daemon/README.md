# Flexiv RDK daemon

This is a Python 3.10-only process. It owns both Flexiv connections and must not
load ROS, Isaac, LeRobot, or any code from another workspace. RDK is pinned to
`flexivrdk==1.9.0`.

The default CLI mode is `--mock`. A live connection additionally requires
`--hardware`. Any call that changes robot state is blocked unless all local write
guards are deliberately enabled. Merely starting the daemon cannot enable, move,
zero, open, or home a robot.

IPC uses `AF_UNIX/SOCK_SEQPACKET`, a `0600` socket, protobuf serialization, a
64 KiB packet limit, monotonic sequence numbers, and schema version checks.

`ZeroFTSensor` is implemented as a guarded maintenance transaction. It is never
available through the remote policy service.
