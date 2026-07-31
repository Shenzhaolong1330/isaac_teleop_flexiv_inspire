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
