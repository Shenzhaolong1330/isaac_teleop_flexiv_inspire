# Safety invariants

These invariants apply to every real-hardware process in this repository. A
failure to prove any invariant is a command rejection and a latched hold, not a
warning.

1. Startup is observation-only. Connecting a robot or a hand does not enable,
   home, open, close, zero, or move it.
2. A new RDK connection, tool/payload configuration change, process restart, or
   new hardware-session UUID invalidates the force/torque-zeroed state.
3. `ZeroFTSensor` is maintenance-only. It requires a local operator token,
   unloaded and stationary arms, stationary hands/cables, and successful
   before/after residual checks on both arms. A remote policy cannot request or
   confirm it.
4. Teleoperation, policy, and replay are mutually exclusive. Control requires a
   local permit, the current session's F/T-zero result, a physical foot pedal,
   a live source heartbeat/deadman, a fresh command TTL, healthy hardware, and
   all safety checks.
5. Invalid Rotation-6D rejects the whole bimanual command. NaN/Inf, a near-zero
   first column, near-collinear columns, failed orthogonalization, or an invalid
   determinant is never replaced silently.
6. Timeouts hold measured arm poses and measured hand angles. They never open
   the hands or replay a stale target.
7. Requested, safe, and sent commands are distinct, timestamped records.
   `safe_command` is the default training action; MCAP remains the immutable
   asynchronous source of truth.
8. A watchdog, communication loss, lease loss, pedal release, source change, or
   hardware fault transitions to `HOLD_LATCHED`. Resuming requires an explicit
   local re-arm; reconnecting alone cannot resume motion.
9. Device and host receive timestamps, sequence numbers, validity, and age are
   retained. Missing data is invalid, never a zero-valued observation.
10. The legacy workspace is neither sourced, imported, linked, nor executed.
