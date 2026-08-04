# PolicyService v1 and PolicyDataService v2

The canonical schemas are in `libs/rpc_interfaces/proto`. Generate both sets
of Python bindings inside `envs/ros-py312` with:

```bash
./scripts/generate_protos.sh
```

The server always uses TLS. Binding outside loopback additionally requires a
client CA and mutual TLS. A remote client can acquire a short lease only while
the local state reports the current session F/T-zeroed, locally policy-armed,
pedal-valid, and all hardware online. The API contains no RPC for F/T zeroing,
robot enable, or local arming.

Client monotonic clocks are diagnostic only. TTL starts at server receipt and
each action point uses a relative `execute_after_s`, so clocks on different
machines are never compared.

PolicyDataService v2 is additive and served on the same TLS port. Read-only
`DescribeSystem`, `SubscribeSamples` and `GetSnapshot` do not acquire a motion
lease. Each channel carries source/mapped/receive timing outside its tensor;
clients should open separate subscriptions for high-rate state and images to
avoid transport head-of-line blocking. `INTERPOLATE` is limited to floating
numeric channels and uses quaternion SLERP for `tcp_pose`.

External LeRobot environments use the portable Python 3.10+ packages in
`libs/policy_contracts` and `libs/policy_client`. They contain no ROS, RDK,
Inspire, or RealSense dependency. `SyncPolicyProfileClient` compiles a named
profile against `DescribeSystem`, rejects missing/stale channels, and produces
the exact `observation.state` and image keys declared by the dataset contract.
It is read-only: P3 exposes no lease or action call.

Smoke the real wire path after starting the policy server:

```bash
flexiv-inspire-policy-data-smoke \
  --server-ca certs/server.crt \
  --duration 10 \
  --subscribe arm.left.q=200 \
  --subscribe camera.head.rgb=15
```

The v2 action method remains subject to the same v1 lease, local authorization,
pedal, TTL and stop authority. Neither version exposes Enable, Home or F/T zero.
