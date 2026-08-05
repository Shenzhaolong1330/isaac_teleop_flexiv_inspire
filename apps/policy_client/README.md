# Copyable remote policy client

This directory is a standalone Python 3.10+ client for the TLS gRPC server in
the sibling `apps/policy_server` directory. It has no ROS, Flexiv RDK, Inspire
or RealSense dependency. Copy this complete directory to the policy machine.

## Install

```bash
cd policy_client
python -m pip install -e .
cp config.example.yaml config.yaml
```

Set the robot-host address and certificate paths in `config.yaml`. A server
bound outside loopback requires a server certificate whose SAN contains that
address and a client certificate signed by the CA configured on the server.

Verify independent fast and slow streams:

```bash
flexiv-policy-client-smoke --config config.yaml --duration 10
```

The server acquires every hardware channel at its native rate. Each client
subscription requests its own maximum rate; the effective rate is
`min(requested_rate, native_rate)`. `DROP_OLDEST` keeps latency bounded.

## Policy integration

```python
from flexiv_policy_client import MultiRatePolicyClient

client = MultiRatePolicyClient(
    target="192.168.110.221:50051",
    server_ca="certs/server-ca.crt",
    client_cert="certs/policy-client.crt",
    client_key="certs/policy-client.key",
    action_client_id="my-policy",
)
client.connect()

fast_channels = {
    "arm.left.q": 200,
    "arm.right.q": 200,
    "arm.left.tcp_pose": 200,
    "arm.right.tcp_pose": 200,
    "hand.left.angle": 200,
    "hand.right.angle": 200,
}
slow_channels = {
    "camera.head.rgb": 15,
    "camera.left_wrist.rgb": 15,
    "camera.right_wrist.rgb": 15,
    "hand.left.tactile": 15,
    "hand.right.tactile": 15,
}
client.subscribe("fast", fast_channels)
client.subscribe("slow", slow_channels)
client.wait_for_channels((*fast_channels, *slow_channels), timeout_s=10)

last_image_sequence = 0
while True:
    # A vision policy normally runs only when a new 15 Hz image arrives.
    image = client.wait_for_update(
        "camera.head.rgb",
        after_sequence=last_image_sequence,
        timeout_s=1.0,
    )
    last_image_sequence = image.sequence
    observation = client.latest_many((*fast_channels, *slow_channels))
    action24 = policy(observation)  # user policy, float32 shape [24]
    client.send_action(action24)
```

The 24D action is world-frame relative left XYZ/rotation-vector, right
XYZ/rotation-vector, then absolute normalized `[0,1]` targets for six left and
six right hand actuators. `send_action_chunk()` also accepts 1..32 future
points with server-relative `execute_after_ns` offsets.

The server also exposes optional 200 Hz force/torque channels: measured,
desired, external and interaction joint torque (`tau`, `tau_des`, `tau_ext`,
`tau_interact`), raw and compensated TCP F/T (`raw_ft`, `tcp_wrench`), and the
six Inspire actuator force values (`actual_force`). The example config puts
these in a separate `force_torque` subscription. They remain outside the
current LeRobot minimal state unless a policy explicitly consumes them.

For exact camera-time alignment, use the camera sample's
`mapped_host_time_ns` as `target_monotonic_ns` in `snapshot()` when requesting
the high-rate channels.

Force and torque are optional native channels and are not inserted into the
current LeRobot state automatically. The example config subscribes to all of
them at 200 Hz:

- `arm.{left,right}.{tau,tau_des,tau_ext,tau_interact}`: seven joint torques
  in Nm.
- `arm.{left,right}.{raw_ft,tcp_wrench}`: `[fx,fy,fz,mx,my,mz]` in N and Nm.
- `hand.{left,right}.actual_force`: six Inspire actuator forces in grams.
