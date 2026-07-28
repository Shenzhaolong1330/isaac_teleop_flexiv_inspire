# PolicyService v1

`interfaces/proto/policy_service_v1.proto` is the only canonical schema. Generate
the Python bindings inside `envs/ros-py312`:

```bash
python -m grpc_tools.protoc \
  -I interfaces/proto \
  --python_out=src/flexiv_inspire_isaac/policy_api/generated \
  --grpc_python_out=src/flexiv_inspire_isaac/policy_api/generated \
  interfaces/proto/policy_service_v1.proto
```

The server always uses TLS. Binding outside loopback additionally requires a
client CA and mutual TLS. A remote client can acquire a short lease only while
the local state reports the current session F/T-zeroed, locally policy-armed,
pedal-valid, and all hardware online. The API contains no RPC for F/T zeroing,
robot enable, or local arming.

Client monotonic clocks are diagnostic only. TTL starts at server receipt and
each action point uses a relative `execute_after_s`, so clocks on different
machines are never compared.
