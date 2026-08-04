# RPC interfaces

Canonical protobuf schemas shared by the Flexiv daemon and external policy API.
Generated files are checked in next to their owning application packages.

- `policy_service_v1.proto`: backward-compatible aggregate observation and
  fixed 30D safe-control API.
- `policy_data_v2.proto`: additive schema discovery, independent multi-rate
  channels, asynchronous subscription, synchronized snapshot and versioned
  action envelopes.

Regenerate all checked-in bindings with `scripts/generate_protos.sh`.
