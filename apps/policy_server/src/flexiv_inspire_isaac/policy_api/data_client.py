"""Reusable Python client and smoke CLI for PolicyDataService v2."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import ipaddress
import json
from pathlib import Path
import time
from typing import AsyncIterator, Iterable

import numpy as np

from .generated import policy_data_v2_pb2 as pb
from .generated import policy_data_v2_pb2_grpc as pb_grpc


_PROTO_TO_DTYPE = {
    pb.BOOL: np.dtype("?"),
    pb.UINT8: np.dtype("u1"),
    pb.UINT16: np.dtype("<u2"),
    pb.UINT32: np.dtype("<u4"),
    pb.UINT64: np.dtype("<u8"),
    pb.INT8: np.dtype("i1"),
    pb.INT16: np.dtype("<i2"),
    pb.INT32: np.dtype("<i4"),
    pb.INT64: np.dtype("<i8"),
    pb.FLOAT16: np.dtype("<f2"),
    pb.FLOAT32: np.dtype("<f4"),
    pb.FLOAT64: np.dtype("<f8"),
}


def decode_tensor(payload: pb.TensorPayload) -> np.ndarray:
    try:
        dtype = _PROTO_TO_DTYPE[payload.dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported tensor dtype enum: {payload.dtype}") from exc
    shape = tuple(int(item) for item in payload.shape)
    if not shape or any(item <= 0 for item in shape):
        raise ValueError("tensor shape is invalid")
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(payload.data) != expected:
        raise ValueError(
            f"tensor payload has {len(payload.data)} bytes, expected {expected}"
        )
    result = np.frombuffer(payload.data, dtype=dtype).reshape(shape)
    result.setflags(write=False)
    return result


class PolicyDataClient:
    def __init__(self, channel) -> None:
        self.channel = channel
        self.stub = pb_grpc.PolicyDataServiceStub(channel)

    async def describe(self) -> pb.SystemDescription:
        return await self.stub.DescribeSystem(pb.DescribeSystemRequest())

    async def snapshot(
        self,
        channels: Iterable[str],
        *,
        schema_hash: str,
        session_id: str,
        target_monotonic_ns: int = 0,
        alignment: int = pb.LATEST_CAUSAL,
        tolerance_ns: int = 0,
        max_age_ns: int = 0,
    ) -> pb.Snapshot:
        return await self.stub.GetSnapshot(
            pb.SnapshotRequest(
                expected_schema_hash=schema_hash,
                expected_session_id=session_id,
                target_monotonic_ns=target_monotonic_ns,
                channels=[
                    pb.SnapshotChannelRequest(
                        channel_id=channel_id,
                        alignment=alignment,
                        tolerance_ns=tolerance_ns,
                        max_age_ns=max_age_ns,
                    )
                    for channel_id in channels
                ],
            )
        )

    def subscribe(
        self,
        subscriptions: Iterable[tuple[str, float, int]],
        *,
        client_id: str,
        schema_hash: str,
        session_id: str,
    ) -> AsyncIterator[pb.SampleEnvelope]:
        return self.stub.SubscribeSamples(
            pb.SubscribeRequest(
                client_id=client_id,
                expected_schema_hash=schema_hash,
                expected_session_id=session_id,
                channels=[
                    pb.ChannelSubscription(
                        channel_id=channel_id,
                        max_rate_hz=rate_hz,
                        drop_policy=drop_policy,
                    )
                    for channel_id, rate_hz, drop_policy in subscriptions
                ],
            )
        )


@dataclass
class ChannelStats:
    count: int = 0
    first_receive_ns: int = 0
    last_receive_ns: int = 0
    age_sum_ns: int = 0
    max_age_ns: int = 0
    invalid: int = 0

    def observe(self, message: pb.SampleEnvelope, receive_ns: int) -> None:
        if self.count == 0:
            self.first_receive_ns = receive_ns
        self.count += 1
        self.last_receive_ns = receive_ns
        self.age_sum_ns += int(message.timing.age_ns)
        self.max_age_ns = max(self.max_age_ns, int(message.timing.age_ns))
        if not message.timing.valid:
            self.invalid += 1

    def report(self) -> dict[str, float | int]:
        duration_s = max(0.0, (self.last_receive_ns - self.first_receive_ns) / 1e9)
        rate_hz = 0.0 if self.count < 2 or duration_s <= 0.0 else (self.count - 1) / duration_s
        return {
            "count": self.count,
            "rate_hz": rate_hz,
            "mean_age_ms": 0.0 if self.count == 0 else self.age_sum_ns / self.count / 1e6,
            "max_age_ms": self.max_age_ns / 1e6,
            "invalid": self.invalid,
        }


def _subscription(value: str) -> tuple[str, float, int]:
    channel_id, separator, raw_rate = value.partition("=")
    if not separator or not channel_id.strip():
        raise argparse.ArgumentTypeError("subscription must be CHANNEL=MAX_HZ")
    try:
        rate = float(raw_rate)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("subscription MAX_HZ must be numeric") from exc
    if rate <= 0.0:
        raise argparse.ArgumentTypeError("subscription MAX_HZ must be positive")
    return channel_id.strip(), rate, pb.DROP_OLDEST


def _parse_target_host(target: str) -> str:
    host = target.rsplit(":", 1)[0].strip("[]")
    if not host:
        raise ValueError("target must include a host")
    return host


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PolicyDataService v2 rate/age smoke")
    parser.add_argument("--target", default="127.0.0.1:50051")
    parser.add_argument("--server-ca", type=Path)
    parser.add_argument("--client-cert", type=Path)
    parser.add_argument("--client-key", type=Path)
    parser.add_argument("--insecure-loopback", action="store_true")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--subscribe",
        action="append",
        type=_subscription,
        default=[],
        metavar="CHANNEL=MAX_HZ",
    )
    return parser.parse_args(argv)


async def _run(options) -> int:
    import grpc

    if options.duration <= 0.0:
        raise ValueError("duration must be positive")
    if options.insecure_loopback:
        host = _parse_target_host(options.target)
        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError("insecure transport is permitted only on loopback")
        channel = grpc.aio.insecure_channel(options.target)
    else:
        if options.server_ca is None:
            raise ValueError("--server-ca is required for TLS")
        client_cert = None if options.client_cert is None else options.client_cert.read_bytes()
        client_key = None if options.client_key is None else options.client_key.read_bytes()
        if (client_cert is None) != (client_key is None):
            raise ValueError("--client-cert and --client-key must be provided together")
        credentials = grpc.ssl_channel_credentials(
            root_certificates=options.server_ca.read_bytes(),
            private_key=client_key,
            certificate_chain=client_cert,
        )
        channel = grpc.aio.secure_channel(options.target, credentials)
    client = PolicyDataClient(channel)
    try:
        description = await client.describe()
        subscriptions = list(options.subscribe)
        if not subscriptions:
            subscriptions = [
                ("arm.left.q", 200.0, pb.DROP_OLDEST),
                ("camera.head.rgb", 15.0, pb.DROP_OLDEST),
            ]
        known = {item.channel_id for item in description.channels}
        unknown = sorted({item[0] for item in subscriptions} - known)
        if unknown:
            raise ValueError("unknown channels: " + ", ".join(unknown))
        call = client.subscribe(
            subscriptions,
            client_id="policy-data-smoke",
            schema_hash=description.schema_hash,
            session_id=description.session_id,
        )
        stats = {channel_id: ChannelStats() for channel_id, _, _ in subscriptions}
        deadline = time.monotonic() + options.duration
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                message = await asyncio.wait_for(call.read(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if message is grpc.aio.EOF:
                break
            stats[message.channel_id].observe(message, time.monotonic_ns())
        call.cancel()
        print(
            json.dumps(
                {
                    "schema_hash": description.schema_hash,
                    "session_id": description.session_id,
                    "control_state": description.control_state,
                    "duration_s": options.duration,
                    "channels": {
                        channel_id: value.report() for channel_id, value in stats.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        await channel.close()
    return 0


def main(argv=None) -> int:
    try:
        return asyncio.run(_run(_parse_args(argv)))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
