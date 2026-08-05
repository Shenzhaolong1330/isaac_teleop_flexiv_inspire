"""Read-only multi-rate wire smoke for a copied policy client directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import yaml

from .client import MultiRatePolicyClient


def _resolve(base: Path, value: str) -> str:
    path = Path(str(value)).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PolicyData v2 multi-rate smoke")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--describe-only", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    tls = document.get("tls", {})
    action = document.get("action", {})
    client = MultiRatePolicyClient(
        target=str(document["target"]),
        server_ca=_resolve(base, tls["server_ca"]),
        client_cert=(
            _resolve(base, tls["client_cert"])
            if str(tls.get("client_cert", "")).strip()
            else None
        ),
        client_key=(
            _resolve(base, tls["client_key"])
            if str(tls.get("client_key", "")).strip()
            else None
        ),
        action_client_id=str(action.get("client_id", "remote-policy")),
        action_ttl_ms=float(action.get("ttl_ms", 250.0)),
        action_lease_ms=int(action.get("lease_ms", 2000)),
    )
    try:
        description = client.connect()
        summary = {
            "schema_hash": description.schema_hash,
            "session_id": description.session_id,
            "control_state": description.control_state,
            "channels": {
                item.channel_id: {
                    "native_rate_hz": item.native_rate_hz,
                    "dtype": item.tensor.dtype,
                    "shape": list(item.tensor.shape),
                    "semantic": item.semantic,
                }
                for item in description.channels
            },
            "actions": {
                item.schema_id: {
                    "rate_hz": item.rate_hz,
                    "shape": list(item.tensor.shape),
                    "representation": item.representation,
                }
                for item in description.action_schemas
            },
        }
        if args.describe_only:
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        subscriptions = document.get("subscriptions", {})
        selected = []
        for name, channels in subscriptions.items():
            normalized = {
                str(channel_id): float(rate_hz)
                for channel_id, rate_hz in channels.items()
            }
            client.subscribe(str(name), normalized)
            selected.extend(normalized)
        if not selected:
            raise ValueError("config subscriptions cannot be empty")
        client.wait_for_channels(selected, timeout_s=10.0, require_valid=True)
        before = client.received_counts()
        started = time.monotonic()
        time.sleep(float(args.duration))
        elapsed = time.monotonic() - started
        after = client.received_counts()
        summary["measured_duration_s"] = elapsed
        summary["measured"] = {
            channel_id: {
                "received": after.get(channel_id, 0) - before.get(channel_id, 0),
                "rate_hz": (
                    after.get(channel_id, 0) - before.get(channel_id, 0)
                )
                / elapsed,
                "latest_sequence": client.latest(channel_id).sequence,
                "latest_age_ms_at_delivery": client.latest(channel_id).age_ns / 1e6,
            }
            for channel_id in selected
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
