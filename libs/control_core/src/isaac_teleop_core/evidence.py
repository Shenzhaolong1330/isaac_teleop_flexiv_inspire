"""Observable actuator receipts; unknown confirmations are never inferred."""

from __future__ import annotations
import time
from .deviceio import record_envelope

SCHEMA = "actuator-evidence-v2"


def command_id(session, source, sequence):
    return f"{session}:{source}:{int(sequence)}"


def receipt(command_id, actuator, status, *, now_ns=None, detail="", target=None):
    if status not in {
        "submitted",
        "accepted",
        "replaced",
        "rejected",
        "timeout",
        "sdk_returned",
        "send_failed",
        "missing",
    }:
        raise ValueError("unknown receipt status")
    return {
        "schema": SCHEMA,
        "command_id": command_id,
        "actuator": actuator,
        "status": status,
        "time_ns": time.monotonic_ns() if now_ns is None else now_ns,
        "detail": detail,
        "accepted_target": target,
        "transport_confirmation": "unknown",
        "controller_confirmation": "unknown",
        "physical_arrival": "unknown",
    }


def emit_receipt(emitter, producer, payload, sequence):
    if emitter is None:
        return
    now = payload["time_ns"]
    try:
        emitter.emit(
            record_envelope(
                producer=producer,
                topic="/control/actuator_evidence",
                source_time_ns=now,
                host_receive_time_ns=now,
                mapped_host_time_ns=now,
                sequence=sequence,
                payload=payload,
            ),
            critical=True,
        )
    except (BufferError, RuntimeError, ValueError):
        # Evidence is best effort here. Recorder completion/consumer reports missing receipts.
        pass


def summarize_receipts(command_id, records, *, deadline_ns, now_ns):
    result = {}
    for actuator in ("left_arm", "right_arm", "left_hand", "right_hand"):
        matches = [
            r
            for r in records
            if r["command_id"] == command_id and r["actuator"] == actuator
        ]
        result[actuator] = (
            matches[-1]["status"]
            if matches
            else ("timeout" if now_ns >= deadline_ns else "missing")
        )
    return {
        "command_id": command_id,
        "actuators": result,
        "physical_arrival": "unknown",
    }
