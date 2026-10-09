"""Written-record quality accounting; missing source evidence remains unknown."""

from bisect import bisect_right
from collections import defaultdict
from .action_contract import boundaries, timestamp


def expected_rates(config):
    sampling = config.get("sampling", {})
    rates = {}
    for side in ("left", "right"):
        for suffix, key in (
            ("arm/state", "arm_observation_hz"),
            ("hand/state", "hand_state_hz"),
            ("hand/tactile_raw", "tactile_hz"),
        ):
            rates[f"robot/{side}_{suffix}"] = float(sampling.get(key, 0))
    for camera in ("head", "left_wrist", "right_wrist"):
        rates[f"camera/{camera}/color/image_raw/compressed"] = float(
            sampling.get("camera_hz", 0)
        )
    rates["control/sent_command"] = float(sampling.get("teleop_command_hz", 0))
    return rates


def written_quality(
    records,
    *,
    config,
    enqueued=None,
    queue_drops=None,
    suppressed=None,
    sequence_rules=None,
):
    cuts = [item[0] for item in boundaries(records)]
    grouped = defaultdict(list)
    rates = expected_rates(config)
    for record in records:
        t = timestamp(record)
        segment = bisect_right(cuts, t) if t is not None else -1
        grouped[(record["topic"].lstrip("/"), segment)].append(
            (t if t is not None else record["host_receive_time_ns"], record)
        )
    streams = {}
    for (topic, segment), values in grouped.items():
        arrival_order = sorted(values, key=lambda item: item[1]["host_receive_time_ns"])
        regressions = sum(
            right[0] < left[0] for left, right in zip(arrival_order, arrival_order[1:])
        )
        values.sort(key=lambda item: item[0])
        times = [t for t, _ in values]
        duration = (times[-1] - times[0]) / 1e9 if segment >= 0 else 0
        count = len(values)
        rule = (sequence_rules or {}).get(topic)
        source_loss = None
        resets = 0
        if rule and rule.get("kind") == "continuous_counter" and not regressions:
            step = int(rule.get("step", 1))
            if step <= 0:
                raise ValueError("sequence step must be positive")
            source_loss = 0
            for (_, left), (_, right) in zip(values, values[1:]):
                delta = right["sequence"] - left["sequence"]
                if delta < 0:
                    resets += 1
                elif delta % step:
                    source_loss = None
                    break
                elif delta > step:
                    source_loss += delta // step - 1
            # This is an observed sequence gap, not proof of source/transport loss.
        entry = streams.setdefault(
            topic,
            {
                "configured_hz": rates.get(topic, 0),
                "written": 0,
                "valid_written": 0,
                "segments": [],
                "produced": None,
                "received": None,
                "source_transport_loss": None,
                "enqueued": (enqueued or {}).get(topic, 0),
                "queue_dropped": (queue_drops or {}).get("/" + topic, 0),
                "gated_suppressed": (suppressed or {}).get(topic, 0),
                "active_downsampled": None,
            },
        )
        valid_count = sum(
            bool(r["valid"]) and timestamp(r) is not None for _, r in values
        )
        entry["written"] += count
        entry["valid_written"] += valid_count
        entry["segments"].append(
            {
                "segment_id": segment,
                "written": count,
                "invalid": count - valid_count,
                "duration_s": duration,
                "observed_hz": (count - 1) / duration if duration > 0 else None,
                "observed_sequence_gaps": source_loss,
                "counter_resets": resets,
                "time_regressions": regressions,
            }
        )
    for topic in (
        set(enqueued or {})
        | set(suppressed or {})
        | {name.lstrip("/") for name in (queue_drops or {})}
    ):
        streams.setdefault(
            topic,
            {
                "configured_hz": rates.get(topic, 0),
                "written": 0,
                "valid_written": 0,
                "segments": [],
                "produced": None,
                "received": None,
                "source_transport_loss": None,
                "enqueued": (enqueued or {}).get(topic, 0),
                "queue_dropped": (queue_drops or {}).get("/" + topic, 0),
                "gated_suppressed": (suppressed or {}).get(topic, 0),
                "active_downsampled": None,
            },
        )
    for topic, entry in streams.items():
        duration = sum(s["duration_s"] for s in entry["segments"])
        intervals = sum(max(0, s["written"] - 1) for s in entry["segments"])
        entry["observed_hz"] = intervals / duration if duration > 0 else None
    return {
        "schema": "written-quality-v2",
        "streams": streams,
        "training_samples": "computed-by-export-validity-gate",
        "frequency_rule": "sum within-segment intervals / sum within-segment source duration; pauses excluded",
        "boundary_basis": "explicit-event" if cuts else "unknown-no-events",
    }
