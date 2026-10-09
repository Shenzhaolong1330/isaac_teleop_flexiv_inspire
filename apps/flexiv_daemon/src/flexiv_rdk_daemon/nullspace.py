"""RDK 1.9.x soft posture adapter. Blocking calls occur only on mode entry."""

from __future__ import annotations
import time
import numpy as np


def validate_nullspace(raw):
    raw = raw or {
        "schema_version": 1,
        "left": {"enabled": False},
        "right": {"enabled": False},
    }
    if raw.get("schema_version") != 1:
        raise ValueError("nullspace schema_version must be 1")
    for side in ("left", "right"):
        cfg = raw[side]
        if not isinstance(cfg.get("enabled"), bool):
            raise ValueError("nullspace enabled must be bool")
        if not cfg["enabled"]:
            continue
        if (
            cfg.get("reference_strategy") not in {"takeover_measured", "manual"}
            or cfg.get("failure_policy") != "hold"
        ):
            raise ValueError(
                "nullspace requires takeover_measured/manual and failure_policy hold"
            )
        if cfg.get("rdk_version") != "1.9.x" or cfg.get("online_update", False):
            raise ValueError(
                "only 1.9.x mode-entry nullspace supported; online update unsupported"
            )
        weights = np.asarray(cfg["weights"], dtype=float)
        if (
            weights.shape != (3,)
            or not np.isfinite(weights).all()
            or np.any(weights[:2] < 0)
            or np.any(weights[:2] > 1)
            or not 0.1 <= weights[2] <= 1
        ):
            raise ValueError("RDK 1.9 weights require [0,1],[0,1],[0.1,1]")
        if cfg["reference_strategy"] == "manual":
            q = np.asarray(cfg["q_ref"], dtype=float)
            if q.shape != (7,) or not np.isfinite(q).all():
                raise ValueError("manual q_ref requires seven finite radians")
    return raw


class NullspaceAdapter:
    def __init__(self, raw=None):
        self.config = validate_nullspace(raw)
        self.reference = {}
        self.results = {}

    def reconnect(self):
        self.reference.clear()
        self.results.clear()

    def mode_entry(self, side, robot, measured_q, q_min, q_max, version):
        cfg = self.config[side]
        if not cfg["enabled"]:
            self.results[side] = {
                "enabled": False,
                "result": "disabled",
                "measured_q": np.asarray(measured_q).tolist(),
            }
            return self.results[side]
        started = time.monotonic_ns()
        record = {
            "enabled": True,
            "mode": "NRT_CARTESIAN_MOTION_FORCE",
            "version": version,
            "strategy": cfg["reference_strategy"],
            "weights": cfg["weights"],
            "measured_q": np.asarray(measured_q).tolist(),
        }
        try:
            if not version.startswith("1.9."):
                raise RuntimeError("unsupported nullspace RDK version")
            methods = []
            for names in (
                ("set_null_space_posture", "SetNullSpacePosture"),
                ("set_null_space_objectives", "SetNullSpaceObjectives"),
            ):
                method = next(
                    (
                        getattr(robot, n)
                        for n in names
                        if callable(getattr(robot, n, None))
                    ),
                    None,
                )
                if method is None:
                    raise RuntimeError(
                        "enabled nullspace interface unsupported by actual binding"
                    )
                methods.append(method)
            if side not in self.reference:
                q = np.asarray(
                    measured_q
                    if cfg["reference_strategy"] == "takeover_measured"
                    else cfg["q_ref"],
                    dtype=float,
                )
                if (
                    q.shape != (7,)
                    or not np.isfinite(q).all()
                    or np.any(q < q_min)
                    or np.any(q > q_max)
                ):
                    raise ValueError("q_ref outside reported robot limits")
                self.reference[side] = q.copy()
            record["q_ref"] = self.reference[side].tolist()
            methods[0](record["q_ref"])
            methods[1](*cfg["weights"])
            record["result"] = "sdk_returned"
        except Exception as exc:
            record.update(result="failed", error=str(exc))
            raise
        finally:
            record["elapsed_ns"] = time.monotonic_ns() - started
            self.results[side] = record
        return record
