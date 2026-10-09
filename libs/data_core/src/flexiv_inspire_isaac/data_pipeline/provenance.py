"""Portable content archives of effective configuration and known assets."""

from __future__ import annotations
import hashlib
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import yaml
from .action_contract import CONTRACT

SECRET = re.compile(
    r"(^|_)(token|password|secret|credential|private_key|api_key|server_key|client_key)(_|$)",
    re.I,
)


def redact(value):
    if isinstance(value, dict):
        return {
            k: "<redacted>" if SECRET.search(str(k)) else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def software_identity():
    detected = {"python": platform.python_version()}
    for package in ("mcap", "numpy", "flexivrdk", "lerobot", "pyrealsense2"):
        try:
            detected[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            detected[package] = "unknown"
    repo = Path(__file__).resolve().parents[5]

    def git(*args):
        result = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return {
        "detected_recorder_environment": detected,
        "producer_runtime_versions": "unknown",
        "declared_dependencies": {
            "flexiv_rdk": "1.9.0",
            "lerobot": "0.6.0",
            "librealsense": "2.57.7",
        },
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty_paths": git("status", "--porcelain"),
        "git_diff_sha256": hashlib.sha256(
            git("diff", "--binary", "HEAD").encode()
        ).hexdigest(),
    }


def archive_episode(directory, *, config=None, assets=None, rendered_dir=None):
    directory = Path(directory)
    root = directory / "provenance"
    root.mkdir(parents=True, exist_ok=False)
    entries = {}

    def save(name, data, required):
        digest = hashlib.sha256(data).hexdigest()
        path = root / "objects" / digest
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        entries[name] = {
            "path": path.relative_to(directory).as_posix(),
            "sha256": digest,
            "required": required,
            "status": "present",
        }

    def yaml_bytes(value):
        return yaml.safe_dump(redact(value), sort_keys=True).encode()

    save("action_contract", json.dumps(CONTRACT, sort_keys=True).encode(), True)
    save("software", json.dumps(software_identity(), sort_keys=True).encode(), True)
    hands = (config or {}).get("inspire", {})
    models = {side: hands.get(side + "_model") for side in ("left", "right")}
    if any(models.values()):
        try:
            from flexiv_inspire_isaac.dftp.profiles import hand_profile

            layouts = {}
            for side, model in models.items():
                if model:
                    profile = hand_profile(model, side=side)
                    layouts[side] = {
                        "declared_model": model,
                        "definition_source": "recorder-shipped-driver-profile",
                        "hardware_identity_probe": "unknown",
                        "profile": asdict(profile),
                        "taxel_count": profile.tactile_taxel_count,
                    }
                else:
                    layouts[side] = {
                        "declared_model": "unknown",
                        "hardware_identity_probe": "unknown",
                    }
            save(
                "hand_models_and_layout",
                json.dumps(layouts, sort_keys=True).encode(),
                True,
            )
        except (ImportError, ValueError) as exc:
            entries["hand_models_and_layout"] = {
                "required": True,
                "status": "missing",
                "reason": str(exc),
            }
    else:
        entries["hand_models_and_layout"] = {
            "required": False,
            "status": "unknown-not-collected",
        }

    if config is None:
        entries["effective_config"] = {"required": True, "status": "missing"}
    else:
        save("effective_config", yaml_bytes(config), True)
    # Referenced YAML assets are copied recursively. Original paths remain lookup keys,
    # and resolve_asset() uses the index instead of deployment paths.
    pending = list((assets or {}).items())
    required_names = set(assets or {})
    if rendered_dir:
        rendered = [
            (f"rendered/{p.name}", p) for p in sorted(Path(rendered_dir).glob("*.yaml"))
        ]
        pending += rendered
        required_names.update(name for name, _ in rendered)
    visited = set()

    def references(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if not SECRET.search(str(key)):
                    yield from references(item)
        elif isinstance(value, list):
            for item in value:
                yield from references(item)
        elif isinstance(value, str) and (
            value.startswith("/")
            or value.endswith((".yaml", ".yml", ".json", ".npz", ".npy"))
        ):
            yield value

    if config:
        pending += [(ref, Path(ref)) for ref in references(config)]

        def required_references(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if SECRET.search(str(key)):
                        continue
                    if (
                        isinstance(item, str)
                        and item
                        and (
                            str(key).endswith("config")
                            or str(key)
                            in {
                                "frame_config",
                                "tool_payload_config",
                                "manus_calibration",
                            }
                        )
                    ):
                        yield item
                    yield from required_references(item)
            elif isinstance(value, list):
                for item in value:
                    yield from required_references(item)

        required_names.update(required_references(config))
    while pending:
        name, source = pending.pop(0)
        if name in visited:
            continue
        visited.add(name)
        source = Path(source)
        required = name in required_names
        if SECRET.search(name) or source.suffix in {".key", ".pem"}:
            entries[name] = {"required": False, "status": "excluded-secret"}
            continue
        if not source.is_file():
            entries[name] = {"required": required, "status": "missing"}
            continue
        if source.suffix in {".yaml", ".yml", ".json"}:
            value = yaml.safe_load(source.read_text())
            data = yaml_bytes(value)
            for ref in references(value):
                refpath = Path(ref)
                pending.append(
                    (ref, refpath if refpath.is_absolute() else source.parent / refpath)
                )
        else:
            data = source.read_bytes()
        save(name, data, required)
        entries[name]["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        entries[name]["normalized_and_redacted"] = source.suffix in {
            ".yaml",
            ".yml",
            ".json",
        }
    for name in ("tactile_calibration", "camera_handeye", "hardware_firmware"):
        entries.setdefault(name, {"required": False, "status": "unknown-not-collected"})
    index = {
        "schema": "episode-provenance-v2",
        "entries": entries,
        "complete": all(
            e["status"] == "present" for e in entries.values() if e["required"]
        ),
    }
    (root / "index.json").write_text(json.dumps(index, indent=2))
    return {
        "path": "provenance/index.json",
        "sha256": hashlib.sha256((root / "index.json").read_bytes()).hexdigest(),
        "complete": index["complete"],
    }


def verify_archive(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    ref = manifest.get("provenance")
    if not ref:
        raise ValueError(
            "legacy episode has no portable provenance; unavailable assets remain unknown"
        )

    def local(path):
        resolved = (directory / path).resolve()
        if not resolved.is_relative_to(directory):
            raise ValueError("archive reference escapes episode")
        return resolved

    index_path = local(ref["path"])
    if hashlib.sha256(index_path.read_bytes()).hexdigest() != ref["sha256"]:
        raise ValueError("provenance index checksum mismatch")
    index = json.loads(index_path.read_text())
    for name, entry in index["entries"].items():
        if entry["status"] != "present":
            if entry["required"]:
                raise ValueError(f"required archive asset missing: {name}")
            continue
        if (
            hashlib.sha256(local(entry["path"]).read_bytes()).hexdigest()
            != entry["sha256"]
        ):
            raise ValueError(f"asset checksum mismatch: {name}")
    return index


def resolve_asset(directory, name):
    index = verify_archive(directory)
    entry = index["entries"][name]
    if entry["status"] != "present":
        raise ValueError(f"asset unavailable: {name}")
    return Path(directory) / entry["path"]


def startup_summary(config, *, profile, task, output, pedal_gate):
    config = redact(config)
    return {
        "task": task,
        "profile": profile,
        "configured_sampling_hz": config.get("sampling", {}),
        "measured_sampling_hz": "unknown-before-capture",
        "modalities": config.get("sensors", config.get("cameras", {})),
        "input": config.get("teleop", {}),
        "hands": config.get("inspire", {}),
        "safety": config.get("safety", config.get("flexiv", {})),
        "pedal_recording_gate": pedal_gate,
        "home_behavior": "existing pause and resume workflow",
        "output": str(output),
        "config_sha256": hashlib.sha256(
            json.dumps(redact(config), sort_keys=True).encode()
        ).hexdigest(),
        "action_contract": CONTRACT,
    }
