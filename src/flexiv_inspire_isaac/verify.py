"""Inspect the isolated bridge, existing workspace, ROS 2, and Isaac prerequisites."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import load_config
from .existing_stack import ExistingStackAdapter

EXPECTED_ISAAC_TAG = "v1.3.131"
EXPECTED_ISAAC_COMMIT = "7002ed63d69454ae4f15c0ee19f803fd2846592b"


def _run(command: list[str], timeout: float = 15.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }


def _version_tuple(text: str) -> tuple[int, ...]:
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not match:
        return ()
    return tuple(int(value) for value in match.groups(default="0"))


def build_report(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    project_root = config.source_path.parent.parent
    upstream = project_root / "upstream" / "IsaacTeleop"
    report: dict[str, Any] = {
        "config": str(config.source_path),
        "command_enabled": config.command_enabled,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "project_root": str(project_root),
    }

    adapter = ExistingStackAdapter(
        config.existing_stack,
        max_translation_step_m=config.safety.max_translation_step_m,
        max_rotation_step_rad=config.safety.max_rotation_step_rad,
    )
    report["existing_stack"] = adapter.inspect()

    isaac_head = _run(["git", "-C", str(upstream), "rev-parse", "HEAD"])
    isaac_tag = _run(
        ["git", "-C", str(upstream), "describe", "--tags", "--exact-match"]
    )
    isaac_status = _run(
        ["git", "-C", str(upstream), "status", "--short", "--branch"]
    )
    report["isaac_upstream"] = {
        "path": str(upstream),
        "exists": upstream.is_dir(),
        "head": isaac_head["stdout"],
        "tag": isaac_tag["stdout"],
        "status": isaac_status["stdout"],
        "expected_head": EXPECTED_ISAAC_COMMIT,
        "expected_tag": EXPECTED_ISAAC_TAG,
        "pinned_ok": (
            isaac_head["stdout"] == EXPECTED_ISAAC_COMMIT
            and isaac_tag["stdout"] == EXPECTED_ISAAC_TAG
        ),
    }

    rclpy = _run(
        [
            "/usr/bin/python3",
            "-c",
            "import rclpy; print(rclpy.__file__)",
        ]
    )
    ros_distro = _run(["ros2", "pkg", "prefix", "rclpy"])
    mcap = _run(["ros2", "pkg", "prefix", "rosbag2_storage_mcap"])
    report["ros2"] = {
        "rclpy": rclpy,
        "rclpy_prefix": ros_distro,
        "mcap_storage": mcap,
        "ready": (
            rclpy["returncode"] == 0
            and ros_distro["returncode"] == 0
            and mcap["returncode"] == 0
        ),
    }

    conda_probe = _run(
        [
            "conda",
            "run",
            "-n",
            config.existing_stack.conda_env,
            "python",
            "-c",
            (
                "import sys,flexivrdk,lerobot;"
                "print(sys.version.split()[0]);"
                "print(lerobot.__file__)"
            ),
        ],
        timeout=30.0,
    )
    report["existing_conda"] = {
        "env": config.existing_stack.conda_env,
        "probe": conda_probe,
        "ready": conda_probe["returncode"] == 0,
    }

    nvcc = _run(["nvcc", "--version"])
    driver = _run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,name,memory.total",
            "--format=csv,noheader",
        ]
    )
    cuda_version = _version_tuple(nvcc["stdout"])
    driver_version = _version_tuple(driver["stdout"])
    report["gpu"] = {
        "nvcc": nvcc,
        "driver": driver,
        "cuda_toolkit_version": cuda_version,
        "driver_version": driver_version,
        "isaac_cuda_requirement": ">=12.8",
        "isaac_driver_requirement": ">=580.95.05",
        "cuda_ready": cuda_version >= (12, 8, 0),
        "driver_ready": driver_version >= (580, 95, 5),
    }

    report["readiness"] = {
        "bridge_config_valid": True,
        "upstream_pinned": report["isaac_upstream"]["pinned_ok"],
        "existing_stack_paths": bool(
            report["existing_stack"]["source_repo_exists"]
            and report["existing_stack"]["robot_config_exists"]
        ),
        "existing_conda_ready": report["existing_conda"]["ready"],
        "ros2_mcap_ready": report["ros2"]["ready"],
        "isaac_runtime_install_ready": bool(
            report["gpu"]["cuda_ready"] and report["gpu"]["driver_ready"]
        ),
        "hardware_command_intentionally_disabled": not config.command_enabled,
    }
    return report


def _print_human(report: dict[str, Any]) -> None:
    readiness = report["readiness"]
    print("Isaac/Flexiv/Inspire isolated bridge verification")
    print(f"  config: {report['config']}")
    print(
        "  Isaac upstream: "
        f"{report['isaac_upstream']['tag']} "
        f"{report['isaac_upstream']['head']} "
        f"(pinned={readiness['upstream_pinned']})"
    )
    print(
        "  existing stack: "
        f"paths={readiness['existing_stack_paths']} "
        f"conda={readiness['existing_conda_ready']}"
    )
    print(f"  ROS 2 + rosbag MCAP: {readiness['ros2_mcap_ready']}")
    print(
        "  NVIDIA: "
        f"CUDA={report['gpu']['cuda_toolkit_version']} "
        f"driver={report['gpu']['driver_version']} "
        f"Isaac-install-ready={readiness['isaac_runtime_install_ready']}"
    )
    print(
        "  hardware commands: "
        + (
            "ENABLED IN CONFIG (still requires two CLI gates)"
            if report["command_enabled"]
            else "disabled (shadow-safe default)"
        )
    )
    if not readiness["isaac_runtime_install_ready"]:
        print(
            "  blocker: system CUDA toolkit is below Isaac Teleop's 12.8 "
            "requirement; the existing environment was not changed."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return non-zero unless every runtime prerequisite is ready",
    )
    args = parser.parse_args(argv)
    report = build_report(args.config)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        _print_human(report)
    if args.strict and not all(
        value
        for key, value in report["readiness"].items()
        if key != "hardware_command_intentionally_disabled"
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

