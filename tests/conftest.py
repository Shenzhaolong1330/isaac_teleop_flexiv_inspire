from __future__ import annotations

from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "core" / "src"))
sys.path.insert(0, str(_ROOT / "rdk_daemon" / "src"))
sys.path.insert(
    0,
    str(_ROOT / "core" / "ros2" / "flexiv_inspire_control"),
)
