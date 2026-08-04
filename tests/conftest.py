from __future__ import annotations

from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "apps" / "session_manager" / "src"))
sys.path.insert(0, str(_ROOT / "libs" / "data_core" / "src"))
sys.path.insert(0, str(_ROOT / "apps" / "policy_server" / "src"))
sys.path.insert(
    0,
    str(_ROOT / "libs" / "policy_contracts" / "src" / "flexiv_inspire_isaac"),
)
sys.path.insert(0, str(_ROOT / "ros2_ws" / "src" / "flexiv_inspire_xr_bridge"))
sys.path.insert(0, str(_ROOT / "libs" / "control_core" / "src"))
sys.path.insert(0, str(_ROOT / "apps" / "flexiv_daemon" / "src"))
sys.path.insert(
    0,
    str(_ROOT / "ros2_ws" / "src" / "flexiv_inspire_control"),
)
