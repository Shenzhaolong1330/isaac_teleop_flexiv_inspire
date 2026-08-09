"""Independent, model-selectable Inspire RH56 Modbus driver.

This package is project code based on the vendor's published Modbus protocol.
It is not vendor-provided ROS 2 source code and has no external workspace
runtime dependency.
"""

from .models import HandCommand, HandState, TactileFrame, TactileSurface
from .profiles import HAND_PROFILES, HandProfile, hand_profile
from .protocol import ACTUATOR_NAMES, TACTILE_LAYOUT, TOTAL_TAXELS
from .worker import DftpHandWorker, LatestOnlyMailbox

__all__ = [
    "ACTUATOR_NAMES",
    "DFTPHandWorker",
    "DftpHandWorker",
    "HandCommand",
    "HAND_PROFILES",
    "HandProfile",
    "HandState",
    "LatestOnlyMailbox",
    "TACTILE_LAYOUT",
    "TOTAL_TAXELS",
    "TactileFrame",
    "TactileSurface",
    "hand_profile",
]

# Backward-friendly spelling without creating another implementation.
DFTPHandWorker = DftpHandWorker
