"""Independent Inspire RH56DFTP-2 driver.

This package is project code based on the vendor's published Modbus protocol.
It is not vendor-provided ROS 2 source code and has no external workspace
runtime dependency.
"""

from .models import HandCommand, HandState, TactileFrame, TactileSurface
from .protocol import ACTUATOR_NAMES, TACTILE_LAYOUT, TOTAL_TAXELS
from .worker import DftpHandWorker, LatestOnlyMailbox

__all__ = [
    "ACTUATOR_NAMES",
    "DFTPHandWorker",
    "DftpHandWorker",
    "HandCommand",
    "HandState",
    "LatestOnlyMailbox",
    "TACTILE_LAYOUT",
    "TOTAL_TAXELS",
    "TactileFrame",
    "TactileSurface",
]

# Backward-friendly spelling without creating another implementation.
DFTPHandWorker = DftpHandWorker
