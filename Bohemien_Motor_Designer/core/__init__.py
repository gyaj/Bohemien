from .specs import DesignSpec, DriveSpec, CoolingSpec, InsulationSpec, EnvelopeConstraints
from .motor import Motor
from .pmsm import PMSM
from .geometry import (SlotProfile, ParallelToothSlot, TrapezoidalSlot, OpenSlot,
                        SPMRotorGeometry, IPMRotorGeometry, IPMBarrier,
                        SquirrelCageRotorGeometry, WindingLayout, StatorGeometry,
                        auto_slot_profile)
__all__ = [
    "DesignSpec", "DriveSpec", "CoolingSpec", "InsulationSpec", "EnvelopeConstraints",
    "Motor", "PMSM",
    "SlotProfile", "ParallelToothSlot", "TrapezoidalSlot", "OpenSlot",
    "SPMRotorGeometry", "IPMRotorGeometry", "IPMBarrier",
    "SquirrelCageRotorGeometry", "WindingLayout", "StatorGeometry", "auto_slot_profile",
]
