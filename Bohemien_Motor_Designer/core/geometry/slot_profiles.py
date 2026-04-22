"""
Parametric slot geometry — shape library for stator slots.

Each profile provides: area(), opening_width(), depth(),
perimeter_wetted(), conductor_area(fill_factor).
"""
from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


class SlotProfile(ABC):
    """Abstract parametric slot cross-section."""

    @abstractmethod
    def area(self) -> float:
        """Total slot area [m²]."""

    @abstractmethod
    def opening_width(self) -> float:
        """Slot mouth (opening) width [m]."""

    @abstractmethod
    def depth(self) -> float:
        """Slot radial depth [m]."""

    def conductor_area(self, fill_factor: float = 0.45) -> float:
        """Usable conductor area [m²]."""
        return self.area() * fill_factor

    def perimeter_wetted(self) -> float:
        """Approximate perimeter for thermal contact [m]."""
        return 2 * self.depth() + self.area() / self.depth()

    def summary(self) -> str:
        return (f"{self.__class__.__name__}: "
                f"area={self.area()*1e6:.1f} mm², "
                f"depth={self.depth()*1e3:.1f} mm, "
                f"opening={self.opening_width()*1e3:.2f} mm")


@dataclass
class ParallelToothSlot(SlotProfile):
    """
    Standard parallel-tooth, near-parallel slot.
    Most common for induction and PMSM motors < 200 kW.

           ___     ___
          |   |   |   |  ← slot opening (narrow)
          |   |   |   |
          |    ___    |  ← slot width (wider)
          |   |   |   |
          |   |   |   |
          |___|   |___|
    """
    slot_width:    float       # mean width [m]
    slot_depth:    float       # radial depth [m]
    slot_opening:  float       # mouth opening width [m]
    wedge_height:  float = 0.0 # magnetic wedge / tooth tip height [m]
    corner_radius: float = 0.0 # corner radius [m]

    def area(self) -> float:
        useful_depth = self.slot_depth - self.wedge_height
        return self.slot_width * useful_depth

    def opening_width(self) -> float:
        return self.slot_opening

    def depth(self) -> float:
        return self.slot_depth


@dataclass
class TrapezoidalSlot(SlotProfile):
    """
    Wider at bore, narrowing toward yoke.
    Used in large machines (> 100 kW) for better copper utilisation.

    Width varies linearly: w(r) = w_bore + (w_yoke - w_bore) * r/h
    """
    width_at_bore:  float   # [m] — widest point (at air-gap side)
    width_at_yoke:  float   # [m] — narrowest point
    slot_depth:     float   # [m]
    slot_opening:   float   # [m]
    wedge_height:   float = 0.0

    def area(self) -> float:
        h = self.slot_depth - self.wedge_height
        return 0.5 * (self.width_at_bore + self.width_at_yoke) * h

    def opening_width(self) -> float:
        return self.slot_opening

    def depth(self) -> float:
        return self.slot_depth

    def width_at(self, r_frac: float) -> float:
        """Width at fractional depth r_frac ∈ [0, 1] from bore."""
        return self.width_at_bore + (self.width_at_yoke - self.width_at_bore) * r_frac


@dataclass
class OpenSlot(SlotProfile):
    """
    Full-open slot — used with pre-formed coils or hairpin windings.
    No tooth-tip narrowing; full slot width at the opening.
    """
    slot_width:  float
    slot_depth:  float
    liner_thickness: float = 0.0002   # slot liner [m]

    def area(self) -> float:
        effective_w = self.slot_width - 2 * self.liner_thickness
        effective_d = self.slot_depth - self.liner_thickness
        return effective_w * effective_d

    def opening_width(self) -> float:
        return self.slot_width   # fully open

    def depth(self) -> float:
        return self.slot_depth


@dataclass
class RoundBottomSlot(SlotProfile):
    """
    Parallel sides with semicircular bottom.
    Common in smaller motors manufactured by punching.
    """
    slot_width:   float
    slot_depth:   float     # to centre of radius
    slot_opening: float
    wedge_height: float = 0.0

    def area(self) -> float:
        r     = self.slot_width / 2
        rect  = self.slot_width * (self.slot_depth - self.wedge_height - r)
        semi  = 0.5 * np.pi * r**2
        return max(rect, 0) + semi

    def opening_width(self) -> float:
        return self.slot_opening

    def depth(self) -> float:
        return self.slot_depth


def auto_slot_profile(power_kW: float, slot_width: float,
                       slot_depth: float, slot_opening: float) -> SlotProfile:
    """
    Heuristic: select slot profile based on motor power rating.

      < 50 kW   → ParallelToothSlot   (punched laminations, random wound)
      50–300 kW → TrapezoidalSlot     (punched or cut, form wound)
      > 300 kW  → OpenSlot            (open slot, pre-formed coils)
    """
    if power_kW < 50:
        return ParallelToothSlot(slot_width, slot_depth, slot_opening)
    elif power_kW < 300:
        return TrapezoidalSlot(
            width_at_bore=slot_width * 1.15,
            width_at_yoke=slot_width * 0.85,
            slot_depth=slot_depth,
            slot_opening=slot_opening,
        )
    else:
        return OpenSlot(slot_width, slot_depth)
