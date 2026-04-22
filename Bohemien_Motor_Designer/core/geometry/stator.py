"""
Parametric stator geometry — complete stator cross-section.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from Bohemien_Motor_Designer.core.geometry.slot_profiles import (
    SlotProfile, ParallelToothSlot, auto_slot_profile)


@dataclass
class StatorGeometry:
    """
    Full parametric stator geometry.

    Parameters
    ----------
    outer_radius     : Stator frame outer radius [m].
    inner_radius     : Bore radius (at air-gap) [m].
    slots            : Number of stator slots.
    slot_profile     : SlotProfile object (shape of each slot).
    lamination       : Lamination material key (e.g. 'M270-35A').
    lamination_thickness: Single lamination thickness [m].
    yoke_material    : Frame / housing material key.
    """
    outer_radius:         float
    inner_radius:         float
    slots:                int
    slot_profile:         SlotProfile
    lamination:           str   = "M270-35A"
    lamination_thickness: float = 0.00035   # 0.35 mm standard
    yoke_material:        str   = "steel"

    # ── Derived geometry ───────────────────────────────────────────────

    @property
    def yoke_thickness(self) -> float:
        """Stator back-yoke radial thickness [m]."""
        return self.outer_radius - self.inner_radius - self.slot_profile.depth()

    @property
    def tooth_width(self) -> float:
        """
        Mean tooth width [m].
        Tooth pitch = bore circumference / slots; tooth = pitch − slot_width.
        """
        slot_pitch = 2 * np.pi * self.inner_radius / self.slots
        return slot_pitch - self.slot_profile.opening_width()

    @property
    def slot_pitch_bore(self) -> float:
        """Slot pitch at bore radius [m]."""
        return 2 * np.pi * self.inner_radius / self.slots

    @property
    def slot_pitch_mean(self) -> float:
        """Slot pitch at mean slot radius [m]."""
        r_mean = self.inner_radius + self.slot_profile.depth() / 2
        return 2 * np.pi * r_mean / self.slots

    @property
    def tooth_area(self) -> float:
        """Cross-sectional area of one tooth [m²]."""
        return self.tooth_width * self.slot_profile.depth()

    @property
    def yoke_area(self) -> float:
        """Cross-sectional area of stator yoke [m²]."""
        return self.yoke_thickness * 1.0  # per unit axial length

    @property
    def total_copper_area(self) -> float:
        """Total copper slot area (all slots, both layers) [m²]. Stack-independent."""
        return self.slots * self.slot_profile.area()

    @property
    def mass_iron_kg_per_m(self) -> float:
        """Iron mass per unit stack length [kg/m]."""
        rho_fe = 7650.0   # kg/m³ electrical steel
        stator_area = np.pi * (self.outer_radius**2 - self.inner_radius**2)
        total_slot_area = self.slots * self.slot_profile.area()
        net_iron_area = stator_area - total_slot_area
        return rho_fe * net_iron_area

    def tooth_flux_density(self, airgap_B_fundamental: float) -> float:
        """
        Peak tooth flux density from airgap fundamental [T].
        B_tooth = B_airgap * slot_pitch / tooth_width
        """
        return airgap_B_fundamental * self.slot_pitch_bore / (
            self.tooth_width + 1e-9)

    def yoke_flux_density(self, airgap_B_fundamental: float,
                           poles: int, stack_length: float,
                           magnet_arc_fraction: float = 1.0) -> float:
        """
        Peak yoke flux density [T].
        φ_pole = B_airgap * α_p * pole_pitch * L
        B_yoke = φ_pole / (2 * t_yoke * L)

        magnet_arc_fraction (α_p): fraction of pole pitch covered by magnet.
        Defaults to 1.0 (conservative). Pass motor.magnet_width_fraction for SPM.
        """
        pole_pitch = np.pi * self.inner_radius / (poles / 2)
        phi_pole   = airgap_B_fundamental * magnet_arc_fraction * pole_pitch * stack_length
        return phi_pole / (2 * self.yoke_thickness * stack_length + 1e-9)

    def validate(self) -> list[str]:
        """Return list of geometry warnings/errors."""
        issues = []
        if self.yoke_thickness < self.slot_profile.depth() * 0.3:
            issues.append(f"Yoke too thin: {self.yoke_thickness*1e3:.1f}mm — saturation risk")
        if self.tooth_width < self.slot_profile.opening_width():
            issues.append("Tooth width < slot opening — negative tooth width")
        if self.slot_profile.depth() > (self.outer_radius - self.inner_radius) * 0.6:
            issues.append("Slot depth > 60% of stator radial build — yoke too thin")
        return issues

    def summary(self) -> str:
        return (
            f"StatorGeometry:\n"
            f"  OD / ID           : {self.outer_radius*1e3:.1f} / {self.inner_radius*1e3:.1f} mm\n"
            f"  Slots             : {self.slots}\n"
            f"  Slot profile      : {self.slot_profile.summary()}\n"
            f"  Tooth width       : {self.tooth_width*1e3:.2f} mm\n"
            f"  Yoke thickness    : {self.yoke_thickness*1e3:.2f} mm\n"
            f"  Slot pitch (bore) : {self.slot_pitch_bore*1e3:.2f} mm\n"
            f"  Lamination        : {self.lamination}"
        )

    @classmethod
    def auto_size(cls, poles: int, slots: int,
                  outer_radius: float, inner_radius: float,
                  power_kW: float, phases: int = 3) -> "StatorGeometry":
        """
        Auto-generate a reasonable slot geometry from overall dimensions.

        Uses classical proportions:
          slot_depth   ≈ 0.35 × (R_outer − R_inner)
          slot_width   ≈ 0.55 × slot_pitch
          slot_opening ≈ 2.5 mm (min for winding insertion)
        """
        r_i = inner_radius
        r_o = outer_radius
        n_s = slots

        slot_pitch  = 2 * np.pi * r_i / n_s
        slot_depth  = min(0.40 * (r_o - r_i), slot_pitch * 2.0)
        slot_width  = slot_pitch * 0.50
        slot_opening= max(0.002, slot_width * 0.35)

        profile = auto_slot_profile(power_kW, slot_width, slot_depth, slot_opening)
        return cls(
            outer_radius=r_o,
            inner_radius=r_i,
            slots=n_s,
            slot_profile=profile,
        )
