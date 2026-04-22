"""
Base Motor class — shared parameters and geometry for all topologies.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from Bohemien_Motor_Designer.core.specs import DesignSpec
from Bohemien_Motor_Designer.core.geometry.stator import StatorGeometry
from Bohemien_Motor_Designer.core.geometry.winding import WindingLayout


@dataclass
class Motor:
    """
    Base motor class.  All topology-specific classes (PMSM, InductionMotor,
    SynRM, WRSM) inherit from this.

    Parameters
    ----------
    poles, slots        : Electromagnetic pole and slot count.
    stator              : StatorGeometry (parametric slot/yoke description).
    rotor_outer_radius  : Rotor outer radius [m].
    rotor_inner_radius  : Shaft radius [m].
    stack_length        : Active axial length [m].
    airgap              : Air-gap length [m].
    rated_speed         : Rated speed [rpm].
    rated_power         : Rated shaft power [W].
    rated_voltage       : Rated phase voltage RMS [V]  — or use spec.
    phases              : Number of electrical phases.
    spec                : Optional DesignSpec.  If supplied, rated_voltage is
                          derived from spec.drive.max_phase_voltage_rms().
    winding             : WindingLayout (generated automatically if None).
    """

    poles:  int   = 8
    slots:  int   = 48

    # ── Geometry ──────────────────────────────────────────────────────────
    stator:              Optional[StatorGeometry] = field(default=None, repr=False)
    rotor_outer_radius:  float = 0.074
    rotor_inner_radius:  float = 0.025
    stack_length:        float = 0.090
    airgap:              float = 0.001

    # ── Electrical ratings ─────────────────────────────────────────────────
    rated_speed:    float = 3000.0
    rated_power:    float = 5000.0
    rated_voltage:  float = 230.0   # phase RMS — overridden if spec provided
    phases:         int   = 3
    connection:     str   = "star"

    # ── Design spec (optional top-down) ───────────────────────────────────
    spec: Optional[DesignSpec] = field(default=None, repr=False)

    # ── Winding (set automatically, or pass in) ───────────────────────────
    winding: Optional[WindingLayout] = field(default=None, repr=False)

    # ── Computed in __post_init__ ─────────────────────────────────────────
    pole_pairs:              int   = field(init=False)
    electrical_frequency:    float = field(init=False)
    slots_per_pole_per_phase: float = field(init=False)

    def __post_init__(self):
        if self.poles % 2 != 0:
            raise ValueError("Number of poles must be even.")

        self.pole_pairs               = self.poles // 2
        self.electrical_frequency     = (self.rated_speed / 60) * self.pole_pairs
        self.slots_per_pole_per_phase = self.slots / (self.poles * self.phases)

        # Derive rated voltage from DriveSpec if spec is provided
        if self.spec is not None:
            self.rated_voltage = self.spec.drive.max_phase_voltage_rms()

        # Auto-generate stator geometry if not provided
        if self.stator is None:
            self.stator = StatorGeometry.auto_size(
                poles=self.poles,
                slots=self.slots,
                outer_radius=self._stator_outer_radius_guess(),
                inner_radius=self.stator_inner_radius,
                power_kW=self.rated_power / 1e3,
                phases=self.phases,
            )

        # Auto-generate winding layout if not provided
        if self.winding is None:
            self.winding = WindingLayout(
                poles=self.poles,
                slots=self.slots,
                phases=self.phases,
            )

        self._validate()

    def _stator_outer_radius_guess(self) -> float:
        """If stator is None, guess OD from rotor radius."""
        return self.stator_inner_radius * 1.6   # typical OD/bore ratio

    @property
    def stator_inner_radius(self) -> float:
        if self.stator is not None:
            return self.stator.inner_radius
        # Fall back to rotor + airgap
        return self.rotor_outer_radius + self.airgap

    @property
    def stator_outer_radius(self) -> float:
        if self.stator is not None:
            return self.stator.outer_radius
        return self.stator_inner_radius * 1.6

    @property
    def airgap_radius(self) -> float:
        return (self.stator_inner_radius + self.rotor_outer_radius) / 2

    @property
    def pole_pitch(self) -> float:
        return np.pi * self.airgap_radius / self.pole_pairs

    @property
    def slot_pitch(self) -> float:
        return 2 * np.pi * self.stator_inner_radius / self.slots

    @property
    def active_volume(self) -> float:
        return np.pi * self.stator_inner_radius**2 * self.stack_length

    @property
    def rated_torque(self) -> float:
        return self.rated_power / (self.rated_speed * 2 * np.pi / 60)

    def winding_factor(self, harmonic: int = 1) -> float:
        return self.winding.winding_factor(harmonic)

    def _validate(self):
        if self.rotor_outer_radius >= self.stator_inner_radius:
            raise ValueError("rotor_outer_radius must be < stator_inner_radius")
        if self.rotor_inner_radius >= self.rotor_outer_radius:
            raise ValueError("rotor_inner_radius must be < rotor_outer_radius")
        if self.airgap > 0.05:
            raise ValueError(f"Air gap {self.airgap*1e3:.1f}mm seems too large (>50mm)")

    def summary(self) -> str:
        w = self.winding
        lines = [
            "=" * 58,
            "  Motor Base Summary",
            "=" * 58,
            f"  Poles / Slots             : {self.poles} / {self.slots}",
            f"  Stator OD / Bore          : {self.stator_outer_radius*1e3:.1f} / "
            f"{self.stator_inner_radius*1e3:.1f} mm",
            f"  Rotor OD / ID             : {self.rotor_outer_radius*1e3:.1f} / "
            f"{self.rotor_inner_radius*1e3:.1f} mm",
            f"  Air gap                   : {self.airgap*1e3:.2f} mm",
            f"  Stack length              : {self.stack_length*1e3:.1f} mm",
            f"  Rated speed               : {self.rated_speed:.0f} rpm",
            f"  Rated power               : {self.rated_power/1e3:.2f} kW",
            f"  Rated voltage (phase RMS) : {self.rated_voltage:.1f} V",
            f"  Electrical frequency      : {self.electrical_frequency:.1f} Hz",
            f"  Slots/pole/phase (q)      : {self.slots_per_pole_per_phase:.3f}",
            f"  Winding factor            : {self.winding_factor():.4f}",
            f"  Series turns / phase      : {w.total_series_turns_per_phase}",
            f"  Slot pitch (bore)         : {self.slot_pitch*1e3:.2f} mm",
            f"  Pole pitch (airgap)       : {self.pole_pitch*1e3:.2f} mm",
        ]
        if self.spec:
            lines.append(f"  DC bus voltage            : {self.spec.drive.dc_bus_voltage:.0f} V")
            lines.append(f"  Cooling                   : {self.spec.cooling.cooling_type}")
        lines.append("=" * 58)
        return "\n".join(lines)
