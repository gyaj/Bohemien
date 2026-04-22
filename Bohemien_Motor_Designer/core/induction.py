"""
Induction Motor (IM) class with squirrel-cage rotor.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from Bohemien_Motor_Designer.core.motor import Motor
from Bohemien_Motor_Designer.core.specs import DesignSpec
from Bohemien_Motor_Designer.core.geometry.rotor import SquirrelCageRotorGeometry


@dataclass
class InductionMotor(Motor):
    """
    Three-phase squirrel-cage induction motor.

    Additional parameters
    --------------------
    rotor_slots       : int   Number of rotor bars
    bar_material      : str   'copper' or 'aluminium'
    bar_width         : float Rotor bar width [m]
    bar_height        : float Rotor bar height [m]
    end_ring_area     : float End ring cross-section [m^2]
    turns_per_coil    : int   Stator coil turns
    conductor_diameter : float Stator wire diameter [m]
    slot_fill_factor  : float Stator slot fill (0-1)
    rated_slip        : float Rated slip (0-0.1). Default 0.03
    skew_slots        : float Rotor skew in slot pitches. Default 1.0
    """

    rotor_slots:       int   = 36
    bar_material:      str   = "aluminium"
    bar_width:         float = 0.006
    bar_height:        float = 0.020
    end_ring_area:     float = 0.0004
    turns_per_coil:    int   = 30
    conductor_diameter: float = 0.0015
    slot_fill_factor:  float = 0.42
    rated_slip:        float = 0.03
    skew_slots:        float = 1.0

    rotor_cage: Optional[SquirrelCageRotorGeometry] = field(default=None, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.rotor_cage is None:
            self.rotor_cage = SquirrelCageRotorGeometry(
                outer_radius=self.rotor_outer_radius,
                inner_radius=self.rotor_inner_radius,
                n_rotor_slots=self.rotor_slots,
                bar_width=self.bar_width,
                bar_height=self.bar_height,
                end_ring_area=self.end_ring_area,
                bar_material=self.bar_material,
                skew_slots=self.skew_slots,
            )

    @property
    def synchronous_speed(self) -> float:
        """Synchronous speed [rpm]."""
        return 60 * self.rated_speed / 60 / (1 - self.rated_slip)

    @property
    def rotor_frequency(self) -> float:
        """Rotor (slip) electrical frequency [Hz]."""
        return self.electrical_frequency * self.rated_slip

    def total_series_turns(self) -> int:
        q = self.slots // (self.poles * self.phases)
        return (q * self.poles * self.turns_per_coil) // 2

    def stator_resistance(self, temp_C: float = 75.0) -> float:
        """Phase resistance [Ohm]."""
        rho_20 = 1.72e-8
        rho    = rho_20 * (1 + 0.00393 * (temp_C - 20))
        N      = self.total_series_turns()
        lt     = self.stack_length + np.pi * self.airgap_radius / self.pole_pairs
        A      = np.pi * (self.conductor_diameter / 2) ** 2
        return rho * N * lt / max(A, 1e-12)

    def rotor_bar_resistance(self, temp_C: float = 75.0) -> float:
        """Single bar resistance referred to rotor [Ohm]."""
        rho_cu  = 1.72e-8; rho_al = 2.82e-8
        rho_20  = rho_cu if self.bar_material == 'copper' else rho_al
        alpha   = 0.00393 if self.bar_material == 'copper' else 0.00403
        rho     = rho_20 * (1 + alpha * (temp_C - 20))
        A_bar   = self.bar_width * self.bar_height
        return rho * self.stack_length / max(A_bar, 1e-12)

    def breakdown_torque(self) -> float:
        """Approximate breakdown (maximum) torque [N*m] via Kloss formula."""
        T_rated = self.rated_torque()
        s_rated = self.rated_slip
        s_break = 0.12   # typical
        return T_rated * (s_break**2 + s_rated**2) / (2 * s_rated * s_break)

    def summary(self) -> str:
        base = super().summary()
        extra = [
            "",
            "  --- Induction Motor Specifics ---",
            f"  Rotor slots              : {self.rotor_slots}",
            f"  Bar material             : {self.bar_material}",
            f"  Rated slip               : {self.rated_slip*100:.1f} %",
            f"  Stator turns/coil        : {self.turns_per_coil}",
            f"  Breakdown torque est.    : {self.breakdown_torque():.1f} N*m",
            "=" * 57,
        ]
        lines = base.split("\n")[:-1] + extra
        return "\n".join(lines)
