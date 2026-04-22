"""
Synchronous Reluctance Motor (SynRel / SyRM).
No permanent magnets — torque from saliency only.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from Bohemien_Motor_Designer.core.motor import Motor


@dataclass
class SynRelMotor(Motor):
    """
    Synchronous Reluctance Motor.

    Key parameters beyond Motor:
    ----------------------------
    n_barriers        : int   Number of flux barriers per pole (2-5)
    barrier_shape     : str   'U' | 'circular' | 'sinusoidal'
    Ld                : float D-axis inductance [H]
    Lq                : float Q-axis inductance [H]
    turns_per_coil    : int   Stator turns per coil
    conductor_diameter : float [m]
    """
    n_barriers:        int   = 3
    barrier_shape:     str   = "U"
    Ld:                float = 0.0
    Lq:                float = 0.0
    turns_per_coil:    int   = 15
    conductor_diameter: float = 0.0015
    slot_fill_factor:  float = 0.42

    def saliency_ratio(self) -> float:
        """Lq / Ld saliency ratio. Larger = more reluctance torque."""
        return self.Lq / max(self.Ld, 1e-9)

    def torque_from_dq(self, Id: float, Iq: float) -> float:
        """Reluctance torque [N*m]."""
        return 1.5 * self.pole_pairs * (self.Ld - self.Lq) * Id * Iq

    def mtpa_angle(self, I_peak: float) -> tuple:
        """MTPA for SynRel: beta = 45 degrees, Id = Iq = I/sqrt(2)."""
        Id = I_peak / np.sqrt(2)
        Iq = I_peak / np.sqrt(2)
        return float(Id), float(Iq)

    def summary(self) -> str:
        base = super().summary()
        xi = self.saliency_ratio()
        extra = [
            "",
            "  --- SynRel Specifics ---",
            f"  Flux barriers/pole       : {self.n_barriers}",
            f"  Barrier shape            : {self.barrier_shape}",
            f"  Ld / Lq                  : {self.Ld*1e3:.2f} / {self.Lq*1e3:.2f} mH",
            f"  Saliency ratio Lq/Ld     : {xi:.2f}",
            f"  Turns per coil           : {self.turns_per_coil}",
            "=" * 57,
        ]
        lines = base.split("\n")[:-1] + extra
        return "\n".join(lines)
