"""
Parametric rotor geometry for different motor topologies.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np


@dataclass
class RotorGeometry:
    """Base rotor geometry."""
    outer_radius:  float
    inner_radius:  float    # shaft radius

    @property
    def yoke_thickness(self) -> float:
        return self.outer_radius - self.inner_radius

    @property
    def active_radius(self) -> float:
        return self.outer_radius


@dataclass
class SPMRotorGeometry(RotorGeometry):
    """
    Surface-Permanent-Magnet rotor.

    Magnets bonded or mechanically retained on outer rotor surface.
    Optional carbon-fibre or stainless retention sleeve for high speed.
    """
    magnet_thickness:       float = 0.005
    magnet_width_fraction:  float = 0.85    # arc fraction of pole pitch
    magnet_material:        str   = "N42SH"
    sleeve_thickness:       float = 0.0     # 0 = no sleeve
    sleeve_material:        str   = "carbon_fibre"
    magnet_type:            str   = "SPM"

    @property
    def magnet_outer_radius(self) -> float:
        return self.outer_radius

    @property
    def magnet_inner_radius(self) -> float:
        return self.outer_radius - self.magnet_thickness

    @property
    def effective_outer_radius(self) -> float:
        """Rotor outer radius including sleeve [m]."""
        return self.outer_radius + self.sleeve_thickness

    def tip_speed(self, speed_rpm: float) -> float:
        """Rotor surface tip speed [m/s]."""
        return self.effective_outer_radius * speed_rpm * 2 * np.pi / 60

    def validate(self, max_tip_speed: float = 200.0, speed_rpm: float = 3000):
        ts = self.tip_speed(speed_rpm)
        if ts > max_tip_speed:
            raise ValueError(f"Tip speed {ts:.1f} m/s > {max_tip_speed} m/s limit. "
                             f"Add retention sleeve or reduce speed/radius.")


@dataclass
class IPMBarrier:
    """Single magnet barrier / pocket for IPM rotor."""
    magnet_width:     float          # [m]
    magnet_thickness: float          # [m]
    barrier_angle_deg: float = 40.0  # V-angle for V-type
    bridge_thickness: float = 0.001  # iron bridge at rotor OD [m]
    rib_thickness:    float = 0.001  # centre rib thickness [m]


@dataclass
class IPMRotorGeometry(RotorGeometry):
    """
    Interior Permanent Magnet rotor.

    Supports V-type, U-type, delta (spoke), and multi-barrier arrangements.
    Each barrier is defined by an IPMBarrier object.
    """
    barrier_type: Literal["V", "U", "delta", "spoke", "multi"] = "V"
    barriers:     list[IPMBarrier] = field(default_factory=list)
    magnet_material: str = "N42SH"
    magnet_type:     str = "IPM"

    def __post_init__(self):
        if not self.barriers:
            # Default single-layer V
            self.barriers = [IPMBarrier(
                magnet_width=0.030,
                magnet_thickness=0.005,
                barrier_angle_deg=40.0,
            )]

    @property
    def n_barriers(self) -> int:
        return len(self.barriers)

    @property
    def magnet_thickness(self) -> float:
        return self.barriers[0].magnet_thickness

    @property
    def magnet_width_fraction(self) -> float:
        """Equivalent arc fraction for analytical models."""
        return min(0.9, self.barriers[0].magnet_width / (
            np.pi * self.outer_radius))

    @property
    def saliency_estimate(self) -> float:
        """Rough Lq/Ld ratio estimate based on barrier count."""
        return 1.0 + 0.8 * self.n_barriers   # grows with barrier layers


@dataclass
class SquirrelCageRotorGeometry(RotorGeometry):
    """
    Squirrel-cage induction motor rotor.
    """
    rotor_slots:  int   = 36
    bar_shape:    Literal["rectangular", "deep_bar", "double_cage", "round"] = "rectangular"
    bar_width:    float = 0.008     # [m]
    bar_height:   float = 0.015     # [m]
    end_ring_area: float = 0.0004   # cross section [m²]
    skew_slots:   int   = 1         # number of stator slots of skew (0 = none)
    bar_material: str   = "aluminium"

    @property
    def bar_area(self) -> float:
        if self.bar_shape in ("rectangular", "deep_bar"):
            return self.bar_width * self.bar_height
        elif self.bar_shape == "round":
            return np.pi * (self.bar_width / 2) ** 2
        return self.bar_width * self.bar_height   # fallback

    @property
    def cage_resistance_ratio(self) -> float:
        """
        R_bar / R_end_ring contribution — deep bar gives higher rotor resistance
        at start (improved starting torque) but lower at rated speed.
        """
        if self.bar_shape == "deep_bar":
            return 2.0   # effective resistance at start
        elif self.bar_shape == "double_cage":
            return 1.5
        return 1.0


@dataclass
class WoundRotorGeometry(RotorGeometry):
    """
    Wound-rotor synchronous motor (WRSM) or slip-ring induction motor.
    Supports salient-pole (large hydro, wind) and cylindrical-rotor (high-speed turbo).
    """
    rotor_type:  Literal["salient_pole", "cylindrical"] = "salient_pole"
    poles:       int   = 4
    field_turns: int   = 100     # turns per pole of field winding
    pole_arc_fraction: float = 0.7
    field_current_A:   float = 10.0   # DC field current

    @property
    def pole_pitch_rad(self) -> float:
        return 2 * np.pi / self.poles
