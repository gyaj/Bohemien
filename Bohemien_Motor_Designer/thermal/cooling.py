"""
Cooling system models — boundary conditions for thermal network.
"""
from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from Bohemien_Motor_Designer.materials.library import MaterialLibrary


class CoolingSystem(ABC):
    """Abstract base cooling model."""

    @abstractmethod
    def wall_resistance(self, stator_outer_radius: float,
                         stack_length: float) -> float:
        """Thermal resistance: stator yoke OD → coolant [K/W]."""

    @abstractmethod
    def max_heat_rejection_W(self) -> float:
        """Maximum steady-state heat rejection capacity [W]."""

    @property
    @abstractmethod
    def inlet_temp_C(self) -> float:
        """Coolant inlet temperature [°C]."""


@dataclass
class WaterJacketCooling(CoolingSystem):
    """
    Helical or axial water jacket around stator OD.
    Most common for EV traction motors and industrial servos above 20 kW.

    Parameters
    ----------
    flow_lpm      : Coolant flow rate [litres/minute].
    inlet_temp_C  : Coolant inlet temperature [°C].
    jacket_width  : Channel width (radial depth of cooling passage) [m].
    n_channels    : Number of parallel axial/helical channels.
    contact_pressure_MPa: Interface contact pressure for contact resistance.
    coolant       : Coolant material key.
    """
    flow_lpm:       float = 10.0
    _inlet_temp_C:  float = 65.0
    jacket_width:   float = 0.008   # 8 mm channel
    n_channels:     int   = 12
    wall_thickness: float = 0.003   # housing wall [m]
    contact_pressure_MPa: float = 0.5
    coolant_key:    str   = "water-glycol-50"

    @property
    def inlet_temp_C(self) -> float:
        return self._inlet_temp_C

    def _coolant(self):
        return MaterialLibrary().coolant(self.coolant_key)

    def _jacket_h(self, stator_outer_radius: float,
                   stack_length: float) -> float:
        """
        Effective convection h at stator OD for a helical water jacket [W/m²·K].

        Uses empirical correlation from Staton & Cavagnino (2008) for
        helical water jackets with turbulators:
          h ≈ 750 × (Q_lpm / 10)^0.8   [W/m²·K]

        This accounts for real jacket geometry (fins, turbulators) which
        gives much higher h than laminar Dittus-Boelter.
        Valid range: 3–30 lpm.
        """
        h_base = 750.0  # W/m²K at 10 lpm reference
        return h_base * (self.flow_lpm / 10.0) ** 0.8

    def wall_resistance(self, stator_outer_radius: float,
                         stack_length: float) -> float:
        """
        Total stator yoke OD → coolant thermal resistance [K/W].
        R = R_conduction(housing) + R_convection(jacket) + R_contact
        """
        A_stator = 2 * np.pi * stator_outer_radius * stack_length

        # Conduction through aluminium housing wall
        k_al   = 160.0
        R_cond = self.wall_thickness / (k_al * A_stator)

        # Jacket convection (empirical)
        h      = self._jacket_h(stator_outer_radius, stack_length)
        R_conv = 1.0 / (h * A_stator)

        # Stator–housing interface contact resistance
        R_contact = 5e-4 / A_stator

        return R_cond + R_conv + R_contact

    def max_heat_rejection_W(self) -> float:
        fluid    = self._coolant()
        flow_m3s = self.flow_lpm / 1000 / 60
        mass_s   = flow_m3s * fluid.density
        delta_T  = 15.0   # allow 15°C rise across jacket
        return mass_s * fluid.specific_heat * delta_T

    def temperature_rise_C(self, total_loss_W: float) -> float:
        """Bulk coolant temperature rise from inlet to outlet [°C]."""
        fluid    = self._coolant()
        flow_m3s = self.flow_lpm / 1000 / 60
        mass_s   = flow_m3s * fluid.density
        return total_loss_W / (mass_s * fluid.specific_heat + 1e-9)

    def summary(self) -> str:
        return (f"WaterJacketCooling: {self.flow_lpm:.1f} lpm, "
                f"inlet={self._inlet_temp_C:.0f}°C, "
                f"max rejection={self.max_heat_rejection_W()/1e3:.1f} kW")


@dataclass
class AirCooling(CoolingSystem):
    """
    TEFC (Totally Enclosed Fan-Cooled) or force-ventilated air cooling.
    Suitable for motors up to ~50 kW.

    Parameters
    ----------
    flow_m3s      : Volumetric air flow rate [m³/s]. 0 = natural convection.
    _inlet_temp_C : Ambient (inlet) temperature [°C].
    fin_area_factor: Ratio of fin area to smooth cylinder area.
    """
    flow_m3s:       float = 0.10
    _inlet_temp_C:  float = 40.0
    fin_area_factor: float = 2.5   # ribbed / finned frame
    frame_material: str   = "aluminium"

    @property
    def inlet_temp_C(self) -> float:
        return self._inlet_temp_C

    def _h_forced(self) -> float:
        """Approximate h for external forced convection over cylinder [W/m²·K]."""
        # TEFC empirical: h ≈ 15–60 W/m²·K depending on fan size
        return 15.0 + 200.0 * self.flow_m3s   # rough linear fit

    def wall_resistance(self, stator_outer_radius: float,
                         stack_length: float) -> float:
        A_smooth = 2 * np.pi * stator_outer_radius * stack_length
        A_fin    = A_smooth * self.fin_area_factor
        h        = self._h_forced()
        return 1.0 / (h * A_fin)

    def max_heat_rejection_W(self) -> float:
        fluid  = MaterialLibrary().coolant("air")
        mass_s = self.flow_m3s * fluid.density
        return mass_s * fluid.specific_heat * 30.0   # 30°C rise

    def summary(self) -> str:
        return (f"AirCooling: {self.flow_m3s*1e3:.1f} L/s, "
                f"inlet={self._inlet_temp_C:.0f}°C, "
                f"max rejection={self.max_heat_rejection_W()/1e3:.1f} kW")


@dataclass
class OilSprayCooling(CoolingSystem):
    """
    Direct ATF oil spray onto end-windings and/or rotor.
    Highest heat rejection per volume — used in high-performance EV motors.

    Typical h = 500–3000 W/m²·K on end-windings.
    """
    flow_lpm:      float = 5.0
    _inlet_temp_C: float = 80.0
    spray_h:       float = 1500.0   # W/m²·K on end-winding surface

    @property
    def inlet_temp_C(self) -> float:
        return self._inlet_temp_C

    def wall_resistance(self, stator_outer_radius: float,
                         stack_length: float) -> float:
        # End-winding spray area (both sides, rough estimate)
        r_ew = stator_outer_radius * 0.5   # mean end-winding radius
        A_ew = 2 * 2 * np.pi * r_ew * 0.03   # ~30mm axial extent per side
        return 1.0 / (self.spray_h * A_ew)

    def max_heat_rejection_W(self) -> float:
        fluid    = MaterialLibrary().coolant("oil-ATF")
        flow_m3s = self.flow_lpm / 1000 / 60
        mass_s   = flow_m3s * fluid.density
        return mass_s * fluid.specific_heat * 30.0

    def summary(self) -> str:
        return (f"OilSprayCooling: {self.flow_lpm:.1f} lpm, "
                f"inlet={self._inlet_temp_C:.0f}°C, "
                f"h={self.spray_h:.0f} W/m²·K")


def make_cooling(spec_cooling) -> CoolingSystem:
    """Factory: create CoolingSystem from CoolingSpec."""
    from Bohemien_Motor_Designer.core.specs import (COOLING_AIR, COOLING_WATER,
                                          COOLING_OIL_SPRAY, COOLING_OIL_FLOOD)
    ct = spec_cooling.cooling_type
    if ct == COOLING_WATER:
        return WaterJacketCooling(
            flow_lpm=max(spec_cooling.coolant_flow_lpm, 5.0),
            _inlet_temp_C=spec_cooling.coolant_temp_C,
        )
    elif ct in (COOLING_OIL_SPRAY, COOLING_OIL_FLOOD):
        return OilSprayCooling(
            flow_lpm=max(spec_cooling.coolant_flow_lpm, 2.0),
            _inlet_temp_C=spec_cooling.coolant_temp_C,
        )
    else:
        return AirCooling(_inlet_temp_C=spec_cooling.ambient_temp_C)
