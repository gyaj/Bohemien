"""
DesignSpec — Motor design requirements.

Separates *what the motor must do* from *how it is built*.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

COOLING_AIR       = "air"
COOLING_WATER     = "water-jacket"
COOLING_OIL_SPRAY = "oil-spray"
COOLING_OIL_FLOOD = "oil-flood"
COOLING_DIRECT    = "direct-water"

INV_2L_VSI = "2L-VSI"
INV_3L_NPC = "3L-NPC"

@dataclass
class EnvelopeConstraints:
    max_outer_diameter_mm: Optional[float] = None
    max_length_mm:         Optional[float] = None
    max_mass_kg:           Optional[float] = None
    mounting:              str = "flange"

@dataclass
class InsulationSpec:
    insulation_class: str   = "H"
    partial_discharge_inception_V: float = 500.0
    surge_voltage_factor:          float = 2.0

    @property
    def max_winding_temp_C(self) -> float:
        return {"B": 130, "F": 155, "H": 180, "C": 220}.get(
            self.insulation_class.upper(), 155)

@dataclass
class CoolingSpec:
    cooling_type:     str   = COOLING_AIR
    coolant_temp_C:   float = 40.0
    coolant_flow_lpm: float = 0.0
    ambient_temp_C:   float = 40.0

    def is_liquid_cooled(self) -> bool:
        return self.cooling_type in (COOLING_WATER, COOLING_OIL_SPRAY,
                                     COOLING_OIL_FLOOD, COOLING_DIRECT)

@dataclass
class DriveSpec:
    dc_bus_voltage:  float = 400.0
    topology:        str   = INV_2L_VSI
    device:          str   = "Si-IGBT"
    switching_freq:  float = 10e3
    modulation:      str   = "SVPWM"
    dead_time_us:    float = 2.0

    def max_phase_voltage_peak(self) -> float:
        if self.modulation == "SVPWM":
            return self.dc_bus_voltage / np.sqrt(3)
        return self.dc_bus_voltage / 2.0

    def max_phase_voltage_rms(self) -> float:
        return self.max_phase_voltage_peak() / np.sqrt(2)

    def validate(self):
        if not (0 <= self.dc_bus_voltage <= 1500):
            raise ValueError(f"DC bus {self.dc_bus_voltage}V outside 0-1500V")

@dataclass
class DesignSpec:
    """
    Complete motor design requirements — not geometry.

    Parameters
    ----------
    power_kW      : Continuous rated output power [kW].
    speed_rpm     : Base (rated) speed [rpm].
    drive         : Inverter / DC bus specification.
    cooling       : Cooling system specification.
    insulation    : Winding insulation class.
    envelope      : Physical size constraints.
    speed_range   : (min_rpm, max_rpm) operating range.
    overload_factor: Peak / continuous torque ratio.
    efficiency_target: Minimum efficiency at rated point.
    phases        : Number of electrical phases.
    duty_cycle    : IEC duty cycle S1/S2/S3/etc.
    """
    power_kW:     float
    speed_rpm:    float

    drive:        DriveSpec         = field(default_factory=DriveSpec)
    cooling:      CoolingSpec       = field(default_factory=CoolingSpec)
    insulation:   InsulationSpec    = field(default_factory=InsulationSpec)
    envelope:     EnvelopeConstraints = field(default_factory=EnvelopeConstraints)

    speed_range:  Optional[tuple]   = None
    peak_torque_Nm: Optional[float] = None
    overload_factor: float          = 2.0
    efficiency_target: float        = 0.93
    power_factor_target: float      = 0.90
    phases:       int               = 3
    connection:   str               = "star"
    duty_cycle:   str               = "S1"

    def __post_init__(self):
        if self.speed_range is None:
            self.speed_range = (self.speed_rpm * 0.1, self.speed_rpm * 3.0)
        self.drive.validate()

    @property
    def rated_torque_Nm(self) -> float:
        return (self.power_kW * 1e3) / (self.speed_rpm * 2 * np.pi / 60)

    @property
    def peak_torque(self) -> float:
        return self.peak_torque_Nm if self.peak_torque_Nm else \
               self.rated_torque_Nm * self.overload_factor

    @property
    def rated_current_estimate(self) -> float:
        P_in = self.power_kW * 1e3 / self.efficiency_target
        return P_in / (self.phases * self.drive.max_phase_voltage_rms())

    @property
    def field_weakening_ratio(self) -> float:
        return self.speed_range[1] / self.speed_rpm

    def summary(self) -> str:
        lines = [
            "=" * 55, "  Design Specification", "=" * 55,
            f"  Rated power          : {self.power_kW:.1f} kW",
            f"  Rated speed          : {self.speed_rpm:.0f} rpm",
            f"  Speed range          : {self.speed_range[0]:.0f} – {self.speed_range[1]:.0f} rpm",
            f"  Rated torque         : {self.rated_torque_Nm:.1f} N·m",
            f"  Peak torque          : {self.peak_torque:.1f} N·m  (×{self.overload_factor:.1f})",
            f"  Efficiency target    : {self.efficiency_target*100:.1f} %",
            f"  Duty cycle           : {self.duty_cycle}",
            "", "  --- Drive ---",
            f"  DC bus voltage       : {self.drive.dc_bus_voltage:.0f} V",
            f"  Max phase V (RMS)    : {self.drive.max_phase_voltage_rms():.1f} V",
            f"  Inverter / device    : {self.drive.topology} / {self.drive.device}",
            f"  Switching freq       : {self.drive.switching_freq/1e3:.0f} kHz",
            f"  CPSR (FW ratio)      : {self.field_weakening_ratio:.1f}×",
            "", "  --- Thermal ---",
            f"  Cooling              : {self.cooling.cooling_type}",
            f"  Coolant inlet temp   : {self.cooling.coolant_temp_C:.0f} °C",
            f"  Insulation class     : {self.insulation.insulation_class}  "
            f"(max {self.insulation.max_winding_temp_C:.0f} °C)",
        ]
        if self.envelope.max_outer_diameter_mm:
            lines.append(f"  Max OD               : {self.envelope.max_outer_diameter_mm:.0f} mm")
        if self.envelope.max_mass_kg:
            lines.append(f"  Max mass             : {self.envelope.max_mass_kg:.1f} kg")
        lines.append("=" * 55)
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, d: dict) -> "DesignSpec":
        d = dict(d)
        drive   = DriveSpec(**d.pop("drive", {}))
        cooling = CoolingSpec(**d.pop("cooling", {}))
        ins     = InsulationSpec(**d.pop("insulation", {}))
        env     = EnvelopeConstraints(**d.pop("envelope", {}))
        return cls(drive=drive, cooling=cooling, insulation=ins, envelope=env, **d)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)
