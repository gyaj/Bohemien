"""
Inverter and drive model — converts DC bus to motor terminals.

Computes:
  - Usable phase voltage from DC bus and modulation scheme
  - Switching and conduction losses
  - Voltage utilisation at each operating point
  - Field-weakening and MTPV limits
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# Device parameter table [Vce_sat V, Vd_fwd V, Eon mJ, Eoff mJ, I_rated A]
_DEVICE_PARAMS = {
    "Si-IGBT":    dict(Vce_sat=1.50, Vd_fwd=1.20, Eon_mJ=5.0,  Eoff_mJ=4.0,  I_nom=400),
    "SiC-MOSFET": dict(Vce_sat=0.30, Vd_fwd=0.90, Eon_mJ=0.6,  Eoff_mJ=0.4,  I_nom=300),
    "GaN":        dict(Vce_sat=0.10, Vd_fwd=0.50, Eon_mJ=0.10, Eoff_mJ=0.08, I_nom=100),
    "SiC-SBD":    dict(Vce_sat=0.20, Vd_fwd=0.70, Eon_mJ=0.3,  Eoff_mJ=0.2,  I_nom=200),
}


@dataclass
class Inverter:
    """
    2-level or 3-level voltage source inverter model.

    Parameters
    ----------
    dc_bus_V      : DC link voltage [V] (0–1500).
    switching_freq: PWM switching frequency [Hz].
    topology      : '2L-VSI' | '3L-NPC'.
    device        : Switch technology key.
    dead_time_us  : Dead time [µs].
    n_parallel    : Number of parallel switch modules per leg.
    """
    dc_bus_V:       float = 400.0
    switching_freq: float = 10e3
    topology:       str   = "2L-VSI"
    device:         str   = "Si-IGBT"
    dead_time_us:   float = 2.0
    n_parallel:     int   = 1

    # Device parameters (auto-set from device key)
    Vce_sat:  float = field(init=False)
    Vd_fwd:   float = field(init=False)
    Eon_mJ:   float = field(init=False)
    Eoff_mJ:  float = field(init=False)
    I_nom:    float = field(init=False)

    def __post_init__(self):
        if self.device not in _DEVICE_PARAMS:
            raise ValueError(f"Unknown device '{self.device}'. "
                             f"Options: {list(_DEVICE_PARAMS)}")
        for k, v in _DEVICE_PARAMS[self.device].items():
            setattr(self, k, v)
        if self.dc_bus_V > 1500:
            raise ValueError(f"DC bus {self.dc_bus_V}V exceeds 1500V limit")

    # ── Voltage limits ─────────────────────────────────────────────────

    def max_phase_voltage_peak(self, modulation: str = "SVPWM") -> float:
        """Maximum phase-to-neutral peak voltage [V]."""
        if self.topology == "3L-NPC":
            return self.dc_bus_V / 2 / np.sqrt(2) * np.sqrt(3)  # SVPWM, 3-level
        # 2L-VSI
        if modulation == "SVPWM":
            return self.dc_bus_V / np.sqrt(3)       # 15% gain over SPWM
        return self.dc_bus_V / 2.0                  # SPWM

    def max_phase_voltage_rms(self, modulation: str = "SVPWM") -> float:
        return self.max_phase_voltage_peak(modulation) / np.sqrt(2)

    def voltage_utilisation(self, motor, speed_rpm: float,
                             Id: float, Iq: float) -> float:
        """
        Fraction of DC bus used: |V_dq| / V_max_peak.
        Values > 1.0 → field weakening required.
        """
        _, _, V_mag = motor.voltage_at(Id, Iq, speed_rpm)
        V_max = self.max_phase_voltage_peak()
        return V_mag / (V_max + 1e-9)

    # ── Loss models ────────────────────────────────────────────────────

    def switching_loss_W(self, I_rms: float, phases: int = 3) -> float:
        """
        Total inverter switching loss [W].

        Scales Eon/Eoff data from I_nom to actual current using linear model.
        """
        I_pk   = I_rms * np.sqrt(2) / self.n_parallel
        ratio  = min(I_pk / (self.I_nom + 1e-9), 3.0)   # cap at 3× rated
        E_tot  = (self.Eon_mJ + self.Eoff_mJ) * 1e-3 * ratio  # J per switch event
        n_sw   = phases * 2   # switches per phase × phases
        return n_sw * E_tot * self.switching_freq

    def conduction_loss_W(self, I_rms: float, power_factor: float = 0.95,
                           phases: int = 3) -> float:
        """IGBT/MOSFET + diode conduction loss [W]."""
        # Simplified: average conduction in upper switch + lower diode
        I_avg = I_rms * np.sqrt(2) / np.pi   # half-wave average of sinusoid
        P_sw  = phases * self.Vce_sat * I_avg
        P_d   = phases * self.Vd_fwd  * I_avg * (1 - power_factor)
        return (P_sw + P_d) / self.n_parallel

    def total_loss_W(self, I_rms: float, power_factor: float = 0.95,
                      phases: int = 3) -> float:
        return (self.switching_loss_W(I_rms, phases) +
                self.conduction_loss_W(I_rms, power_factor, phases))

    def efficiency(self, output_power_W: float, I_rms: float,
                   power_factor: float = 0.95) -> float:
        P_loss = self.total_loss_W(I_rms, power_factor)
        return output_power_W / (output_power_W + P_loss + 1e-9)

    # ── PWM harmonic spectrum ──────────────────────────────────────────

    def voltage_harmonics(self, m_index: float = 0.9) -> list[tuple[int, float]]:
        """
        Dominant PWM voltage harmonics as (order, V_pk_fraction) pairs.
        Returns harmonics relative to fundamental using standard SVPWM spectrum.
        m_index: modulation index (0–1).
        """
        f_sw_ratio = self.switching_freq   # ratio to fundamental (caller scales)
        # Main SVPWM sidebands at n*fsw ± m*f1 (m=1,3,5 for n=1,2,...)
        harmonics = [
            (1, 1.0),                     # fundamental
            (int(f_sw_ratio - 2), 0.32 * m_index),
            (int(f_sw_ratio),     0.60 * (1 - m_index**2)**0.5),
            (int(f_sw_ratio + 2), 0.32 * m_index),
            (int(2 * f_sw_ratio - 1), 0.12),
            (int(2 * f_sw_ratio + 1), 0.12),
        ]
        return [(h, v) for h, v in harmonics if h > 1]

    def summary(self) -> str:
        return (
            f"Inverter [{self.topology} / {self.device}]\n"
            f"  DC bus             : {self.dc_bus_V:.0f} V\n"
            f"  Max phase V (RMS)  : {self.max_phase_voltage_rms():.1f} V\n"
            f"  Switching freq     : {self.switching_freq/1e3:.0f} kHz\n"
            f"  Dead time          : {self.dead_time_us:.1f} µs\n"
            f"  Parallel modules   : {self.n_parallel}"
        )
