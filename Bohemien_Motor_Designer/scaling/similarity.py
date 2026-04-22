"""
Motor scaling laws — dimensional analysis and similarity relations.

Based on Esson's output coefficient and classical electromagnetic
scaling theory. Allows rapid feasibility assessment across the
1 kW – 1 MW power range before running FEA.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


# Esson's output coefficient C [kN·m/m³] by cooling type
# Source: Hendershot & Miller, "Design of Brushless PM Motors" (2010)
ESSON_COEFFICIENT = {
    "natural_convection":  8,
    "TEFC_air":           15,
    "forced_air":         25,
    "water-jacket":       50,
    "oil-spray":          80,
    "direct-water":       150,
}

# Typical current density limits [A/mm²] by cooling type
CURRENT_DENSITY_LIMIT = {
    "natural_convection":  3.0,
    "TEFC_air":            4.5,
    "forced_air":          7.0,
    "water-jacket":       12.0,
    "oil-spray":          18.0,
    "direct-water":       25.0,
}

# Linear current density A_s [kA/m] at bore
LINEAR_CURRENT_DENSITY = {
    "natural_convection": 20,
    "TEFC_air":           35,
    "forced_air":         50,
    "water-jacket":       80,
    "oil-spray":         120,
    "direct-water":      200,
}


@dataclass
class SizeEstimate:
    """Result of a motor size estimate from scaling laws."""
    power_kW:         float
    speed_rpm:        float
    cooling:          str
    torque_Nm:        float
    outer_diameter_mm: float
    stack_length_mm:  float
    aspect_ratio:     float   # L/D
    active_volume_L:  float
    shear_stress_kPa: float
    current_density_A_mm2: float
    linear_current_density_kA_m: float
    pole_pairs_estimate:  int
    mass_estimate_kg: float

    def summary(self) -> str:
        return (
            f"Size Estimate ({self.power_kW:.0f} kW, {self.speed_rpm:.0f} rpm, {self.cooling})\n"
            f"  Torque            : {self.torque_Nm:.1f} N·m\n"
            f"  Outer diameter    : {self.outer_diameter_mm:.0f} mm\n"
            f"  Stack length      : {self.stack_length_mm:.0f} mm\n"
            f"  L/D aspect ratio  : {self.aspect_ratio:.2f}\n"
            f"  Active volume     : {self.active_volume_L:.2f} L\n"
            f"  Torque density    : {self.torque_Nm / self.active_volume_L * 1e3:.1f} N·m/m³  "
            f"({self.shear_stress_kPa:.1f} kPa)\n"
            f"  Current density   : {self.current_density_A_mm2:.1f} A/mm²\n"
            f"  Pole pairs (est.) : {self.pole_pairs_estimate}"
        )


class MotorScalingLaws:
    """
    Dimensional analysis and similarity scaling for electric motors.

    Key scaling relationships:
      Torque   ∝ D² · L             (Esson's output coefficient)
      Speed    ∝ 1/√(pole_pitch)    (EMF / pole pitch constraint)
      Loss     ∝ D³                 (volume losses, fixed loss density)
      Current  ∝ D                  (linear current density = const)
      Voltage  ∝ D · n · N_turns    (EMF)
      Mass     ∝ D² · L             (same topology, same materials)
    """

    @staticmethod
    def size_estimate(power_kW: float, speed_rpm: float,
                       cooling: str = "water-jacket",
                       aspect_ratio: float = 1.2) -> SizeEstimate:
        """
        Estimate motor outer diameter and stack length from power and speed.

        Parameters
        ----------
        power_kW     : Required output power [kW].
        speed_rpm    : Required speed [rpm].
        cooling      : Cooling type key.
        aspect_ratio : Desired L/D ratio (typically 0.8–2.0).

        Returns
        -------
        SizeEstimate with geometry and electromagnetic estimates.
        """
        if cooling not in ESSON_COEFFICIENT:
            raise ValueError(f"Unknown cooling type '{cooling}'. "
                             f"Options: {list(ESSON_COEFFICIENT)}")

        C      = ESSON_COEFFICIENT[cooling] * 1e3   # N·m/m³
        omega  = speed_rpm * 2 * np.pi / 60
        torque = power_kW * 1e3 / (omega + 1e-9)

        # Esson: T = C_o * (D_bore/2)² * L_stack
        # With L = LD * D:   T = C_o * (LD/4π) * D³
        # → D = (4π * T / (C * LD))^(1/3) ... (full cylinder approximation)
        # More precisely for the bore (not OD):
        D_bore = (8 * torque / (np.pi * C * aspect_ratio + 1e-9)) ** (1/3)
        L      = D_bore * aspect_ratio
        D_OD   = D_bore * 1.55   # typical OD ≈ 1.55 × bore for inrunner

        # Active volume
        V_active = np.pi * (D_bore/2)**2 * L

        # Current density from thermal limit
        J_max  = CURRENT_DENSITY_LIMIT[cooling]      # A/mm²
        A_s    = LINEAR_CURRENT_DENSITY[cooling]     # kA/m

        # Pole count estimate: choose p so that tip speed ≈ 100–150 m/s (SPM)
        # or frequency ≈ 100–200 Hz
        f_target = 150.0   # Hz
        p_est    = max(1, round(f_target * 60 / speed_rpm))
        p_est    = max(1, p_est)

        # Mass estimate: iron + copper = ~7000 kg/m³ equivalent
        mass_est = V_active * 7000

        shear_stress = C / 2 / 1e3   # kPa

        return SizeEstimate(
            power_kW=power_kW,
            speed_rpm=speed_rpm,
            cooling=cooling,
            torque_Nm=torque,
            outer_diameter_mm=D_OD * 1e3,
            stack_length_mm=L * 1e3,
            aspect_ratio=aspect_ratio,
            active_volume_L=V_active * 1e3,
            shear_stress_kPa=shear_stress,
            current_density_A_mm2=J_max,
            linear_current_density_kA_m=A_s,
            pole_pairs_estimate=p_est,
            mass_estimate_kg=mass_est,
        )

    @staticmethod
    def scale_from_reference(ref_power_kW: float, ref_speed_rpm: float,
                               ref_OD_mm: float, ref_L_mm: float,
                               new_power_kW: float, new_speed_rpm: float,
                               same_poles: bool = True) -> dict:
        """
        Scale all dimensions from a known reference motor design.

        Assumes geometric similarity and same cooling type.

        Returns scaling factors for geometry, current, voltage, turns.
        """
        omega_ref = ref_speed_rpm * 2 * np.pi / 60
        omega_new = new_speed_rpm  * 2 * np.pi / 60
        T_ref     = ref_power_kW * 1e3 / (omega_ref + 1e-9)
        T_new     = new_power_kW * 1e3 / (omega_new + 1e-9)

        # Torque ∝ D² * L → scaling for same L/D: D ∝ T^(1/3)
        k_T = T_new / (T_ref + 1e-9)
        k_D = k_T ** (1/3)   # all linear dimensions scale equally
        k_L = k_D

        # Speed scaling
        k_n = new_speed_rpm / (ref_speed_rpm + 1e-9)

        # EMF ∝ D * n * N  → if V stays same, N ∝ 1/(D * n)
        k_V = 1.0  # assume same DC bus → same phase voltage
        k_N = k_V / (k_D * k_n + 1e-9)   # turns scale to maintain voltage

        # Current ∝ D (linear current density = const)
        k_I = k_D

        # Power ∝ V * I = (k_V) * (k_D) → check
        k_P = k_V * k_I * k_n   # power = torque × speed

        return {
            "diameter_factor":      k_D,
            "length_factor":        k_L,
            "speed_factor":         k_n,
            "torque_factor":        k_T,
            "current_factor":       k_I,
            "voltage_factor":       k_V,
            "turns_factor":         k_N,
            "power_factor":         k_P,
            "new_OD_mm":           ref_OD_mm * k_D,
            "new_L_mm":            ref_L_mm  * k_L,
            "description": (
                f"Scale {ref_power_kW:.0f}kW/{ref_speed_rpm:.0f}rpm → "
                f"{new_power_kW:.0f}kW/{new_speed_rpm:.0f}rpm: "
                f"D×{k_D:.2f}, L×{k_L:.2f}, I×{k_I:.2f}, N×{k_N:.2f}"
            )
        }

    @staticmethod
    def compare_cooling(power_kW: float, speed_rpm: float) -> list[SizeEstimate]:
        """
        Compare motor size for different cooling options at same power/speed.
        Returns list of SizeEstimate sorted by volume (smallest first).
        """
        results = []
        for cooling in ESSON_COEFFICIENT:
            try:
                est = MotorScalingLaws.size_estimate(power_kW, speed_rpm, cooling)
                results.append(est)
            except Exception:
                pass
        results.sort(key=lambda x: x.active_volume_L)
        return results

    @staticmethod
    def feasibility_check(power_kW: float, speed_rpm: float,
                           max_OD_mm: float, max_L_mm: float,
                           cooling: str = "water-jacket") -> dict:
        """
        Check whether a motor of given power fits in an envelope.

        Returns dict with feasible (bool), size, margin, and recommendation.
        """
        est = MotorScalingLaws.size_estimate(power_kW, speed_rpm, cooling)
        fits_OD = est.outer_diameter_mm <= max_OD_mm
        fits_L  = est.stack_length_mm  <= max_L_mm
        fits    = fits_OD and fits_L

        margin_OD = (max_OD_mm - est.outer_diameter_mm) / max_OD_mm * 100
        margin_L  = (max_L_mm  - est.stack_length_mm)  / max_L_mm  * 100

        if not fits:
            # Try denser cooling
            for better in ["oil-spray", "direct-water"]:
                if better != cooling:
                    est2 = MotorScalingLaws.size_estimate(power_kW, speed_rpm, better)
                    if est2.outer_diameter_mm <= max_OD_mm and est2.stack_length_mm <= max_L_mm:
                        recommendation = f"Switch to {better} cooling to fit in envelope"
                        break
            else:
                recommendation = "Motor cannot fit in envelope with any standard cooling"
        else:
            recommendation = "Motor fits with current cooling"

        return {
            "feasible":        fits,
            "size":            est,
            "OD_margin_pct":   margin_OD,
            "L_margin_pct":    margin_L,
            "recommendation":  recommendation,
        }
