"""
Performance analysis — efficiency map, torque-speed envelope, MTPA trajectory.
"""
from __future__ import annotations
import numpy as np
from typing import Optional


class PerformanceAnalyzer:
    """
    Compute motor performance maps across speed and torque range.

    Parameters
    ----------
    motor        : Motor instance.
    material_lib : MaterialLibrary.
    inverter     : Optional Inverter (for switching losses + V-limit).
    """

    def __init__(self, motor, material_lib=None, inverter=None):
        self.motor    = motor
        self.inverter = inverter
        from Bohemien_Motor_Designer.materials.library import MaterialLibrary
        self.mlib = material_lib or MaterialLibrary()
        from Bohemien_Motor_Designer.analysis.losses import LossCalculator
        self.loss_calc = LossCalculator(motor, self.mlib,
                                         temperature=75.0, inverter=inverter)

    def efficiency_at(self, speed_rpm: float, torque_Nm: float) -> float:
        """Efficiency [0–1] at a single operating point."""
        if speed_rpm < 10 or torque_Nm < 0.01:
            return 0.0
        lb = self.loss_calc.loss_budget(speed_rpm, torque_Nm)
        return lb.efficiency

    def pmsm_efficiency_map(self, speed_range: tuple = None,
                              n_speed: int = 30, n_torque: int = 25,
                              T_max: float = None) -> dict:
        """
        Compute efficiency map on a speed × torque grid.

        Returns dict with: speed_rpm, torque_Nm, efficiency, power_W grids.
        """
        m = self.motor
        if speed_range is None:
            speed_range = (100, m.rated_speed * 3)
        if T_max is None:
            T_max = m.rated_torque * 2.5

        speeds  = np.linspace(speed_range[0], speed_range[1], n_speed)
        torques = np.linspace(0.5, T_max, n_torque)

        eff_map  = np.full((n_torque, n_speed), np.nan)
        pwr_map  = np.full((n_torque, n_speed), np.nan)

        for i, T in enumerate(torques):
            for j, n in enumerate(speeds):
                omega = n * 2 * np.pi / 60
                P_out = T * omega
                if P_out < 1.0:
                    continue
                try:
                    eff = self.efficiency_at(n, T)
                    if 0.1 < eff <= 1.0:
                        eff_map[i, j] = eff
                        pwr_map[i, j] = P_out
                except Exception:
                    pass

        return {
            "speed_rpm":   speeds,
            "torque_Nm":   torques,
            "efficiency":  eff_map,
            "power_W":     pwr_map,
            "peak_eff":    float(np.nanmax(eff_map)) if not np.all(np.isnan(eff_map)) else 0,
            "rated_speed": m.rated_speed,
            "rated_torque": m.rated_torque,
        }

    def mtpa_trajectory(self, n_points: int = 40) -> dict:
        """
        MTPA current trajectory from zero to rated torque.
        Returns Id, Iq, I_total, torque arrays.
        """
        m = self.motor
        psi_m = m.back_emf_constant() if hasattr(m, "back_emf_constant") else 0.1
        T_max = m.rated_torque * 2.0

        torques = np.linspace(0.1, T_max, n_points)
        Id_arr  = np.zeros(n_points)
        Iq_arr  = np.zeros(n_points)

        for i, T in enumerate(torques):
            Iq = T / (1.5 * m.pole_pairs * psi_m + 1e-9)
            if hasattr(m, "mtpa_angle") and hasattr(m, "Ld") and m.Ld > 1e-6:
                I_pk = Iq   # rough estimate
                Id, Iq = m.mtpa_angle(I_pk)
            else:
                Id = 0.0
            Id_arr[i] = Id
            Iq_arr[i] = Iq

        return {
            "Id":       Id_arr,
            "Iq":       Iq_arr,
            "I_total":  np.sqrt(Id_arr**2 + Iq_arr**2),
            "torque_Nm":torques,
        }

    def torque_speed_envelope(self, n_points: int = 80,
                               I_max: float = None) -> dict:
        """
        Maximum torque-speed envelope using field-weakening controller.
        """
        m = self.motor
        if I_max is None:
            omega_r = m.rated_speed * 2 * np.pi / 60
            psi_m   = m.back_emf_constant() if hasattr(m, "back_emf_constant") else 0.1
            I_max   = m.rated_torque / (1.5 * m.pole_pairs * psi_m) * 2.5

        if self.inverter and hasattr(m, "Ld") and m.Ld > 0:
            from Bohemien_Motor_Designer.drive.field_weakening import FieldWeakeningController
            fw = FieldWeakeningController(m, self.inverter, I_max)
            return fw.torque_speed_envelope(n_points)

        # Simple approximation without full FW controller
        n_base  = m.rated_speed
        n_max   = n_base * 3.0
        speeds  = np.linspace(50, n_max, n_points)
        T_rated = m.rated_torque
        T_arr   = np.where(speeds <= n_base,
                           T_rated * 2.0,
                           T_rated * 2.0 * n_base / speeds)
        return {
            "speed_rpm": speeds,
            "T_max_Nm":  T_arr,
            "power_W":   T_arr * speeds * 2 * np.pi / 60,
        }

    def continuous_rating_check(self, thermal_network,
                                 duty_cycle: list[dict]) -> dict:
        """
        Check if motor can sustain a drive cycle without exceeding temperature limits.

        Returns dict with pass/fail, peak winding temp, time to overheat.
        """
        m      = self.motor
        T_max  = 155.0   # class F default
        if m.spec:
            T_max = m.spec.insulation.max_winding_temp_C

        history = thermal_network.transient(
            duty_cycle, dt_s=2.0, loss_calculator=self.loss_calc)

        T_peak    = float(np.max(history["T_winding_C"]))
        overheat_t= None
        for t, T in zip(history["time_s"], history["T_winding_C"]):
            if T > T_max and overheat_t is None:
                overheat_t = t

        return {
            "pass":          T_peak <= T_max,
            "T_winding_peak_C": T_peak,
            "T_limit_C":     T_max,
            "overheat_time_s": overheat_t,
            "history":       history,
        }
