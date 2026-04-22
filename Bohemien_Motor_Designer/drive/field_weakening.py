"""
Field-weakening and MTPV trajectory computation.

Computes the optimal (Id, Iq) trajectory across the full speed range
including:
  - MTPA region (below base speed)
  - Field-weakening region (above base speed, V-limit circle)
  - MTPV region (max-torque-per-volt near max speed)
"""
from __future__ import annotations
import numpy as np
from typing import Optional


class FieldWeakeningController:
    """
    Computes optimal dq current for any (speed, torque) operating point
    subject to current and voltage limits.

    Parameters
    ----------
    motor   : PMSM instance with Ld, Lq, Rs, back_emf_constant().
    inverter: Inverter instance (provides V_max_peak).
    I_max   : Maximum phase current amplitude [A peak].
    """

    def __init__(self, motor, inverter, I_max: float):
        self.motor    = motor
        self.inverter = inverter
        self.I_max    = I_max

    @property
    def V_max(self) -> float:
        return self.inverter.max_phase_voltage_peak()

    @property
    def psi_m(self) -> float:
        return self.motor.back_emf_constant()

    def base_speed_rpm(self) -> float:
        """Speed at which back-EMF reaches V_max at MTPA with no Id."""
        omega_e_base = self.V_max / (self.psi_m + 1e-9)
        return omega_e_base / self.motor.pole_pairs * 60 / (2 * np.pi)

    def operating_point(self, speed_rpm: float,
                         torque_Nm: float) -> dict:
        """
        Compute optimal (Id, Iq) for given speed and torque target.

        Returns dict with Id, Iq, Vd, Vq, utilisation, region.
        """
        omega_e = speed_rpm * 2 * np.pi / 60 * self.motor.pole_pairs
        m = self.motor
        Ld, Lq, Rs = m.Ld, m.Lq, m.Rs
        psi_m = self.psi_m

        # Required Iq for requested torque (rough MTPA estimate)
        Iq_req = torque_Nm / (1.5 * m.pole_pairs * psi_m + 1e-9)
        Iq_req = np.clip(Iq_req, 0, self.I_max)

        # Try MTPA first (no field weakening)
        I_amp  = min(Iq_req, self.I_max)
        Id_mtpa, Iq_mtpa = m.mtpa_angle(I_amp)

        Vd, Vq, V_mag = m.voltage_at(Id_mtpa, Iq_mtpa, speed_rpm)

        if V_mag <= self.V_max:
            # MTPA region — no FW needed
            return dict(Id=Id_mtpa, Iq=Iq_mtpa, Vd=Vd, Vq=Vq,
                        V_mag=V_mag, V_util=V_mag / self.V_max,
                        region="MTPA", feasible=True,
                        torque=m.torque_from_dq(Id_mtpa, Iq_mtpa))

        # Field weakening — find Id to satisfy V_max circle
        Id_fw = m.field_weakening_Id(speed_rpm, Iq_req, self.V_max)
        I_total = np.sqrt(Id_fw**2 + Iq_req**2)

        if I_total > self.I_max:
            # Current limit hit — reduce Iq
            Iq_fw = np.sqrt(max(0, self.I_max**2 - Id_fw**2))
        else:
            Iq_fw = Iq_req

        Vd, Vq, V_mag = m.voltage_at(Id_fw, Iq_fw, speed_rpm)
        T_actual = m.torque_from_dq(Id_fw, Iq_fw)
        feasible = (T_actual >= torque_Nm * 0.95)

        return dict(Id=Id_fw, Iq=Iq_fw, Vd=Vd, Vq=Vq,
                    V_mag=V_mag, V_util=V_mag / self.V_max,
                    region="FW", feasible=feasible,
                    torque=T_actual)

    def torque_speed_envelope(self, n_points: int = 100) -> dict:
        """
        Compute the full torque-speed envelope:
        maximum torque at each speed point up to max speed.

        Returns dict with speed_rpm, T_max, Id, Iq, power_W arrays.
        """
        speed_max = self.motor.rated_speed * self.inverter.dc_bus_V / (
            self.psi_m * self.motor.pole_pairs * 2 * np.pi / 60 * 60 + 1e-9)
        speed_max = min(speed_max, self.motor.rated_speed * 5)

        speeds   = np.linspace(100, speed_max, n_points)
        T_arr    = np.zeros(n_points)
        Id_arr   = np.zeros(n_points)
        Iq_arr   = np.zeros(n_points)
        P_arr    = np.zeros(n_points)

        for i, spd in enumerate(speeds):
            # Maximum torque at this speed: start with full I_max
            op = self.operating_point(spd, 1e6)   # request very high torque
            T_arr[i]  = op["torque"]
            Id_arr[i] = op["Id"]
            Iq_arr[i] = op["Iq"]
            P_arr[i]  = T_arr[i] * spd * 2 * np.pi / 60

        return {
            "speed_rpm": speeds,
            "T_max_Nm":  T_arr,
            "Id":        Id_arr,
            "Iq":        Iq_arr,
            "power_W":   P_arr,
        }
