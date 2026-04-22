"""
Lumped-Parameter Thermal Network (LPTN) for electric motors.

6-node network:
  0 — Winding (copper)
  1 — Stator teeth
  2 — Stator yoke
  3 — Cooling jacket / housing
  4 — Rotor iron + magnets
  5 — Air gap (convection link)

Resistances are computed from geometry and material properties.
Supports both steady-state and transient (drive-cycle) analysis.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from Bohemien_Motor_Designer.thermal.cooling import CoolingSystem


NODE_WINDING = 0
NODE_TEETH   = 1
NODE_YOKE    = 2
NODE_JACKET  = 3
NODE_ROTOR   = 4
N_NODES      = 5


@dataclass
class ThermalResult:
    T_winding_C:  float
    T_teeth_C:    float
    T_yoke_C:     float
    T_jacket_C:   float
    T_rotor_C:    float
    T_coolant_out_C: float

    def summary(self) -> str:
        return (
            f"Thermal Result (steady state):\n"
            f"  Winding        : {self.T_winding_C:6.1f} °C\n"
            f"  Stator teeth   : {self.T_teeth_C:6.1f} °C\n"
            f"  Stator yoke    : {self.T_yoke_C:6.1f} °C\n"
            f"  Cooling jacket : {self.T_jacket_C:6.1f} °C\n"
            f"  Rotor / magnet : {self.T_rotor_C:6.1f} °C\n"
            f"  Coolant outlet : {self.T_coolant_out_C:6.1f} °C"
        )

    def max_temp_C(self) -> float:
        return max(self.T_winding_C, self.T_teeth_C, self.T_yoke_C,
                   self.T_jacket_C, self.T_rotor_C)


class ThermalNetwork:
    """
    6-node LPTN for a liquid or air-cooled PMSM/induction motor.

    Usage::

        from Bohemien_Motor_Designer.thermal.lumped_model import ThermalNetwork
        from Bohemien_Motor_Designer.thermal.cooling import WaterJacketCooling

        cooling = WaterJacketCooling(flow_lpm=12, _inlet_temp_C=65)
        therm   = ThermalNetwork(motor, cooling, material_lib)
        result  = therm.steady_state(loss_budget)
        print(result.summary())
    """

    def __init__(self, motor, cooling: CoolingSystem, material_lib=None):
        self.motor   = motor
        self.cooling = cooling
        from Bohemien_Motor_Designer.materials.library import MaterialLibrary
        self.mlib = material_lib or MaterialLibrary()
        self._build()

    def _build(self):
        """Compute all thermal resistances [K/W] from geometry."""
        m  = self.motor
        st = m.stator

        # Lamination steel properties
        k_fe   = 30.0   # W/m·K — axial is lower, radial ≈ 30
        k_cu   = 400.0
        rho_fe = 7650.0

        # Geometry (fall back to simple estimates if stator not fully defined)
        r_o   = m.stator_outer_radius
        r_i   = m.stator_inner_radius
        L     = m.stack_length
        slots = m.slots

        if st is not None:
            slot_depth  = st.slot_profile.depth()
            slot_area   = st.slot_profile.area()
            tooth_w     = st.tooth_width
            yoke_th     = st.yoke_thickness
            liner_th    = 0.0003   # [m] slot liner (paper or varnish)
            k_liner     = 0.2      # W/m·K
        else:
            slot_depth  = (r_o - r_i) * 0.40
            slot_area   = (r_i * 2 * np.pi / slots) * 0.5 * slot_depth
            tooth_w     = (r_i * 2 * np.pi / slots) * 0.4
            yoke_th     = (r_o - r_i) * 0.30
            liner_th    = 0.0003
            k_liner     = 0.2

        # Slot area for heat conduction (insulation surface)
        A_liner = slots * 2 * slot_depth * L   # two sides of slot per slot

        # 1) Winding → Teeth  (through slot liner)
        self.R_winding_tooth = liner_th / (k_liner * A_liner + 1e-9)

        # 2) Teeth → Yoke  (radial conduction through tooth body)
        A_teeth  = slots * tooth_w * L
        self.R_tooth_yoke = slot_depth / (k_fe * A_teeth + 1e-9)

        # 3) Yoke → Jacket  (radial conduction + contact + convection)
        self.R_yoke_jacket = self.cooling.wall_resistance(r_o, L)

        # 4) Rotor → Air gap  (Taylor-Couette forced convection)
        self.R_rotor_airgap = self._taylor_couette_resistance()

        # Thermal capacitances [J/K] — for transient analysis
        cp_cu  = 385.0
        cp_fe  = 460.0

        mass_cu = slots * slot_area * 0.45 * 8960 * L      # copper mass [kg]
        mass_fe_tooth = A_teeth * slot_depth * 0.5 * rho_fe * L
        mass_fe_yoke  = np.pi * (r_o**2 - (r_i + slot_depth)**2) * rho_fe * L

        self.C_winding = mass_cu * cp_cu
        self.C_teeth   = mass_fe_tooth * cp_fe
        self.C_yoke    = mass_fe_yoke * cp_fe
        self.C_rotor   = (np.pi * (m.rotor_outer_radius**2 -
                                    m.rotor_inner_radius**2) * L * rho_fe * cp_fe)

    def _taylor_couette_resistance(self) -> float:
        """
        Convection resistance: rotor surface → stator bore (air-gap convection).
        Uses Taylor-Couette empirical correlation for rotating cylinder.
        """
        m   = self.motor
        r_r = m.rotor_outer_radius
        r_s = m.stator_inner_radius
        L   = m.stack_length
        g   = m.airgap
        omega = m.rated_speed * 2 * np.pi / 60

        # Taylor number
        rho_air = 1.15; mu_air = 1.85e-5
        Ta = rho_air * omega * r_r * g / mu_air

        # Nusselt correlation (Gazley / Hayase approximation)
        if Ta < 41:
            Nu = 2.0          # laminar Couette
        elif Ta < 100:
            Nu = 0.202 * Ta**0.63
        else:
            Nu = 0.386 * Ta**0.5

        k_air = 0.026   # W/m·K
        h_gap = Nu * k_air / (g + 1e-9)
        A_rot = 2 * np.pi * r_r * L
        return 1.0 / (h_gap * A_rot + 1e-9)

    # ── Steady-state solver ───────────────────────────────────────────────

    def steady_state(self, loss_budget: dict) -> ThermalResult:
        """
        Solve thermal network at steady state.

        Parameters
        ----------
        loss_budget : dict with keys:
            copper_loss_W, stator_iron_W, rotor_iron_W, friction_W, stray_W

        Returns
        -------
        ThermalResult with node temperatures.
        """
        P_cu  = loss_budget.get("copper_loss_W",  0.0)
        P_fe_s = loss_budget.get("stator_iron_W", 0.0)
        P_fe_r = loss_budget.get("rotor_iron_W",  0.0)
        P_mech = loss_budget.get("friction_W",    0.0) + \
                 loss_budget.get("windage_W",      0.0)
        P_tot  = P_cu + P_fe_s + P_fe_r + P_mech

        T0 = self.cooling.inlet_temp_C

        # Solve from coolant inward
        T_coolant_rise = self.cooling.temperature_rise_C(P_tot) \
            if hasattr(self.cooling, "temperature_rise_C") else P_tot * 0.01
        T_jacket = T0 + T_coolant_rise / 2   # mid-point temperature

        # Yoke = jacket + (stator iron + copper) × R_yoke_jacket
        T_yoke   = T_jacket + (P_cu + P_fe_s) * self.R_yoke_jacket

        # Teeth = yoke + copper × R_tooth_yoke  (iron losses generated in tooth too)
        T_teeth  = T_yoke   + (P_cu + P_fe_s * 0.6) * self.R_tooth_yoke

        # Winding = teeth + copper × R_winding_tooth
        T_winding = T_teeth + P_cu * self.R_winding_tooth

        # Rotor isolated by air gap
        T_rotor  = T_jacket + P_fe_r * self.R_rotor_airgap

        return ThermalResult(
            T_winding_C=T_winding,
            T_teeth_C=T_teeth,
            T_yoke_C=T_yoke,
            T_jacket_C=T_jacket,
            T_rotor_C=T_rotor,
            T_coolant_out_C=T0 + T_coolant_rise,
        )

    # ── Transient solver (drive cycle) ────────────────────────────────────

    def transient(self, duty_cycle: list[dict], dt_s: float = 1.0,
                  loss_calculator=None) -> dict:
        """
        Integrate thermal ODE over a drive cycle.

        Parameters
        ----------
        duty_cycle : list of dicts, each with:
            {'torque_Nm': float, 'speed_rpm': float, 'duration_s': float}
        dt_s       : Integration time step [s].
        loss_calculator: LossCalculator instance (to compute losses at each step).

        Returns
        -------
        dict with 'time_s', 'T_winding', 'T_rotor', 'T_magnet' arrays.
        """
        # State vector: [T_winding, T_teeth, T_yoke, T_jacket, T_rotor]
        C = np.array([self.C_winding, self.C_teeth, self.C_yoke,
                       max(self.C_rotor, 1.0), max(self.C_rotor, 1.0)])
        T = np.full(N_NODES, self.cooling.inlet_temp_C + 20.0)

        time_history, T_wind_hist, T_rot_hist = [], [], []
        t_total = 0.0

        for step in duty_cycle:
            torque = step.get("torque_Nm", 0.0)
            speed  = step.get("speed_rpm", 0.0)
            dur    = step.get("duration_s", 10.0)

            # Compute losses for this operating point
            if loss_calculator is not None:
                lb = loss_calculator.loss_budget(speed_rpm=speed, torque_Nm=torque)
                P_cu   = lb.copper_loss_W
                P_fe_s = lb.stator_iron_W
                P_fe_r = lb.rotor_iron_W
            else:
                P_cu = P_fe_s = P_fe_r = 0.0

            Q = np.array([P_cu, P_fe_s * 0.6, P_fe_s * 0.4, 0.0, P_fe_r])

            # Use smaller sub-steps if needed for stability
            tau_min = min(self.C_winding * self.R_winding_tooth,
                          self.C_teeth   * self.R_tooth_yoke)
            sub_dt  = min(dt_s, tau_min * 0.3)
            n_steps = max(1, int(dur / sub_dt))
            sub_dt  = dur / n_steps   # evenly divide

            for _ in range(n_steps):
                T0 = self.cooling.inlet_temp_C
                dT = np.zeros(N_NODES)

                # Heat flow: source Q, conduction along chain
                dT[NODE_WINDING] = (Q[NODE_WINDING]
                    - (T[NODE_WINDING] - T[NODE_TEETH]) / self.R_winding_tooth
                ) / C[NODE_WINDING]
                dT[NODE_TEETH] = (Q[NODE_TEETH]
                    + (T[NODE_WINDING] - T[NODE_TEETH]) / self.R_winding_tooth
                    - (T[NODE_TEETH]   - T[NODE_YOKE])  / self.R_tooth_yoke
                ) / C[NODE_TEETH]
                dT[NODE_YOKE] = (Q[NODE_YOKE]
                    + (T[NODE_TEETH] - T[NODE_YOKE]) / self.R_tooth_yoke
                    - (T[NODE_YOKE]  - T0)           / self.R_yoke_jacket
                ) / C[NODE_YOKE]
                dT[NODE_JACKET] = 0.0
                dT[NODE_ROTOR]  = (Q[NODE_ROTOR]
                    - (T[NODE_ROTOR] - T0) / self.R_rotor_airgap
                ) / C[NODE_ROTOR]

                T += dT * sub_dt
                T[NODE_JACKET] = T0   # enforce coolant BC
                T = np.clip(T, T0 - 5, 500.0)   # physical bounds

                t_total  += dt_s
                time_history.append(t_total)
                T_wind_hist.append(T[NODE_WINDING])
                T_rot_hist.append(T[NODE_ROTOR])

        return {
            "time_s":     np.array(time_history),
            "T_winding_C":np.array(T_wind_hist),
            "T_rotor_C":  np.array(T_rot_hist),
            "T_final":    ThermalResult(
                T_winding_C=T[NODE_WINDING],
                T_teeth_C=T[NODE_TEETH],
                T_yoke_C=T[NODE_YOKE],
                T_jacket_C=T[NODE_JACKET],
                T_rotor_C=T[NODE_ROTOR],
                T_coolant_out_C=self.cooling.inlet_temp_C + 5,
            )
        }
