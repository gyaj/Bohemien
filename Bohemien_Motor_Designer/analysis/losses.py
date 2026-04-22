"""
Loss calculation — copper, iron, mechanical, inverter, stray.

Supports:
  - Harmonic-aware iron losses (Bertotti three-term per harmonic)
  - AC copper losses (Dowell's method for skin + proximity)
  - PWM harmonic copper losses
  - Temperature-corrected resistances
  - dq-model current derivation (consistent with efficiency map)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from Bohemien_Motor_Designer.materials.library import MaterialLibrary


@dataclass
class LossBudget:
    speed_rpm:       float
    torque_Nm:       float
    output_power_W:  float
    copper_loss_W:   float
    stator_iron_W:   float
    rotor_iron_W:    float
    friction_W:      float
    windage_W:       float
    stray_W:         float
    inverter_loss_W: float = 0.0
    ac_factor:       float = 1.0

    @property
    def total_loss_W(self) -> float:
        return (self.copper_loss_W + self.stator_iron_W + self.rotor_iron_W +
                self.friction_W + self.windage_W + self.stray_W + self.inverter_loss_W)

    @property
    def input_power_W(self) -> float:
        return self.output_power_W + self.total_loss_W

    @property
    def efficiency(self) -> float:
        return self.output_power_W / (self.input_power_W + 1e-9)

    def to_dict(self) -> dict:
        return {
            "speed_rpm":       self.speed_rpm,
            "torque_Nm":       self.torque_Nm,
            "output_power_W":  self.output_power_W,
            "copper_loss_W":   self.copper_loss_W,
            "stator_iron_W":   self.stator_iron_W,
            "rotor_iron_W":    self.rotor_iron_W,
            "friction_W":      self.friction_W,
            "windage_W":       self.windage_W,
            "stray_W":         self.stray_W,
            "inverter_loss_W": self.inverter_loss_W,
            "total_loss_W":    self.total_loss_W,
            "efficiency":      self.efficiency,
        }

    def print_summary(self):
        print(f"\n{'='*50}")
        print(f"  Loss Budget  @  {self.speed_rpm:.0f} rpm  |  {self.torque_Nm:.1f} N·m")
        print(f"  Output power     : {self.output_power_W/1e3:.3f} kW")
        print(f"{'='*50}")
        print(f"  Copper loss      : {self.copper_loss_W:.1f} W  (AC factor: {self.ac_factor:.2f}x)")
        print(f"  Stator iron      : {self.stator_iron_W:.1f} W")
        print(f"  Rotor iron       : {self.rotor_iron_W:.1f} W")
        print(f"  Friction         : {self.friction_W:.1f} W")
        print(f"  Windage          : {self.windage_W:.1f} W")
        print(f"  Stray            : {self.stray_W:.1f} W")
        if self.inverter_loss_W > 0:
            print(f"  Inverter         : {self.inverter_loss_W:.1f} W")
        print(f"  {'─'*44}")
        print(f"  Total losses     : {self.total_loss_W:.1f} W")
        print(f"  Input power      : {self.input_power_W/1e3:.3f} kW")
        print(f"  EFFICIENCY       : {self.efficiency*100:.2f} %")
        print(f"{'='*50}\n")


class LossCalculator:
    """
    Compute all motor losses at any operating point.

    Parameters
    ----------
    motor        : Motor instance.
    material_lib : MaterialLibrary (or None to use defaults).
    temperature  : Winding temperature [°C] for copper resistance.
    inverter     : Optional Inverter instance for switching losses.
    """

    def __init__(self, motor, material_lib=None,
                 temperature: float = 75.0, inverter=None):
        self.motor       = motor
        self.mlib        = material_lib or MaterialLibrary()
        self.temperature = temperature
        self.inverter    = inverter

    # ── Resistance ────────────────────────────────────────────────────────

    def phase_resistance(self) -> float:
        """DC phase resistance at operating temperature [Ω].

        Conductor cross-section is derived from slot geometry (fill × slot_area / turns),
        which correctly accounts for stranded / parallel-wire conductors.
        """
        m    = self.motor
        cond = self.mlib.conductor("copper")
        rho  = cond.resistivity_at(self.temperature)

        # Mean turn length: axial slot + two end-winding half-turns
        r_ew   = m.stator_inner_radius * 0.7   # end-winding mean radius
        L_turn = 2 * m.stack_length + np.pi * r_ew

        N      = m.winding.total_series_turns_per_phase

        # Conductor cross-section per series turn — derived from slot geometry.
        # Each slot-side has turns_per_coil conductors sharing the copper area.
        if m.stator is not None:
            slot_area  = m.stator.slot_profile.area()
        else:
            slot_area  = 100e-6   # fallback 100 mm²
        fill    = getattr(m, "slot_fill_factor", 0.45)
        t_coil  = getattr(m, "turns_per_coil", 1)
        # Each slot-side copper area ÷ turns_per_coil = area per conductor
        A_c = slot_area * fill / max(t_coil, 1)

        return rho * N * L_turn / (A_c + 1e-12)

    def ac_resistance_factor(self, freq: float = None) -> float:
        """Rac/Rdc using Dowell's method for the fundamental frequency."""
        m = self.motor
        if freq is None:
            freq = m.electrical_frequency
        if freq < 1:
            return 1.0
        cond   = self.mlib.conductor("copper")
        d_c    = getattr(m, "conductor_diameter", 0.0012)
        n_lay  = getattr(m, "winding", None)
        n_lay  = n_lay.layers if n_lay else 2
        return cond.ac_factor_dowell(d_c, freq, n_lay)

    # ── Copper loss ───────────────────────────────────────────────────────

    def _dq_currents(self, speed_rpm: float, torque_Nm: float) -> tuple[float, float]:
        """
        Derive (Id, Iq) peak currents from operating point.
        Uses MTPA below base speed, field-weakening above.
        """
        m     = self.motor
        psi_m = m.back_emf_constant() if hasattr(m, "back_emf_constant") else 0.1
        Iq    = torque_Nm / (1.5 * m.pole_pairs * psi_m + 1e-9)

        # Field weakening if voltage exceeded
        if self.inverter and hasattr(m, "Ld") and m.Ld > 0:
            V_max = self.inverter.max_phase_voltage_peak()
            _, _, V_mag = m.voltage_at(0.0, Iq, speed_rpm)
            if V_mag > V_max:
                Id = m.field_weakening_Id(speed_rpm, Iq, V_max)
                I_tot = np.sqrt(Id**2 + Iq**2)
                I_lim = getattr(m, "I_max", 3 * Iq)
                if I_tot > I_lim:
                    Iq = np.sqrt(max(0, I_lim**2 - Id**2))
                return Id, Iq
        return 0.0, Iq

    def copper_loss(self, speed_rpm: float, torque_Nm: float,
                    include_ac: bool = True) -> tuple[float, float]:
        """
        Phase copper loss [W] and AC factor.
        Returns (P_cu, ac_factor).
        """
        _, Iq = self._dq_currents(speed_rpm, torque_Nm)
        I_rms = Iq / np.sqrt(2)   # peak → RMS for SPM MTPA (Id≈0)

        R_dc  = self.phase_resistance()
        kac   = self.ac_resistance_factor() if include_ac else 1.0
        P_cu  = 3 * (I_rms**2) * R_dc * kac
        return P_cu, kac

    # ── Iron losses ───────────────────────────────────────────────────────

    def _airgap_B_fundamental(self) -> float:
        """
        Fundamental air-gap flux density [T] — Carter formula PM field.
        """
        m = self.motor
        if not hasattr(m, "back_emf_constant"):
            return 0.7  # fallback
        kw   = m.winding_factor()
        N    = m.winding.total_series_turns_per_phase
        psi_m= m.back_emf_constant()
        Lz   = m.stack_length
        tau_p= m.pole_pitch
        # Invert: psi_m = (2/π) * N * kw * B * tau_p * Lz
        B_gap = psi_m * np.pi / (2 * N * kw * tau_p * Lz + 1e-9)
        return np.clip(B_gap, 0.05, 2.0)

    def stator_iron_loss(self, speed_rpm: float,
                          harmonics: Optional[list] = None) -> float:
        """Stator iron loss [W] at given speed."""
        m   = self.motor
        st  = m.stator
        f   = speed_rpm / 60 * m.pole_pairs
        B_g = self._airgap_B_fundamental()

        try:
            lam_key = st.lamination if st else "M270-35A"
            lam = self.mlib.lamination(lam_key)
        except Exception:
            lam = self.mlib.lamination("M270-35A")

        # Tooth and yoke flux densities
        if st:
            B_tooth = st.tooth_flux_density(B_g)
            B_yoke  = st.yoke_flux_density(B_g, m.poles, m.stack_length)
        else:
            B_tooth = B_g * 1.5
            B_yoke  = B_g * 1.2

        B_tooth = np.clip(B_tooth, 0.1, lam.B_sat)
        B_yoke  = np.clip(B_yoke,  0.1, lam.B_sat)

        # Iron volume from geometry
        if st:
            r_i = m.stator_inner_radius
            r_o = m.stator_outer_radius
            L   = m.stack_length
            V_tooth = m.slots * st.tooth_width * st.slot_profile.depth() * L
            V_yoke  = np.pi * (r_o**2 - (r_i + st.slot_profile.depth())**2) * L
        else:
            V_tooth = m.active_volume * 0.25
            V_yoke  = m.active_volume * 0.15

        rho_fe   = 7650.0
        mass_tooth = V_tooth * rho_fe
        mass_yoke  = V_yoke  * rho_fe

        p_tooth = lam.loss_density(B_tooth, f, harmonics)
        p_yoke  = lam.loss_density(B_yoke,  f, harmonics)

        return mass_tooth * p_tooth + mass_yoke * p_yoke

    def rotor_iron_loss(self, speed_rpm: float) -> float:
        """Rotor iron loss [W] — mainly eddy currents from MMF harmonics."""
        m = self.motor
        # Rotor sees stator slot harmonics at much higher frequency
        f_slot = speed_rpm / 60 * m.slots   # slot harmonic frequency
        B_rotor = self._airgap_B_fundamental() * 0.05   # small harmonic content

        try:
            lam = self.mlib.lamination("M270-35A")
        except Exception:
            return 0.0

        V_rotor  = np.pi * (m.rotor_outer_radius**2 -
                             m.rotor_inner_radius**2) * m.stack_length * 0.5
        mass_rot = V_rotor * 7650.0
        return mass_rot * lam.loss_density(B_rotor, f_slot)

    # ── Mechanical losses ─────────────────────────────────────────────────

    def friction_loss(self, speed_rpm: float) -> float:
        """Bearing friction + windage [W]."""
        m       = self.motor
        omega   = speed_rpm * 2 * np.pi / 60
        r_shaft = m.rotor_inner_radius
        # Empirical: T_friction ≈ 0.001 × P_rated / omega_rated (NEMA)
        T_f     = 0.001 * m.rated_power / (m.rated_speed * 2 * np.pi / 60 + 1e-9)
        return T_f * omega

    def windage_loss(self, speed_rpm: float) -> float:
        """Windage loss [W] — significant above ~6000 rpm."""
        m     = self.motor
        omega = speed_rpm * 2 * np.pi / 60
        r     = m.rotor_outer_radius
        L     = m.stack_length
        rho   = 1.15   # air density kg/m³
        Cf    = 0.01   # friction coefficient (smooth rotor)
        return 0.5 * Cf * rho * omega**3 * r**4 * np.pi * L

    # ── Full loss budget ──────────────────────────────────────────────────

    def loss_budget(self, speed_rpm: float = None, torque_Nm: float = None,
                    temperature: float = None) -> LossBudget:
        """
        Compute complete loss budget at given operating point.

        Uses motor rated values if speed_rpm / torque_Nm not provided.
        """
        m = self.motor
        if speed_rpm is None:
            speed_rpm = m.rated_speed
        if torque_Nm is None:
            torque_Nm = m.rated_torque
        if temperature is not None:
            self.temperature = temperature

        omega = speed_rpm * 2 * np.pi / 60
        P_out = torque_Nm * omega

        P_cu, kac = self.copper_loss(speed_rpm, torque_Nm)
        P_fe_s    = self.stator_iron_loss(speed_rpm)
        P_fe_r    = self.rotor_iron_loss(speed_rpm)
        P_fric    = self.friction_loss(speed_rpm)
        P_wind    = self.windage_loss(speed_rpm)
        P_stray   = (P_cu + P_fe_s) * 0.005   # 0.5% of electrical losses

        P_inv = 0.0
        if self.inverter:
            _, Iq = self._dq_currents(speed_rpm, torque_Nm)
            P_inv = self.inverter.total_loss_W(Iq / np.sqrt(2))

        return LossBudget(
            speed_rpm=speed_rpm,
            torque_Nm=torque_Nm,
            output_power_W=P_out,
            copper_loss_W=P_cu,
            stator_iron_W=P_fe_s,
            rotor_iron_W=P_fe_r,
            friction_W=P_fric,
            windage_W=P_wind,
            stray_W=P_stray,
            inverter_loss_W=P_inv,
            ac_factor=kac,
        )



def cogging_torque_Nm(motor, B_gap_override: float = None) -> dict:
    """
    Cogging torque estimate using the Zhu & Howe (1993) spectral method.

    Sums contributions from the spatial harmonics of the PM field that
    interact with the slot harmonics (LCM orders).

    Method (Zhu & Howe, IEEE Trans. Mag. 1993 / IEE Proc.-B 1994):

        T_cog_pp = (2π/Qs) × (R² L / 2μ₀) × Σ_ν [ B_ν² × |k_ν| ] × 2

    where:
        ν     = 3, 9, 15, ... (odd PM harmonic orders; × pole_pairs gives
                spatial cogging order at LCM, 3×LCM, ...)
        B_ν   = (4 B_gap / π) × sin(ν π α_p / 2) / ν  [νth PM spatial harmonic]
        k_ν   = (2/π) × sin(k π b_s / (2 τ_s Qs))     [slot harmonic factor]
        k     = ν × pole_pairs  [spatial order]

    The LCM cancellation is implicit: only harmonics where
    k = ν × p = LCM / pole_pairs × integer contribute, so machines with high
    LCM naturally produce small cogging.

    Parameters
    ----------
    B_gap_override : float, optional
        If provided, overrides the magnet-circuit B_gap with an FEM-derived
        value (e.g. from airgap flux extraction).

    Returns
    -------
    dict : Tcog_pp_Nm, Tcog_pp_pct, cogging_period_deg, lcm_slots_poles,
           B_gap_used, harmonics (list of per-harmonic contributions)
    """
    from math import gcd

    MU0 = 4e-7 * np.pi
    m   = motor

    r_bore  = m.stator.inner_radius if m.stator else (m.rotor_outer_radius + m.airgap)
    b_s     = m.stator.slot_profile.opening_width() if m.stator else 0.003
    tau_s   = 2 * np.pi * r_bore / m.slots
    Qs      = m.slots
    pp      = m.pole_pairs

    # ── Airgap flux density (magnet circuit) ─────────────────────────────
    Br      = m._get_Br() if hasattr(m, "_get_Br") else 1.2
    rg      = m.rotor_geo
    t_m     = getattr(rg, "magnet_thickness", getattr(m, "magnet_thickness", 0.005))
    alpha_p = getattr(rg, "magnet_width_fraction", getattr(m, "magnet_width_fraction", 0.85))
    mu_r    = getattr(m, "_mu_r", 1.05)

    # Correct magnet-circuit formula: B_gap = Br × t_m / (t_m + μr × g)
    B_gap = B_gap_override if B_gap_override is not None else (
        Br * t_m / (t_m + mu_r * m.airgap + 1e-12)
    )

    lcm_val        = (Qs * m.poles) // gcd(Qs, m.poles)
    cog_period_deg = 360.0 / lcm_val

    # ── Zhu-Howe spectral summation ───────────────────────────────────────
    # Odd PM harmonic orders ν that give spatial cogging orders at multiples
    # of LCM. For typical SPM: ν = 3, 9, 15 (=3, 9, 15 × pole_pairs spatial).
    # Higher harmonics decay rapidly with B_ν ∝ 1/ν.
    T_cog_pp  = 0.0
    harmonics = []
    for nu in range(3, 40, 6):          # ν = 3, 9, 15, 21, ...
        k   = nu * pp                   # spatial order
        B_k = (4 * B_gap / np.pi) * np.sin(nu * np.pi * alpha_p / 2) / nu
        # Slot harmonic factor: Carter-Schwarz slot function at order k
        arg = k * np.pi * b_s / (2 * tau_s * Qs)
        k_s = abs((2 / np.pi) * np.sin(arg)) if arg != 0 else 0.0
        T_k = (2 * np.pi / Qs) * (r_bore**2 * m.stack_length / (2 * MU0)) * B_k**2 * k_s * 2
        T_cog_pp += T_k
        harmonics.append({"nu": nu, "k": k, "B_k": float(B_k), "k_s": float(k_s),
                           "T_k_Nm": float(T_k)})
        if T_k < 1e-5:          # converged
            break

    T_rated = m.rated_torque
    pct     = 100.0 * T_cog_pp / (T_rated + 1e-9)

    return {
        "Tcog_pp_Nm":         float(T_cog_pp),
        "Tcog_pp_pct":        float(pct),
        "cogging_period_deg": float(cog_period_deg),
        "lcm_slots_poles":    int(lcm_val),
        "B_gap_used":         float(B_gap),
        "harmonics":          harmonics,
    }
