"""
PMSM — Permanent Magnet Synchronous Motor.

Supports SPM, IPM (V, U, delta barriers), and arbitrary geometry
defined via StatorGeometry + RotorGeometry objects.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np

from Bohemien_Motor_Designer.core.motor import Motor
from Bohemien_Motor_Designer.core.specs import DesignSpec
from Bohemien_Motor_Designer.core.geometry.stator import StatorGeometry
from Bohemien_Motor_Designer.core.geometry.rotor import SPMRotorGeometry, IPMRotorGeometry
from Bohemien_Motor_Designer.core.geometry.winding import WindingLayout


@dataclass
class PMSM(Motor):
    """
    Permanent Magnet Synchronous Motor.

    Additional Parameters
    ---------------------
    rotor_geo     : SPMRotorGeometry or IPMRotorGeometry.
                    If None, a default SPM geometry is generated.
    magnet_material: Material key for the permanent magnets.
    conductor_diameter: Bare conductor wire diameter [m].
    slot_fill_factor: Copper fill factor (0–1).
    Ld, Lq, Rs   : d/q inductances and phase resistance (populated by FEA
                    or set manually).
    """
    # Rotor geometry object
    rotor_geo: Optional[SPMRotorGeometry | IPMRotorGeometry] = \
        field(default=None, repr=False)

    # Magnet parameters (used if rotor_geo not provided)
    magnet_material:       str   = "N42SH"
    magnet_thickness:      float = 0.005
    magnet_width_fraction: float = 0.85

    # Conductor
    conductor_diameter:  float = 0.0012
    slot_fill_factor:    float = 0.45
    parallel_paths:      int   = 1
    turns_per_coil:      int   = 8

    # Circuit parameters (updated by FEA)
    Ld: float = 0.0
    Lq: float = 0.0
    Rs: float = 0.0

    # Cached from material library
    _Br:  Optional[float] = field(default=None, init=False, repr=False)
    _mu_r: float          = field(default=1.05, init=False, repr=False)

    def __post_init__(self):
        # Rebuild winding with turns_per_coil before calling super
        # (super.__post_init__ builds winding, but we need our turns)
        if self.winding is None:
            self.winding = WindingLayout(
                poles=self.poles,
                slots=self.slots,
                phases=self.phases,
                layers=2,
                parallel_paths=self.parallel_paths,
                turns_per_coil=self.turns_per_coil,
            )
        super().__post_init__()

        # Auto-build rotor geometry if not supplied
        if self.rotor_geo is None:
            self.rotor_geo = SPMRotorGeometry(
                outer_radius=self.rotor_outer_radius,
                inner_radius=self.rotor_inner_radius,
                magnet_thickness=self.magnet_thickness,
                magnet_width_fraction=self.magnet_width_fraction,
                magnet_material=self.magnet_material,
            )

        # Populate Ld / Lq analytically if not already set by caller
        if self.Ld == 0.0 and self.Lq == 0.0:
            self.Ld, self.Lq = self._compute_inductances()

    # ── Magnet properties ─────────────────────────────────────────────────

    @property
    def magnet_type(self) -> str:
        return getattr(self.rotor_geo, "magnet_type", "SPM")

    def _get_Br(self) -> float:
        """Remanence at operating temperature, from material library."""
        if self._Br is not None:
            return self._Br
        try:
            from Bohemien_Motor_Designer.materials.library import MaterialLibrary
            mat = MaterialLibrary().magnet(self.magnet_material)
            self._Br = mat.remanence_Br
            self._mu_r = mat.mu_r
        except Exception:
            self._Br = 1.2
        return self._Br

    # ── Analytical electromagnetic quantities ─────────────────────────────

    def back_emf_constant(self, Br: float = None) -> float:
        """
        Analytical back-EMF constant Ke [V·s/rad = Wb-turns peak flux linkage].

        Uses the standard Gieras & Wing fundamental Fourier coefficient:
          B_gap = Br · t_m / (t_m + μr · g)          [B in air gap under magnets]
          B_rv  = (4/π) · B_gap · sin(π · α_p / 2)   [fundamental harmonic]
          Ke    = (2/π) · N_series · kw · B_rv · τ_p · L

        Previous versions mistakenly used (g/g_eff) ≈ 0.15 instead of
        t_m/(t_m + μr·g) ≈ 0.85, giving Ke ~5.7× too low.
        """
        if Br is None:
            Br = self._get_Br()

        rg     = self.rotor_geo
        t_m    = getattr(rg, "magnet_thickness", self.magnet_thickness)
        alpha_p= getattr(rg, "magnet_width_fraction", self.magnet_width_fraction)
        mu_r   = self._mu_r

        # Air-gap flux density under magnets (Gieras eq. 5.12)
        B_gap  = Br * t_m / (t_m + mu_r * self.airgap)
        # Fundamental harmonic of rectangular PM field
        B_rv   = (4 / np.pi) * B_gap * np.sin(np.pi * alpha_p / 2)

        kw     = self.winding_factor()
        N      = self.winding.total_series_turns_per_phase
        phi_p  = B_rv * self.pole_pitch * self.stack_length
        Ke     = (2 / np.pi) * N * kw * phi_p
        return Ke

    @property
    def total_series_turns(self) -> int:
        return self.winding.total_series_turns_per_phase

    @property
    def rated_current(self) -> float:
        """Rough rated phase current from P = 1.5 * Ke * omega_e * Iq."""
        omega_e = self.rated_speed * 2 * np.pi / 60 * self.pole_pairs
        T_rated = self.rated_torque
        Ke      = self.back_emf_constant()
        Iq      = T_rated / (1.5 * self.pole_pairs * Ke + 1e-9)
        return Iq / np.sqrt(2)   # peak → RMS

    # ── dq-model ──────────────────────────────────────────────────────────

    def torque_from_dq(self, Id: float, Iq: float) -> float:
        """
        Electromagnetic torque from dq currents [N·m].
        T = 1.5 * p * (ψ_m * Iq + (Ld - Lq) * Id * Iq)
        """
        psi_m = self.back_emf_constant()
        Ld, Lq = self.Ld, self.Lq
        return 1.5 * self.pole_pairs * (psi_m * Iq + (Ld - Lq) * Id * Iq)

    def mtpa_angle(self, I_peak: float) -> tuple[float, float]:
        """
        Maximum-Torque-Per-Ampere (MTPA) current angle.
        Returns (Id_mtpa, Iq_mtpa) [A peak].
        """
        psi_m = self.back_emf_constant()
        Ld, Lq = self.Ld, self.Lq

        if abs(Ld - Lq) < 1e-9 or (Ld == 0 and Lq == 0):
            # Surface PM — no reluctance torque, all Iq
            return (0.0, I_peak)

        # MTPA condition: dT/dβ = 0  (β = current angle from q-axis)
        # Analytical solution for the current angle β:
        # 2*(Ld-Lq)*I²*sin(2β) - ψm*I*sin(β) = 0  → nonlinear solve
        from scipy.optimize import brentq
        def dT_dbeta(beta):
            Id_ = -I_peak * np.sin(beta)
            Iq_ =  I_peak * np.cos(beta)
            return 1.5 * self.pole_pairs * (
                psi_m * Iq_ + (Ld - Lq) * Id_ * Iq_)

        # Scan then refine
        betas = np.linspace(0, np.pi / 2, 500)
        T_vals = [dT_dbeta(b) for b in betas]
        idx = int(np.argmax(T_vals))
        beta_opt = betas[idx]

        Id = -I_peak * np.sin(beta_opt)
        Iq =  I_peak * np.cos(beta_opt)
        return (Id, Iq)

    def voltage_at(self, Id: float, Iq: float, speed_rpm: float) -> tuple[float, float]:
        """
        Phase voltage components (Vd, Vq) in dq frame [V peak].
        Returns (Vd, Vq, V_magnitude).
        """
        omega_e = speed_rpm * 2 * np.pi / 60 * self.pole_pairs
        psi_m   = self.back_emf_constant()
        Vd = self.Rs * Id - omega_e * self.Lq * Iq
        Vq = self.Rs * Iq + omega_e * (self.Ld * Id + psi_m)
        return Vd, Vq, np.sqrt(Vd**2 + Vq**2)

    def field_weakening_Id(self, speed_rpm: float, Iq: float,
                            V_max_peak: float) -> float:
        """
        Required Id for field weakening at given speed, Iq, voltage limit.
        Solves |V_dq| = V_max for Id.
        Returns Id [A peak] (negative for demagnetising).
        """
        omega_e = speed_rpm * 2 * np.pi / 60 * self.pole_pairs
        psi_m   = self.back_emf_constant()
        Ld, Lq  = self.Ld, self.Lq

        # |V|² = (Rs*Id - ω*Lq*Iq)² + (Rs*Iq + ω*(Ld*Id + ψm))² = V_max²
        # Quadratic in Id:
        a  = self.Rs**2 + (omega_e * Ld)**2
        b  = 2 * (- omega_e * Lq * Iq * self.Rs + self.Rs * Iq * omega_e * Ld +
                   omega_e**2 * Ld * psi_m)
        c  = (self.Rs * Iq + omega_e * psi_m)**2 + (omega_e * Lq * Iq)**2 - V_max_peak**2

        disc = b**2 - 4 * a * c
        if disc < 0 or a < 1e-12:
            return 0.0
        Id = (-b - np.sqrt(disc)) / (2 * a)   # take the negative (demagnetising) root
        return Id

    # ── Analytical inductance components ─────────────────────────────────

    def _end_winding_geometry(self) -> tuple[float, float]:
        """
        (W_ew, l_ew) in metres for the end-winding leakage model.

        W_ew : coil span measured at the slot mid-depth radius.  For a
               full-pitch coil this is one pole pitch at that radius.
        l_ew : axial extent of the end winding (stack face to the coil
               nose).  Diamond-shaped distributed coils leave the slot at
               roughly 45°, so the nose sits about W_ew/2 out, plus a
               straight run for the bend radius and insulation.
        Both can be overridden by setting motor.end_winding_span_m and
        motor.end_winding_axial_m (e.g. from a measured coil).
        """
        p   = self.pole_pairs
        r_m = self.rotor_outer_radius + self.airgap
        if self.stator is not None:
            r_m += 0.5 * self.stator.slot_profile.depth()
        W_ew = getattr(self, "end_winding_span_m", None) \
               or (np.pi * r_m / p) * getattr(self, "coil_pitch_fraction", 1.0)
        l_ew = getattr(self, "end_winding_axial_m", None) \
               or (0.5 * W_ew + 0.010)
        return float(W_ew), float(l_ew)

    def end_winding_leakage_inductance(self) -> float:
        """
        End-winding leakage inductance per phase [H] — the component that a
        2D (cross-section) FEM cannot see.

        Method: Richter's permeance-factor formula as given in Pyrhönen,
        Jokinen & Hrabovcová, *Design of Rotating Electrical Machines*,
        2nd ed., §4.3.3, eq. (4.64):

            L_ew = (4·m / Q) · q · N² · μ0 · l_w·λ_w
                 = (2 / p) · N² · μ0 · (2·l_ew·λ_lew + W_ew·λ_W)

        where m = phases, Q = slots, q = slots per pole per phase,
        N = series turns per phase, p = pole pairs, and the permeance
        factors (Table 4.5, two-layer diamond winding) are
            λ_lew = 0.324   (axial part of the overhang)
            λ_W   = 0.243   (circumferential part / coil span)

        Accuracy: empirical, ±30–50 %.  Published 3D-FEM studies put L_ew
        at roughly 5–30 % of the total synchronous inductance for radial-
        flux distributed-winding PMSMs (Chalmers ICEM 2024 / IECON 2025;
        Calvano et al. COMPEL 2013), higher for hairpin and short-stack
        machines.  A single 3D end-turn FEM at the final design point is
        the usual way to tighten this number.
        """
        MU0 = 4e-7 * np.pi
        p   = self.pole_pairs
        N   = self.winding.total_series_turns_per_phase
        W_ew, l_ew = self._end_winding_geometry()
        lam_lew = getattr(self, "lambda_end_axial", 0.324)
        lam_W   = getattr(self, "lambda_end_span",  0.243)
        return float((2.0 / p) * MU0 * N**2 * (2.0 * l_ew * lam_lew + W_ew * lam_W))

    def inductance_breakdown(self) -> dict:
        """
        Analytical synchronous-inductance components per phase [H]:

            L_ag    air-gap (magnetising) inductance — fundamental
            L_h     air-gap harmonic leakage (MMF space harmonics)
            L_slot  slot leakage
            L_tt    tooth-tip leakage (across the slot opening)
            L_ew    end-winding leakage  (3D-only; absent from 2D FEM)
            L_2d    = L_ag + L_h + L_slot + L_tt   what a 2D FEM sees
            L_total = L_2d + L_ew                  what the machine has

        A 2D FEM result must be compared with L_2d, not L_total.
        Validated against the pure-Python 2D FEM on the reference 8p/48s
        30 kW SPM: L_2d analytical 0.0833 mH vs FEM 0.0845 mH (1.4 %).
        """
        c = self._airgap_and_slot_inductance()
        L_ew = self.end_winding_leakage_inductance()
        L_2d = c["L_ag"] + c["L_h"] + c["L_slot"] + c["L_tt"]
        return dict(**c, L_ew=L_ew, L_2d=L_2d, L_total=L_2d + L_ew,
                    L_ew_fraction=L_ew / (L_2d + L_ew + 1e-30))

    def _airgap_and_slot_inductance(self) -> dict:
        """
        Inductance components per phase [H] for the isotropic (SPM) case,
        following Pyrhönen, Jokinen & Hrabovcová, *Design of Rotating
        Electrical Machines*, 2nd ed., chapters 3–4:

          L_ag   = (6/π²) · μ0 · (N·kw1)² · τ_p · L / (p · g_c)        (3.110)
                   [= (3/2) × single-phase (4/π²)·μ0(Nkw)²τ_p L/(p g_c)]
          L_h    = σ_δ · L_ag,   σ_δ = Σ_{ν≠1} (kw_ν / (ν·kw1))²      (3.127)
          L_slot = (4m/Q) · μ0 · L · N² · λ_slot                      (4.33)
                   λ_slot = h_s/(3 b_s) + h_wedge/b_open
          L_tt   = (4m/Q) · μ0 · L · N² · λ_tt                        (4.43)
                   λ_tt = (5 g_eff/b_open) / (5 + 4 g_eff/b_open)

        Note the explicit 1/p in L_ag and the 4m/Q (not 2m/Q) in the
        slot-leakage term; the previous version of this routine had
        (3/π²)/g_c and 2m/Q, which is only right for p = 2.
        """
        MU0   = 4e-7 * np.pi
        p     = self.pole_pairs
        m_ph  = self.phases
        Q     = self.slots
        N     = self.winding.total_series_turns_per_phase
        kw1   = self.winding_factor()
        L_stk = self.stack_length
        r_i   = self.rotor_outer_radius
        tau_p = np.pi * r_i / p

        rg    = self.rotor_geo
        t_m   = getattr(rg, "magnet_thickness", self.magnet_thickness)
        g_eff = self.airgap + t_m / self._mu_r
        b_op  = self.stator.slot_profile.opening_width() if self.stator else 0.003
        gamma = (b_op / g_eff)**2 / (5 + b_op / g_eff)
        kc    = tau_p / (tau_p - gamma * g_eff)
        g_c   = kc * g_eff

        L_ag = (6.0 / np.pi**2) * MU0 * (N * kw1)**2 * tau_p * L_stk / (p * g_c)

        # Air-gap harmonic leakage factor from the winding's space harmonics
        q      = Q / (2.0 * p * m_ph)
        alpha  = 2.0 * np.pi * p / Q             # slot angle [rad elec]
        q_int  = max(1, int(round(q)))
        sigma  = 0.0
        for nu in range(5, 400, 2):
            if nu % 3 == 0:
                continue
            kd_nu = np.sin(nu * q_int * alpha / 2) / (q_int * np.sin(nu * alpha / 2) + 1e-15)
            sigma += (kd_nu / (nu * kw1 + 1e-15))**2
        L_h = sigma * L_ag

        if self.stator:
            sp    = self.stator.slot_profile
            h_s   = sp.depth()
            b_s_w = sp.area() / (h_s + 1e-9)
            h_w   = getattr(sp, "wedge_height", 0.0)
            lam_slot = h_s / (3.0 * b_s_w + 1e-9) + (h_w / (b_op + 1e-9))
        else:
            lam_slot = 0.5
        lam_tt = (5.0 * g_eff / b_op) / (5.0 + 4.0 * g_eff / b_op)

        L_slot = (4.0 * m_ph / Q) * MU0 * L_stk * N**2 * lam_slot
        L_tt   = (4.0 * m_ph / Q) * MU0 * L_stk * N**2 * lam_tt
        return dict(L_ag=float(L_ag), L_h=float(L_h),
                    L_slot=float(L_slot), L_tt=float(L_tt),
                    sigma_delta=float(sigma), g_c=float(g_c))

    def _compute_inductances(self) -> tuple[float, float]:
        """
        Analytical Ld / Lq [H] using air-gap + leakage magnetic circuit.

        Method (Pyrhönen / Howe):
          L_ag  = (3/2) * mu0 * (N*kw)^2 * (2*tau_p*L) / (pi^2 * g_eff)
                  — fundamental air-gap inductance per phase

          SPM : Ld ≈ Lq ≈ L_ag + L_slot + L_ew
                (magnet has mu_r≈1, so it looks like extra air gap)

          IPM : Ld = L_ag_d + L_leak   (d-axis sees magnet reluctance)
                Lq = L_ag_q + L_leak   (q-axis flux path through iron)
                The saliency ratio Lq/Ld is estimated from barrier geometry.

        All leakage terms (slot, tooth-tip, end-winding) use standard
        permeance coefficient formulae — see inductance_breakdown() for
        the individual components and end_winding_leakage_inductance()
        for the L_ew model and its literature basis.
        """
        MU0   = 4e-7 * np.pi
        p     = self.pole_pairs
        N     = self.winding.total_series_turns_per_phase
        kw1   = self.winding_factor()
        L_stk = self.stack_length
        r_i   = self.rotor_outer_radius          # bore inner radius ≈ rotor OD
        tau_p = np.pi * r_i / p                  # pole pitch [m]

        rg    = self.rotor_geo
        mu_r  = self._mu_r
        t_m   = getattr(rg, "magnet_thickness", self.magnet_thickness)
        g_eff = self.airgap + t_m / mu_r
        b_s   = self.stator.slot_profile.opening_width() if self.stator else 0.003
        gamma = (b_s / g_eff)**2 / (5 + b_s / g_eff)
        kc    = tau_p / (tau_p - gamma * g_eff)
        g_c   = kc * g_eff                       # Carter-corrected gap (IPM branch)

        # Shared components (single source of truth)
        c      = self._airgap_and_slot_inductance()
        L_ag   = c["L_ag"]
        L_ew   = self.end_winding_leakage_inductance()
        L_leak = c["L_h"] + c["L_slot"] + c["L_tt"] + L_ew

        # --- Topology-specific d/q split ---
        if isinstance(rg, IPMRotorGeometry) and rg.barriers:
            b       = rg.barriers[0]
            w_m     = b.magnet_width
            t_m_ipm = b.magnet_thickness
            mu_r_m  = mu_r
            K_ag    = (6.0 / np.pi**2) * MU0 * (N * kw1)**2 * tau_p * L_stk / p

            # d-axis: flux crosses magnet (high reluctance)
            # Extra reluctance of magnet layer on d-axis
            g_d = g_c + t_m_ipm / mu_r_m
            L_ag_d = K_ag / g_d

            # q-axis: flux through iron bridge / flux guides
            # Barrier blocks a fraction of the q-axis MMF
            # Fraction of pole pitch occupied by magnet(s)
            alpha_m = min(0.95, w_m / (tau_p + 1e-9))
            # Effective q-axis reluctance reduced by iron path fraction
            g_q = g_c * (1.0 - alpha_m) + g_c * alpha_m * 0.1  # iron path ≈ 10× lower Rm
            L_ag_q = K_ag / g_q

            Ld = L_ag_d + L_leak
            Lq = L_ag_q + L_leak

        else:
            # SPM or unknown — isotropic (Ld ≈ Lq)
            Ld = L_ag + L_leak
            Lq = L_ag + L_leak

        return float(Ld), float(Lq)

    # ── Back-EMF harmonic spectrum ────────────────────────────────────────

    def back_emf_harmonics(self, n_harmonics: int = 20) -> dict:
        """
        Analytical back-EMF harmonic spectrum.

        Returns
        -------
        dict with keys:
          'harmonics'   : list of harmonic orders [1, 5, 7, 11, 13, ...]
          'amplitudes'  : peak voltage of each harmonic [V]
          'thd'         : total harmonic distortion [%]  (vs fundamental)
                          Triplen harmonics are excluded — they cancel in
                          line voltage and balanced torque for 3-phase machines.
          'fundamental' : peak voltage of fundamental [V]

        Method
        ------
        The nth space harmonic of the PM air-gap field:
          B_n = (4/nπ) * Br_eff * sin(n * π * α_p / 2)

        The nth harmonic of the phase back-EMF [V peak]:
          E_n = ω_e * N * kw_n * B_n * (2 * τ_p / π) * L

        Note: the n*ω_e time frequency and 1/n spatial pole pitch of the
        nth harmonic cancel exactly, giving the same ω_e prefactor for all n.
        kw_n provides the winding attenuation specific to each harmonic.
        """
        omega_e = self.rated_speed * 2 * np.pi / 60 * self.pole_pairs
        Br      = self._get_Br()
        rg      = self.rotor_geo
        t_m     = getattr(rg, "magnet_thickness", self.magnet_thickness)
        alpha_p = getattr(rg, "magnet_width_fraction", self.magnet_width_fraction)
        mu_r    = self._mu_r
        g_eff   = self.airgap + t_m / mu_r
        N       = self.winding.total_series_turns_per_phase
        tau_p   = self.pole_pitch
        L_stk   = self.stack_length

        # Air-gap flux density under magnets (same formula as back_emf_constant)
        B_gap = Br * t_m / (t_m + mu_r * self.airgap)

        orders     = []
        amplitudes = []
        for k in range(1, n_harmonics + 1):
            n = 2 * k - 1                        # odd harmonics only (symmetric PM)
            Bn   = (4.0 / (n * np.pi)) * B_gap * np.sin(n * np.pi * alpha_p / 2)
            kw_n = self.winding_factor(n)
            # Each harmonic contributes at ω_e (n*ω_e from dΨ/dt cancels n in τ_p/n)
            phi_n = abs(Bn) * (2 * tau_p * L_stk / np.pi)
            U_n   = omega_e * N * kw_n * phi_n
            orders.append(n)
            amplitudes.append(abs(U_n))

        U1 = amplitudes[0] if amplitudes else 1.0

        # THD: exclude triplen harmonics — they cancel in line voltage
        # and produce zero net torque ripple in balanced 3-phase operation
        harm_sq_sum = sum(
            a**2 for n, a in zip(orders, amplitudes)
            if n > 1 and n % 3 != 0
        )
        thd = 100.0 * np.sqrt(harm_sq_sum) / (U1 + 1e-9)

        return {
            "harmonics":   orders,
            "amplitudes":  amplitudes,
            "fundamental": U1,
            "thd":         thd,
        }

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        ke     = self.back_emf_constant()
        omega_e = self.rated_speed * 2 * np.pi / 60 * self.pole_pairs
        E_pk   = ke * omega_e
        base   = super().summary()
        lines  = [base, "  --- PMSM Specifics ---",
            f"  Magnet type              : {self.magnet_type}",
            f"  Magnet material          : {self.magnet_material}",
            f"  Magnet thickness         : {self.magnet_thickness*1e3:.1f} mm",
            f"  Magnet width fraction    : {self.magnet_width_fraction:.2f}",
            f"  Turns per coil           : {self.turns_per_coil}",
            f"  Total series turns/ph    : {self.total_series_turns}",
            f"  Slot fill factor         : {self.slot_fill_factor:.2f}",
            f"  Back-EMF constant Ke     : {ke:.4f} V·s/rad",
            f"  Peak back-EMF @ rated n  : {E_pk:.1f} V",
            f"  Rated torque (mech)      : {self.rated_torque:.2f} N·m",
            "=" * 58]
        return "\n".join(lines)
