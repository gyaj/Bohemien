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

    # ── Analytical inductance computation ────────────────────────────────

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
        permeance coefficient formulae.
        """
        MU0   = 4e-7 * np.pi
        p     = self.pole_pairs
        N     = self.winding.total_series_turns_per_phase
        kw1   = self.winding_factor()
        L_stk = self.stack_length
        r_i   = self.rotor_outer_radius          # bore inner radius ≈ rotor OD
        tau_p = np.pi * r_i / p                  # pole pitch [m]

        # --- Effective air gap (Carter + magnet layer) ---
        rg    = self.rotor_geo
        t_m   = getattr(rg, "magnet_thickness", self.magnet_thickness)
        mu_r  = self._mu_r
        g_eff = self.airgap + t_m / mu_r         # magnetically effective gap [m]

        # Carter factor for slot opening
        b_s   = self.stator.slot_profile.opening_width() if self.stator else 0.003
        gamma = (b_s / g_eff)**2 / (5 + b_s / g_eff)
        kc    = tau_p / (tau_p - gamma * g_eff)
        g_c   = kc * g_eff                       # Carter-corrected gap

        # --- Fundamental air-gap inductance ---
        # L_ag = (3/pi^2) * mu0 * (N*kw)^2 * tau_p * L / g_c
        L_ag = (3.0 / np.pi**2) * MU0 * (N * kw1)**2 * tau_p * L_stk / g_c

        # --- Slot leakage permeance coefficient ---
        # Lambda_slot ≈ h_slot/(3*b_slot) + h_wedge/b_wedge  (parallel-tooth approx)
        if self.stator:
            sp    = self.stator.slot_profile
            h_s   = sp.depth()
            b_s_w = sp.area() / (h_s + 1e-9)    # mean slot width
            b_op  = sp.opening_width()
            h_w   = getattr(sp, "wedge_height", 0.0)
            lam_slot = h_s / (3.0 * b_s_w + 1e-9) + (h_w / (b_op + 1e-9))
        else:
            lam_slot = 0.5                        # generic default

        q     = self.slots / (self.poles * self.phases)   # slots/pole/phase
        L_slot = 2 * MU0 * L_stk * N**2 * lam_slot / (self.slots / self.phases)

        # --- End-winding leakage (Richter formula, simplified) ---
        # L_ew ≈ 0.3 * mu0 * N^2 * tau_p  (per phase)
        L_ew  = 0.3 * MU0 * N**2 * tau_p / (self.poles / 2)

        L_leak = L_slot + L_ew

        # --- Topology-specific d/q split ---
        if isinstance(rg, IPMRotorGeometry) and rg.barriers:
            b       = rg.barriers[0]
            w_m     = b.magnet_width
            t_m_ipm = b.magnet_thickness
            mu_r_m  = mu_r

            # d-axis: flux crosses magnet (high reluctance)
            # Extra reluctance of magnet layer on d-axis
            g_d = g_c + t_m_ipm / mu_r_m
            L_ag_d = (3.0 / np.pi**2) * MU0 * (N * kw1)**2 * tau_p * L_stk / g_d

            # q-axis: flux through iron bridge / flux guides
            # Barrier blocks a fraction of the q-axis MMF
            # Fraction of pole pitch occupied by magnet(s)
            alpha_m = min(0.95, w_m / (tau_p + 1e-9))
            # Effective q-axis reluctance reduced by iron path fraction
            g_q = g_c * (1.0 - alpha_m) + g_c * alpha_m * 0.1  # iron path ≈ 10× lower Rm
            L_ag_q = (3.0 / np.pi**2) * MU0 * (N * kw1)**2 * tau_p * L_stk / g_q

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
