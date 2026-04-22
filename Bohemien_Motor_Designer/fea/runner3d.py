"""
runner3d.py — High-level 3D FEM workflow for PMSM / SynRel motors.

Coordinates mesh building, static solve, post-processing, and
Ke/Ld/Lq extraction from the 3D Nédélec edge-element solver.

Usage
-----
    from Bohemien_Motor_Designer.fea.runner3d import FEMRunner3D

    runner = FEMRunner3D(motor, n_angular_per_slot=4, n_axial=6)
    runner.build_mesh()
    result = runner.run_static(progress_cb=print)
    print(result['Ke_3d_Wb'], result['Ld_mH'], result['Lq_mH'])

Mesh-size guidance
------------------
  n_angular_per_slot=4, n_axial=4   →  ~75k DOF,  ~40s per solve,  Ke ~±25%
  n_angular_per_slot=4, n_axial=6   →  ~100k DOF, ~60s per solve,  Ke ~±20%
  n_angular_per_slot=6, n_axial=6   →  ~200k DOF, ~150s per solve, Ke ~±15%
  n_angular_per_slot=8, n_axial=8   →  ~500k DOF, ~600s per solve, Ke ~±8%

Ke comparison to 2D
-------------------
  The 3D solver captures end-winding flux leakage and axial fringing.
  At mid-stack the field matches the 2D solver within mesh discretisation error.
  Ke_3d is expected to be 5–20% below Ke_2d on typical PM motors because the
  3D B-field has axial components not present in the 2D cross-section model.
"""
from __future__ import annotations
import numpy as np
import time
from typing import Optional, Callable


class FEMRunner3D:
    """
    3D edge-element FEM runner for PMSM and SynRel motors.

    Parameters
    ----------
    motor                : PMSM or SynRelMotor instance
    n_angular_per_slot   : angular mesh density (4 = fast/coarse, 8 = accurate)
    n_axial              : axial layers in active stack (4–10)
    n_end_winding        : axial layers in each end-winding extension (1–3)
    n_radial_airgap      : radial layers across the airgap (2–6)
    """

    def __init__(self,
                 motor,
                 n_angular_per_slot: int = 4,
                 n_axial:            int = 6,
                 n_end_winding:      int = 2,
                 n_radial_airgap:    int = 3):
        self.motor              = motor
        self.n_ang              = n_angular_per_slot
        self.n_axial            = n_axial
        self.n_ew               = n_end_winding
        self.n_ag               = n_radial_airgap
        self._mesh              = None
        self._geom              = None
        self._last_static: Optional[dict]   = None
        self._last_loaded: Optional[dict]   = None

    # ── Mesh ──────────────────────────────────────────────────────────────────

    def build_mesh(self, progress_cb: Optional[Callable] = None) -> None:
        """Build the 3D tetrahedral mesh (call once, reuse for multiple solves)."""
        from Bohemien_Motor_Designer.fea.mesh3d import build_motor_mesh_3d

        def _log(msg, f=None):
            if progress_cb:
                progress_cb(msg, f)

        _log("Building 3D tetrahedral mesh...", 0.0)
        t0 = time.time()
        self._mesh = build_motor_mesh_3d(
            self.motor,
            n_radial_airgap=self.n_ag,
            n_angular_per_slot=self.n_ang,
            n_axial=self.n_axial,
            n_end_winding=self.n_ew,
        )
        dt = time.time() - t0
        _log(
            f"3D mesh: {self._mesh.n_nodes:,} nodes  "
            f"{self._mesh.n_tets:,} tets  "
            f"{self._mesh.n_edges:,} edge DOFs  ({dt:.1f}s)",
            0.05,
        )

    # ── Static (no-load) solve → Ke, Ld, Lq ──────────────────────────────────

    def run_static(self,
                   rotor_angle: float = 0.0,
                   progress_cb: Optional[Callable] = None) -> dict:
        """
        Single magnetostatic solve at the given rotor angle with Id=Iq=0.

        Extracts:
          - Ke_3d   : peak flux linkage (proxy for back-EMF constant) [Wb]
          - theta_w : winding-axis Park offset [deg electrical]
          - Ld, Lq  : incremental inductances via Park perturbation [mH]
          - B_gap   : mean |B| in airgap at mid-stack [T]

        Returns dict with all results plus solve timing.
        """
        from Bohemien_Motor_Designer.fea.solver3d import (
            solve_magnetostatic_3d, _precompute_geometry,
            compute_flux_linkage_3d, compute_B_field_3d,
        )

        def _log(msg, f=None):
            if progress_cb:
                progress_cb(msg, f)

        if self._mesh is None:
            self.build_mesh(progress_cb)

        motor = self.motor
        pp    = motor.pole_pairs

        t0_total = time.time()

        # ── Step 1: Calibration solve (no-load) ───────────────────────────
        _log("3D FEM: calibration solve (no-load)...", 0.05)
        t0 = time.time()
        A_cal = solve_magnetostatic_3d(
            self._mesh, motor,
            rotor_angle=rotor_angle, Id=0.0, Iq=0.0,
            progress_cb=lambda m, f: _log(f"  {m}", 0.05 + 0.35 * f),
        )
        dt_cal = time.time() - t0
        _log(f"  Calibration solve: {dt_cal:.1f}s", 0.40)

        if self._geom is None:
            self._geom = _precompute_geometry(self._mesh)

        psi_cal = compute_flux_linkage_3d(self._mesh, A_cal, motor, self._geom)

        # Park offset and Ke from calibration
        pA, pB, pC = psi_cal['psi_A'], psi_cal['psi_B'], psi_cal['psi_C']
        th_e = 0.0
        c = 2.0 / 3.0
        pd_raw = c * (pA * np.cos(th_e)
                    + pB * np.cos(th_e - 2*np.pi/3)
                    + pC * np.cos(th_e + 2*np.pi/3))
        pq_raw = c * (-pA * np.sin(th_e)
                     - pB * np.sin(th_e - 2*np.pi/3)
                     - pC * np.sin(th_e + 2*np.pi/3))
        theta_w   = float(np.arctan2(-pq_raw, pd_raw))
        Ke_3d     = float(np.sqrt(pd_raw**2 + pq_raw**2))

        _log(f"  Ke_3d={Ke_3d:.4f}Wb  θ_w={np.degrees(theta_w):.1f}°e  "
             f"(analytical Ke={motor.back_emf_constant():.4f}Wb)", 0.42)

        # ── Step 2: B-field quality ────────────────────────────────────────
        _log("3D FEM: computing B-field...", 0.44)
        B = compute_B_field_3d(self._mesh, A_cal, self._geom)

        poles = motor.poles; ag_tag = 3 + poles + 1
        ag_idx = self._mesh.groups.get(ag_tag, np.array([], dtype=np.int32))
        nodes  = self._mesh.nodes; tets = self._mesh.tets
        L_stack = motor.stack_length

        if len(ag_idx) > 0:
            cz_ag = nodes[tets[ag_idx], 2].mean(axis=1)
            mid   = np.abs(cz_ag - L_stack / 2) < 0.02
            if mid.sum() == 0:
                mid = np.ones(len(ag_idx), dtype=bool)
            Bmag  = np.sqrt((B[ag_idx[mid]]**2).sum(axis=1))
            B_gap = float(Bmag.mean())
            B_gap_max = float(Bmag.max())
        else:
            B_gap = 0.0; B_gap_max = 0.0

        _log(f"  B_gap mid-stack: mean={B_gap:.3f}T  max={B_gap_max:.3f}T", 0.48)

        # ── Step 3: Ld/Lq perturbation ────────────────────────────────────
        # Use ZERO background current for both perturbation solves.
        # Rationale: with a large Iq background (e.g. 170 A rated current),
        # even a 1% Park-projection error contaminates the tiny dI signal
        # (e.g. 8 A) by a factor of 20x, making Ld blow up by 400x.
        # Perturbing from (Id=0, Iq=0) avoids this entirely.
        # The no-load solve (calibration) already gives psi_d0 = Ke_3d.
        # Incremental Ld at the PM operating point is what matters for
        # field weakening and saliency — this is the physically correct value.
        _log("3D FEM: Ld/Lq perturbation (2 solves, zero-background)...", 0.50)

        # Choose dI to give ~5% of Ke as flux change: dI = 0.05*Ke / Ld_est
        # Use analytical Ld as the estimate to size dI appropriately.
        Ld_est = max(getattr(motor, 'Ld', 1e-4), 1e-5)
        dI     = max(0.05 * Ke_3d / Ld_est, 1.0)
        # Cap at rated current to avoid nonlinear saturation
        I_rated_est = motor.rated_power / (3 * 200 + 1)  # rough peak current
        dI = min(dI, I_rated_est)
        th_e_pert = rotor_angle * pp - theta_w   # d-axis aligned frame

        def _psi_dq_zero(Id_, Iq_):
            """Flux linkage dq components with zero background current."""
            A_ = solve_magnetostatic_3d(
                self._mesh, motor,
                rotor_angle=rotor_angle, Id=Id_, Iq=Iq_,
                electrical_angle=th_e_pert,
            )
            p_ = compute_flux_linkage_3d(self._mesh, A_, motor, self._geom)
            pA_, pB_, pC_ = p_['psi_A'], p_['psi_B'], p_['psi_C']
            c_ = 2.0 / 3.0
            pd = c_ * (pA_*np.cos(th_e_pert) + pB_*np.cos(th_e_pert-2*np.pi/3)
                      + pC_*np.cos(th_e_pert+2*np.pi/3))
            pq = c_ * (-pA_*np.sin(th_e_pert) - pB_*np.sin(th_e_pert-2*np.pi/3)
                      - pC_*np.sin(th_e_pert+2*np.pi/3))
            return float(pd), float(pq)

        # No-load psi_d0 = Ke_3d (already computed from calibration solve);
        # reuse it directly for the d-axis baseline — saves one solve.
        pd0 = Ke_3d   # aligned d-axis, no current → psi_d = Ke
        pq0 = 0.0     # no q-axis flux at no load in aligned frame

        pd_d, _  = _psi_dq_zero(dI,  0.0)   # d-axis perturbation
        _,  pq_q = _psi_dq_zero(0.0, dI)    # q-axis perturbation

        Ld = (pd_d - pd0) / dI
        Lq = (pq_q - pq0) / dI

        # Clamp to physically sensible range (avoids sign flips from noise)
        Ld = max(Ld, 1e-6)
        Lq = max(Lq, 1e-6)

        dt_total = time.time() - t0_total
        _log(
            f"3D static done: Ke={Ke_3d:.4f}Wb  Ld={Ld*1e3:.2f}mH  Lq={Lq*1e3:.2f}mH  "
            f"[{dt_total:.0f}s]",
            1.0,
        )

        result = dict(
            # Ke
            Ke_3d_Wb     = Ke_3d,
            Ke_anal_Wb   = float(motor.back_emf_constant()),
            theta_w_deg  = float(np.degrees(theta_w)),
            # Inductances
            Ld_H         = float(Ld),
            Lq_H         = float(Lq),
            Ld_mH        = float(Ld * 1e3),
            Lq_mH        = float(Lq * 1e3),
            Ld_anal_mH   = float(motor.Ld * 1e3),
            Lq_anal_mH   = float(motor.Lq * 1e3),
            # B-field
            B_gap_mean_T = B_gap,
            B_gap_max_T  = B_gap_max,
            B_gap_anal_T = float(motor.back_emf_constant() * np.pi /
                                 (2 * motor.winding.total_series_turns_per_phase *
                                  motor.winding_factor() *
                                  motor.pole_pitch * motor.stack_length + 1e-9)),
            # Mesh info
            n_nodes      = self._mesh.n_nodes,
            n_tets       = self._mesh.n_tets,
            n_edges      = self._mesh.n_edges,
            # Timing
            solve_time_s = float(dt_total),
            method       = "3d_nedelec",
        )
        self._last_static = result
        return result

    # ── Mesh summary ──────────────────────────────────────────────────────────

    def mesh_summary(self) -> str:
        if self._mesh is None:
            return "Mesh not built."
        from Bohemien_Motor_Designer.fea.mesh3d import mesh_report_3d
        return mesh_report_3d(self._mesh)
