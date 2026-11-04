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

What this solver is — and is not
--------------------------------
  By default (n_end_winding = 0) the 3D model is a pure axial extrusion
  of the 2D cross-section with A_t = 0 on both stack faces.  It is a
  genuinely 3D Nédélec edge-element solve and serves as an independent
  check on the 2D solver: Ke_3d and Ld_3d must match Ke_2d and Ld_2d to
  within mesh discretisation error (they do, on the reference motor).

  It does NOT model the end turns, so it cannot see end-winding leakage.
  L_ew is added analytically (PMSM.end_winding_leakage_inductance,
  Pyrhönen/Richter) to both 2D and 3D results:   L_total = L_3d + L_ew.

  n_end_winding > 0 enables an experimental 2.5-D model in which the
  slot conductors are continued straight through an air end region to
  the axial boundary.  That captures axial fringing but adds the
  inductance of straight conductors in air, which is NOT end-turn
  leakage; treat those numbers as indicative only.  Proper end-turn
  geometry (circumferential arcs) is on the roadmap.

  The d-axis inductance is cross-checked internally by the gauge-
  invariant stored-energy method (result['Ld_energy_check'] ≈ 1.00).

Mesh-size guidance (reference 8p/48s motor, pure extrusion)
-----------------------------------------------------------
  n_angular_per_slot=4, n_axial=2, n_radial_airgap=3  →   55k edges,  ~1 s/solve,  Ke ≈ 0.70×
  n_angular_per_slot=4, n_axial=4, n_radial_airgap=4  →  ~130k edges,  ~10 s/solve
  n_angular_per_slot=8, n_axial=4, n_radial_airgap=8  →  ~500k edges,  minutes/solve, Ke → 1.00×
  Convergence is governed mainly by the radial airgap layers (as in 2D:
  n_radial_airgap = 8 brings Ke to within 0.1 % of analytical).
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
    n_axial              : axial layers in active stack (2–8)
    n_end_winding        : axial layers in each air end region.
                           0 (default) = pure extrusion, validated against 2D.
                           >0 = experimental straight-conductor end model.
    n_radial_airgap      : radial layers across the airgap (3–8; this is the
                           main convergence knob — 8 gives Ke within 0.1 %)
    """

    def __init__(self,
                 motor,
                 n_angular_per_slot: int = 4,
                 n_axial:            int = 4,
                 n_end_winding:      int = 0,
                 n_radial_airgap:    int = 4):
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

        # Fixed dI = 20 A: small enough for linear regime (<<< rated current),
        # large enough that delta_psi = Ld * dI ~ 0.5 mWb >> mesh noise.
        # Do NOT derive dI from Ke or Iq — that causes saturation or
        # Park-projection contamination from large background currents.
        dI = 20.0   # A — fixed, linear, above noise floor
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

        # d-axis perturbation.  Also compute Ld by the stored-energy method
        # (L = 4W/3I²), which is gauge-invariant and independent of the
        # flux-linkage bookkeeping: the two must agree to ~0.1 %.
        from Bohemien_Motor_Designer.fea.solver3d import _build_nu
        _Br_backup = getattr(motor, "_get_Br", None)
        motor._get_Br = lambda: 0.0                    # PM off for energy check
        A_d = solve_magnetostatic_3d(self._mesh, motor, rotor_angle=rotor_angle,
                                     Id=dI, Iq=0.0, electrical_angle=th_e_pert)
        if _Br_backup is None:
            del motor._get_Br
        else:
            motor._get_Br = _Br_backup
        p_d = compute_flux_linkage_3d(self._mesh, A_d, motor, self._geom)
        c_  = 2.0 / 3.0
        pd_d = c_ * (p_d['psi_A']*np.cos(th_e_pert)
                     + p_d['psi_B']*np.cos(th_e_pert-2*np.pi/3)
                     + p_d['psi_C']*np.cos(th_e_pert+2*np.pi/3))
        B_d  = compute_B_field_3d(self._mesh, A_d, self._geom)
        nu   = _build_nu(self._mesh, motor)
        W_d  = 0.5 * float(np.sum(nu * (B_d**2).sum(axis=1) * np.abs(self._geom["vol"])))
        Ld_energy = 4.0 * W_d / (3.0 * dI**2)
        Ld        = pd_d / dI                           # PM off → no baseline needed

        # q-axis perturbation (PM on, baseline pq0 = 0 in aligned frame)
        _, pq_q = _psi_dq_zero(0.0, dI)
        Lq = (pq_q - pq0) / dI

        Ld = max(Ld, 1e-6)
        Lq = max(Lq, 1e-6)
        energy_check = Ld / (Ld_energy + 1e-30)

        # ── Step 4: reconcile with 2D and analytical ──────────────────────
        # This 3D model is a pure axial extrusion of the 2D cross-section
        # (plus, if n_end_winding > 0, straight conductors in an air end
        # region — an experimental 2.5-D model, NOT end-turn geometry).
        # It therefore cannot see end-winding leakage; L_ew is added
        # analytically, exactly as for the 2D result.  See PMSM.
        # end_winding_leakage_inductance() for the model and literature.
        L_ew  = float(motor.end_winding_leakage_inductance())
        brk   = motor.inductance_breakdown()

        dt_total = time.time() - t0_total
        _log(
            f"3D static done: Ke={Ke_3d:.4f}Wb  Ld={Ld*1e3:.4f}mH  Lq={Lq*1e3:.4f}mH  "
            f"(energy check {energy_check:.3f})  +L_ew={L_ew*1e3:.4f}mH  [{dt_total:.0f}s]",
            1.0,
        )

        result = dict(
            # Ke
            Ke_3d        = Ke_3d,
            Ke_3d_Wb     = Ke_3d,
            Ke_anal_Wb   = float(motor.back_emf_constant()),
            theta_w_deg  = float(np.degrees(theta_w)),
            # Inductances — raw 3D (extrusion: no end-winding leakage)
            Ld_H         = float(Ld),
            Lq_H         = float(Lq),
            Ld_mH        = float(Ld * 1e3),
            Lq_mH        = float(Lq * 1e3),
            Ld_energy_mH = float(Ld_energy * 1e3),
            Ld_energy_check = float(energy_check),      # Ld_fluxlink / Ld_energy, expect 1.00
            # End-winding correction and totals
            L_ew_mH      = L_ew * 1e3,
            Ld_total_mH  = float((Ld + L_ew) * 1e3),
            Lq_total_mH  = float((Lq + L_ew) * 1e3),
            # Analytical comparison (same decomposition)
            L_2d_anal_mH    = brk["L_2d"] * 1e3,
            L_total_anal_mH = brk["L_total"] * 1e3,
            Ld_anal_mH   = float(motor.Ld * 1e3),
            Lq_anal_mH   = float(motor.Lq * 1e3),
            end_region_model = ("straight-conductor (experimental)"
                                if self.n_ew > 0 else "none (pure extrusion)"),
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
