"""
py_runner.py - Pure-Python FEA orchestrator.

No GMSH, ElmerGrid, or ElmerSolver required.
Only numpy + scipy.

Capabilities
------------
The structured polar mesh gives accurate results for:
  - Average electromagnetic torque vs rotor position
  - Per-phase flux linkage -> back-EMF waveform and THD
  - Ld/Lq from d/q current perturbation (~10% accuracy)

Cogging torque from the structured mesh is NOT reliable because
the stairstepped PM boundaries dominate.  The runner therefore
reports the analytical cogging estimate (Ishikawa-Slemon formula)
from losses.py alongside the FEA loaded-torque results.

Usage
-----
    from Bohemien_Motor_Designer.fea.py_runner import PythonFEARunner

    runner = PythonFEARunner(motor)
    results = runner.run_cogging()   # analytical (fast)
    results = runner.run_loaded()    # FEA (1-3 min)
"""
from __future__ import annotations
import math
import time
import numpy as np
from typing import Optional, Callable

from Bohemien_Motor_Designer.fea.py_mesh   import build_motor_mesh, mesh_report
from Bohemien_Motor_Designer.fea.py_solver import solve_magnetostatic, compute_B_field
from Bohemien_Motor_Designer.fea.py_torque import compute_torque, compute_flux_linkage, extract_back_emf


class PythonFEARunner:
    """
    Pure-Python FEM runner.

    Parameters
    ----------
    motor            : PMSM instance
    n_radial_airgap  : radial elements across air gap.  THE main convergence
                       knob: 4 → Ke ≈ 0.8×, 8 → Ke within 0.1 % (reference motor)
    n_ang_per_slot   : angular elements per slot pitch (8–16)
    """

    def __init__(self, motor,
                 n_radial_airgap:   int = 8,
                 n_ang_per_slot:    int = 12):
        self.motor  = motor
        self._nr    = n_radial_airgap
        self._na    = n_ang_per_slot
        self._mesh  = None
        self._last_cogging: Optional[dict] = None
        self._last_loaded:  Optional[dict] = None

    # ── Mesh ───────────────────────────────────────────────────────────────────

    def build_mesh(self, progress_cb: Optional[Callable] = None) -> None:
        if progress_cb:
            progress_cb("Building structured mesh...", 0.0)
        t0 = time.time()
        self._mesh = build_motor_mesh(self.motor,
                                      n_radial_airgap=self._nr,
                                      n_angular_per_slot=self._na)
        dt = time.time() - t0
        if progress_cb:
            progress_cb(f"Mesh: {self._mesh.n_nodes} nodes / "
                        f"{self._mesh.n_elems} elems  ({dt:.2f}s)", 0.05)

    def mesh_summary(self) -> str:
        return mesh_report(self._mesh) if self._mesh else "Mesh not built."

    # ── Cogging (analytical) ───────────────────────────────────────────────────

    def run_cogging(self,
                    n_positions: int = 31,
                    progress_cb: Optional[Callable] = None) -> dict:
        """
        FEM-assisted cogging torque: one magnetostatic solve + Zhu-Howe spectral formula.

        Workflow
        --------
        1. Build mesh (if not already built).
        2. Solve the FEM at rotor_angle=0, Id=Iq=0.
        3. Extract the airgap radial flux density Br(θ) from the solve.
        4. Compute the fundamental PM spatial harmonic amplitude B_gap_fem
           via discrete Fourier transform of Br(θ).
        5. Feed B_gap_fem into the Zhu-Howe spectral cogging formula, which
           correctly accounts for LCM harmonic cancellation.
        6. Synthesise a multi-harmonic waveform over one cogging period.

        Why not a direct FEM sweep?
        ---------------------------
        The structured polar mesh assigns element tags by centroid angle.
        As the rotor sweeps, PM-boundary elements flip between PM and iron
        discretely, causing torque jumps ±10–50 × T_rated — far larger than
        actual cogging.  Resolving cogging directly requires a body-fitted
        rotor mesh (GMSH/Elmer path) or a sliding-layer formulation.
        The FEM-assisted spectral approach gives physically correct results
        with a single solve in O(1 s).
        """
        from math import gcd
        from Bohemien_Motor_Designer.fea.py_solver import solve_magnetostatic

        if self._mesh is None:
            self.build_mesh(progress_cb)

        motor   = self.motor
        Qs      = motor.slots
        poles   = motor.poles
        pp      = motor.pole_pairs
        rated_T = motor.rated_power / (motor.rated_speed * 2*np.pi/60 + 1e-9)

        lcm_val        = (Qs * poles) // gcd(Qs, poles)
        cog_period_rad = 2 * np.pi / lcm_val

        # ── Step 1: One FEM solve at reference position ────────────────────
        if progress_cb:
            progress_cb("FEM solve (Id=Iq=0, theta=0) — extracting airgap flux...", 0.05)

        t0 = time.time()
        try:
            Az = solve_magnetostatic(self._mesh, motor,
                                     rotor_angle=0.0, Id=0.0, Iq=0.0,
                                     electrical_angle=0.0)
            fem_ok = True
        except Exception as exc:
            if progress_cb:
                progress_cb(f"FEM solve failed ({exc}) — using magnet-circuit B_gap.", 0.3)
            Az     = None
            fem_ok = False

        # ── Step 2: Extract airgap Br(θ) and compute fundamental harmonic ──
        B_gap_fem = None
        if fem_ok and Az is not None:
            try:
                ag_idx = self._mesh.airgap_elems
                e  = self._mesh.elems[ag_idx]
                x  = self._mesh.nodes[:, 0][e]
                y  = self._mesh.nodes[:, 1][e]
                x0,x1,x2 = x[:,0],x[:,1],x[:,2]
                y0,y1,y2 = y[:,0],y[:,1],y[:,2]
                two_A = (x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)
                inv2A = 1.0/(two_A+1e-30)
                dNdx = np.column_stack([(y1-y2)*inv2A,(y2-y0)*inv2A,(y0-y1)*inv2A])
                dNdy = np.column_stack([(x2-x1)*inv2A,(x0-x2)*inv2A,(x1-x0)*inv2A])
                Az_e = Az[e]
                Bx = np.sum(Az_e*dNdy, axis=1)
                By = -np.sum(Az_e*dNdx, axis=1)
                cx = (x0+x1+x2)/3;  cy = (y0+y1+y2)/3
                th = np.arctan2(cy, cx)
                Br_ag = Bx*np.cos(th) + By*np.sin(th)

                # Sort by angle for Fourier
                idx_s  = np.argsort(th)
                th_s   = th[idx_s]
                Br_s   = Br_ag[idx_s]

                # Fundamental PM harmonic (spatial order = pole_pairs)
                # via discrete Fourier at the pole_pairs frequency
                try:
                    _trapz = np.trapezoid          # NumPy ≥ 2.0
                except AttributeError:
                    _trapz = np.trapz              # NumPy < 2.0
                B_cos = _trapz(Br_s * np.cos(pp * th_s), th_s) / np.pi
                B_sin = _trapz(Br_s * np.sin(pp * th_s), th_s) / np.pi
                B_fundamental = np.sqrt(B_cos**2 + B_sin**2)

                # Convert fundamental amplitude back to equivalent flat-top B_gap
                # For a flat-top PM: B_1 = (4/pi)*B_gap*sin(pi*alpha_p/2)/1
                alpha_p = getattr(motor, "magnet_width_fraction", 0.83)
                sin_factor = np.sin(np.pi * alpha_p / 2)
                if sin_factor > 0.05:
                    B_gap_fem = B_fundamental * np.pi / (4 * sin_factor)

                if progress_cb:
                    progress_cb(
                        f"  Airgap B_fund={B_fundamental:.3f}T  "
                        f"→ B_gap_fem={B_gap_fem:.3f}T  (FEM-derived)", 0.50)
            except Exception as exc2:
                if progress_cb:
                    progress_cb(f"  Airgap extraction failed ({exc2})", 0.50)
                B_gap_fem = None

        dt_fem = time.time() - t0

        # ── Step 3: Zhu-Howe spectral cogging formula ──────────────────────
        if progress_cb:
            progress_cb("Computing Zhu-Howe spectral cogging torque...", 0.60)

        from Bohemien_Motor_Designer.analysis.losses import cogging_torque_Nm
        cog = cogging_torque_Nm(motor, B_gap_override=B_gap_fem)
        T_pp = cog["Tcog_pp_Nm"]
        T_pct = T_pp / (rated_T + 1e-9) * 100

        # ── Step 4: Multi-harmonic waveform synthesis ──────────────────────
        angles_rad = np.linspace(0.0, cog_period_rad, n_positions, endpoint=False)
        torques    = np.zeros(n_positions)
        for h in cog.get("harmonics", []):
            nu  = h["nu"]
            T_k = h["T_k_Nm"]
            # Phase: each harmonic is sinusoidal at ν × Qs × θ with amplitude T_k/2
            torques += (T_k / 2) * np.sin(nu * Qs * angles_rad)

        # Normalise so that peak-to-peak of synthesised waveform = T_pp
        actual_pp = np.max(torques) - np.min(torques)
        if actual_pp > 1e-9:
            torques *= T_pp / actual_pp

        method = "fem_assisted_spectral"
        source_label = (f"B_gap={cog['B_gap_used']:.3f}T (FEM)"
                        if B_gap_fem is not None
                        else f"B_gap={cog['B_gap_used']:.3f}T (magnet circuit)")

        if progress_cb:
            progress_cb(
                f"Cogging T_pp={T_pp:.3f} Nm ({T_pct:.2f}% rated)  "
                f"[Zhu-Howe spectral  {source_label}  {dt_fem:.1f}s FEM]",
                1.0)

        result = dict(
            theta_deg    = np.degrees(angles_rad),
            angles_deg   = np.degrees(angles_rad),
            torque_Nm    = torques,
            Tcog_pp_Nm   = float(T_pp),
            Tcog_pp_pct  = float(T_pct),
            T_pp_Nm      = float(T_pp),
            T_pp_pct     = float(T_pct),
            rated_T_Nm   = float(rated_T),
            solve_time_s = float(dt_fem),
            method       = method,
            B_gap_used   = float(cog["B_gap_used"]),
            B_gap_fem    = float(B_gap_fem) if B_gap_fem else None,
            lcm_val      = int(lcm_val),
            cog_period_deg = float(np.degrees(cog_period_rad)),
            harmonics    = cog.get("harmonics", []),
        )
        self._last_cogging = result
        return result


    def run_loaded(self,
                   n_steps: int = 24,
                   progress_cb: Optional[Callable] = None) -> dict:
        """
        Loaded electromagnetic analysis over one magnetic period.

        Workflow
        --------
        1. Calibration solve (no-load, θ_m=0) to measure:
             • Ke_fem        — peak PM flux linkage as seen by the FEM mesh
             • theta_w       — winding-axis electrical angle offset (the mechanical
                               angle mismatch between rotor θ=0 and the phase-A axis
                               in the winding layout)
           This calibration is essential because:
             a) The mesh assigns coil sides by slot angle; phase A may not be at
                exactly 0° mechanical, so the Park frame needs correcting.
             b) The analytical back_emf_constant() and the FEM Ke can differ by
                10–30 % due to mesh discretisation, no Carter factor, etc.  Using
                Ke_fem ensures the FEM current level is self-consistent with the FEM
                flux linkage, giving T_avg ≈ T_rated.

        2. Sweep one magnetic period (π/pole_pairs mechanical) — the geometry is
           symmetric with period = one pole pitch, so sweeping two periods (the full
           electrical period 2π/pp) merely repeats computations.

        3. All Park transforms and current injection use the calibrated angle
             θ_e = θ_m · pp − θ_w
           which keeps the d-axis aligned with the PM flux vector throughout.

        4. Ld/Lq extraction via Park-frame perturbation at the calibrated d-axis.
        """
        from Bohemien_Motor_Designer.fea.py_solver import solve_magnetostatic

        if self._mesh is None:
            self.build_mesh(progress_cb)

        motor   = self.motor
        pp      = motor.pole_pairs
        rated_T = motor.rated_power / (motor.rated_speed * 2*np.pi/60 + 1e-9)

        # ── Step 0: Calibration (no-load solve at θ_m = 0) ─────────────────
        if progress_cb:
            progress_cb("Calibration solve (no-load θ_m=0) — measuring Ke_fem...", 0.02)

        Az_cal  = solve_magnetostatic(self._mesh, motor,
                                      rotor_angle=0.0, Id=0.0, Iq=0.0,
                                      electrical_angle=0.0)
        psi_cal = compute_flux_linkage(self._mesh, Az_cal, motor)

        def _park(psi_dict, th_e):
            """Park transform (amplitude-invariant)."""
            pA = psi_dict["psi_A"]; pB = psi_dict["psi_B"]; pC = psi_dict["psi_C"]
            c  = 2.0 / 3.0
            pd = c * (pA*np.cos(th_e) + pB*np.cos(th_e-2*np.pi/3)
                     + pC*np.cos(th_e+2*np.pi/3))
            pq = c * (-pA*np.sin(th_e) - pB*np.sin(th_e-2*np.pi/3)
                      - pC*np.sin(th_e+2*np.pi/3))
            return pd, pq

        pd_raw, pq_raw = _park(psi_cal, 0.0)
        # Winding offset: the d-axis is at theta_w electrical from the θ_e=0 frame.
        # theta_w = atan2(-pq_raw, pd_raw)  so that Park at θ_e = -theta_w gives pq=0.
        theta_w = float(np.arctan2(-pq_raw, pd_raw))
        Ke_fem  = float(np.sqrt(pd_raw**2 + pq_raw**2))   # amplitude independent of frame

        # Iq from FEM Ke — self-consistent with the FEM field
        try:
            Iq_rated = rated_T / (1.5 * pp * Ke_fem + 1e-12)
            if hasattr(motor, "mtpa_angle") and motor.Ld != motor.Lq:
                # Use analytical MTPA angle if IPM saliency is meaningful
                mtpa_I  = rated_T / (1.5 * pp * motor.back_emf_constant() + 1e-12)
                Id_mtpa, Iq_mtpa = motor.mtpa_angle(mtpa_I)
                # Preserve MTPA angle; scale magnitude to Ke_fem
                I_mtpa = np.sqrt(Id_mtpa**2 + Iq_mtpa**2)
                ang_mtpa = np.arctan2(Id_mtpa, Iq_mtpa)  # negative = demagnetising
                I_scaled = rated_T / (1.5 * pp * Ke_fem + 1e-12)
                Id = I_scaled * np.sin(ang_mtpa)
                Iq = I_scaled * np.cos(ang_mtpa)
            else:
                Id, Iq = 0.0, Iq_rated
        except Exception:
            Id, Iq = 0.0, Ke_fem and rated_T / (1.5 * pp * Ke_fem) or 50.0

        if progress_cb:
            progress_cb(
                f"  Ke_fem={Ke_fem:.4f}Wb  θ_w={np.degrees(theta_w):.1f}°e  "
                f"Iq={Iq:.1f}A Id={Id:.1f}A  (analytical Ke={motor.back_emf_constant():.4f}Wb)",
                0.05)

        # ── Step 1: Sweep one magnetic period (π/pp mechanical) ─────────────
        # The FEM solution repeats with period = one pole pitch = π/pp rad mechanical.
        # Sweeping the full electrical period (2π/pp) just repeats computations.
        mag_period = np.pi / pp               # one pole pitch [rad mechanical]
        angles     = np.linspace(0.0, mag_period, n_steps, endpoint=False)

        omega_e = motor.rated_speed * 2*np.pi/60 * pp
        dt_mech = angles[1] - angles[0] if len(angles) > 1 else 1e-4
        dt_time = dt_mech / (omega_e + 1e-9)

        if progress_cb:
            progress_cb(f"Loaded sweep: {n_steps} steps over {np.degrees(mag_period):.1f}°  "
                        f"Id={Id:.1f}A Iq={Iq:.1f}A", 0.06)

        t0      = time.time()
        torques = np.zeros(n_steps)
        psi_hist = []

        for k, theta_m in enumerate(angles):
            # Calibrated electrical angle: shifts Park frame to d-axis
            theta_e = theta_m * pp - theta_w

            Az  = solve_magnetostatic(self._mesh, motor,
                                      rotor_angle=theta_m,
                                      Id=Id, Iq=Iq,
                                      electrical_angle=theta_e)
            psi = compute_flux_linkage(self._mesh, Az, motor)
            psi_hist.append(psi)

            pd, pq = _park(psi, theta_e)
            torques[k] = 1.5 * pp * (pd * Iq - pq * Id)

            frac = 0.06 + 0.75 * (k+1) / n_steps
            if progress_cb:
                progress_cb(f"  [{k+1}/{n_steps}] T={torques[k]:.2f} Nm  "
                            f"ψ_d={pd:.4f} ψ_q={pq:.4f} Wb", frac)

        # ── Step 2: Ke (from calibration — no extra solve needed) ───────────
        # Ke_fem already computed above; just sanity-check with no-load flux
        # at the corrected d-axis angle.
        if progress_cb:
            progress_cb("Computing Ke (from calibration solve)...", 0.82)

        # Ke_fea = peak psi_d at d-axis (= Ke_fem from calibration)
        # Also re-check: at theta_m=0, theta_e = -theta_w → pq=0, pd=Ke_fem
        Ke_fea = Ke_fem

        # ── Step 3: Back-EMF and THD ─────────────────────────────────────────
        psi_A_arr = np.array([p["psi_A"] for p in psi_hist])
        psi_B_arr = np.array([p["psi_B"] for p in psi_hist])
        psi_C_arr = np.array([p["psi_C"] for p in psi_hist])

        emf_fund_V = Ke_fea * omega_e

        # THD from FEM requires at least one full electrical period (2π/pp mechanical).
        # The sweep covers one magnetic period (π/pp), which is only HALF an electrical
        # period.  Use the analytical back-EMF harmonics for THD instead.
        try:
            h_anal     = motor.back_emf_harmonics(n_harmonics=15)
            emf_thd_pct = float(h_anal["thd"])
            emf_A_wave  = np.array([h_anal.get("fundamental", emf_fund_V)
                                    * np.cos(float(th) * pp * np.pi / 180.0)
                                    for th in np.degrees(angles)])
        except Exception:
            emf_thd_pct = 0.0
            emf_A_wave  = np.zeros(n_steps)

        # Still run extract_back_emf for emf_A/B/C waveform display
        try:
            emf_raw = extract_back_emf(psi_hist, dt_time, motor)
            emf_A_wave = emf_raw.get("emf_A", emf_A_wave)
            emf_B_wave = emf_raw.get("emf_B", np.zeros(n_steps))
            emf_C_wave = emf_raw.get("emf_C", np.zeros(n_steps))
        except Exception:
            emf_B_wave = emf_C_wave = np.zeros(n_steps)

        T_avg    = float(torques.mean())
        T_ripple = (torques.max() - torques.min()) / (abs(T_avg) + 1e-9) * 100

        # ── Step 4: Ld/Lq via Park-frame perturbation at d-axis ─────────────
        if progress_cb:
            progress_cb("Extracting Ld/Lq (2 perturbation solves at calibrated d-axis)...", 0.85)

# ── Ld/Lq: zero-background perturbation ──────────────────────────────────
        # Use Id=Iq=0 baseline (no-load) so there is no large background
        # current whose Park projection error can contaminate the tiny
        # delta_psi signal. The no-load psi_d0 = Ke_fea (already computed).
        #
        # dI is fixed at a value that gives a flux change well above
        # numerical noise (~0.001 mWb) but small enough to stay in the
        # linear regime (<<< rated current).
        # delta_psi = Ld * dI; need dI > noise/Ld ~ 0.001mWb / 0.05mH = 20A
        # Use 20 A fixed — well below rated current for any motor in this range.
        dI = 20.0   # A — fixed perturbation, linear regime, above noise floor

        theta_m_pert = 0.0
        theta_e_pert = theta_m_pert * pp - theta_w  # d-axis aligned

        def _psi_at_zero_bg(Id_, Iq_):
            """Flux linkage with zero background current."""
            Az = solve_magnetostatic(self._mesh, motor,
                                     rotor_angle=theta_m_pert, Id=Id_, Iq=Iq_,
                                     electrical_angle=theta_e_pert)
            return compute_flux_linkage(self._mesh, Az, motor)

        # Baseline: reuse no-load psi_d = Ke_fea (saves one solve)
        pd0 = Ke_fea   # psi_d at no-load, d-axis aligned
        pq0 = 0.0      # psi_q = 0 at no-load in d-axis frame

        p_d = _psi_at_zero_bg(dI,  0.0)   # d-axis perturbation only
        p_q = _psi_at_zero_bg(0.0, dI)    # q-axis perturbation only

        pdd, _  = _park(p_d, theta_e_pert)
        _,  pqq = _park(p_q, theta_e_pert)

        Ld = (pdd - pd0) / dI
        Lq = (pqq - pq0) / dI

        # Clamp: inductance must be positive and physically bounded
        Ld = max(Ld, 1e-7)
        Lq = max(Lq, 1e-7)

        # ── Step 5: mesh-convergence check ──────────────────────────────────
        # Re-solve Ld on a mesh with half the angular density AND half the
        # radial airgap layers.  The FEM energy converges from below, so
        # Ld(fine)/Ld(coarse) > 1; if it exceeds ~1.10 the fine mesh is
        # itself still far from converged and the Ld/Lq numbers should be
        # treated as lower bounds.  (On the reference motor the radial
        # airgap layers are the controlling parameter: n_radial_airgap=8
        # brings Ke to 0.1 % and Ld to ~1 % of the converged value.)
        if progress_cb:
            progress_cb("Mesh-convergence check (coarser mesh, 2 solves)...", 0.95)
        conv = self._convergence_check(Ld, theta_w, dI)

        # ── Step 6: end-winding correction ──────────────────────────────────
        # A 2D model cannot see end-winding leakage.  Add the analytical
        # L_ew (PMSM.end_winding_leakage_inductance, Pyrhönen/Richter) to
        # give the total synchronous inductance.  The analytical
        # decomposition is reported alongside for a like-for-like comparison:
        #   Ld_fem        ↔  L_2d_anal   (= L_ag + L_h + L_slot + L_tt)
        #   Ld_fem + L_ew ↔  L_total_anal
        try:
            brk  = motor.inductance_breakdown()
            L_ew = float(brk["L_ew"])
        except Exception:
            brk, L_ew = {}, 0.0

        elapsed = time.time() - t0
        if progress_cb:
            progress_cb(
                f"Loaded done: T_avg={T_avg:.1f}Nm  Ke={Ke_fea:.4f}Wb  "
                f"Ld={Ld*1e3:.4f}mH Lq={Lq*1e3:.4f}mH  (+L_ew {L_ew*1e3:.4f})  "
                f"conv={conv['ratio']:.3f}{' ⚠' if not conv['converged'] else ''}  "
                f"THD={emf_thd_pct:.1f}%  [{elapsed:.0f}s]", 1.0)

        result = dict(
            theta_deg      = np.degrees(angles),
            angles_deg     = np.degrees(angles),
            torque_Nm      = torques,
            torque_avg_Nm  = T_avg,
            T_avg_Nm       = T_avg,
            T_ripple_pct   = float(T_ripple),
            rated_T_Nm     = float(rated_T),
            psi_A          = psi_A_arr,
            psi_B          = psi_B_arr,
            psi_C          = psi_C_arr,
            emf_A          = emf_A_wave,
            emf_B          = emf_B_wave,
            emf_C          = emf_C_wave,
            emf_fund_V     = emf_fund_V,
            emf_thd_pct    = emf_thd_pct,
            emf_waveform   = {
                "voltage":  emf_A_wave,
                "thd_pct":  emf_thd_pct,
                "fund_V":   emf_fund_V,
            },
            Ke_Wb          = Ke_fea,
            Ke_fem_Wb      = Ke_fem,
            theta_w_deg    = float(np.degrees(theta_w)),
            # Raw 2D FEM inductances (no end-winding leakage by construction)
            Ld_H           = float(Ld),
            Lq_H           = float(Lq),
            Ld_mH          = float(Ld * 1e3),
            Lq_mH          = float(Lq * 1e3),
            # End-winding correction and totals
            L_ew_mH        = L_ew * 1e3,
            Ld_total_mH    = float((Ld + L_ew) * 1e3),
            Lq_total_mH    = float((Lq + L_ew) * 1e3),
            # Analytical decomposition for like-for-like comparison
            L_2d_anal_mH   = float(brk.get("L_2d", 0.0) * 1e3),
            L_total_anal_mH= float(brk.get("L_total", 0.0) * 1e3),
            L_ag_anal_mH   = float(brk.get("L_ag", 0.0) * 1e3),
            L_slot_anal_mH = float((brk.get("L_slot", 0.0) + brk.get("L_tt", 0.0)
                                    + brk.get("L_h", 0.0)) * 1e3),
            # Mesh convergence
            conv_ratio     = conv["ratio"],
            conv_converged = conv["converged"],
            conv_coarse_Ld_mH = conv["Ld_coarse_mH"],
            conv_mesh_fine = conv["mesh_fine"],
            conv_mesh_coarse = conv["mesh_coarse"],
            solve_time_s   = float(elapsed),
            method         = "python_fem",
        )
        self._last_loaded = result
        return result

    # ── Mesh-convergence check ──────────────────────────────────────────────

    def _convergence_check(self, Ld_fine: float, theta_w: float,
                           dI: float, tol: float = 1.10) -> dict:
        """
        Re-extract Ld on a coarser mesh (half angular density, half radial
        airgap layers) and return the ratio Ld_fine / Ld_coarse.

        FEM energy converges from below, so the ratio is ≥ 1.  A ratio
        above `tol` (default 1.10) means the fine mesh is still changing
        by more than ~10 % per refinement step and the reported Ld/Lq are
        lower bounds; increase n_radial_airgap first, then n_ang_per_slot.
        """
        from Bohemien_Motor_Designer.fea.py_solver import solve_magnetostatic
        motor = self.motor
        n_ang_c = max(2, self._na // 2)
        n_ag_c  = max(2, self._nr // 2)
        try:
            mesh_c = build_motor_mesh(motor, n_radial_airgap=n_ag_c,
                                      n_angular_per_slot=n_ang_c)
            th_e = -theta_w
            Az0 = solve_magnetostatic(mesh_c, motor, rotor_angle=0.0,
                                      Id=0.0, Iq=0.0, electrical_angle=0.0)
            p0  = compute_flux_linkage(mesh_c, Az0, motor)
            c   = 2.0 / 3.0
            pd0 = c * (p0["psi_A"]*np.cos(0.0) + p0["psi_B"]*np.cos(-2*np.pi/3)
                       + p0["psi_C"]*np.cos(2*np.pi/3))
            pq0 = c * (-p0["psi_B"]*np.sin(-2*np.pi/3) - p0["psi_C"]*np.sin(2*np.pi/3))
            Ke_c = float(np.hypot(pd0, pq0))
            Az_d = solve_magnetostatic(mesh_c, motor, rotor_angle=0.0,
                                       Id=dI, Iq=0.0, electrical_angle=th_e)
            p_d  = compute_flux_linkage(mesh_c, Az_d, motor)
            pdd  = c * (p_d["psi_A"]*np.cos(th_e) + p_d["psi_B"]*np.cos(th_e-2*np.pi/3)
                        + p_d["psi_C"]*np.cos(th_e+2*np.pi/3))
            Ld_c = max((pdd - Ke_c) / dI, 1e-9)
            ratio = float(Ld_fine / Ld_c)
            return dict(ratio=ratio, converged=bool(ratio <= tol),
                        Ld_coarse_mH=float(Ld_c * 1e3),
                        mesh_fine=f"{self._na} ang/slot, {self._nr} AG layers",
                        mesh_coarse=f"{n_ang_c} ang/slot, {n_ag_c} AG layers")
        except Exception as exc:  # never let the check break the main result
            return dict(ratio=float("nan"), converged=False,
                        Ld_coarse_mH=float("nan"),
                        mesh_fine=f"{self._na} ang/slot, {self._nr} AG layers",
                        mesh_coarse=f"check failed: {exc}")


    def summary(self) -> str:
        m = self.motor
        Ke_ana = m.back_emf_constant()
        lines  = ["=== PythonFEA Results ===",
                  f"Analytical Ke       = {Ke_ana:.4f} Vs/rad"]
        try:
            lines.append(f"Analytical Ld       = {m.Ld*1e3:.3f} mH")
            lines.append(f"Analytical Lq       = {m.Lq*1e3:.3f} mH")
        except Exception:
            pass

        if self._last_cogging:
            r = self._last_cogging
            lines.append(f"Cogging T_pp (ana.) = {r['T_pp_Nm']:.3f} Nm "
                         f"({r['T_pp_pct']:.2f}% rated)")

        if self._last_loaded:
            r = self._last_loaded
            Ke_err = abs(r['Ke_Wb'] - Ke_ana) / (Ke_ana+1e-9) * 100
            lines.append(f"FEA T_avg           = {r['T_avg_Nm']:.1f} Nm  "
                         f"(rated {r['rated_T_Nm']:.1f} Nm)")
            lines.append(f"FEA Torque ripple   = {r['T_ripple_pct']:.1f}%")
            lines.append(f"FEA Ke              = {r['Ke_Wb']:.4f} Wb  "
                         f"(err {Ke_err:.1f}% vs analytical)")
            lines.append("— Inductance (2D FEM has no end-winding term) —")
            lines.append(f"FEA Ld (2D raw)     = {r['Ld_mH']:.4f} mH   "
                         f"analytical L_2d = {r.get('L_2d_anal_mH', 0):.4f} mH")
            lines.append(f"FEA Lq (2D raw)     = {r['Lq_mH']:.4f} mH")
            lines.append(f"+ L_ew (analytical) = {r.get('L_ew_mH', 0):.4f} mH")
            lines.append(f"Ld total            = {r.get('Ld_total_mH', r['Ld_mH']):.4f} mH   "
                         f"analytical L_total = {r.get('L_total_anal_mH', 0):.4f} mH")
            lines.append(f"Lq total            = {r.get('Lq_total_mH', r['Lq_mH']):.4f} mH")
            if 'conv_ratio' in r:
                flag = "ok" if r['conv_converged'] else "NOT converged — refine mesh"
                lines.append(f"Mesh convergence    = {r['conv_ratio']:.3f}  "
                             f"[{r['conv_mesh_fine']} vs {r['conv_mesh_coarse']}]  {flag}")
            lines.append(f"FEA BEMF fund       = {r['emf_fund_V']:.1f} V pk")
            lines.append(f"FEA BEMF THD        = {r['emf_thd_pct']:.1f}%")
            lines.append(f"Solve time          = {r['solve_time_s']:.0f}s")

        return "\n".join(lines)
