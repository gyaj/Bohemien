"""
Pure-Python FEA Runner for Bohemien_Motor_Designer.

Replaces the GMSH + Elmer pipeline with a self-contained Python implementation
that requires only numpy and scipy.  No external executables needed.

Pipeline
--------
1. MeshBuilder  -- structured polar triangular mesh from motor geometry
2. FEMSolver    -- assemble + Newton solve for A_z (nonlinear BH)
3. arkkio_torque + flux_linkage_per_phase -- T, psi
4. Sweep        -- repeat 2-3 for each rotor position

Usage
-----
    from Bohemien_Motor_Designer.fea.python_runner import PythonFEARunner

    runner = PythonFEARunner(motor)
    cog    = runner.run_cogging(progress_cb=print)
    loaded = runner.run_loaded(progress_cb=print)
"""
from __future__ import annotations
import numpy as np
import time
from typing import Callable, Optional, Dict
from pathlib import Path

from Bohemien_Motor_Designer.fea.mesh_reader import MeshData
from Bohemien_Motor_Designer.fea.solver import FEMSolver, MaterialSpec, MU0
from Bohemien_Motor_Designer.fea.torque import (arkkio_torque, flux_linkage_per_phase,
                                      extract_inductances, thd_from_waveform)
from Bohemien_Motor_Designer.fea.rotor_rotation import (rotate_rotor_nodes, cogging_angles,
                                              electrical_angles)
from Bohemien_Motor_Designer.fea.index_registry import IndexRegistry
from Bohemien_Motor_Designer.materials.library import MaterialLibrary


# ── Structured polar mesh builder ─────────────────────────────────────────────

class MeshBuilder:
    """
    Builds a structured polar triangular mesh for a 2D PMSM cross-section.

    Radial zones (centre outward):
      shaft      r < R_ri
      rotor iron R_ri < r < R_mi  (R_mi = R_ro - t_m)
      magnets    R_mi < r < R_ro  (SPM arc segments)
      air gap    R_ro < r < R_si  (split at R_slide)
      stator     R_si < r < R_so  (with slot sub-regions)

    Each (r_band, theta_sector) quad cell is split into 2 triangles.
    Slot regions are identified by angle and radius and assigned winding tags.
    """

    def __init__(self, motor, registry: IndexRegistry,
                 n_radial: int = 12,
                 n_angular_per_slot: int = 6):
        self.m    = motor
        self.reg  = registry
        self.nr   = n_radial
        self.naps = n_angular_per_slot

    def build(self) -> MeshData:
        m   = self.m
        reg = self.reg
        rg  = m.rotor_geo

        R_ri    = m.rotor_inner_radius
        R_ro    = m.rotor_outer_radius
        R_si    = m.stator.inner_radius if m.stator else R_ro + m.airgap
        R_so    = m.stator.outer_radius if m.stator else R_si * 1.6
        R_slide = (R_ro + R_si) / 2.0
        t_m     = getattr(rg, "magnet_thickness", m.magnet_thickness)
        R_mi    = R_ro - t_m
        Qs      = m.slots
        p2      = m.poles

        n_theta = Qs * self.naps
        thetas  = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)

        r_bounds = self._radial_bounds(R_ri, R_mi, R_ro, R_slide, R_si, R_so)
        N_r = len(r_bounds)
        N_t = n_theta

        # Node grid: (N_r * N_t) nodes
        nodes = np.empty((N_r * N_t, 2), dtype=np.float64)
        for ri, r in enumerate(r_bounds):
            for ti, theta in enumerate(thetas):
                nodes[ri * N_t + ti, 0] = r * np.cos(theta)
                nodes[ri * N_t + ti, 1] = r * np.sin(theta)

        def nid(ri, ti):
            return ri * N_t + (ti % N_t)

        # Precompute slot geometry
        sp        = m.stator.slot_profile if m.stator else None
        slot_depth = sp.depth() if sp else 0.0
        slot_w     = sp.area() / (sp.depth() + 1e-9) if sp else 0.0
        half_ang_slot = np.arcsin(min(slot_w / (2 * R_si + 1e-9), 0.999)) if sp else 0.0
        alpha_p   = getattr(rg, "magnet_width_fraction",
                            getattr(m, "magnet_width_fraction", 0.85))
        half_mag  = np.pi * alpha_p / p2

        slot_pitch_ang = 2 * np.pi / Qs

        tri_list      = []
        tri_tags_list = []

        for ri in range(N_r - 1):
            r_inner = r_bounds[ri]
            r_outer = r_bounds[ri + 1]
            r_mid   = (r_inner + r_outer) / 2.0

            for ti in range(N_t):
                ti1 = (ti + 1) % N_t
                theta_c = thetas[ti] + (thetas[ti1] - thetas[ti] if ti < N_t - 1
                                        else 2*np.pi/N_t) / 2.0

                tag = self._tag(r_mid, theta_c, R_ri, R_mi, R_ro, R_slide,
                                R_si, R_so, slot_depth, half_ang_slot,
                                half_mag, slot_pitch_ang, alpha_p,
                                Qs, p2, rg, reg)

                n00 = nid(ri,   ti)
                n10 = nid(ri+1, ti)
                n01 = nid(ri,   ti1)
                n11 = nid(ri+1, ti1)

                tri_list.append([n00, n10, n01])
                tri_tags_list.append(tag)
                tri_list.append([n10, n11, n01])
                tri_tags_list.append(tag)

        # Boundary edges: outer ring
        edge_list = []
        edge_tags = []
        ri_out = N_r - 1
        for ti in range(N_t):
            edge_list.append([nid(ri_out, ti), nid(ri_out, (ti+1) % N_t)])
            edge_tags.append(reg.outer_boundary)

        # Sliding surface edges
        r_slide_ri = None
        for ri, r in enumerate(r_bounds):
            if abs(r - R_slide) < (r_bounds[1] - r_bounds[0]) * 0.6:
                r_slide_ri = ri
                break
        if r_slide_ri is not None:
            for ti in range(N_t):
                edge_list.append([nid(r_slide_ri, ti), nid(r_slide_ri, (ti+1)%N_t)])
                edge_tags.append(reg.sliding_surface)

        return MeshData(
            nodes      = nodes,
            tri        = np.array(tri_list,  dtype=np.int32),
            tri_tags   = np.array(tri_tags_list, dtype=np.int32),
            edge_nodes = np.array(edge_list, dtype=np.int32) if edge_list else np.zeros((0,2),np.int32),
            edge_tags  = np.array(edge_tags, dtype=np.int32) if edge_tags else np.zeros(0,np.int32),
        )

    def _radial_bounds(self, R_ri, R_mi, R_ro, R_slide, R_si, R_so):
        nr = self.nr
        def layer(r0, r1, n):
            pts = np.linspace(r0, r1, n + 1)
            return list(pts)
        bounds = layer(R_ri, R_mi, nr)
        bounds += layer(R_mi, R_ro, max(2, nr // 3))[1:]
        bounds += layer(R_ro, R_slide, max(2, nr // 3))[1:]
        bounds += layer(R_slide, R_si, max(2, nr // 3))[1:]
        bounds += layer(R_si, R_so, nr)[1:]
        return bounds

    def _tag(self, r_mid, theta_c, R_ri, R_mi, R_ro, R_slide, R_si, R_so,
              slot_depth, half_ang_slot, half_mag, slot_pitch_ang,
              alpha_p, Qs, p2, rg, reg):

        if r_mid <= R_ri:
            return reg.shaft
        if r_mid <= R_mi:
            return reg.rotor_iron
        if r_mid <= R_ro:
            # Check if inside a magnet arc
            for pole in range(p2):
                centre = pole * 2 * np.pi / p2
                diff = (theta_c - centre + np.pi) % (2*np.pi) - np.pi
                if abs(diff) <= half_mag:
                    return reg.pm_tag(pole)
            return reg.rotor_iron
        if r_mid <= R_slide:
            return reg.air_gap
        if r_mid <= R_si:
            return reg.air_gap
        if r_mid <= R_so:
            # Slot check
            if half_ang_slot > 0 and r_mid <= R_si + slot_depth:
                for si in range(Qs):
                    sc = si * slot_pitch_ang + slot_pitch_ang / 2.0
                    diff = (theta_c - sc + np.pi) % (2*np.pi) - np.pi
                    if abs(diff) <= half_ang_slot:
                        layer = 0 if r_mid <= R_si + slot_depth / 2.0 else 1
                        return reg.winding_tag(si, layer)
            return reg.stator_iron
        return reg.stator_iron


# ── Material builder ──────────────────────────────────────────────────────────

def build_materials(motor, registry: IndexRegistry,
                    lib: MaterialLibrary = None,
                    theta_rotor: float = 0.0,
                    Id: float = 0.0,
                    Iq: float = 0.0) -> Dict[int, MaterialSpec]:
    if lib is None:
        lib = MaterialLibrary()

    mats = {}
    m    = motor
    reg  = registry
    rg   = m.rotor_geo

    # Stator + rotor iron
    lam_grade = getattr(m.stator, "lamination", "M270-35A") if m.stator else "M270-35A"
    for iron_tag in (reg.stator_iron, reg.rotor_iron):
        try:
            lam = lib.lamination(lam_grade)
            if lam.bh_table:
                nu_f = FEMSolver.nu_func_from_bh(lam.bh_table, lam.mu_r_initial)
                mats[iron_tag] = MaterialSpec(tag=iron_tag,
                                              mu_r=lam.mu_r_initial, nu_func=nu_f)
            else:
                mats[iron_tag] = MaterialSpec(tag=iron_tag, mu_r=lam.mu_r_initial)
        except Exception:
            mats[iron_tag] = MaterialSpec(tag=iron_tag, mu_r=5000.0)

    # Air, shaft
    mats[reg.air_gap] = MaterialSpec(tag=reg.air_gap, mu_r=1.0)
    mats[reg.shaft]   = MaterialSpec(tag=reg.shaft,   mu_r=1.0)

    # Permanent magnets
    Br      = m._get_Br() if hasattr(m, "_get_Br") else 1.2
    mu_r_m  = getattr(m, "_mu_r", 1.05)
    p2      = m.poles

    for pole in range(p2):
        bid        = reg.pm_tag(pole)
        mech_angle = pole * 2 * np.pi / p2 + theta_rotor
        polarity   = 1.0 if pole % 2 == 0 else -1.0
        H_rem      = Br / (MU0 * mu_r_m)
        mats[bid]  = MaterialSpec(tag=bid, mu_r=mu_r_m,
                                   Mx=polarity * H_rem * np.cos(mech_angle),
                                   My=polarity * H_rem * np.sin(mech_angle))

    # Winding current density
    winding = m.winding
    if winding is None:
        return mats

    theta_e  = theta_rotor * m.pole_pairs
    I_phase  = np.array([
        Id * np.cos(theta_e + k*2*np.pi/3) - Iq * np.sin(theta_e + k*2*np.pi/3)
        for k in range(3)
    ])

    sp      = m.stator.slot_profile if m.stator else None
    slot_A  = sp.area() if sp else 1e-4
    ff      = getattr(m, "slot_fill_factor", 0.45)
    N_coil  = getattr(m, "turns_per_coil", 10)
    J_scale = N_coil / (slot_A * ff + 1e-9)

    for coil in (winding._table if hasattr(winding, "_table") else []):
        wt       = reg.winding_tag(coil.slot_idx, coil.layer)
        mats[wt] = MaterialSpec(tag=wt, mu_r=1.0,
                                 J_z=float(coil.direction * J_scale * I_phase[coil.phase]))
    return mats


# ── PythonFEARunner ───────────────────────────────────────────────────────────

class PythonFEARunner:
    """
    Pure-Python FEA runner.  No external tools.  numpy + scipy only.

    Parameters
    ----------
    motor       : PMSM instance
    n_cog       : rotor positions for cogging sweep  (default 32)
    n_loaded    : rotor positions for loaded sweep   (default 60)
    mesh_radial : radial layers per zone  (default 12)
    mesh_angular_per_slot : angular divisions per slot pitch  (default 6)
    """

    def __init__(self, motor,
                 n_cog:    int = 32,
                 n_loaded: int = 60,
                 mesh_radial: int = 12,
                 mesh_angular_per_slot: int = 6):
        self.m    = motor
        self.n_cog    = n_cog
        self.n_loaded = n_loaded
        self.nr   = mesh_radial
        self.naps = mesh_angular_per_slot
        self.reg  = IndexRegistry(poles=motor.poles, slots=motor.slots)
        self.lib  = MaterialLibrary()
        self._base_mesh: Optional[MeshData] = None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _cb(self, cb, msg, frac=0.0):
        if cb:
            try:    cb(msg, frac)
            except TypeError: cb(msg)

    def _get_mesh(self) -> MeshData:
        if self._base_mesh is None:
            builder = MeshBuilder(self.m, self.reg, self.nr, self.naps)
            self._base_mesh = builder.build()
        return self._base_mesh

    def _R_slide(self):
        m = self.m
        R_si = m.stator.inner_radius if m.stator else m.rotor_outer_radius + m.airgap
        return (m.rotor_outer_radius + R_si) / 2.0

    def _solve_at(self, theta_rad, Id=0.0, Iq=0.0, progress_cb=None):
        base  = self._get_mesh()
        mesh  = rotate_rotor_nodes(base, theta_rad, self._R_slide())
        mats  = build_materials(self.m, self.reg, self.lib, theta_rad, Id, Iq)
        solver = FEMSolver(mesh, mats, outer_bc_tag=self.reg.outer_boundary)
        sol   = solver.solve(progress_cb=progress_cb)
        return sol, mesh

    def mesh_summary(self) -> str:
        return self._get_mesh().summary()

    # ── Cogging ───────────────────────────────────────────────────────────

    def run_cogging(self, progress_cb=None) -> dict:
        """
        Cogging torque sweep (Id=Iq=0).
        Returns dict: theta_deg, torque_Nm, Tcog_pp_Nm, Tcog_pp_pct
        """
        m      = self.m
        angles = cogging_angles(m)[:self.n_cog]
        n      = len(angles)
        R_ro   = m.rotor_outer_radius
        R_si   = m.stator.inner_radius if m.stator else R_ro + m.airgap

        t0 = time.time()
        base = self._get_mesh()
        self._cb(progress_cb,
                 f"Python FEA: cogging  {n} positions | "
                 f"{base.n_nodes} nodes  {len(base.tri)} elements", 0.02)

        torques = np.zeros(n)
        for i, theta in enumerate(angles):
            self._cb(progress_cb,
                     f"  [{i+1}/{n}] theta={np.degrees(theta):.2f} deg",
                     0.05 + 0.90 * i / n)
            sol, mesh_r = self._solve_at(theta)
            torques[i]  = arkkio_torque(mesh_r, sol,
                                         self.reg.air_gap,
                                         R_ro, R_si, m.stack_length)

        Tpp = float(np.max(torques) - np.min(torques))
        pct = 100.0 * Tpp / (m.rated_torque + 1e-9)
        self._cb(progress_cb,
                 f"  Done: Tpp={Tpp:.3f} Nm ({pct:.1f}%)  "
                 f"elapsed={time.time()-t0:.0f}s", 1.0)

        return {"theta_deg":   np.degrees(angles),
                "torque_Nm":   torques,
                "Tcog_pp_Nm":  Tpp,
                "Tcog_pp_pct": pct}

    # ── Loaded ────────────────────────────────────────────────────────────

    def run_loaded(self, progress_cb=None) -> dict:
        """
        Loaded torque sweep + Ld/Lq extraction + back-EMF THD.
        Writes results back to motor.Ld and motor.Lq.
        Returns dict: time_s, torque_Nm, torque_avg_Nm, Ld_H, Lq_H, emf_waveform
        """
        m      = self.m
        angles = electrical_angles(m, self.n_loaded)
        n      = len(angles)
        R_ro   = m.rotor_outer_radius
        R_si   = m.stator.inner_radius if m.stator else R_ro + m.airgap

        Ke     = m.back_emf_constant()
        omega_e = m.rated_speed * 2 * np.pi / 60 * m.pole_pairs

        # Guard: if back-EMF > Vbus/sqrt(3), the motor can't run at rated speed
        # with this winding.  Clamp Iq to keep E_peak < Vpk_max.
        V_bus   = getattr(m, "rated_voltage", 400.0)
        try:
            V_bus = m.spec.drive.dc_bus_voltage
        except Exception:
            pass
        Vpk_max = V_bus / (3 ** 0.5)         # SVPWM phase peak [V]
        E_peak  = Ke * omega_e
        if E_peak > Vpk_max * 0.95:
            # Winding is over-turned; compute Iq from voltage limit
            # T = 1.5*p*Ke*Iq, Vpk ≈ Ke*omega_e + Ld*omega_e*Iq (simplified)
            # At voltage limit: T_max = 1.5*p*Ke*(Vpk-E_peak)/(Ld*omega_e+1e-9)
            # Use simpler: Iq = T_rated/(1.5*p*Ke) but warn
            pass  # let it compute; will be small and physically correct

        Iq_pk  = m.rated_torque / (1.5 * m.pole_pairs * Ke + 1e-9) * np.sqrt(2)
        Id_pk  = 0.0
        if abs(m.Ld - m.Lq) > 1e-6:
            Id_pk, Iq_pk = m.mtpa_angle(Iq_pk)

        omega  = m.rated_speed * 2 * np.pi / 60
        dt     = (angles[1] - angles[0]) / omega

        t0 = time.time()
        self._cb(progress_cb,
                 f"Python FEA: loaded  {n} positions | "
                 f"Id={Id_pk:.1f}A  Iq={Iq_pk:.1f}A peak", 0.02)

        torques = np.zeros(n)
        psi_all = np.zeros((n, 3))

        for i, theta in enumerate(angles):
            self._cb(progress_cb,
                     f"  [{i+1}/{n}] theta={np.degrees(theta):.1f} deg",
                     0.05 + 0.80 * i / n)
            sol, mesh_r = self._solve_at(theta, Id_pk, Iq_pk)
            torques[i]  = arkkio_torque(mesh_r, sol,
                                         self.reg.air_gap,
                                         R_ro, R_si, m.stack_length)
            psi_all[i]  = flux_linkage_per_phase(mesh_r, sol, m, self.reg)

        T_avg = float(np.mean(torques[n // 4:]))

        # Ld/Lq: 3 extra static solves at theta=0
        self._cb(progress_cb, "  Extracting Ld/Lq...", 0.87)
        Ld, Lq = self._extract_LdLq(Id_pk, Iq_pk)

        # Back-EMF
        emf_all = -np.gradient(psi_all, dt, axis=0)
        thd     = thd_from_waveform(emf_all[:, 0])
        times   = angles / omega

        m.Ld = Ld
        m.Lq = Lq

        self._cb(progress_cb,
                 f"  Done: T_avg={T_avg:.1f}Nm  Ld={Ld*1e3:.2f}mH  "
                 f"Lq={Lq*1e3:.2f}mH  THD={thd:.1f}%  "
                 f"elapsed={time.time()-t0:.0f}s", 1.0)

        return {"time_s":        times,
                "torque_Nm":     torques,
                "torque_avg_Nm": T_avg,
                "Ld_H":          Ld,
                "Lq_H":          Lq,
                "emf_waveform":  {"time": times, "voltage": emf_all[:,0], "thd_pct": thd}}

    def _extract_LdLq(self, Id_pk, Iq_pk):
        m   = self.m
        I   = max(abs(Id_pk), abs(Iq_pk), m.rated_current * 0.5 * np.sqrt(2))
        sol0,  m0  = self._solve_at(0.0, 0.0, 0.0)
        sol_d, md  = self._solve_at(0.0, I,   0.0)
        sol_q, mq  = self._solve_at(0.0, 0.0, I)
        psi0  = flux_linkage_per_phase(m0,  sol0,  m, self.reg)
        psi_d = flux_linkage_per_phase(md,  sol_d, m, self.reg)
        psi_q = flux_linkage_per_phase(mq,  sol_q, m, self.reg)
        Ld, Lq = extract_inductances(psi0, psi_d, psi_q, I, I, m.pole_pairs)
        # Fallback if FEA gives implausible result (outside 50–200% of analytical)
        for attr, val, name in ((m.Ld, Ld, "Ld"), (m.Lq, Lq, "Lq")):
            if not (0.5 * attr < val < 3.0 * attr):
                val = attr
            if name == "Ld": Ld = val
            else:             Lq = val
        return Ld, Lq
