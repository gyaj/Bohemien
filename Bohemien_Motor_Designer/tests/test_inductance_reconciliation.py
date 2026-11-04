"""
Inductance reconciliation tests: analytical vs 2D FEM vs 3D FEM.

Run:
    python -m pytest Bohemien_Motor_Designer/tests -v         (from repo root)
    python Bohemien_Motor_Designer/tests/test_inductance_reconciliation.py

Every assertion below encodes a relation that was broken before the
Nov-2026 fixes and is now verified on the reference 8p/48s 30 kW SPM:

  T1  winding bookkeeping  : N_series = coils_per_phase × N_coil / a   (was /2 low)
  T2  analytical breakdown : L_2d = L_ag + L_h + L_slot + L_tt,  L_ew ≈ 5–30 % of total
  T3  2D energy check      : Ld(flux linkage) == Ld(stored energy)      (was ×1/ff off)
  T4  2D SPM symmetry      : Lq/Ld ≈ 1 (0.85–1.05)                      (was 1.6–1.9)
  T5  2D vs analytical     : Ke_fem/Ke_anal ≈ 1 ±3 %;  Ld_2d/L_2d_anal ≈ 1 ±8 %
  T6  3D mesh conformity   : no interior crack faces                    (was 75 % cracked)
  T7  3D energy check      : Ld(flux linkage) == Ld(stored energy)      (gauge invariance)
  T8  3D vs 2D             : Ke_3d/Ke_2d and Ld_3d/Ld_2d within ±4 %    (was >10×)

Total run time ≈ 60 s (the 3D cases dominate).  Mesh densities are the
smallest that give the quoted tolerances; tighten the meshes, not the
tolerances, if you want a stricter test.
"""
from __future__ import annotations
import sys, os, time
import numpy as np

# Make the package importable whether run from repo root or elsewhere
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PKG)
for p in (_ROOT, _PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

_PKGNAME = os.path.basename(_PKG)
import importlib
_core   = importlib.import_module(f"{_PKGNAME}.core.pmsm")
_stat   = importlib.import_module(f"{_PKGNAME}.core.geometry.stator")
_slot   = importlib.import_module(f"{_PKGNAME}.core.geometry.slot_profiles")
_rot    = importlib.import_module(f"{_PKGNAME}.core.geometry.rotor")
_wind   = importlib.import_module(f"{_PKGNAME}.core.geometry.winding")
_mesh2  = importlib.import_module(f"{_PKGNAME}.fea.py_mesh")
_sol2   = importlib.import_module(f"{_PKGNAME}.fea.py_solver")
_tq2    = importlib.import_module(f"{_PKGNAME}.fea.py_torque")
_mesh3  = importlib.import_module(f"{_PKGNAME}.fea.mesh3d")
_sol3   = importlib.import_module(f"{_PKGNAME}.fea.solver3d")
_run3   = importlib.import_module(f"{_PKGNAME}.fea.runner3d")
_run2   = importlib.import_module(f"{_PKGNAME}.fea.py_runner")


# ── Reference motor ──────────────────────────────────────────────────────────

def reference_motor(turns_per_coil: int = 1):
    """8p/48s, 30 kW, 4000 rpm, 400 V SPM.  N_series = 16 × turns_per_coil."""
    slot = _slot.ParallelToothSlot(slot_width=0.0080, slot_depth=0.0240,
                                   slot_opening=0.0030, wedge_height=0.0010)
    stator = _stat.StatorGeometry(outer_radius=0.1125, inner_radius=0.082,
                                  slots=48, slot_profile=slot, lamination="M270-35A")
    rotor_geo = _rot.SPMRotorGeometry(outer_radius=0.081, inner_radius=0.030,
                                      magnet_thickness=0.006, magnet_width_fraction=0.83,
                                      magnet_material="N42SH")
    winding = _wind.WindingLayout(poles=8, slots=48, phases=3, layers=2,
                                  turns_per_coil=turns_per_coil, parallel_paths=1)
    return _core.PMSM(poles=8, slots=48, stator=stator, rotor_geo=rotor_geo,
                      rotor_outer_radius=0.081, rotor_inner_radius=0.030,
                      stack_length=0.130, airgap=0.001,
                      rated_speed=4000.0, rated_power=30000.0,
                      magnet_material="N42SH", magnet_thickness=0.006,
                      magnet_width_fraction=0.83, turns_per_coil=turns_per_coil,
                      conductor_diameter=0.0009, slot_fill_factor=0.48,
                      parallel_paths=1, winding=winding)


def _park(psi, th):
    c = 2.0 / 3.0
    A, B, C = psi["psi_A"], psi["psi_B"], psi["psi_C"]
    pd = c * (A*np.cos(th) + B*np.cos(th - 2*np.pi/3) + C*np.cos(th + 2*np.pi/3))
    pq = c * (-A*np.sin(th) - B*np.sin(th - 2*np.pi/3) - C*np.sin(th + 2*np.pi/3))
    return pd, pq


# ── T1: winding bookkeeping ──────────────────────────────────────────────────

def test_T1_series_turns_match_coil_side_count():
    m = reference_motor(2)
    w = m.winding
    sides = w.coil_sides_for_phase(0)
    # Each coil has 2 sides; the layout lists every side with N_coil turns
    coils_per_phase = len(sides) // 2
    assert coils_per_phase == w.coils_per_phase == 16
    assert w.total_series_turns_per_phase == coils_per_phase * w.turns_per_coil == 32
    assert reference_motor(1).total_series_turns == 16


# ── T2: analytical decomposition ─────────────────────────────────────────────

def test_T2_analytical_breakdown():
    m = reference_motor(1)
    b = m.inductance_breakdown()
    assert abs(b["L_2d"] - (b["L_ag"] + b["L_h"] + b["L_slot"] + b["L_tt"])) < 1e-15
    assert abs(b["L_total"] - (b["L_2d"] + b["L_ew"])) < 1e-15
    assert 0.03 <= b["L_ew_fraction"] <= 0.35, b["L_ew_fraction"]
    assert abs(m.Ld - b["L_total"]) < 1e-12          # property uses the same components
    # Corrected prefactor: (6/π²)·μ0·(N kw)²·τ_p·L/(p·g_c).  Sanity band from
    # independent derivation (Hendershot & Miller / Pyrhönen): 0.055–0.065 mH
    assert 0.055e-3 <= b["L_ag"] <= 0.065e-3, b["L_ag"]


# ── T3/T4/T5: 2D FEM ─────────────────────────────────────────────────────────

def _solve_2d(m, n_ang=12, n_ag=8):
    mesh = _mesh2.build_motor_mesh(m, n_radial_airgap=n_ag, n_angular_per_slot=n_ang)
    Az0  = _sol2.solve_magnetostatic(mesh, m, rotor_angle=0.0, Id=0.0, Iq=0.0, electrical_angle=0.0)
    pd0, pq0 = _park(_tq2.compute_flux_linkage(mesh, Az0, m), 0.0)
    th_e = -np.arctan2(-pq0, pd0)
    Ke   = float(np.hypot(pd0, pq0))
    dI   = 20.0
    Az_d = _sol2.solve_magnetostatic(mesh, m, rotor_angle=0.0, Id=dI, Iq=0.0, electrical_angle=th_e)
    Az_q = _sol2.solve_magnetostatic(mesh, m, rotor_angle=0.0, Id=0.0, Iq=dI, electrical_angle=th_e)
    pdd, _ = _park(_tq2.compute_flux_linkage(mesh, Az_d, m), th_e)
    _, pqq = _park(_tq2.compute_flux_linkage(mesh, Az_q, m), th_e)
    return dict(mesh=mesh, Ke=Ke, Ld=(pdd - Ke)/dI, Lq=pqq/dI, th_e=th_e)


def test_T3_2d_energy_method_matches_flux_linkage():
    m = reference_motor(1)
    mesh = _mesh2.build_motor_mesh(m, n_radial_airgap=4, n_angular_per_slot=8)
    m._get_Br = lambda: 0.0                                   # PM off
    I  = 20.0
    Az = _sol2.solve_magnetostatic(mesh, m, rotor_angle=0.0, Id=I, Iq=0.0, electrical_angle=0.0)
    pd, _ = _park(_tq2.compute_flux_linkage(mesh, Az, m), 0.0)
    geom = _sol2._precompute_geometry(mesh.nodes, mesh.elems)
    nu   = _sol2._build_nu(mesh.tags, m, mesh)
    Bx, By, _ = _sol2.compute_B_field(mesh, Az)
    W  = 0.5 * np.sum(nu * (Bx**2 + By**2) * np.abs(geom["area"])) * m.stack_length
    del m._get_Br
    Ld_fl, Ld_en = pd / I, 4.0 * W / (3.0 * I**2)
    assert abs(Ld_fl / Ld_en - 1.0) < 1e-3, (Ld_fl, Ld_en)


def test_T4_T5_2d_vs_analytical():
    m = reference_motor(1)
    r = _solve_2d(m, n_ang=12, n_ag=8)
    b = m.inductance_breakdown()
    assert 0.85 <= r["Lq"] / r["Ld"] <= 1.05, "SPM must have Lq ≈ Ld (inter-pole gap is air)"
    assert abs(r["Ke"] / m.back_emf_constant() - 1.0) < 0.03, r["Ke"]
    assert abs(r["Ld"] / b["L_2d"] - 1.0) < 0.08, (r["Ld"], b["L_2d"])


def test_T5b_2d_runner_reports_corrections():
    m = reference_motor(1)
    runner = _run2.PythonFEARunner(m, n_radial_airgap=8, n_ang_per_slot=12)
    res = runner.run_loaded(n_steps=4)
    for k in ("L_ew_mH", "Ld_total_mH", "L_2d_anal_mH", "L_total_anal_mH", "conv_ratio"):
        assert k in res, k
    assert res["conv_converged"], res["conv_ratio"]
    assert abs(res["Ld_total_mH"] - (res["Ld_mH"] + res["L_ew_mH"])) < 1e-9


# ── T6/T7/T8: 3D FEM ─────────────────────────────────────────────────────────

def test_T6_3d_mesh_is_conforming():
    m = reference_motor(1)
    mesh = _mesh3.build_motor_mesh_3d(m, n_angular_per_slot=4, n_axial=2,
                                      n_radial_airgap=3, n_end_winding=1)
    t = mesh.tets
    faces = np.sort(np.concatenate([t[:, [0,1,2]], t[:, [0,1,3]], t[:, [0,2,3]], t[:, [1,2,3]]]), axis=1)
    u, cnt = np.unique(faces, axis=0, return_counts=True)
    assert cnt.max() == 2
    r = np.hypot(mesh.nodes[u, 0], mesh.nodes[u, 1]); z = mesh.nodes[u, 2]
    R_so = m.stator.outer_radius; R_sh = m.rotor_inner_radius
    on_bnd = ((r.min(axis=1) > R_so*0.999) | (r.max(axis=1) < R_sh*1.001) |
              np.all(z < mesh.nodes[:, 2].min() + 1e-9, axis=1) |
              np.all(z > mesh.nodes[:, 2].max() - 1e-9, axis=1))
    cracks = int(((cnt == 1) & ~on_bnd).sum())
    assert cracks == 0, f"{cracks} interior crack faces — checkerboard tet split broken"
    vol = _sol3._precompute_geometry(mesh)["vol"]
    assert (vol > 0).all()


def test_T7_T8_3d_energy_check_and_3d_vs_2d():
    m = reference_motor(1)
    r2 = _solve_2d(m, n_ang=8, n_ag=8)
    runner = _run3.FEMRunner3D(m, n_angular_per_slot=8, n_axial=2,
                               n_radial_airgap=8, n_end_winding=0)
    runner.build_mesh()
    r3 = runner.run_static()
    # T7: gauge-invariant energy method agrees with extended-conductor flux linkage
    assert abs(r3["Ld_energy_check"] - 1.0) < 2e-3, r3["Ld_energy_check"]
    # T8: pure extrusion reproduces 2D at the same mesh density
    assert abs(r3["Ke_3d"] / r2["Ke"] - 1.0) < 0.04, (r3["Ke_3d"], r2["Ke"])
    assert abs(r3["Ld_mH"] / (r2["Ld"] * 1e3) - 1.0) < 0.04, (r3["Ld_mH"], r2["Ld"] * 1e3)
    assert abs(r3["Lq_mH"] / (r2["Lq"] * 1e3) - 1.0) < 0.04, (r3["Lq_mH"], r2["Lq"] * 1e3)
    assert r3["end_region_model"].startswith("none")


# ── Script entry point (no pytest needed) ────────────────────────────────────

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn(); print(f"PASS  {name}  ({time.time()-t0:.1f}s)")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1; print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
