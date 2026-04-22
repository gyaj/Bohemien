"""
py_solver.py - 2D magnetostatic FEM solver for PMSM cross-section.

Solves:  div(nu * grad(A_z)) = -J_z  (Poisson, A_z formulation)

where
  nu  = reluctivity (1/mu) per element
  J_z = winding current density + PM equivalent source

The PM source uses the weak-form curl(M) approach:
  f_i += nu_pm * area_e * (Brx*dNi/dy - Bry*dNi/dx)

Rotor rotation is handled by mesh.update_rotor_tags(angle) which
reclassifies PM/iron elements at each angular position before assembly.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from typing import Optional, Callable

MU0 = 4e-7 * np.pi


# ── Public API ─────────────────────────────────────────────────────────────────

def solve_magnetostatic(
    mesh,
    motor,
    rotor_angle:     float = 0.0,
    Id:              float = 0.0,
    Iq:              float = 0.0,
    electrical_angle: float = 0.0,
    nonlinear:       bool  = False,
    max_iter:        int   = 8,
    tol:             float = 1e-4,
    progress_cb: Optional[Callable] = None,
) -> np.ndarray:
    """
    Solve 2D magnetostatics and return nodal A_z [Wb/m].

    Parameters
    ----------
    mesh            : MotorMesh (will be mutated: update_rotor_tags called)
    motor           : PMSM instance
    rotor_angle     : mechanical rotor position [rad]
    Id, Iq          : d/q currents [A peak]
    electrical_angle: electrical angle for winding excitation [rad]
    nonlinear       : enable Newton-Raphson BH iteration
    """
    # 1. Rotate rotor tags to current position
    mesh.update_rotor_tags(rotor_angle)

    nodes = mesh.nodes
    elems = mesh.elems
    tags  = mesh.tags
    N     = len(nodes)

    geom = _precompute_geometry(nodes, elems)

    # 2. Reluctivity map
    nu = _build_nu(tags, motor, mesh)

    # 3. Winding current density (constant during iteration)
    J = _build_J_winding(mesh, motor, Id, Iq, electrical_angle)

    # 4. PM source (position-dependent, built from updated tags)
    f_pm = _build_pm_source(mesh, motor, geom, rotor_angle)

    # 5. Linear solve (single iteration for SPM - air-gap dominated)
    K = _assemble_K(nodes, elems, geom, nu)
    f = _assemble_f(nodes, elems, geom, J) + f_pm
    K, f = _apply_dirichlet(K, f, mesh.outer_nodes)

    A_z = spsolve(K, f)
    if not np.isfinite(A_z).all():
        A_z = np.zeros(N)

    # 6. Optional Newton-Raphson for nonlinear BH
    if nonlinear:
        bh = _get_bh_table(motor)
        if bh is not None:
            for it in range(max_iter):
                nu_new = _update_nu(nodes, elems, geom, tags, A_z, bh, motor)
                K2 = _assemble_K(nodes, elems, geom, nu_new)
                f2 = _assemble_f(nodes, elems, geom, J) + f_pm
                K2, f2 = _apply_dirichlet(K2, f2, mesh.outer_nodes)
                A_new = spsolve(K2, f2)
                delta = np.max(np.abs(A_new - A_z)) / (np.max(np.abs(A_new)) + 1e-12)
                A_z = A_new
                if progress_cb:
                    progress_cb(f"  NR iter {it+1}: delta={delta:.2e}", 0.5+0.5*(it+1)/max_iter)
                if delta < tol:
                    break

    if progress_cb:
        progress_cb(f"  theta={np.degrees(rotor_angle):.1f}deg  |A|_max={np.max(np.abs(A_z)):.5f}", 1.0)

    return A_z


def compute_B_field(mesh, A_z):
    """Return (Bx, By, B_mag) per element from nodal A_z."""
    geom  = _precompute_geometry(mesh.nodes, mesh.elems)
    Az_e  = A_z[mesh.elems]
    Bx    = np.sum(Az_e * geom["dN_dy"], axis=1)
    By    = -np.sum(Az_e * geom["dN_dx"], axis=1)
    B_mag = np.sqrt(Bx**2 + By**2)
    return Bx, By, B_mag


# ── Geometry pre-computation ───────────────────────────────────────────────────

def _precompute_geometry(nodes, elems):
    x = nodes[:, 0][elems]
    y = nodes[:, 1][elems]
    x0,x1,x2 = x[:,0],x[:,1],x[:,2]
    y0,y1,y2 = y[:,0],y[:,1],y[:,2]
    two_area  = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)
    area      = two_area / 2.0
    inv2A     = 1.0 / (two_area + 1e-30)
    dN_dx = np.column_stack([(y1-y2)*inv2A, (y2-y0)*inv2A, (y0-y1)*inv2A])
    dN_dy = np.column_stack([(x2-x1)*inv2A, (x0-x2)*inv2A, (x1-x0)*inv2A])
    cx = (x0+x1+x2)/3
    cy = (y0+y1+y2)/3
    return dict(area=area, dN_dx=dN_dx, dN_dy=dN_dy, cx=cx, cy=cy)


# ── Stiffness matrix ───────────────────────────────────────────────────────────

def _assemble_K(nodes, elems, geom, nu):
    N     = len(nodes)
    area  = geom["area"]
    dN_dx = geom["dN_dx"]
    dN_dy = geom["dN_dy"]
    coeff = np.abs(area) * nu
    rows, cols, vals = [], [], []
    for i in range(3):
        for j in range(3):
            k_ij = coeff * (dN_dx[:,i]*dN_dx[:,j] + dN_dy[:,i]*dN_dy[:,j])
            rows.append(elems[:,i])
            cols.append(elems[:,j])
            vals.append(k_ij)
    K = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(N, N)).tocsr()
    return K


# ── Load vector ────────────────────────────────────────────────────────────────

def _assemble_f(nodes, elems, geom, J):
    """P1 consistent load: each node gets 1/3 of element contribution."""
    N = len(nodes)
    contrib = np.abs(geom["area"]) * J / 3.0
    f = np.zeros(N)
    for i in range(3):
        np.add.at(f, elems[:,i], contrib)
    return f


# ── PM source term ─────────────────────────────────────────────────────────────

def _build_pm_source(mesh, motor, geom, rotor_angle: float) -> np.ndarray:
    """
    Weak-form PM source:  f_i = nu_pm * area * (Brx*dNi/dy - Bry*dNi/dx)

    Magnetisation direction: radially outward for N poles, inward for S poles,
    oriented at pole-centre angle in the rotor frame.
    """
    N     = len(mesh.nodes)
    f_pm  = np.zeros(N)
    poles = motor.poles
    Br    = getattr(motor, "magnet_Br", 1.22)
    try:
        Br = motor._get_Br()
    except Exception:
        pass
    mu_r_pm = getattr(motor, "magnet_mu_r", 1.05)
    nu_pm   = 1.0 / (MU0 * mu_r_pm)

    area  = geom["area"]
    dN_dx = geom["dN_dx"]
    dN_dy = geom["dN_dy"]
    cx    = geom["cx"]
    cy    = geom["cy"]
    tags  = mesh.tags

    for pole in range(poles):
        pm_tag = 3 + pole
        idx    = mesh.groups.get(pm_tag, np.array([], int))
        if len(idx) == 0:
            continue

        # Pole-centre angle in lab frame = sector_centre + rotor_angle
        pole_centre = (pole + 0.5) * 2*np.pi/poles + rotor_angle
        polarity    = -1 if pole % 2 == 0 else 1   # sign fix: even poles are S, odd are N in our convention

        # Use centroid angle for each element to get local radial direction
        # This gives curved magnet magnetization following pole shape
        th_e  = np.arctan2(cy[idx], cx[idx])
        Brx_e = polarity * Br * np.cos(th_e)
        Bry_e = polarity * Br * np.sin(th_e)

        coeff = nu_pm * np.abs(area[idx])
        for i in range(3):
            contrib = coeff * (Brx_e * dN_dy[idx, i] - Bry_e * dN_dx[idx, i])
            np.add.at(f_pm, mesh.elems[idx, i], contrib)

    return f_pm


# ── Winding current density ────────────────────────────────────────────────────

def _build_J_winding(mesh, motor, Id, Iq, electrical_angle):
    E  = len(mesh.elems)
    J  = np.zeros(E)
    if abs(Id) < 1e-9 and abs(Iq) < 1e-9:
        return J

    ia = Id*np.cos(electrical_angle)         - Iq*np.sin(electrical_angle)
    ib = Id*np.cos(electrical_angle-2*np.pi/3) - Iq*np.sin(electrical_angle-2*np.pi/3)
    ic = Id*np.cos(electrical_angle+2*np.pi/3) - Iq*np.sin(electrical_angle+2*np.pi/3)
    i_phase = [ia, ib, ic]

    sp     = motor.stator.slot_profile if motor.stator else None
    A_slot = sp.area() if sp else 1e-4
    ff     = getattr(motor, "slot_fill_factor", 0.45)
    N_coil = getattr(motor, "turns_per_coil", 10)
    n_lay  = motor.winding.layers if motor.winding else 2
    A_coil = A_slot / n_lay
    J_scale = N_coil / (A_coil * ff + 1e-12)

    for coil in mesh.coil_data:
        J[coil["elem_idx"]] = coil["direction"] * i_phase[coil["phase"]] * J_scale

    return J


# ── Reluctivity ────────────────────────────────────────────────────────────────

def _build_nu(tags, motor, mesh):
    E  = len(tags)
    nu = np.full(E, 1.0/MU0)     # air default

    # Iron
    try:
        from Bohemien_Motor_Designer.materials.library import MaterialLibrary
        lib = MaterialLibrary()
        lam_name = getattr(motor.stator, "lamination", "M270-35A") if motor.stator else "M270-35A"
        mat = lib.lamination(lam_name)
        nu_iron = 1.0 / (MU0 * mat.mu_r_initial)
    except Exception:
        nu_iron = 1.0 / (MU0 * 2000)

    nu[(tags == 1) | (tags == 2)] = nu_iron

    # PM
    mu_r_pm = getattr(motor, "magnet_mu_r", 1.05)
    nu_pm   = 1.0 / (MU0 * mu_r_pm)
    for p in range(motor.poles):
        nu[tags == 3+p] = nu_pm

    return nu


def _get_bh_table(motor):
    try:
        from Bohemien_Motor_Designer.materials.library import MaterialLibrary
        lib = MaterialLibrary()
        lam = getattr(motor.stator, "lamination", "M270-35A") if motor.stator else "M270-35A"
        mat = lib.lamination(lam)
        return list(zip(mat.bh_B, mat.bh_H))
    except Exception:
        return None


def _update_nu(nodes, elems, geom, tags, A_z, bh_table, motor):
    nu = _build_nu(tags, motor, type('M', (), {'stator': motor.stator,
                                                'poles': motor.poles,
                                                'winding': None})())
    # Faster: just re-call _build_nu
    nu = np.full(len(tags), 1.0/MU0)
    mu_r_pm = getattr(motor, "magnet_mu_r", 1.05)
    nu_pm   = 1.0 / (MU0 * mu_r_pm)
    for p in range(motor.poles):
        nu[tags == 3+p] = nu_pm

    B_arr = np.array([row[0] for row in bh_table])
    H_arr = np.array([row[1] for row in bh_table])

    Az_e  = A_z[elems]
    Bx    = np.sum(Az_e * geom["dN_dy"], axis=1)
    By    = -np.sum(Az_e * geom["dN_dx"], axis=1)
    B_mag = np.sqrt(Bx**2 + By**2)

    iron  = (tags == 1) | (tags == 2)
    if np.any(iron):
        Bi = np.clip(B_mag[iron], B_arr[0]+1e-6, B_arr[-1]-1e-6)
        Hi = np.interp(Bi, B_arr, H_arr)
        nu[iron] = np.where(Bi > 1e-6, Hi / Bi, 1.0/(MU0*2000))

    return nu


# ── Dirichlet BC ───────────────────────────────────────────────────────────────

def _apply_dirichlet(K, f, bc_nodes, value=0.0):
    K = K.tolil()
    for n in bc_nodes:
        K[n, :]  = 0.0
        K[:, n]  = 0.0
        K[n, n]  = 1.0
        f[n]     = value
    return K.tocsr(), f
