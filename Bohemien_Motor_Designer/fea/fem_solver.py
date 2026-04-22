"""
2D Magnetostatic FEM Solver.

Solves:   ∇·(ν ∇A_z) = -J_z + ∇×(ν·M)

using linear triangle elements (Galerkin), assembled into a sparse
CSR system and solved with scipy.sparse.linalg.spsolve.

Nonlinearity (iron BH curve) is handled by Newton relaxation:
  at each iteration, reluctivity ν is updated from |B| per element
  using the material BH table, then the system is reassembled and resolved.

PM contribution uses the equivalent magnetisation current formulation:
  f_PM_i = ν_PM · (Br_x·c_i - Br_y·b_i) / 2
where b_i, c_i are the shape function gradient coefficients for node i.

References
----------
Meeker, "Finite Element Method Magnetics", v4.2 manual (2015).
Salon, "Finite Element Analysis of Electrical Machines", Kluwer 1995.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Optional, Callable

from Bohemien_Motor_Designer.fea.fem_mesh import (
    MotorMesh, SHAFT, ROTOR_IRON, PM_BASE, AIR_ROTOR,
    AIR_STATOR, TOOTH, YOKE, WINDING_BASE,
)

MU0 = 4e-7 * math.pi


# ── Material reluctivity helpers ──────────────────────────────────────────────

def _nu_linear(mu_r: float) -> float:
    return 1.0 / (MU0 * mu_r)


def _nu_from_B(B: np.ndarray, bh_table) -> np.ndarray:
    """
    Compute reluctivity ν [1/(H/m)] from |B| [T] array using BH table.
    BH table: list of (B [T], H [A/m]) pairs, monotone.
    ν = H / B  (with safe handling at B→0).
    """
    if bh_table is None or len(bh_table) == 0:
        return np.full_like(B, _nu_linear(1000.0))

    Bs = np.array([row[0] for row in bh_table])
    Hs = np.array([row[1] for row in bh_table])

    H_interp = np.interp(B, Bs, Hs, left=Hs[0], right=Hs[-1])
    # ν = H/B, but at B=0 use initial slope H[1]/B[1]
    nu_init = Hs[1] / (Bs[1] + 1e-12) if len(Bs) > 1 else _nu_linear(5000)
    nu = np.where(B > 1e-6, H_interp / B, nu_init)
    return nu


# ── Triangle FEM helpers ──────────────────────────────────────────────────────

def _triangle_grad_coeffs(xy: np.ndarray):
    """
    Compute gradient coefficients b, c for a linear triangle.

    Parameters
    ----------
    xy : (3, 2) array of node coordinates (CCW)

    Returns
    -------
    b  : (3,) array  — ∂N_i/∂x = b_i / (2·Area)
    c  : (3,) array  — ∂N_i/∂y = c_i / (2·Area)
    area : signed area (positive if CCW)
    """
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]
    b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=np.float64)
    c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=np.float64)
    area = 0.5 * (b[0] * c[1] - b[1] * c[0])
    return b, c, area


def _assemble(mesh: MotorMesh, nu_e: np.ndarray,
              Jz_e: np.ndarray, Brx_e: np.ndarray,
              Bry_e: np.ndarray, nu_pm_e: np.ndarray) -> tuple:
    """
    Assemble global stiffness matrix K and load vector f.

    Uses vectorised element loop for speed:
      K_ij^e = ν_e/Area_e * (b_i·b_j + c_i·c_j) / 4
      f_J^e_i = J_e·Area_e / 3
      f_PM^e_i = ν_PM_e·(Brx_e·c_i - Bry_e·b_i) / 2

    Returns
    -------
    K : sparse CSR matrix (N, N)
    f : dense array (N,)
    """
    nodes  = mesh.nodes
    elems  = mesh.elems
    N      = len(nodes)
    M      = len(elems)

    # Pre-compute all triangle geometry at once
    xy = nodes[elems]                      # (M, 3, 2)
    x  = xy[:, :, 0]                       # (M, 3)
    y  = xy[:, :, 1]                       # (M, 3)

    # b_i = y_j - y_k  (cyclic: (1,2,0) and (2,0,1))
    b = np.stack([y[:, 1] - y[:, 2],
                  y[:, 2] - y[:, 0],
                  y[:, 0] - y[:, 1]], axis=1)   # (M, 3)
    c = np.stack([x[:, 2] - x[:, 1],
                  x[:, 0] - x[:, 2],
                  x[:, 1] - x[:, 0]], axis=1)   # (M, 3)

    # Signed area from b, c of node 0
    area2 = b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]   # (M,)  = 2*Area
    area  = area2 / 2.0

    # K^e_ij = ν_e / (4·Area_e) · (b_i·b_j + c_i·c_j)
    # Assemble 9 entries per element into COO format
    rows_list, cols_list, vals_list = [], [], []
    for i in range(3):
        for j in range(3):
            k_val = (nu_e / (2.0 * np.abs(area2) + 1e-30)) * (
                b[:, i] * b[:, j] + c[:, i] * c[:, j])
            rows_list.append(elems[:, i])
            cols_list.append(elems[:, j])
            vals_list.append(k_val)

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)
    K    = csr_matrix((vals, (rows, cols)), shape=(N, N))

    # Load vector f
    f = np.zeros(N, dtype=np.float64)

    # Current density source: f_i += J_e·|Area_e|/3
    J_contrib = Jz_e * np.abs(area) / 3.0
    for i in range(3):
        np.add.at(f, elems[:, i], J_contrib)

    # PM source: f_i += ν_PM · (Brx·c_i - Bry·b_i) / 2
    pm_mask = nu_pm_e != 0.0
    if pm_mask.any():
        for i in range(3):
            pm_f = nu_pm_e * (Brx_e * c[:, i] - Bry_e * b[:, i]) / 2.0
            np.add.at(f, elems[:, i], pm_f)

    return K, f


def _apply_dirichlet(K, f: np.ndarray,
                     fixed_nodes: np.ndarray,
                     fixed_values: Optional[np.ndarray] = None):
    """
    Apply Dirichlet BC by row zeroing and diagonal = 1.
    Modifies K (as lil_matrix or csr) and f in-place.
    """
    if fixed_values is None:
        fixed_values = np.zeros(len(fixed_nodes))

    # Convert to LIL for efficient row modification
    K_lil = K.tolil()
    for idx, node in enumerate(fixed_nodes):
        K_lil.rows[node] = [node]
        K_lil.data[node] = [1.0]
        f[node]           = fixed_values[idx]

    return K_lil.tocsr(), f


# ── Public solver API ─────────────────────────────────────────────────────────

def solve_magnetostatic(
        mesh:       MotorMesh,
        motor,
        Jz_e:       np.ndarray,
        rotor_angle: float = 0.0,
        bh_table=None,
        mu_r_pm:    float = 1.05,
        Br_mag:     float = 1.2,
        max_iter:   int   = 25,
        tol:        float = 1e-5,
        relax:      float = 0.7,
        progress_cb: Optional[Callable] = None,
) -> np.ndarray:
    """
    Solve the 2D magnetostatic problem for one rotor position.

    Parameters
    ----------
    mesh        : MotorMesh from fem_mesh.build_mesh()
    motor       : PMSM instance (for geometry/material info)
    Jz_e        : (M,) imposed current density per element [A/m²]
    rotor_angle : rotor mechanical angle [rad] for PM magnetisation direction
    bh_table    : list of (B [T], H [A/m]) pairs for iron BH curve
                  None → linear iron with mu_r_initial from motor materials
    mu_r_pm     : relative permeability of PM material
    Br_mag      : remanence [T] of PM material
    max_iter    : Newton iteration limit
    tol         : convergence tolerance on ||ΔA||/||A||
    relax       : Newton relaxation factor (0.5–1.0)
    progress_cb : optional callback(iter, residual)

    Returns
    -------
    A_z : (N,) nodal values of magnetic vector potential [Wb/m]
    """
    poles  = mesh.poles
    nu_pm  = _nu_linear(mu_r_pm)

    # ── PM magnetisation directions ──
    Brx_e = np.zeros(len(mesh.elems))
    Bry_e = np.zeros(len(mesh.elems))
    nu_pm_e = np.zeros(len(mesh.elems))

    for pole in range(poles):
        pm_mask = mesh.regions == PM_BASE + pole
        if not pm_mask.any():
            continue
        # Mechanical angle of pole centre + rotor_angle offset
        pole_centre = pole * 2 * math.pi / poles + rotor_angle
        # Alternating polarity: even poles outward, odd inward
        polarity = 1.0 if pole % 2 == 0 else -1.0
        Brx_e[pm_mask] = polarity * Br_mag * math.cos(pole_centre)
        Bry_e[pm_mask] = polarity * Br_mag * math.sin(pole_centre)
        nu_pm_e[pm_mask] = nu_pm

    # ── Initial reluctivity (linear) ──
    mu_r_init = 5000.0   # initial iron permeability
    nu_iron   = _nu_linear(mu_r_init)

    nu_e = np.full(len(mesh.elems), _nu_linear(1.0))   # air = mu0 permeability

    # Iron regions
    iron_mask = ((mesh.regions == ROTOR_IRON) |
                 (mesh.regions == TOOTH)      |
                 (mesh.regions == YOKE))
    nu_e[iron_mask] = nu_iron

    # PM regions: PM reluctivity
    pm_any = (mesh.regions >= PM_BASE) & (mesh.regions < PM_BASE + poles)
    nu_e[pm_any] = nu_pm

    # ── Newton iteration ──────────────────────────────────────────────────
    A = np.zeros(len(mesh.nodes))

    # All outer boundary nodes → A = 0
    fixed_nodes = mesh.outer_nodes

    for iteration in range(max_iter):
        # Assemble
        K, f = _assemble(mesh, nu_e, Jz_e, Brx_e, Bry_e, nu_pm_e)
        K, f = _apply_dirichlet(K, f, fixed_nodes)

        # Solve
        A_new = spsolve(K, f)

        # Convergence check
        delta    = np.linalg.norm(A_new - A)
        norm_A   = np.linalg.norm(A_new) + 1e-15
        residual = delta / norm_A

        if progress_cb:
            progress_cb(iteration, residual)

        if iteration > 0 and residual < tol:
            A = A_new
            break

        A = relax * A_new + (1 - relax) * A

        # Update reluctivity from |B| if BH table provided
        if bh_table is not None and iron_mask.any():
            Bx_e, By_e = _compute_B_elements(mesh, A)
            B_mag = np.sqrt(Bx_e**2 + By_e**2)
            nu_iron_new = _nu_from_B(B_mag, bh_table)
            nu_e[iron_mask] = nu_iron_new[iron_mask]

    return A


def _compute_B_elements(mesh: MotorMesh, A: np.ndarray) -> tuple:
    """
    Compute B_x, B_y per element from nodal A_z.

    B = curl(A_z ẑ) = (∂A_z/∂y, -∂A_z/∂x)
    For linear triangles: ∂A/∂x = Σ A_i·b_i / (2·Area)

    Returns
    -------
    Bx_e, By_e : (M,) arrays
    """
    nodes = mesh.nodes
    elems = mesh.elems
    xy    = nodes[elems]
    x     = xy[:, :, 0]
    y     = xy[:, :, 1]

    b = np.stack([y[:, 1] - y[:, 2],
                  y[:, 2] - y[:, 0],
                  y[:, 0] - y[:, 1]], axis=1)
    c = np.stack([x[:, 2] - x[:, 1],
                  x[:, 0] - x[:, 2],
                  x[:, 1] - x[:, 0]], axis=1)
    area2 = b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]

    A_e   = A[elems]                       # (M, 3) nodal A values
    dAdx  = (A_e * b).sum(axis=1) / (area2 + 1e-30)
    dAdy  = (A_e * c).sum(axis=1) / (area2 + 1e-30)

    Bx_e  =  dAdy    # ∂A/∂y
    By_e  = -dAdx    # -∂A/∂x
    return Bx_e, By_e


def compute_B_field(mesh: MotorMesh, A: np.ndarray) -> tuple:
    """
    Public: compute (Bx, By, |B|) per element.

    Returns
    -------
    Bx_e, By_e, B_mag : (M,) arrays
    """
    Bx, By = _compute_B_elements(mesh, A)
    return Bx, By, np.sqrt(Bx**2 + By**2)


def build_current_density(mesh: MotorMesh, motor,
                           Id: float, Iq: float,
                           elec_angle: float = 0.0) -> np.ndarray:
    """
    Build per-element current density Jz [A/m²] from dq currents.

    Uses inverse Park transform to get phase currents, then assigns
    current density to each slot element based on winding table.

    Parameters
    ----------
    mesh       : MotorMesh
    motor      : PMSM instance
    Id, Iq     : d-q axis currents [A peak]
    elec_angle : electrical angle [rad] (rotor_angle * pole_pairs)

    Returns
    -------
    Jz_e : (M,) array of current density [A/m²]
    """
    # Inverse Park: i_abc from Id, Iq at electrical angle
    # ia = Id·cos(θ_e) - Iq·sin(θ_e)
    # ib = Id·cos(θ_e - 2π/3) - Iq·sin(θ_e - 2π/3)
    # ic = -(ia + ib)
    angles = elec_angle - np.array([0.0, 2*math.pi/3, 4*math.pi/3])
    I_phase = Id * np.cos(angles) - Iq * np.sin(angles)   # [ia, ib, ic]

    # Current density J = N_coil * I / (slot_area * fill_factor)
    sp       = motor.stator.slot_profile
    slot_A   = sp.area()
    ff       = getattr(motor, "slot_fill_factor", 0.45)
    N_coil   = getattr(motor, "turns_per_coil", 10)

    Jz_e = np.zeros(len(mesh.elems))

    # Get coil direction from winding table
    coil_dir = {}    # (slot_idx, layer) -> direction (+1/-1)
    if motor.winding:
        for cs in motor.winding._table:
            coil_dir[(cs.slot_idx, cs.layer)] = cs.direction

    for s in range(motor.slots):
        for layer in range(2):
            ph  = mesh.regions[mesh.regions == (WINDING_BASE + s * 3)]
            # Find winding region code for this slot
            # codes: WINDING_BASE + s*3 + phase
            for phase in range(3):
                code = WINDING_BASE + s * 3 + phase
                elem_mask = mesh.regions == code
                if not elem_mask.any():
                    continue
                direction = coil_dir.get((s, layer), 1)
                J = direction * N_coil * I_phase[phase] / (slot_A * ff + 1e-9)
                Jz_e[elem_mask] = J
                break   # one phase per slot code

    return Jz_e
