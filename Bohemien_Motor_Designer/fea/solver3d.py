"""
solver3d.py — 3D Nédélec edge-element magnetostatic FEM solver.

Solves:
    curl(ν curl A) = J_s + curl(M)

using first-order Nédélec (Whitney) H(curl) edge elements on linear
tetrahedra.  Tree-cotree gauge enforces uniqueness of A.

Mathematical background
-----------------------
The vector magnetic potential A satisfies curl B = μ J_s + curl M
with B = curl A.  In the weak form with Nédélec test functions W_e:

    ∫_Ω ν (curl A · curl W_e) dV = ∫_Ω (J_s · W_e + ν M · curl W_e) dV

Local element matrix (6×6 per tetrahedron):
    K_ij^e = ν_e V_e (curl W_i · curl W_j)

Whitney edge basis for edge e = (a, b):
    W_e = λ_a ∇λ_b − λ_b ∇λ_a
    curl W_e = 2 ∇λ_a × ∇λ_b

where λ_a, λ_b are P1 barycentric coordinates.

Gauge condition
---------------
Without additional constraints the curl-curl matrix is singular
(gradient fields lie in its null space).  Tree-cotree decomposition
identifies a spanning tree of the mesh graph; tree-edge DOFs are
pinned to zero, removing the null space exactly without altering
the physics solution for the non-tree (cotree) edges.

Boundary conditions
-------------------
Dirichlet:  A_tangential = 0 on outer stator surface
            (implemented by zeroing corresponding edge DOFs)
Neumann:    natural (flux parallel to boundary — automatic)

Post-processing
---------------
    B = curl A  computed from nodal A values via element gradients
    Torque via Maxwell stress tensor on airgap surface
    Flux linkage via coil volume integral
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Optional, Callable

MU0 = 4e-7 * np.pi

# Local edge connectivity for a tetrahedron (6 edges, each = pair of local node indices)
_TET_LOCAL_EDGES = np.array([
    [0, 1], [0, 2], [0, 3],
    [1, 2], [1, 3], [2, 3],
], dtype=np.int32)


# ── Public API ────────────────────────────────────────────────────────────────

def solve_magnetostatic_3d(
        mesh,
        motor,
        rotor_angle:     float = 0.0,
        Id:              float = 0.0,
        Iq:              float = 0.0,
        electrical_angle: float = 0.0,
        progress_cb: Optional[Callable] = None,
) -> np.ndarray:
    """
    Solve 3D magnetostatics and return edge DOF vector A [Wb/m²·m].

    Parameters
    ----------
    mesh          : MotorMesh3D
    motor         : PMSM instance
    rotor_angle   : mechanical rotor position [rad]  (for tag-based rotation)
    Id, Iq        : d/q stator currents [A peak]
    electrical_angle : electrical angle for winding excitation [rad]
    progress_cb   : optional callback(message, fraction)

    Returns
    -------
    A_edge : (n_edges,) float64  — edge DOF values [Wb/m]
    """
    def _log(msg, frac=None):
        if progress_cb:
            progress_cb(msg, frac)

    _log("Pre-computing element geometry...", 0.02)
    geom = _precompute_geometry(mesh)

    _log("Building reluctivity map...", 0.08)
    nu = _build_nu(mesh, motor)

    _log("Assembling curl-curl stiffness matrix K...", 0.12)
    K = _assemble_K(mesh, geom, nu)

    _log("Building PM source vector...", 0.35)
    f_pm = _build_pm_source(mesh, motor, geom, rotor_angle)

    _log("Building winding current source vector...", 0.45)
    f_J = _build_J_source(mesh, motor, geom, Id, Iq, electrical_angle)

    f = f_pm + f_J

    _log("Applying tree-cotree gauge + Dirichlet BC...", 0.50)
    K_bc, f_bc, free_dofs = _apply_boundary_conditions(K, f, mesh)

    _log(f"Solving {len(free_dofs):,}-DOF sparse system...", 0.55)
    A_free = spsolve(K_bc, f_bc)

    if not np.isfinite(A_free).all():
        _log("WARNING: solver returned non-finite values — check BCs", 0.99)
        A_free = np.zeros_like(A_free)

    # Reconstruct full edge DOF vector
    A_edge = np.zeros(mesh.n_edges, dtype=np.float64)
    A_edge[free_dofs] = A_free

    _log(f"Done. |A|_max={np.max(np.abs(A_edge)):.4e}", 1.0)
    return A_edge


# ── Element geometry pre-computation ─────────────────────────────────────────

def _precompute_geometry(mesh) -> dict:
    """
    Pre-compute per-element geometry: volumes, barycentric gradients.

    Returns dict with:
        vol        : (E,)    element volumes [m³]
        grad_lam   : (E,4,3) barycentric coordinate gradients per element
    """
    nodes = mesh.nodes   # (N, 3)
    tets  = mesh.tets    # (E, 4)

    v0 = nodes[tets[:, 0]]   # (E, 3)
    v1 = nodes[tets[:, 1]]
    v2 = nodes[tets[:, 2]]
    v3 = nodes[tets[:, 3]]

    # T[:,  :, k] = v_{k+1} - v_0  (each edge vector is a COLUMN)
    # shape (E, 3, 3): T[e] has columns = edge vectors
    T = np.stack([v1 - v0, v2 - v0, v3 - v0], axis=2)   # (E, 3, 3)

    # Volume = det(T) / 6
    det = np.linalg.det(T)          # (E,)
    vol = det / 6.0                 # positive by construction (mesh3d enforces this)

    # Barycentric gradients from inv(T).
    # For point x: [lambda_1, lambda_2, lambda_3] = inv(T) · (x - v0)
    # So  grad(lambda_{k+1}) = row k of inv(T)  →  G[k, :] = grad(lambda_{k+1})
    # grad(lambda_0) = -sum of grad(lambda_1..3)
    G = np.linalg.inv(T)            # (E, 3, 3), G[e, k, :] = grad(lambda_{k+1})

    grad_lam = np.zeros((len(tets), 4, 3), dtype=np.float64)
    # grad_lam[e, k+1, :] = G[e, k, :]  (row k of G, NOT column k)
    for k in range(3):
        grad_lam[:, k + 1, :] = G[:, k, :]   # G[:,k,:] = all-elements, row k, all coords
    grad_lam[:, 0, :] = -np.sum(grad_lam[:, 1:, :], axis=1)

    return dict(vol=vol, grad_lam=grad_lam)


# ── Stiffness matrix assembly ─────────────────────────────────────────────────

def _assemble_K(mesh, geom, nu: np.ndarray) -> csr_matrix:
    """
    Assemble global curl-curl stiffness matrix.

    K_ij = Σ_e ν_e V_e (curl W_i^e · curl W_j^e)

    Whitney curl:  curl W_e = 2 ∇λ_a × ∇λ_b  for edge e = (a, b)
    """
    E     = mesh.n_tets
    M     = mesh.n_edges
    vol   = geom["vol"]        # (E,)
    grad  = geom["grad_lam"]   # (E, 4, 3)

    # Compute curl of all 6 Whitney basis functions per element: (E, 6, 3)
    # curl W_e = 2 * grad_lam[a] × grad_lam[b]  for edge (a,b)
    le    = _TET_LOCAL_EDGES   # (6, 2)
    curls = np.empty((E, 6, 3), dtype=np.float64)
    for k, (a, b) in enumerate(le):
        # cross product: (E, 3) × (E, 3) → (E, 3)
        curls[:, k, :] = 2.0 * np.cross(grad[:, a, :], grad[:, b, :])

    # Local 6×6 matrix: K_ij^e = nu_e * vol_e * (curl_i · curl_j)
    # curl_dot[e, i, j] = curls[e, i, :] · curls[e, j, :]
    curl_dot = np.einsum('eik,ejk->eij', curls, curls)   # (E, 6, 6)
    coeff    = (nu * vol)[:, None, None]                  # (E, 1, 1)
    K_local  = coeff * curl_dot                           # (E, 6, 6)

    # Assemble into global COO format
    te   = mesh.tet_edges    # (E, 6) global edge indices
    sign = mesh.edge_signs   # (E, 6) ±1 orientation

    # Global indices: gi = te[e, i], gj = te[e, j]
    # Contribution: sign_i * sign_j * K_local[e, i, j]
    rows, cols, vals = [], [], []
    for i in range(6):
        for j in range(6):
            row_idx = te[:, i]
            col_idx = te[:, j]
            val     = sign[:, i] * sign[:, j] * K_local[:, i, j]
            rows.append(row_idx)
            cols.append(col_idx)
            vals.append(val)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)

    K = coo_matrix((vals, (rows, cols)), shape=(M, M)).tocsr()
    return K


# ── PM source term ────────────────────────────────────────────────────────────

def _build_pm_source(mesh, motor, geom, rotor_angle: float) -> np.ndarray:
    """
    PM source vector:  f_i = Σ_e ν_pm V_e (M_e · curl W_i^e)

    M is the PM remanent magnetisation vector (radially outward for N,
    inward for S, in the rotor frame).
    """
    M_edge = mesh.n_edges
    f_pm   = np.zeros(M_edge, dtype=np.float64)

    poles    = motor.poles
    Br       = motor._get_Br() if hasattr(motor, "_get_Br") else 1.2
    mu_r_pm  = getattr(motor, "magnet_mu_r", 1.05)
    nu_pm    = 1.0 / (MU0 * mu_r_pm)

    vol     = geom["vol"]
    grad    = geom["grad_lam"]

    le    = _TET_LOCAL_EDGES
    curls = np.empty((mesh.n_tets, 6, 3), dtype=np.float64)
    for k, (a, b) in enumerate(le):
        curls[:, k, :] = 2.0 * np.cross(grad[:, a, :], grad[:, b, :])

    for pole in range(poles):
        pm_tag = 3 + pole
        idx    = mesh.groups.get(pm_tag, np.array([], dtype=np.int32))
        if len(idx) == 0:
            continue

        # Centroid of each PM element
        cx = mesh.nodes[mesh.tets[idx, :], 0].mean(axis=1)
        cy = mesh.nodes[mesh.tets[idx, :], 1].mean(axis=1)

        # Radial direction at centroid (lab frame)
        th_c = np.arctan2(cy, cx)

        # N / S polarity
        polarity = -1 if pole % 2 == 0 else 1

        # Magnetisation vector: radially outward for N, inward for S
        Mx = polarity * Br * np.cos(th_c)   # (n_pm,)
        My = polarity * Br * np.sin(th_c)
        Mz = np.zeros_like(Mx)
        M_vec = np.stack([Mx, My, Mz], axis=1)   # (n_pm, 3)

        # f_i^e = nu_pm * vol_e * M · curl_W_i
        coeff = nu_pm * vol[idx]   # (n_pm,)
        # M_dot_curl[e, i] = M_vec[e, :] · curls[idx[e], i, :]
        M_dot_curl = np.einsum('ei,eji->ej', M_vec,
                                curls[idx])   # (n_pm, 6)

        contrib = coeff[:, None] * M_dot_curl   # (n_pm, 6)

        # Scatter to global DOFs with sign correction
        te_pm   = mesh.tet_edges[idx]    # (n_pm, 6)
        sign_pm = mesh.edge_signs[idx]   # (n_pm, 6)

        for k in range(6):
            np.add.at(f_pm, te_pm[:, k], sign_pm[:, k] * contrib[:, k])

    return f_pm


# ── Winding current source ────────────────────────────────────────────────────

def _build_J_source(mesh, motor, geom, Id, Iq, electrical_angle) -> np.ndarray:
    """
    Winding current source vector:  f_i = Σ_e J_e (W_i^e · ẑ) V_e

    For the axially-wound conductors in the active stack, the current
    flows in the ±z direction.  The Whitney edge basis projected onto ẑ
    is the z-component of W_e = λ_a ∇λ_b − λ_b ∇λ_a, evaluated at
    the element centroid (consistent approximation).
    """
    M_edge = mesh.n_edges
    f_J    = np.zeros(M_edge, dtype=np.float64)

    if abs(Id) < 1e-12 and abs(Iq) < 1e-12:
        return f_J

    # Phase currents (inverse Park)
    ia = Id * np.cos(electrical_angle)           - Iq * np.sin(electrical_angle)
    ib = Id * np.cos(electrical_angle-2*np.pi/3) - Iq * np.sin(electrical_angle-2*np.pi/3)
    ic = Id * np.cos(electrical_angle+2*np.pi/3) - Iq * np.sin(electrical_angle+2*np.pi/3)
    i_phase = [ia, ib, ic]

    st      = motor.stator
    sp      = st.slot_profile if st else None
    A_slot  = sp.area() if sp else 1e-4
    ff      = getattr(motor, "slot_fill_factor", 0.45)
    N_coil  = getattr(motor, "turns_per_coil", 2)
    n_lay   = motor.winding.layers if motor.winding else 2
    A_layer = A_slot / n_lay
    J_scale = N_coil / (A_layer * ff + 1e-15)

    vol      = geom["vol"]
    grad_lam = geom["grad_lam"]

    # For each winding coil region:
    for coil in mesh.coil_data:
        phase     = coil["phase"]
        direction = coil["direction"]
        idx       = coil["elem_idx"]   # tet indices in this slot+layer
        if len(idx) == 0:
            continue

        J_z = direction * i_phase[phase] * J_scale   # scalar current density

        # Whitney basis z-projection at centroid:
        # W_e · ẑ at centroid ≈ (λ_a ∇λ_b − λ_b ∇λ_a)_z at centroid
        # At centroid: λ_a = λ_b = 1/4 for all nodes, so:
        # W_e · ẑ = (1/4)(∂λ_b/∂z) - (1/4)(∂λ_a/∂z)
        # = (1/4)(grad_lam[b,z] - grad_lam[a,z])
        le = _TET_LOCAL_EDGES
        for k, (a, b) in enumerate(le):
            # W_z contribution at centroid for this tet subset
            w_z = 0.25 * (grad_lam[idx, b, 2] - grad_lam[idx, a, 2])  # (n,)
            contrib = J_z * vol[idx] * w_z  # (n,)

            te_k    = mesh.tet_edges[idx, k]    # (n,)
            sign_k  = mesh.edge_signs[idx, k]   # (n,)
            np.add.at(f_J, te_k, sign_k * contrib)

    return f_J


# ── Reluctivity ────────────────────────────────────────────────────────────────

def _build_nu(mesh, motor) -> np.ndarray:
    """Per-element reluctivity [m/H]."""
    E   = mesh.n_tets
    nu  = np.full(E, 1.0 / MU0, dtype=np.float64)   # air default

    # Stator and rotor iron
    try:
        from Bohemien_Motor_Designer.materials.library import MaterialLibrary
        lib     = MaterialLibrary()
        lam_key = (motor.stator.lamination
                   if motor.stator and hasattr(motor.stator, "lamination")
                   else "M270-35A")
        mat = lib.lamination(lam_key)
        nu_iron = 1.0 / (MU0 * mat.mu_r_initial)
    except Exception:
        nu_iron = 1.0 / (MU0 * 2000)

    nu[mesh.tags == 1] = nu_iron   # stator iron
    nu[mesh.tags == 2] = nu_iron   # rotor iron

    # PM: reluctivity includes magnet recoil
    mu_r_pm = getattr(motor, "magnet_mu_r", 1.05)
    nu_pm   = 1.0 / (MU0 * mu_r_pm)
    for p in range(motor.poles):
        mask = mesh.tags == (3 + p)
        nu[mask] = nu_pm

    return nu


# ── Boundary conditions ────────────────────────────────────────────────────────

def _apply_boundary_conditions(K: csr_matrix, f: np.ndarray, mesh) -> tuple:
    """
    Apply Dirichlet BCs and tree-cotree gauge, return reduced system.

    Constrained DOFs (set to zero):
    1. Outer surface edges (A_tangential = 0)
    2. Tree edges from spanning-tree gauge

    Returns
    -------
    K_red     : reduced csr_matrix
    f_red     : reduced RHS vector
    free_dofs : indices of unconstrained DOFs in original system
    """
    M = mesh.n_edges
    constrained = set(mesh.outer_edges.tolist())

    # ── Tree-cotree gauge ─────────────────────────────────────────────────
    # Build a spanning tree of the node graph using BFS.
    # Each tree edge is fixed to zero, removing the gradient null space.
    tree_edges = _spanning_tree_edges(mesh)
    constrained.update(tree_edges)

    free_dofs = np.array(sorted(set(range(M)) - constrained), dtype=np.int32)
    n_free    = len(free_dofs)

    # Extract submatrix and RHS
    K_red = K[free_dofs][:, free_dofs]
    f_red = f[free_dofs]

    return K_red, f_red, free_dofs


def _spanning_tree_edges(mesh) -> set:
    """
    BFS spanning tree of the node graph.
    Returns the set of global edge indices that form the spanning tree.
    These edges are pinned to zero for the Coulomb gauge.
    """
    N      = mesh.n_nodes
    edges  = mesh.edges   # (M, 2)

    # Build adjacency list: node → [(neighbour, edge_idx)]
    adj = [[] for _ in range(N)]
    for ei, (a, b) in enumerate(edges):
        adj[a].append((b, ei))
        adj[b].append((a, ei))

    visited    = np.zeros(N, dtype=bool)
    tree_edges = set()

    # BFS from node 0
    queue = [0]
    visited[0] = True
    while queue:
        next_queue = []
        for node in queue:
            for nb, ei in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    tree_edges.add(ei)
                    next_queue.append(nb)
        queue = next_queue

    return tree_edges


# ── Post-processing ───────────────────────────────────────────────────────────

def compute_B_field_3d(mesh, A_edge: np.ndarray, geom: dict) -> np.ndarray:
    """
    Compute element-centroid B = curl A [T].

    B = Σ_e A_e * curl W_e   summed over 6 edges of each tet.

    Returns
    -------
    B : (E, 3) float64
    """
    E     = mesh.n_tets
    grad  = geom["grad_lam"]  # (E, 4, 3)
    te    = mesh.tet_edges    # (E, 6)
    sign  = mesh.edge_signs   # (E, 6)

    B = np.zeros((E, 3), dtype=np.float64)
    le = _TET_LOCAL_EDGES
    for k, (a, b) in enumerate(le):
        curl_k  = 2.0 * np.cross(grad[:, a, :], grad[:, b, :])  # (E, 3)
        A_k     = sign[:, k] * A_edge[te[:, k]]                  # (E,)
        B += A_k[:, None] * curl_k

    return B


def compute_torque_3d(mesh, A_edge: np.ndarray, motor, geom: dict) -> float:
    """
    Maxwell stress tensor torque integrated over airgap surface [N·m].

        T = (L_ax / μ₀) ∫_S (Br · Bθ) r dS

    For the 3D case the airgap 'surface' is a set of triangular faces
    and the stack length is already accounted for in the face areas.
    """
    faces = mesh.airgap_faces   # (F, 3)
    if len(faces) == 0:
        return 0.0

    nodes = mesh.nodes

    # B at face centroids — interpolate from element B
    B_elem = compute_B_field_3d(mesh, A_edge, geom)   # (E, 3)

    # Map faces to element centroids
    # For each face find the airgap element that contains it
    # (faster: use the pre-tagged airgap tet centroids directly)
    ag_tag = 3 + motor.poles + 1
    ag_idx = mesh.groups.get(ag_tag, np.array([], dtype=np.int32))
    if len(ag_idx) == 0:
        return 0.0

    # Centroid of each airgap tet
    tets   = mesh.tets
    ag_cx  = nodes[tets[ag_idx, :], 0].mean(axis=1)
    ag_cy  = nodes[tets[ag_idx, :], 1].mean(axis=1)
    ag_r   = np.sqrt(ag_cx**2 + ag_cy**2)
    ag_th  = np.arctan2(ag_cy, ag_cx)

    # For torque use all airgap elements on the midline
    r_slide = mesh.meta["r_slide"]
    near_mid = np.abs(ag_r - r_slide) < (mesh.meta["r_slide"] * 0.15)
    if not np.any(near_mid):
        near_mid = np.ones(len(ag_idx), dtype=bool)

    mid_idx = ag_idx[near_mid]
    B_mid   = B_elem[mid_idx]    # (n_mid, 3)
    cx_mid  = ag_cx[near_mid]
    cy_mid  = ag_cy[near_mid]
    r_mid   = ag_r[near_mid]
    th_mid  = ag_th[near_mid]

    # Cylindrical components
    Br   =  B_mid[:, 0] * np.cos(th_mid) + B_mid[:, 1] * np.sin(th_mid)
    Bth  = -B_mid[:, 0] * np.sin(th_mid) + B_mid[:, 1] * np.cos(th_mid)

    # Element volume (proxy for face area × dr — uses Arkkio variant)
    vol_mid = geom["vol"][mid_idx]
    R_ro    = motor.rotor_outer_radius
    R_si    = motor.stator.inner_radius if motor.stator else R_ro + motor.airgap
    delta_r = R_si - R_ro

    T = -(1.0 / MU0) * (r_slide / (delta_r + 1e-9)) * np.sum(Br * Bth * vol_mid)
    return float(T)


def compute_flux_linkage_3d(mesh, A_edge: np.ndarray, motor, geom: dict) -> dict:
    """
    Per-phase flux linkage [Wb] via coil volume integral.

        ψ_k = N_coil * (1/A_layer) ∫_coil A · ẑ dV

    A · ẑ is the z-component of the vector potential, extracted from
    edge DOFs at element centroids.
    """
    psi = np.zeros(3, dtype=np.float64)
    vol     = geom["vol"]
    grad    = geom["grad_lam"]

    N_coil  = getattr(motor, "turns_per_coil", 2)
    st      = motor.stator
    sp      = st.slot_profile if st else None
    A_slot  = sp.area() if sp else 1e-4
    n_lay   = motor.winding.layers if motor.winding else 2
    A_layer = A_slot / n_lay

    # Centroid value of A_z via edge DOF projection
    # A at centroid: A_z ≈ Σ_e A_e * W_e_z(centroid)
    # W_e_z at centroid = (1/4)(grad_lam[b,z] - grad_lam[a,z])
    le = _TET_LOCAL_EDGES
    te = mesh.tet_edges    # (E, 6)
    sg = mesh.edge_signs   # (E, 6)

    # A_z at centroid per element
    A_z_cent = np.zeros(mesh.n_tets, dtype=np.float64)
    for k, (a, b) in enumerate(le):
        w_z = 0.25 * (grad[:, b, 2] - grad[:, a, 2])   # (E,)
        A_z_cent += sg[:, k] * A_edge[te[:, k]] * w_z

    for coil in mesh.coil_data:
        phase     = coil["phase"]
        direction = coil["direction"]
        idx       = coil["elem_idx"]
        if len(idx) == 0:
            continue

        Az_c   = A_z_cent[idx]
        e_area = vol[idx]
        # Correct formula: psi = N * (1/A_layer) * (1/L) * integral(A_z dV)
        # = N * sum(A_z_cent * dV) / (A_layer * L_stack)
        dpsi   = (N_coil * direction
                  * np.sum(Az_c * e_area)
                  / (A_layer * motor.stack_length + 1e-15))
        psi[phase] += dpsi

    return {"psi_A": float(psi[0]), "psi_B": float(psi[1]), "psi_C": float(psi[2])}
