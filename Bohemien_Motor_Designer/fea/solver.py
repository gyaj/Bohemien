"""
2D Magnetostatic FEM Solver (A-formulation).

Solves:   nabla . (nu * nabla A_z) = -J_z

where:
  A_z   = magnetic vector potential [Wb/m]
  nu    = reluctivity = 1/(mu0 * mu_r(B)) [m/H]
  J_z   = imposed current density [A/m^2]

Supports:
  - Linear materials (constant mu_r)
  - Nonlinear BH materials via Newton-Raphson iteration
  - Permanent magnets via equivalent current density
  - Dirichlet boundary conditions (A_z = 0 at outer boundary)

Implementation
--------------
Uses standard P1 (linear) triangular elements.
Assembly: element stiffness matrix K_e (3x3) + load vector f_e (3).
Global system: K * A = f  (sparse CSR, solved with scipy spsolve).

Nonlinear iteration (Newton-Raphson):
  K(A^n) * dA = -R(A^n)   where R = K(A^n)*A^n - f
  A^{n+1} = A^n + dA
  Converges when ||R|| / ||f|| < tol (typically 1e-6 in 5-15 iterations).

References
----------
  Sadiku, "Numerical Techniques in Electromagnetics", 3rd ed.
  Meeker, FEMM 4.2 theory manual.
  Jin, "The Finite Element Method in Electromagnetics", 3rd ed.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass
from typing import Dict, Callable, Optional

from Bohemien_Motor_Designer.fea.mesh_reader import MeshData


MU0 = 4e-7 * np.pi


@dataclass
class MaterialSpec:
    """
    Material definition for one physical surface tag.

    For nonlinear materials supply a nu_func; for linear supply mu_r only.
    For permanent magnets supply Mx, My (magnetisation vector [A/m]).
    """
    tag:      int
    mu_r:     float  = 1000.0
    # Optional nonlinear reluctivity: nu_func(B_sq) -> (nu, d_nu/d(B^2))
    # B_sq = |B|^2 = |nabla A|^2
    nu_func:  Optional[Callable] = None
    # PM magnetisation vector [A/m]  (0 if not a magnet)
    Mx:       float = 0.0
    My:       float = 0.0
    # Imposed current density [A/m^2]
    J_z:      float = 0.0

    @property
    def nu_linear(self) -> float:
        return 1.0 / (MU0 * self.mu_r)


@dataclass
class SolveResult:
    A_z:     np.ndarray   # (N,) magnetic vector potential per node [Wb/m]
    B_x:     np.ndarray   # (E,) Bx per element centroid [T]
    B_y:     np.ndarray   # (E,) By per element centroid [T]
    B_mag:   np.ndarray   # (E,) |B| per element centroid [T]
    n_iter:  int          # Newton iterations taken
    residual: float       # final relative residual


class FEMSolver:
    """
    2D Magnetostatic FEM solver.

    Parameters
    ----------
    mesh      : MeshData from mesh_reader.read_msh()
    materials : dict  {tag: MaterialSpec}
    outer_bc_tag : physical line tag where A_z = 0 (Dirichlet)
    """

    def __init__(self, mesh: MeshData,
                 materials: Dict[int, MaterialSpec],
                 outer_bc_tag: int = 100):
        self.mesh     = mesh
        self.mats     = materials
        self.bc_tag   = outer_bc_tag

        # Pre-compute element geometry (constant for a given mesh position)
        self._precompute_geometry()

    # ── Geometry pre-computation ──────────────────────────────────────────

    def _precompute_geometry(self):
        """
        Compute and cache per-element geometric quantities.
        These are cheap (O(E)) and only need recomputing when nodes move
        (i.e. once per rotor position).
        """
        mesh  = self.mesh
        xy    = mesh.nodes
        tri   = mesh.tri
        E     = len(tri)

        # Node coordinates for each element: (E,3,2)
        x = xy[tri, 0]   # (E,3)
        y = xy[tri, 1]   # (E,3)

        # Element area: A = 0.5 * |det([x1-x0, y1-y0; x2-x0, y2-y0])|
        self._area = 0.5 * np.abs(
            (x[:,1] - x[:,0]) * (y[:,2] - y[:,0])
          - (x[:,2] - x[:,0]) * (y[:,1] - y[:,0])
        )   # (E,)

        # Shape function gradients: b_i = (y_j - y_k)/2A, c_i = (x_k - x_j)/2A
        # Stored as (E, 3) each
        inv2A = 1.0 / (2.0 * self._area + 1e-30)
        self._b = np.empty((E, 3))
        self._c = np.empty((E, 3))
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            self._b[:, i] = (y[:, j] - y[:, k]) * inv2A
            self._c[:, i] = (x[:, k] - x[:, j]) * inv2A

        # Element centroids
        self._cx = x.mean(axis=1)   # (E,)
        self._cy = y.mean(axis=1)

        # Material tag per element (0 if tag not in materials dict)
        self._etag = mesh.tri_tags   # (E,)

    # ── Assembly ──────────────────────────────────────────────────────────

    def _assemble(self, A_z: np.ndarray):
        """
        Assemble global stiffness matrix K and load vector f.
        A_z is the current solution vector (used for nonlinear nu evaluation).

        Returns K (sparse CSR), f (dense).
        """
        mesh  = self.mesh
        N     = mesh.n_nodes
        tri   = mesh.tri
        E     = len(tri)

        # Compute B^2 at each element from current A_z
        # B = nabla A_z  (in 2D, only x,y components)
        # Bx = sum_i(b_i * A_i),  By = sum_i(c_i * A_i)
        Az_elem = A_z[tri]   # (E, 3)
        Bx = np.einsum('ei,ei->e', self._b, Az_elem)   # (E,)
        By = np.einsum('ei,ei->e', self._c, Az_elem)   # (E,)
        B2 = Bx**2 + By**2                              # (E,)

        # Reluctivity per element
        nu, dnu_dB2 = self._get_nu(B2)   # (E,), (E,)

        # Element stiffness matrices: K_e[i,j] = nu * area * (b_i*b_j + c_i*c_j)
        # Shape: (E, 3, 3)
        b = self._b   # (E, 3)
        c = self._c
        a = self._area

        # (E,3,3): outer product b_i*b_j + c_i*c_j, scaled by nu*area
        # Standard P1 stiffness: K_e_ij = nu * A_e * (b_i b_j + c_i c_j)
        bb = np.einsum('ei,ej->eij', b, b)
        cc = np.einsum('ei,ej->eij', c, c)
        Ke = (nu * a)[:, None, None] * (bb + cc)   # (E,3,3)

        # Nonlinear correction (Newton tangent): extra term from d(nu)/d(B^2)
        # Tangent K_t = K_secant + dnu/dB^2 * A_e * (B . grad_N_i)(B . grad_N_j) * 2
        # grad_N_i dot B = b_i*Bx + c_i*By
        gNB = b * Bx[:,None] + c * By[:,None]   # (E,3)  grad(N_i) . B
        Kt_extra = (2.0 * dnu_dB2 * a)[:, None, None] * np.einsum('ei,ej->eij', gNB, gNB)
        Ke += Kt_extra

        # Load vector: PM contribution + imposed J
        # f_e_i = area/3 * J_z  (constant J over element, linear shape functions)
        fe = np.zeros((E, 3))

        # PM load:  f_pm_i = area * (Mx * c_i - My * b_i)
        # (comes from integration by parts of curl(M) source term)
        Mx_e = np.zeros(E)
        My_e = np.zeros(E)
        Jz_e = np.zeros(E)

        for tag, mat in self.mats.items():
            mask = (self._etag == tag)
            if mat.Mx != 0.0:
                Mx_e[mask] = mat.Mx
            if mat.My != 0.0:
                My_e[mask] = mat.My
            if mat.J_z != 0.0:
                Jz_e[mask] = mat.J_z

        # PM: f_i = A_e * (Mx * c_i - My * b_i)
        fe += (a * Mx_e)[:, None] * c - (a * My_e)[:, None] * b
        # J_z: f_i = J_z * A_e / 3
        fe += (Jz_e * a / 3.0)[:, None]

        # Assemble into global K (COO -> CSR) and f
        rows = np.repeat(tri[:, :, None], 3, axis=2).reshape(-1)   # (E*9,)
        cols = np.repeat(tri[:, None, :], 3, axis=1).reshape(-1)
        vals = Ke.reshape(-1)

        K = csr_matrix((vals, (rows, cols)), shape=(N, N))
        f = np.zeros(N)
        for i in range(3):
            np.add.at(f, tri[:, i], fe[:, i])

        return K, f

    def _get_nu(self, B2: np.ndarray):
        """
        Compute reluctivity nu and d(nu)/d(B^2) for each element.
        Uses nonlinear BH curves where provided, linear otherwise.
        Returns (nu, dnu_dB2) both (E,) arrays.
        """
        E   = len(B2)
        nu  = np.empty(E)
        dnu = np.zeros(E)

        # Default: air
        nu[:] = 1.0 / MU0

        for tag, mat in self.mats.items():
            mask = (self._etag == tag)
            if not np.any(mask):
                continue

            if mat.nu_func is not None:
                nu_m, dnu_m = mat.nu_func(B2[mask])
                nu[mask]  = nu_m
                dnu[mask] = dnu_m
            else:
                nu[mask] = mat.nu_linear

        return nu, dnu

    # ── Boundary conditions ────────────────────────────────────────────────

    def _apply_dirichlet(self, K: csr_matrix, f: np.ndarray, bc_nodes: np.ndarray):
        """
        Apply A_z = 0 Dirichlet BC.

        For zero-value Dirichlet we only need to zero the row and set
        diagonal to 1.  Column zeroing is NOT needed for zero BC values
        and can cause numerical issues with sparse solvers.

        Sets K[n, :] = 0, K[n, n] = 1, f[n] = 0 for each BC node n.
        """
        if len(bc_nodes) == 0:
            return K, f

        # Convert to LIL for row assignment, then back to CSR
        K_lil = K.tolil()
        for n in bc_nodes:
            K_lil.rows[n] = [n]
            K_lil.data[n] = [1.0]
            f[n] = 0.0
        return K_lil.tocsr(), f

    def _get_bc_nodes(self) -> np.ndarray:
        """Return node indices where A_z = 0 (outer boundary)."""
        mesh = self.mesh
        if self.bc_tag not in mesh.btag_map:
            return np.array([], dtype=np.int32)
        edge_indices = mesh.btag_map[self.bc_tag]
        bc_edges = mesh.edge_nodes[edge_indices]
        return np.unique(bc_edges.ravel())

    # ── Newton solver ──────────────────────────────────────────────────────

    def solve(self, max_iter: int = 50, tol: float = 1e-4,
              relax: float = 0.4,
              progress_cb=None) -> SolveResult:
        """
        Solve using a two-phase linear predictor + Picard (secant) approach.

        Newton-Raphson with full tangent stiffness diverges on highly
        nonlinear BH curves.  The secant (Picard) method with under-relaxation
        is more robust at the cost of more iterations.

        Phase 1:  Linear solve with nu = nu_initial (constant).
        Phase 2:  Picard iteration with relax=0.4 under-relaxation.
                  Aitken delta-squared acceleration is applied every 3 steps.
        """
        N        = self.mesh.n_nodes
        bc_nodes = self._get_bc_nodes()

        def cb(msg, frac):
            if progress_cb:
                try:    progress_cb(msg, frac)
                except TypeError: progress_cb(msg)

        # ── Phase 1: Linear predictor ──────────────────────────────────────
        cb("  FEM: linear predictor...", 0.0)
        saved = {}
        for tag, mat in self.mats.items():
            if mat.nu_func is not None:
                saved[tag] = mat.nu_func
                mat.nu_func = None
        A_z = np.zeros(N)
        K, f = self._assemble(A_z)
        K, f_bc = self._apply_dirichlet(K, f.copy(), bc_nodes)
        A_z = spsolve(K, f_bc)
        A_z[bc_nodes] = 0.0
        if not np.isfinite(A_z).all():
            A_z = np.zeros(N)
        for tag, nf in saved.items():
            self.mats[tag].nu_func = nf

        if not saved:
            Az_elem = A_z[self.mesh.tri]
            Bx = np.einsum('ei,ei->e', self._b, Az_elem)
            By = np.einsum('ei,ei->e', self._c, Az_elem)
            return SolveResult(A_z=A_z, B_x=Bx, B_y=By,
                               B_mag=np.sqrt(Bx**2 + By**2),
                               n_iter=1, residual=0.0)

        # ── Phase 2: Picard with under-relaxation ──────────────────────────
        A_prev  = A_z.copy()
        A_pprev = A_z.copy()
        residual = 1.0
        n_iter   = 1
        omega    = relax   # initial relaxation

        with np.errstate(invalid='ignore', divide='ignore'):
          for iteration in range(max_iter):
            # Compute nu at current A_z from nonlinear BH
            Az_elem = A_z[self.mesh.tri]
            Bx_e = np.einsum('ei,ei->e', self._b, Az_elem)
            By_e = np.einsum('ei,ei->e', self._c, Az_elem)
            B2_e = Bx_e**2 + By_e**2
            nu_e, _ = self._get_nu(B2_e)   # secant only (no dnu)

            # Assemble secant stiffness
            bb  = np.einsum('ei,ej->eij', self._b, self._b)
            cc  = np.einsum('ei,ej->eij', self._c, self._c)
            Ke  = (nu_e * self._area)[:, None, None] * (bb + cc)

            # Load vector
            fe    = np.zeros((len(self.mesh.tri), 3))
            Mx_e  = np.zeros(len(self.mesh.tri))
            My_e  = np.zeros_like(Mx_e)
            Jz_e  = np.zeros_like(Mx_e)
            for tag, mat in self.mats.items():
                mask = (self._etag == tag)
                if mat.Mx: Mx_e[mask] = mat.Mx
                if mat.My: My_e[mask] = mat.My
                if mat.J_z: Jz_e[mask] = mat.J_z
            fe += (self._area * Mx_e)[:, None] * self._c \
                - (self._area * My_e)[:, None] * self._b
            fe += (Jz_e * self._area / 3.0)[:, None]

            rows = np.repeat(self.mesh.tri[:,:,None], 3, axis=2).reshape(-1)
            cols = np.repeat(self.mesh.tri[:,None,:], 3, axis=1).reshape(-1)
            K_s  = csr_matrix((Ke.reshape(-1), (rows, cols)),
                               shape=(N, N))
            f_s  = np.zeros(N)
            for i in range(3):
                np.add.at(f_s, self.mesh.tri[:, i], fe[:, i])

            K_s, f_bc = self._apply_dirichlet(K_s, f_s.copy(), bc_nodes)
            A_new = spsolve(K_s, f_bc)
            if not np.isfinite(A_new).all():
                break
            A_new[bc_nodes] = 0.0

            # Aitken delta^2 acceleration every 3 steps
            if iteration > 0 and iteration % 3 == 0:
                d1 = A_z   - A_prev
                d2 = A_new - A_z
                dd = d2 - d1
                dd_norm = np.dot(dd, dd)
                if dd_norm > 1e-30:
                    omega = -omega * np.dot(d1, dd) / dd_norm
                    omega = float(np.clip(omega, 0.1, 0.9))

            # Under-relaxed update
            A_z = A_prev + omega * (A_new - A_prev)
            A_z[bc_nodes] = 0.0

            # Convergence check: relative change in A_z
            dA_norm   = np.linalg.norm(A_z - A_prev)
            Az_norm   = np.linalg.norm(A_z) + 1e-30
            residual  = dA_norm / Az_norm

            cb(f"  FEM Picard {iteration+1:2d}: dA/A={residual:.2e}  "
               f"B_max={np.sqrt(B2_e.max() if np.isfinite(B2_e.max()) else 0):.2f}T  "
               f"w={omega:.2f}",
               0.15 + 0.80 * iteration / max_iter)

            if residual < tol:
                n_iter = iteration + 2
                break

            A_pprev = A_prev.copy()
            A_prev  = A_z.copy()
            n_iter  = iteration + 2

        # ── Final B field ──────────────────────────────────────────────────
        Az_elem = A_z[self.mesh.tri]
        Bx = np.einsum('ei,ei->e', self._b, Az_elem)
        By = np.einsum('ei,ei->e', self._c, Az_elem)

        cb(f"  FEM done: {n_iter} iters  dA/A={residual:.2e}", 1.0)

        return SolveResult(A_z=A_z, B_x=Bx, B_y=By,
                           B_mag=np.sqrt(Bx**2 + By**2),
                           n_iter=n_iter, residual=residual)

    # ── Utility: build nu_func from BH table ──────────────────────────────

    @staticmethod
    def nu_func_from_bh(bh_table, mu_r_initial: float = 5000.0) -> Callable:
        """
        Build a nu_func callable from a LaminationMaterial.bh_table.

        bh_table is list of (B [T], H [A/m]) pairs.
        Returns nu_func(B2_array) -> (nu, dnu_dB2).

        The (B=0, H=0) point is excluded — reluctivity nu = H/B is
        ill-defined at B=0.  For B below the first non-zero tabulated
        value the initial permeability nu_0 = 1/(mu0*mu_r_initial) is used.
        """
        # Skip B=0 entries (H/B is undefined there)
        pairs = [(B, H) for B, H in bh_table if B > 1e-9]
        if not pairs:
            # Fallback: linear material
            nu0 = 1.0 / (MU0 * mu_r_initial)
            def nu_linear(B2):
                return np.full_like(B2, nu0), np.zeros_like(B2)
            return nu_linear

        Bs = np.array([p[0] for p in pairs], dtype=float)
        Hs = np.array([p[1] for p in pairs], dtype=float)
        nu_curve = Hs / Bs   # reluctivity H/B

        # Initial reluctivity for B below first tabulated point
        nu0 = 1.0 / (MU0 * mu_r_initial)

        # Derivative of nu w.r.t. B for Newton tangent
        if len(Bs) > 1:
            dnu_dB = np.gradient(nu_curve, Bs)
        else:
            dnu_dB = np.zeros_like(nu_curve)

        def nu_func(B2: np.ndarray):
            B   = np.sqrt(np.clip(B2, 0.0, None))
            # For B below first table point use initial permeability
            nu       = np.where(B < Bs[0], nu0, np.interp(B, Bs, nu_curve))
            dnuB     = np.where(B < Bs[0], 0.0, np.interp(B, Bs, dnu_dB))
            dnu_dB2  = dnuB / (2.0 * np.maximum(B, 1e-9))
            return nu, dnu_dB2

        return nu_func
