"""
Torque and flux linkage extraction from FEM solution.

Two methods:
  1. Arkkio's Maxwell Stress Tensor method -- torque from a circular
     contour in the air gap (the sliding surface).
  2. Flux linkage by coil area integration -- phase voltage and Ld/Lq.

Arkkio method (IEEE Trans. Magnetics, 1987)
-------------------------------------------
The electromagnetic torque per unit axial length is:

  T/L = (1/mu0) * (1/(R2^2 - R1^2))
        * integral_{R1}^{R2} r * Br * Bt * dA

where:
  R1, R2  = inner and outer radii of the air gap annulus
  Br, Bt  = radial and tangential components of B
  dA      = area element

In the FEM implementation this integral is evaluated over all triangular
elements in the air gap region (physical surface tag = reg.air_gap).

The result is multiplied by stack_length to give torque in N.m.

Flux linkage
------------
Phase flux linkage psi_k is the area integral of A_z over the coil cross-
sections, weighted by conductor density (N * kw / slot_area):

  psi_k = (N_series / slot_area) * sum_over_slots(sign_k * integral(A_z dA))

where sign_k = +1 for go conductors, -1 for return conductors of phase k.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional

from Bohemien_Motor_Designer.fea.mesh_reader import MeshData
from Bohemien_Motor_Designer.fea.solver import SolveResult
from Bohemien_Motor_Designer.fea.index_registry import IndexRegistry


MU0 = 4e-7 * np.pi


def arkkio_torque(mesh: MeshData, sol: SolveResult,
                  air_gap_tag: int,
                  R_inner: float, R_outer: float,
                  stack_length: float) -> float:
    """
    Compute electromagnetic torque using Arkkio's method.

    Parameters
    ----------
    mesh        : MeshData
    sol         : SolveResult from FEMSolver.solve()
    air_gap_tag : physical surface tag of the air gap region
    R_inner     : inner radius of air gap annulus [m]  (rotor OD)
    R_outer     : outer radius of air gap annulus [m]  (stator bore)
    stack_length: axial length of motor [m]

    Returns
    -------
    Torque in N.m (positive = CCW).
    """
    if air_gap_tag not in mesh.tag_map:
        return 0.0

    elem_indices = np.array(mesh.tag_map[air_gap_tag], dtype=np.int32)
    if len(elem_indices) == 0:
        return 0.0

    tri  = mesh.tri[elem_indices]
    area = _element_areas(mesh.nodes, tri)
    cx   = mesh.nodes[tri, 0].mean(axis=1)   # centroid x
    cy   = mesh.nodes[tri, 1].mean(axis=1)   # centroid y
    r    = np.sqrt(cx**2 + cy**2)
    cos_t = cx / (r + 1e-30)
    sin_t = cy / (r + 1e-30)

    # B field at centroids (from SolveResult element arrays)
    # SolveResult B arrays are indexed over ALL elements; need to select by elem_indices
    Bx = sol.B_x[elem_indices]
    By = sol.B_y[elem_indices]

    # Convert to radial / tangential components
    Br = Bx * cos_t + By * sin_t
    Bt = -Bx * sin_t + By * cos_t

    # Arkkio integrand: r * Br * Bt per element, weighted by area
    integrand = r * Br * Bt * area

    # Normalization factor
    # T/L = 1/mu0 * 1/(R2^2 - R1^2) * integral
    R2sq_minus_R1sq = R_outer**2 - R_inner**2
    T_per_length = (1.0 / MU0) * (1.0 / R2sq_minus_R1sq) * integrand.sum()

    return -T_per_length * stack_length


def flux_linkage_per_phase(mesh: MeshData, sol: SolveResult,
                            motor, registry: IndexRegistry) -> np.ndarray:
    """
    Compute per-phase flux linkage psi [Wb] by area integration of A_z
    over winding slots.

    Parameters
    ----------
    mesh     : MeshData
    sol      : SolveResult
    motor    : PMSM instance (needs winding, turns_per_coil, slot fill factor)
    registry : IndexRegistry

    Returns
    -------
    psi : (3,) array  [phase_A, phase_B, phase_C] flux linkage [Wb]
    """
    winding = motor.winding
    if winding is None:
        return np.zeros(3)

    sp      = motor.stator.slot_profile if motor.stator else None
    slot_A  = sp.area() if sp else 1e-4
    ff      = getattr(motor, "slot_fill_factor", 0.45)
    N_coil  = getattr(motor, "turns_per_coil", 10)
    L_stk   = motor.stack_length
    n_layers = winding.layers if hasattr(winding, "layers") else 2
    # Each layer covers 1/n_layers of the slot area in our structured mesh
    slot_A_per_layer = slot_A / n_layers

    psi = np.zeros(3)
    coil_table = winding._table if hasattr(winding, "_table") else []

    for coil in coil_table:
        slot_idx  = coil.slot_idx
        layer     = coil.layer
        phase     = coil.phase
        direction = coil.direction

        wt = registry.winding_tag(slot_idx, layer)
        if wt not in mesh.tag_map:
            continue

        elem_indices = np.array(mesh.tag_map[wt], dtype=np.int32)
        if len(elem_indices) == 0:
            continue

        tri      = mesh.tri[elem_indices]
        area     = _element_areas(mesh.nodes, tri)
        Az_elem  = sol.A_z[tri].mean(axis=1)
        flux_slot = np.dot(Az_elem, area)   # integral(A_z dA) [Wb/m * m^2 = Wb.m]

        # psi = N_coil * L * (1/A_layer) * integral(A_z dA)
        # Note: ff is NOT in the denominator — it is already accounted for
        # in the FEM source current density J = N*I/(A_layer*ff), so A_z
        # implicitly reflects the actual conductor fill.
        psi[phase] += direction * (N_coil / (slot_A_per_layer + 1e-9)) * flux_slot * L_stk

    return psi


def extract_inductances(psi_no_current: np.ndarray,
                         psi_with_Id: np.ndarray,
                         psi_with_Iq: np.ndarray,
                         Id_applied: float,
                         Iq_applied: float,
                         pole_pairs: int) -> tuple[float, float]:
    """
    Extract Ld and Lq from three static FEM solutions:
      1. No armature current (open circuit) -> psi_no_current
      2. d-axis current Id, Iq=0           -> psi_with_Id
      3. q-axis current Iq, Id=0           -> psi_with_Iq

    Uses:
      Ld = (psi_d_with_Id - psi_d_no_current) / Id
      Lq = (psi_q_with_Iq - psi_q_no_current) / Iq

    All psi arrays are (3,) phase flux linkages.
    d/q projection uses the Park transform at theta=0.
    """
    def park_d(psi_abc):
        # d-axis projection at theta=0: psi_d = (2/3)(psi_a - psi_b/2 - psi_c/2)
        return (2.0/3.0) * (psi_abc[0] - 0.5*psi_abc[1] - 0.5*psi_abc[2])

    def park_q(psi_abc):
        # q-axis projection: psi_q = (2/3)(sqrt(3)/2)*(psi_b - psi_c)
        return (2.0/3.0) * (np.sqrt(3)/2) * (psi_abc[1] - psi_abc[2])

    psi_d0 = park_d(psi_no_current)
    psi_q0 = park_q(psi_no_current)

    psi_d_id = park_d(psi_with_Id)
    psi_q_iq = park_q(psi_with_Iq)

    Ld = (psi_d_id - psi_d0) / (Id_applied + 1e-30)
    Lq = (psi_q_iq - psi_q0) / (Iq_applied + 1e-30)

    return float(Ld), float(Lq)


def back_emf_from_flux_linkage(psi_time_series: np.ndarray,
                                dt: float) -> np.ndarray:
    """
    Compute phase back-EMF from time series of flux linkage.

    psi_time_series : (T, 3) flux linkage at T time instants
    dt              : time step [s]
    Returns         : (T, 3) phase back-EMF [V]
    """
    return -np.gradient(psi_time_series, dt, axis=0)


def thd_from_waveform(voltage: np.ndarray, n_harmonics: int = 20) -> float:
    """
    Compute THD of a periodic waveform, excluding triplen harmonics.

    voltage : 1D array, one full period
    Returns : THD [%]
    """
    N    = len(voltage)
    fft  = np.fft.rfft(voltage)
    amps = np.abs(fft) * 2.0 / N
    fund = amps[1] if len(amps) > 1 else 1.0

    # Exclude dc (0), fundamental (1), and triplen harmonics (3,6,9,...)
    harm_sq = sum(
        amps[k]**2
        for k in range(2, min(n_harmonics + 2, len(amps)))
        if k % 3 != 0
    )
    return 100.0 * np.sqrt(harm_sq) / (fund + 1e-9)


# ── Helper ────────────────────────────────────────────────────────────────────

def _element_areas(nodes: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Compute areas of triangular elements."""
    x = nodes[tri, 0]
    y = nodes[tri, 1]
    return 0.5 * np.abs(
        (x[:,1] - x[:,0]) * (y[:,2] - y[:,0])
      - (x[:,2] - x[:,0]) * (y[:,1] - y[:,0])
    )
