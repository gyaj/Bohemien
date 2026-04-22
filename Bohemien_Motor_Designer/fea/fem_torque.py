"""
Electromagnetic Torque and Flux Linkage Computation.

Arkkio Maxwell Stress Tensor method for torque:
  T = (L_stk / μ₀) * (1/(R_out² - R_in²))
      * ∫∫_airgap  r · B_r · B_θ  dA

where B_r, B_θ are radial and tangential B components in the air gap annulus.

For linear triangles, B is constant per element and the integral reduces to:
  T = (L_stk / μ₀) * (1/(R_out² - R_in²))
      * Σ_e  [ 2·|Area_e| · r_e · B_r_e · B_θ_e ]

Flux linkage per phase:
  Ψ_ph = N_coil · Σ_{slots in phase} ±  ∫∫_slot  A_z  dA / slot_area

The sign follows the coil direction (+1 going, -1 return).

References
----------
Arkkio, "Analysis of Induction Motors Based on the Numerical Solution
of the Magnetic Field and Circuit Equations", Acta Polytechnica Scand. 1987.
"""
from __future__ import annotations
import math
import numpy as np
from Bohemien_Motor_Designer.fea.fem_mesh import (
    MotorMesh, AIR_ROTOR, AIR_STATOR, WINDING_BASE
)

MU0 = 4e-7 * math.pi


def arkkio_torque(mesh: MotorMesh, A: np.ndarray,
                  stack_length: float) -> float:
    """
    Compute electromagnetic torque using the Arkkio method.

    Integrates the Maxwell stress tensor over the air gap annulus
    (both AIR_ROTOR and AIR_STATOR regions).

    Parameters
    ----------
    mesh         : MotorMesh
    A            : (N,) nodal A_z solution [Wb/m]
    stack_length : axial stack length [m]

    Returns
    -------
    T : torque [N·m]  (positive = motoring CCW)
    """
    from Bohemien_Motor_Designer.fea.fem_solver import _compute_B_elements

    nodes = mesh.nodes
    elems = mesh.elems
    regs  = mesh.regions

    # Air gap elements only
    ag_mask = (regs == AIR_ROTOR) | (regs == AIR_STATOR)
    ag_idx  = np.where(ag_mask)[0]

    if len(ag_idx) == 0:
        return 0.0

    # B field per element
    Bx_e, By_e = _compute_B_elements(mesh, A)

    # Element centroids in polar coords
    c_xy    = nodes[elems[ag_idx]].mean(axis=1)
    r_e     = np.sqrt(c_xy[:, 0]**2 + c_xy[:, 1]**2)
    theta_e = np.arctan2(c_xy[:, 1], c_xy[:, 0])

    # Convert B to polar components
    Bx = Bx_e[ag_idx]
    By = By_e[ag_idx]
    cos_t = np.cos(theta_e)
    sin_t = np.sin(theta_e)
    B_r   =  Bx * cos_t + By * sin_t    # radial
    B_t   = -Bx * sin_t + By * cos_t    # tangential

    # Element areas
    xy   = nodes[elems[ag_idx]]
    x    = xy[:, :, 0];  y = xy[:, :, 1]
    b0   = y[:, 1] - y[:, 2]
    c0   = x[:, 2] - x[:, 1]
    b1   = y[:, 2] - y[:, 0]
    c1   = x[:, 0] - x[:, 2]
    area = np.abs(b0 * c1 - b1 * c0) / 2.0

    # Arkkio formula
    R_in  = mesh.R_r
    R_out = mesh.R_si
    denom = R_out**2 - R_in**2

    integrand = 2.0 * area * r_e * B_r * B_t
    T = (stack_length / (MU0 * denom)) * integrand.sum()
    return float(T)


def flux_linkage(mesh: MotorMesh, A: np.ndarray,
                 motor, n_phases: int = 3) -> np.ndarray:
    """
    Compute per-phase flux linkage Ψ [Wb-turns].

    Method: for each slot winding element, integrate A_z over the
    element area, weight by ±N_coil/slot_area, sum over all slots
    belonging to each phase.

    Parameters
    ----------
    mesh     : MotorMesh
    A        : (N,) nodal A_z [Wb/m]
    motor    : PMSM instance
    n_phases : number of phases (default 3)

    Returns
    -------
    psi : (n_phases,) flux linkage [Wb-turns]
    """
    nodes    = mesh.nodes
    elems    = mesh.elems
    regions  = mesh.regions
    N_coil   = getattr(motor, "turns_per_coil", 10)
    sp       = motor.stator.slot_profile
    slot_A   = sp.area()
    slots    = motor.slots
    L_stk    = motor.stack_length

    # Coil direction from winding table
    coil_dir   = {}   # (slot_idx, layer) -> direction
    coil_phase = {}   # slot_idx -> phase (from first layer found)
    if motor.winding:
        for cs in motor.winding._table:
            coil_dir[(cs.slot_idx, cs.layer)] = cs.direction
            if cs.slot_idx not in coil_phase:
                coil_phase[cs.slot_idx] = cs.phase

    psi = np.zeros(n_phases)

    for s in range(slots):
        for phase in range(n_phases):
            code = WINDING_BASE + s * 3 + phase
            emask = regions == code
            if not emask.any():
                continue

            # Integral of A_z over winding elements
            # ∫∫ A dA ≈ Σ_e (A_centroid_e * |Area_e|)
            xy_e  = nodes[elems[emask]]
            A_e   = A[elems[emask]].mean(axis=1)   # centroid A_z per elem

            x     = xy_e[:, :, 0];  y_c = xy_e[:, :, 1]
            b0    = y_c[:, 1] - y_c[:, 2]
            c0    = x[:, 2] - x[:, 1]
            b1    = y_c[:, 2] - y_c[:, 0]
            c1    = x[:, 0] - x[:, 2]
            area  = np.abs(b0 * c1 - b1 * c0) / 2.0

            integral_A = (A_e * area).sum()

            # Direction: take from layer 0 of this slot
            direction = coil_dir.get((s, 0), 1)

            # Flux linkage contribution: N * L * direction * ∫A dA / slot_area
            psi[phase] += N_coil * L_stk * direction * integral_A / slot_A

    return psi


def back_emf_from_flux(psi_history: np.ndarray,
                        angle_history: np.ndarray,
                        omega_e: float) -> dict:
    """
    Compute back-EMF waveform and THD from flux linkage history.

    Parameters
    ----------
    psi_history   : (N_pos, n_phases) flux linkage array
    angle_history : (N_pos,) rotor electrical angle [rad]
    omega_e       : electrical angular velocity [rad/s]

    Returns
    -------
    dict with 'emf', 'thd_pct', 'fundamental_V', 'harmonics'
    """
    # EMF = -dΨ/dt = -ω_e · dΨ/dθ_e
    dtheta = np.gradient(angle_history)
    emf    = np.zeros_like(psi_history)
    for ph in range(psi_history.shape[1]):
        dpsi_dtheta = np.gradient(psi_history[:, ph], angle_history)
        emf[:, ph]  = -omega_e * dpsi_dtheta

    # Use phase A for THD computation
    e_a   = emf[:, 0]
    N     = len(e_a)
    fft_e = np.fft.rfft(e_a)
    amps  = np.abs(fft_e) / (N / 2)
    fund  = amps[1] if len(amps) > 1 else 1.0
    thd   = 100.0 * math.sqrt(sum(a**2 for a in amps[2:])) / (fund + 1e-9)

    return {
        "emf":          emf,        # (N_pos, n_phases)
        "thd_pct":      thd,
        "fundamental_V": fund,
        "harmonics":    amps,
    }


def inductance_from_flux(psi_Iq: np.ndarray, psi_Id: np.ndarray,
                          psi_0:  np.ndarray,
                          Iq_peak: float, Id_peak: float,
                          pole_pairs: int) -> tuple:
    """
    Extract Ld, Lq from flux linkage at three operating points.

    Method: ΔΨ/ΔI at d-axis and q-axis current injection.

      Lq = (Ψ_q(Iq) - Ψ_q(0)) / Iq
      Ld = (Ψ_d(Id) - Ψ_d(0)) / Id

    where Ψ_d, Ψ_q are d/q components of the phase flux linkage vector.

    Parameters
    ----------
    psi_Iq : (n_phases,) flux linkage with Iq only [Wb-turns]
    psi_Id : (n_phases,) flux linkage with Id only
    psi_0  : (n_phases,) flux linkage at no-load (Id=Iq=0)
    Iq_peak, Id_peak : injected peak currents [A]
    pole_pairs : p

    Returns
    -------
    Ld, Lq : inductances [H]
    """
    # Phase A flux linkage (3-phase balanced, phase A is reference)
    # Ψ_a(Iq) = ψ_m·cos(0) - Lq·Iq·sin(0) + ... → simplified:
    # dΨ_a/dIq ≈ Lq  (at zero d-axis angle)
    dpsi_Iq = psi_Iq[0] - psi_0[0]
    dpsi_Id = psi_Id[0] - psi_0[0]

    Lq = abs(dpsi_Iq / (Iq_peak + 1e-9))
    Ld = abs(dpsi_Id / (Id_peak + 1e-9))

    return float(Ld), float(Lq)
