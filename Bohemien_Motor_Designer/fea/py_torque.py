"""
py_torque.py - Post-processing: torque and flux linkage from A_z.

Arkkio torque (IEEE Trans. Mag. 1987):
    T = (L/mu0) * (R_slide/(R_si - R_ro)) * integral_airgap(Br*Bth dA)

Flux linkage:
    psi_k = N_coil * (1/A_coil) * integral_coilside(A_z dA) * L_stack
"""
from __future__ import annotations
import numpy as np
from typing import Optional

MU0 = 4e-7 * np.pi


def compute_torque(mesh, A_z, motor) -> float:
    """
    Arkkio Maxwell stress tensor torque [N*m].

    Parameters
    ----------
    mesh  : MotorMesh
    A_z   : (N,) nodal A_z [Wb/m]
    motor : PMSM instance
    """
    elems = mesh.elems
    nodes = mesh.nodes
    ag    = mesh.airgap_elems
    if len(ag) == 0:
        return 0.0

    # Element geometry (just for airgap elements is sufficient)
    e   = elems[ag]
    x   = nodes[:, 0][e]
    y   = nodes[:, 1][e]
    x0,x1,x2 = x[:,0],x[:,1],x[:,2]
    y0,y1,y2 = y[:,0],y[:,1],y[:,2]
    two_A = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)
    area  = np.abs(two_A)/2
    inv2A = 1.0/(two_A+1e-30)
    dNdx  = np.column_stack([(y1-y2)*inv2A, (y2-y0)*inv2A, (y0-y1)*inv2A])
    dNdy  = np.column_stack([(x2-x1)*inv2A, (x0-x2)*inv2A, (x1-x0)*inv2A])

    Az_e = A_z[e]
    Bx   = np.sum(Az_e * dNdy, axis=1)
    By   = -np.sum(Az_e * dNdx, axis=1)

    cx   = (x0+x1+x2)/3
    cy   = (y0+y1+y2)/3
    th   = np.arctan2(cy, cx)

    Br   =  Bx*np.cos(th) + By*np.sin(th)
    Bth  = -Bx*np.sin(th) + By*np.cos(th)

    R_ro    = motor.rotor_outer_radius
    R_si    = motor.stator.inner_radius if motor.stator else R_ro + motor.airgap
    delta_r = R_si - R_ro

    T = -((motor.stack_length / MU0)
          * (mesh.r_slide / (delta_r + 1e-9))
          * np.sum(Br * Bth * area))

    return float(T)


def compute_flux_linkage(mesh, A_z, motor) -> dict:
    """
    Per-phase flux linkage [Wb].

    psi_k = N_coil * (stack_length/A_coil) * integral(A_z dA)  [sum over coil sides]

    Returns {'psi_A', 'psi_B', 'psi_C'}
    """
    elems  = mesh.elems
    nodes  = mesh.nodes
    psi    = np.zeros(3)

    N_coil = getattr(motor, "turns_per_coil", 10)
    sp     = motor.stator.slot_profile if motor.stator else None
    A_slot = sp.area() if sp else 1e-4
    n_lay  = motor.winding.layers if motor.winding else 2
    # Homogenized conductor model: psi = N*L*(1/A_layer)*integral(A_z dA)
    # A_layer = A_slot/n_layers  (no fill factor — ff is implicit in A_z field)
    A_layer = A_slot / n_lay

    for coil in mesh.coil_data:
        phase  = coil["phase"]
        dirsgn = coil["direction"]
        idx    = coil["elem_idx"]
        if len(idx) == 0:
            continue

        # Centroid A_z = mean of 3 nodal values
        Az_c    = A_z[elems[idx]].mean(axis=1)

        # Element areas
        e = elems[idx]
        x = nodes[:,0][e]; y = nodes[:,1][e]
        two_A = ((x[:,1]-x[:,0])*(y[:,2]-y[:,0]) -
                 (x[:,2]-x[:,0])*(y[:,1]-y[:,0]))
        e_area = np.abs(two_A)/2

        # psi contribution: N * L * (1/A_coil) * sum(Az * dA)
        dpsi = N_coil * motor.stack_length * dirsgn * np.sum(Az_c * e_area) / (A_layer + 1e-15)
        psi[phase] += dpsi

    return {"psi_A": psi[0], "psi_B": psi[1], "psi_C": psi[2]}


def extract_back_emf(psi_history: list, dt: float, motor) -> dict:
    """
    Back-EMF from flux linkage time series.

    Uses FFT on the flux linkage waveform (more robust than differencing)
    to extract the fundamental and harmonics, then multiplies by omega_e.

    Parameters
    ----------
    psi_history : list of dicts {'psi_A','psi_B','psi_C'}
    dt          : time step per electrical-period step [s]

    Returns dict with emf_A, emf_B, emf_C arrays, fundamental_V, thd_pct.
    """
    pA = np.array([p["psi_A"] for p in psi_history])
    pB = np.array([p["psi_B"] for p in psi_history])
    pC = np.array([p["psi_C"] for p in psi_history])

    n   = len(pA)
    # electrical angular frequency
    omega_e = motor.rated_speed * 2 * np.pi / 60 * motor.pole_pairs

    # FFT of psi_A to get harmonic amplitudes then multiply by n*omega_e
    sp_psi = np.abs(np.fft.rfft(pA)) * 2.0 / n
    # EMF harmonic k = omega_e * k * |psi_k|  (since e = -dpsi/dt)
    freqs  = np.fft.rfftfreq(n, d=1.0/n)   # harmonic numbers 0,1,2,...
    emf_sp = np.zeros_like(sp_psi)
    for k, fk in enumerate(freqs):
        if fk > 0:
            emf_sp[k] = sp_psi[k] * omega_e * fk

    # Fundamental = first harmonic
    h1 = float(emf_sp[1]) if len(emf_sp) > 1 else 1.0

    # Time-domain EMF (for waveform display)
    eA = -np.gradient(pA, dt)
    eB = -np.gradient(pB, dt)
    eC = -np.gradient(pC, dt)

    # THD from FFT (exclude DC and triplens)
    harm_sq_sum = sum(
        emf_sp[k]**2
        for k in range(2, len(emf_sp))
        if int(freqs[k]) % 3 != 0 and freqs[k] == int(freqs[k])
    )
    thd = 100.0 * np.sqrt(harm_sq_sum) / (h1 + 1e-9)

    return {
        "emf_A":         eA,
        "emf_B":         eB,
        "emf_C":         eC,
        "fundamental_V": h1,
        "thd_pct":       float(thd),
    }


def extract_Ld_Lq(mesh, motor, solver_fn) -> dict:
    """
    Extract Ld, Lq from two static solves at rated MTPA condition.

    Ld = (psi_d(Id+dI, Iq) - psi_d(Id, Iq)) / dI
    Lq = (psi_q(Id, Iq+dI) - psi_q(Id, Iq)) / dI

    Parameters
    ----------
    solver_fn : callable(rotor_angle, Id, Iq, elec_angle) -> A_z
    """
    try:
        from Bohemien_Motor_Designer.drive.field_weakening import mtpa_currents
        Id0, Iq0 = mtpa_currents(motor)
    except Exception:
        # fallback: pure Iq (SPM MTPA with Id=0)
        I_pk = motor.rated_current_peak() if hasattr(motor, "rated_current_peak") else 50.0
        Id0, Iq0 = 0.0, I_pk

    dI    = max(abs(Iq0) * 0.05, 1.0)   # 5% perturbation
    theta = 0.0                           # rotor at zero

    # Solve at base and perturbed
    A0   = solver_fn(theta, Id0,    Iq0,    theta)
    A_dq = solver_fn(theta, Id0+dI, Iq0,    theta)
    A_qq = solver_fn(theta, Id0,    Iq0+dI, theta)

    def psi_dq(A_z):
        psi = compute_flux_linkage(mesh, A_z, motor)
        psi_a = psi["psi_A"]
        # Park transform at elec_angle=0: psi_d = psi_A, psi_q = -psi_B (approx)
        return psi_a, -psi["psi_B"]

    psi_d0, psi_q0 = psi_dq(A0)
    psi_dd, _      = psi_dq(A_dq)
    _,      psi_qq = psi_dq(A_qq)

    Ld = (psi_dd - psi_d0) / dI
    Lq = (psi_qq - psi_q0) / dI

    return {"Ld_mH": float(Ld * 1e3),
            "Lq_mH": float(Lq * 1e3)}
