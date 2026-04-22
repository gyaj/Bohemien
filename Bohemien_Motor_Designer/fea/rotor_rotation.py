"""
Rotor rotation for position sweep.

Instead of a sliding mesh (complex mortar BC), we rotate the rotor node
coordinates for each angular position and re-assemble the FEM system.
This is the standard approach for 2D static solvers.

For a cogging sweep of 31 positions this takes 31 assemblies + solves,
which at ~2-5 seconds each on a 40k-element mesh gives ~1-3 minutes total.

Implementation
--------------
The mesh is split into two zones at assembly time:
  - Stator zone: nodes with radius > R_slide  (never move)
  - Rotor zone:  nodes with radius < R_slide  (rotate by theta)
  - Gap nodes:   nodes exactly on the sliding surface (interpolated)

For simplicity (and because GMSH always creates distinct node sets on each
side of an interface), we rotate all nodes with r < R_slide.

The sliding surface itself has matching nodes on both sides.  We handle this
by identifying pairs of nodes at the same (r, theta) on the sliding surface
and imposing continuity of A_z via a tie constraint.

This is equivalent to the Mortar BC but simpler: for P1 elements on a
circular sliding surface, node positions match exactly after rotation to
the identity position, so we just rotate the inner nodes and re-use the
same node numbering.
"""
from __future__ import annotations
import numpy as np
from typing import Optional

from Bohemien_Motor_Designer.fea.mesh_reader import MeshData


def rotate_rotor_nodes(mesh: MeshData,
                        theta_rad: float,
                        R_slide: float,
                        R_tol:   float = 1e-4) -> MeshData:
    """
    Return a new MeshData with all rotor-side nodes rotated by theta_rad.

    Rotor-side: nodes with radial coordinate < R_slide - R_tol.
    Sliding surface nodes (R_slide - R_tol <= r <= R_slide + R_tol)
    are also rotated (they belong to the rotor side in our convention).

    Parameters
    ----------
    mesh      : original MeshData (not modified)
    theta_rad : rotor angle to rotate to [rad]  (absolute, not delta)
    R_slide   : sliding surface radius [m]
    R_tol     : tolerance for classifying nodes on the sliding surface [m]

    Returns
    -------
    New MeshData with rotated node coordinates.
    """
    xy      = mesh.nodes.copy()
    r       = np.sqrt(xy[:, 0]**2 + xy[:, 1]**2)
    rotor_mask = r < (R_slide + R_tol)

    if theta_rad == 0.0:
        return MeshData(
            nodes      = xy,
            tri        = mesh.tri,
            tri_tags   = mesh.tri_tags,
            edge_nodes = mesh.edge_nodes,
            edge_tags  = mesh.edge_tags,
        )

    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    x_rot = xy[rotor_mask, 0] * cos_t - xy[rotor_mask, 1] * sin_t
    y_rot = xy[rotor_mask, 0] * sin_t + xy[rotor_mask, 1] * cos_t

    xy[rotor_mask, 0] = x_rot
    xy[rotor_mask, 1] = y_rot

    return MeshData(
        nodes      = xy,
        tri        = mesh.tri,
        tri_tags   = mesh.tri_tags,
        edge_nodes = mesh.edge_nodes,
        edge_tags  = mesh.edge_tags,
    )


def sliding_surface_nodes(mesh: MeshData, R_slide: float,
                           R_tol: float = 5e-4) -> np.ndarray:
    """
    Return indices of nodes on the sliding surface.

    Parameters
    ----------
    mesh    : MeshData
    R_slide : expected radius of the sliding surface [m]
    R_tol   : radial tolerance [m]

    Returns
    -------
    1D array of node indices on the sliding surface.
    """
    r = np.sqrt(mesh.nodes[:, 0]**2 + mesh.nodes[:, 1]**2)
    return np.where(np.abs(r - R_slide) < R_tol)[0]


def cogging_angles(motor) -> np.ndarray:
    """
    Compute the canonical set of rotor angles for a cogging torque sweep.

    The cogging period is 2*pi / LCM(slots, poles) mechanical radians.
    We sweep one full period + a few extra points for FFT padding.

    Returns array of mechanical angles [rad].
    """
    from math import gcd
    Qs      = motor.slots
    p2      = motor.poles
    lcm_val = (Qs * p2) // gcd(Qs, p2)
    period  = 2 * np.pi / lcm_val          # one cogging period [rad mech]
    n_pts   = 32                            # points per period (power of 2 for FFT)
    return np.linspace(0.0, period, n_pts, endpoint=False)


def electrical_angles(motor, n_steps: int = 60) -> np.ndarray:
    """
    Compute rotor angles covering one full electrical period.

    Used for the loaded torque sweep (averages to rated torque,
    gives back-EMF waveform).

    Returns array of mechanical angles [rad].
    """
    elec_period_mech = 2 * np.pi / motor.pole_pairs
    return np.linspace(0.0, elec_period_mech, n_steps, endpoint=False)
