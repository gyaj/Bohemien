"""
Structured Polar Mesh Generator for 2D Motor FEA.

Generates a triangular mesh directly from Bohemien_Motor_Designer geometry objects —
no GMSH or external meshing tool required. The mesh is a polar structured
grid (r, theta) subdivided into triangles, with element regions determined
analytically from motor geometry.

Mesh structure (radial layers, from shaft outward):
  0. Shaft fill         (r < R_ri)       — Dirichlet A=0, not solved
  1. Rotor iron         (R_ri .. R_r)    — nonlinear BH, PMs embedded
  2. Air gap rotor side (R_r .. R_slide) — linear, mu0
  3. Air gap stator side(R_slide .. R_si)— linear, mu0 [sliding surface here]
  4. Stator slots+teeth (R_si .. R_si+d) — slots: current; teeth: nonlinear
  5. Stator yoke        (R_si+d .. R_so) — nonlinear BH

Region codes (stored per element):
  SHAFT=0, ROTOR_IRON=1, PM=2..2+poles-1,
  AIR_ROTOR=10, AIR_STATOR=11,
  WINDING=20..20+3*Qs-1  (slot_idx*3 + phase),
  TOOTH=30, YOKE=31

The rotor angle is parameterised separately — rotating the rotor simply
shifts the angular coordinate of rotor elements (no re-meshing needed).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Region code constants ────────────────────────────────────────────────────
SHAFT        = 0
ROTOR_IRON   = 1
PM_BASE      = 2      # PM pole k → code PM_BASE + k
AIR_ROTOR    = 10
AIR_STATOR   = 11
TOOTH        = 30
YOKE         = 31
WINDING_BASE = 100    # slot s, phase ph → WINDING_BASE + s*3 + ph


@dataclass
class MotorMesh:
    """
    Complete triangular mesh for a 2D motor cross-section.

    Attributes
    ----------
    nodes   : (N, 2) float array — (x, y) coordinates of all nodes
    elems   : (M, 3) int array  — node indices of each triangle (CCW)
    regions : (M,)  int array   — region code of each element
    r_nodes : (N,)  float array — radius of each node
    theta_nodes : (N,) float  — angle of each node [rad]
    n_r     : number of radial layers
    n_theta : number of circumferential divisions (periodic)
    outer_nodes : indices of nodes on stator OD (Dirichlet A=0)
    shaft_nodes : indices of nodes on shaft bore
    slide_layer : radial layer index of sliding surface
    """
    nodes:       np.ndarray
    elems:       np.ndarray
    regions:     np.ndarray
    r_nodes:     np.ndarray
    theta_nodes: np.ndarray
    n_r:         int
    n_theta:     int
    outer_nodes: np.ndarray
    shaft_nodes: np.ndarray
    slide_layer: int

    # Motor geometry (stored for downstream use)
    R_ri:    float = 0.0
    R_r:     float = 0.0
    R_slide: float = 0.0
    R_si:    float = 0.0
    R_so:    float = 0.0
    poles:   int   = 8
    slots:   int   = 48

    def n_nodes(self) -> int:
        return len(self.nodes)

    def n_elems(self) -> int:
        return len(self.elems)

    def elem_centroids(self) -> np.ndarray:
        """(M, 2) centroid (x, y) of each element."""
        return self.nodes[self.elems].mean(axis=1)

    def elem_r(self) -> np.ndarray:
        """(M,) radial coordinate of each element centroid."""
        c = self.elem_centroids()
        return np.sqrt(c[:, 0]**2 + c[:, 1]**2)

    def elem_theta(self) -> np.ndarray:
        """(M,) angular coordinate of each element centroid in [0, 2pi)."""
        c = self.elem_centroids()
        return np.arctan2(c[:, 1], c[:, 0]) % (2 * np.pi)

    def air_gap_mask(self) -> np.ndarray:
        """Boolean mask: True for elements in either air gap region."""
        return (self.regions == AIR_ROTOR) | (self.regions == AIR_STATOR)

    def rotor_mask(self) -> np.ndarray:
        """Boolean mask: True for all rotor elements (iron + PMs)."""
        return (self.regions == ROTOR_IRON) | (
            (self.regions >= PM_BASE) & (self.regions < PM_BASE + self.poles))

    def winding_mask(self, phase: Optional[int] = None) -> np.ndarray:
        """Boolean mask for winding elements. phase=None → all phases."""
        mask = (self.regions >= WINDING_BASE) & \
               (self.regions < WINDING_BASE + self.slots * 3)
        if phase is not None:
            phase_code = self.regions[mask] % 3
            sub = np.zeros(len(self.regions), dtype=bool)
            idx = np.where(mask)[0]
            sub[idx[phase_code == phase]] = True
            return sub
        return mask


def build_mesh(motor, n_r_rotor=15, n_r_gap=6, n_r_slot=12,
               n_r_yoke=10, n_theta=None) -> MotorMesh:
    """
    Build a structured polar triangular mesh from a PMSM motor object.

    Parameters
    ----------
    motor      : PMSM instance
    n_r_rotor  : radial layers inside rotor
    n_r_gap    : radial layers in each air gap half (×2 total)
    n_r_slot   : radial layers in stator slot region
    n_r_yoke   : radial layers in stator yoke
    n_theta    : circumferential divisions (default: 6×slots)

    Returns
    -------
    MotorMesh
    """
    # ── Geometry ──────────────────────────────────────────────────────────
    stator   = motor.stator
    rg       = motor.rotor_geo
    R_ri     = motor.rotor_inner_radius
    R_r      = motor.rotor_outer_radius
    R_si     = stator.inner_radius
    R_so     = stator.outer_radius
    R_slide  = (R_r + R_si) / 2.0
    slot_d   = stator.slot_profile.depth()
    R_slot_b = R_si + slot_d          # bottom of slot / top of yoke

    poles    = motor.poles
    slots    = motor.slots
    phases   = motor.phases

    if n_theta is None:
        n_theta = 6 * slots             # ensures slot opening is resolved

    # ── Radial layer boundaries ───────────────────────────────────────────
    r_layers = np.concatenate([
        np.linspace(R_ri,    R_r,      n_r_rotor + 1),
        np.linspace(R_r,     R_slide,  n_r_gap   + 1)[1:],
        np.linspace(R_slide, R_si,     n_r_gap   + 1)[1:],
        np.linspace(R_si,    R_slot_b, n_r_slot  + 1)[1:],
        np.linspace(R_slot_b, R_so,    n_r_yoke  + 1)[1:],
    ])
    # Index of sliding surface layer
    slide_layer = n_r_rotor + n_r_gap

    n_r    = len(r_layers)         # number of radial levels
    n_theta_pts = n_theta          # circumferential divisions (periodic)

    # ── Node coordinates (polar → Cartesian) ──────────────────────────────
    # theta: 0 .. 2pi with n_theta points (periodic, so last = first)
    theta_arr = np.linspace(0, 2 * np.pi, n_theta_pts + 1)[:-1]  # [0, 2pi)

    # Node array shape: (n_r, n_theta) → (n_r * n_theta, 2)
    R_grid, Th_grid = np.meshgrid(r_layers, theta_arr, indexing="ij")
    X = R_grid * np.cos(Th_grid)
    Y = R_grid * np.sin(Th_grid)

    nodes       = np.column_stack([X.ravel(), Y.ravel()])
    r_nodes     = R_grid.ravel()
    theta_nodes = Th_grid.ravel()

    N       = len(nodes)          # total nodes
    n_idx   = lambda ir, it: ir * n_theta_pts + (it % n_theta_pts)

    # ── Element connectivity (2 triangles per quad, CCW orientation) ───────
    elem_list = []
    for ir in range(n_r - 1):
        for it in range(n_theta_pts):
            i00 = n_idx(ir,     it)
            i01 = n_idx(ir,     it + 1)
            i10 = n_idx(ir + 1, it)
            i11 = n_idx(ir + 1, it + 1)
            # Lower triangle: (i00, i10, i11)
            elem_list.append([i00, i10, i11])
            # Upper triangle: (i00, i11, i01)
            elem_list.append([i00, i11, i01])

    elems = np.array(elem_list, dtype=np.int32)

    # ── Region assignment ─────────────────────────────────────────────────
    # Compute centroid (r, theta) for each element
    c_xy    = nodes[elems].mean(axis=1)             # (M, 2)
    c_r     = np.sqrt(c_xy[:, 0]**2 + c_xy[:, 1]**2)
    c_theta = np.arctan2(c_xy[:, 1], c_xy[:, 0]) % (2 * np.pi)

    regions = np.full(len(elems), SHAFT, dtype=np.int32)

    # Rotor iron and PMs
    mask_rotor = (c_r >= R_ri) & (c_r < R_r)
    regions[mask_rotor] = ROTOR_IRON

    # PMs: assign based on pole sector and magnet arc fraction
    import math
    alpha_p  = getattr(rg, "magnet_width_fraction", 0.85)
    half_pm  = alpha_p * math.pi / poles         # half-angle of PM arc [rad]
    t_m      = getattr(rg, "magnet_thickness", 0.005)
    R_m_inner = R_r - t_m

    for pole in range(poles):
        pole_centre = pole * 2 * math.pi / poles
        ang_lo = pole_centre - half_pm
        ang_hi = pole_centre + half_pm

        # Angular distance from pole centre (wrapped)
        d_ang = np.abs(((c_theta - pole_centre + math.pi) % (2 * math.pi)) - math.pi)
        pm_mask = mask_rotor & (c_r >= R_m_inner) & (d_ang <= half_pm)
        regions[pm_mask] = PM_BASE + pole

    # Air gap
    regions[(c_r >= R_r)   & (c_r < R_slide)] = AIR_ROTOR
    regions[(c_r >= R_slide) & (c_r < R_si)]  = AIR_STATOR

    # Stator region
    mask_stator_full = c_r >= R_si

    # Yoke (above slot bottom)
    regions[mask_stator_full & (c_r >= R_slot_b)] = YOKE

    # Slot region: assign each element to slot or tooth
    mask_slot_region = mask_stator_full & (c_r < R_slot_b)

    # Slot geometry
    sp          = stator.slot_profile
    slot_width  = sp.area() / (sp.depth() + 1e-9)   # mean slot width [m]
    slot_open   = sp.opening_width()
    slot_pitch  = 2 * math.pi / slots

    # Winding phase/direction from WindingLayout
    coil_table = {}     # (slot_idx, layer) -> (phase, direction)
    if motor.winding:
        for cs in motor.winding._table:
            coil_table[(cs.slot_idx, cs.layer)] = (cs.phase, cs.direction)

    for elem_idx in np.where(mask_slot_region)[0]:
        th = c_theta[elem_idx]
        r  = c_r[elem_idx]

        # Which slot sector?
        slot_idx  = int(th / slot_pitch) % slots
        slot_ctr  = (slot_idx + 0.5) * slot_pitch
        dth       = abs(((th - slot_ctr + math.pi) % (2 * math.pi)) - math.pi)

        # Angular half-width of slot at this radius
        half_sw_ang = math.asin(min(slot_width / (2 * r + 1e-9), 0.999))

        # Wedge (opening) zone at bore
        depth_frac  = (r - R_si) / (slot_d + 1e-9)   # 0 at bore, 1 at bottom

        if dth < half_sw_ang:
            # Inside slot body — assign layer by radial depth
            layer = 0 if depth_frac < 0.5 else 1
            ph, _ = coil_table.get((slot_idx, layer), (0, 1))
            regions[elem_idx] = WINDING_BASE + slot_idx * 3 + ph
        else:
            regions[elem_idx] = TOOTH

    # ── Boundary node sets ─────────────────────────────────────────────────
    outer_nodes = np.where(np.abs(r_nodes - R_so) < R_so * 1e-6)[0]
    shaft_nodes = np.where(np.abs(r_nodes - R_ri) < R_ri * 1e-6)[0]

    return MotorMesh(
        nodes=nodes,
        elems=elems,
        regions=regions,
        r_nodes=r_nodes,
        theta_nodes=theta_nodes,
        n_r=n_r,
        n_theta=n_theta_pts,
        outer_nodes=outer_nodes,
        shaft_nodes=shaft_nodes,
        slide_layer=slide_layer,
        R_ri=R_ri, R_r=R_r, R_slide=R_slide, R_si=R_si, R_so=R_so,
        poles=poles, slots=slots,
    )


def mesh_summary(mesh: MotorMesh) -> str:
    reg_counts = {
        "Shaft":       np.sum(mesh.regions == SHAFT),
        "Rotor iron":  np.sum(mesh.regions == ROTOR_IRON),
        "PM total":    np.sum((mesh.regions >= PM_BASE) &
                              (mesh.regions < PM_BASE + mesh.poles)),
        "Air gap":     np.sum(mesh.air_gap_mask()),
        "Winding":     np.sum(mesh.winding_mask()),
        "Tooth":       np.sum(mesh.regions == TOOTH),
        "Yoke":        np.sum(mesh.regions == YOKE),
    }
    lines = [
        f"MotorMesh: {mesh.n_nodes():,} nodes  {mesh.n_elems():,} elements",
        f"  Radial layers   : {mesh.n_r}",
        f"  Theta divisions : {mesh.n_theta}",
        f"  Outer BC nodes  : {len(mesh.outer_nodes)}",
    ]
    for name, count in reg_counts.items():
        lines.append(f"  {name:15s} : {count:6d} elems")
    return "\n".join(lines)
