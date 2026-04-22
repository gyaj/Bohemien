"""
mesh3d.py — Structured 3D tetrahedral mesh for PMSM cross-section.

Extrudes the 2D polar cross-section axially into a 3D volume mesh of
linear tetrahedra (P1 nodal / Nédélec edge elements).

Coordinate system
-----------------
  x, y  — cross-section plane  (same as 2D mesh)
  z      — axial direction, z=0 at drive end, z=L_stack at non-drive end
  End-winding regions extend to z < 0 and z > L_stack

Mesh topology
-------------
  Nodes: indexed [ir, it, iz] → flattened to 1D
  Hexahedral cells split into 5 tetrahedra each (Freudenthal type-A)
  All tets sharing a hex face are consistently oriented

Material tags  (same scheme as py_mesh.py, extended to 3D)
--------------
  1           stator iron
  2           rotor iron
  3..2+poles  PM bodies (pole 0..p-1)
  3+poles     shaft / bore
  3+poles+1   airgap
  200+        winding conductors (slot*2 + layer)
  500         end-winding (coil overhangs outside stack)

Edge orientation
----------------
  Each global edge is stored as (i_lo, i_hi) with i_lo < i_hi.
  Whitney basis signs are adjusted accordingly during assembly.

Usage
-----
    from Bohemien_Motor_Designer.fea.mesh3d import build_motor_mesh_3d
    mesh = build_motor_mesh_3d(motor,
                               n_radial_airgap=4,
                               n_angular_per_slot=8,
                               n_axial=10,
                               n_end_winding=2)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ── Material tag helpers (consistent with 2D mesh) ────────────────────────────
TAG_STATOR_IRON = 1
TAG_ROTOR_IRON  = 2
TAG_PM_BASE     = 3
TAG_END_WINDING = 500

def _pm_tag(pole: int) -> int:       return TAG_PM_BASE + pole
def _shaft_tag(poles: int) -> int:   return TAG_PM_BASE + poles
def _airgap_tag(poles: int) -> int:  return TAG_PM_BASE + poles + 1
def _winding_tag(slot: int, layer: int, Qs: int) -> int:
    return 200 + (slot % Qs) * 2 + layer

# 5-tet Freudenthal decomposition of a hexahedron.
# Hex corner numbering:
#   bottom face (z=k):   0=(i,j,k)  1=(i+1,j,k)  2=(i+1,j+1,k)  3=(i,j+1,k)
#   top face    (z=k+1): 4=(i,j,k+1) 5=(i+1,j,k+1) 6=(i+1,j+1,k+1) 7=(i,j+1,k+1)
HEX_TO_TET_5 = np.array([
    [1, 0, 3, 4],   # tet 0  (orientation reversed from naive Freudenthal)
    [2, 1, 3, 6],   # tet 1
    [4, 1, 5, 6],   # tet 2
    [4, 3, 6, 7],   # tet 3
    [3, 1, 4, 6],   # tet 4  (centre)
], dtype=np.int32)

# All 6 local edges of a tetrahedron (node-pair indices)
TET_LOCAL_EDGES = np.array([
    [0, 1], [0, 2], [0, 3],
    [1, 2], [1, 3], [2, 3],
], dtype=np.int32)


# ── Mesh data structure ───────────────────────────────────────────────────────

@dataclass
class MotorMesh3D:
    """
    3D tetrahedral mesh of a PMSM cross-section.

    Attributes
    ----------
    nodes        : (N, 3) float64  — x, y, z node coordinates [m]
    tets         : (E, 4) int32    — node indices per tetrahedron
    tags         : (E,)   int32    — material tag per tetrahedron
    edges        : (M, 2) int32    — global edge node pairs (i_lo < i_hi)
    tet_edges    : (E, 6) int32    — global edge index per tet local edge
    edge_signs   : (E, 6) int32    — +1 or -1: local edge orientation vs global
    groups       : dict[int, np.ndarray]   — tag → tet indices
    airgap_faces : (F, 3) int32    — node triples of airgap surface triangles
    outer_nodes  : (No,)  int32    — Dirichlet BC nodes (stator OD surface)
    outer_edges  : (Me,)  int32    — Dirichlet BC edges on outer surface
    stack_faces  : dict    — 'z0' and 'zL' face node sets
    coil_data    : list[dict]      — winding slot data (same format as 2D)
    meta         : dict            — poles, radii, stack_length, n_axial etc.
    """
    nodes:        np.ndarray
    tets:         np.ndarray
    tags:         np.ndarray
    edges:        np.ndarray
    tet_edges:    np.ndarray
    edge_signs:   np.ndarray
    groups:       dict
    airgap_faces: np.ndarray
    outer_nodes:  np.ndarray
    outer_edges:  np.ndarray
    stack_faces:  dict
    coil_data:    list
    meta:         dict

    @property
    def n_nodes(self) -> int:    return len(self.nodes)
    @property
    def n_tets(self) -> int:     return len(self.tets)
    @property
    def n_edges(self) -> int:    return len(self.edges)


# ── Public API ────────────────────────────────────────────────────────────────

def build_motor_mesh_3d(
        motor,
        n_radial_airgap:   int   = 4,
        n_angular_per_slot: int  = 8,
        n_axial:           int   = 10,
        n_end_winding:     int   = 2,
        rotor_angle:       float = 0.0) -> MotorMesh3D:
    """
    Build a 3D tetrahedral mesh of the motor cross-section.

    Parameters
    ----------
    motor              : PMSM or similar motor object
    n_radial_airgap    : radial layers across the 1-mm airgap
    n_angular_per_slot : angular elements per slot pitch
    n_axial            : layers within the active stack
    n_end_winding      : axial layers in each end-winding extension
    rotor_angle        : initial rotor mechanical angle [rad]

    Returns
    -------
    MotorMesh3D
    """
    m  = motor
    st = m.stator

    # ── Radial and angular layout (same as 2D) ────────────────────────────
    R_shaft = m.rotor_inner_radius
    R_ro    = m.rotor_outer_radius
    R_si    = st.inner_radius if st else R_ro + m.airgap
    R_so    = st.outer_radius if st else R_si * 1.6
    Qs      = m.slots
    poles   = m.poles

    sp      = st.slot_profile if st else None
    h_slot  = sp.depth()                 if sp else 0.022
    b_slot  = sp.area() / (h_slot+1e-9) if sp else 0.008
    b_open  = sp.opening_width()         if sp else 0.003
    h_wedge = getattr(sp, "wedge_height", 0.0)

    from Bohemien_Motor_Designer.core.geometry.rotor import IPMRotorGeometry
    is_ipm  = isinstance(getattr(m, "rotor_geo", None), IPMRotorGeometry)
    alpha_p = getattr(m, "magnet_width_fraction", 0.80)
    t_m     = getattr(m, "magnet_thickness", 0.006)
    R_mi    = R_ro - t_m

    R_layers = _radial_layers(R_shaft, R_mi, R_ro, R_si, R_so,
                               n_radial_airgap, is_ipm, t_m)
    nr       = len(R_layers)
    n_theta  = Qs * n_angular_per_slot

    # ── Axial layout ──────────────────────────────────────────────────────
    z_layers = _axial_layers(m.stack_length, n_axial, n_end_winding)
    nz       = len(z_layers)

    # ── Node coordinates ──────────────────────────────────────────────────
    thetas = np.linspace(0.0, 2 * np.pi, n_theta + 1)[:-1]  # exclude 2π (= 0)
    N_total = nr * n_theta * nz
    nodes   = np.empty((N_total, 3), dtype=np.float64)

    for iz, z in enumerate(z_layers):
        for ir, r in enumerate(R_layers):
            base = (iz * nr + ir) * n_theta
            nodes[base : base + n_theta, 0] = r * np.cos(thetas)
            nodes[base : base + n_theta, 1] = r * np.sin(thetas)
            nodes[base : base + n_theta, 2] = z

    # Node index helper: [ir, it, iz] → global
    def nidx(ir, it, iz):
        return (iz * nr + ir) * n_theta + (it % n_theta)

    # ── Tetrahedra and tags ───────────────────────────────────────────────
    tets_list = []
    tags_list = []

    for iz in range(nz - 1):
        z_mid = (z_layers[iz] + z_layers[iz + 1]) / 2.0
        in_stack = (0.0 <= z_mid <= m.stack_length)

        for ir in range(nr - 1):
            r_mid = (R_layers[ir] + R_layers[ir + 1]) / 2.0

            for it in range(n_theta):
                itn = (it + 1) % n_theta
                th_mid = (thetas[it] + thetas[itn]) / 2.0

                # 8 hex corners
                hex_n = np.array([
                    nidx(ir,   it,  iz),   # 0
                    nidx(ir,   itn, iz),   # 1  ← angular+1
                    nidx(ir+1, itn, iz),   # 2
                    nidx(ir+1, it,  iz),   # 3
                    nidx(ir,   it,  iz+1), # 4
                    nidx(ir,   itn, iz+1), # 5
                    nidx(ir+1, itn, iz+1), # 6
                    nidx(ir+1, it,  iz+1), # 7
                ], dtype=np.int32)

                tag = _classify_3d(
                    r_mid, th_mid, z_mid,
                    poles=poles, R_shaft=R_shaft, R_mi=R_mi,
                    R_ro=R_ro, R_si=R_si, R_so=R_so,
                    is_ipm=is_ipm, alpha_p=alpha_p,
                    Qs=Qs, h_slot=h_slot, b_slot=b_slot,
                    b_open=b_open, h_wedge=h_wedge,
                    stack_length=m.stack_length,
                    rotor_angle=rotor_angle,
                    in_stack=in_stack,
                )

                for local_tet in HEX_TO_TET_5:
                    tets_list.append(hex_n[local_tet])
                    tags_list.append(tag)

    tets = np.array(tets_list, dtype=np.int32)
    tags = np.array(tags_list, dtype=np.int32)

    # ── Edge table ────────────────────────────────────────────────────────
    edges, tet_edges, edge_signs = _build_edge_table(tets)

    # ── Groups ────────────────────────────────────────────────────────────
    groups = {int(t): np.where(tags == t)[0]
              for t in np.unique(tags)}

    # ── Airgap surface faces ───────────────────────────────────────────────
    airgap_faces = _extract_airgap_faces(tets, tags, nodes,
                                          poles, m.stack_length)

    # ── Outer Dirichlet boundary ───────────────────────────────────────────
    outer_nodes, outer_edges = _outer_boundary(
        nodes, edges, R_so * 0.999)

    # ── Stack end faces ───────────────────────────────────────────────────
    stack_faces = _stack_end_faces(nodes, tets)

    # ── Coil data (winding assignment) ────────────────────────────────────
    coil_data = _build_coil_data_3d(motor, tags, tets, Qs, m.stack_length)

    meta = dict(
        poles=poles, Qs=Qs,
        R_shaft=R_shaft, R_mi=R_mi, R_ro=R_ro, R_si=R_si, R_so=R_so,
        alpha_p=alpha_p, is_ipm=is_ipm, t_m=t_m,
        stack_length=m.stack_length,
        n_radial=nr, n_theta=n_theta, n_axial=nz,
        r_slide=(R_ro + R_si) / 2.0,
        z_layers=z_layers,
    )

    return MotorMesh3D(
        nodes=nodes, tets=tets, tags=tags,
        edges=edges, tet_edges=tet_edges, edge_signs=edge_signs,
        groups=groups, airgap_faces=airgap_faces,
        outer_nodes=outer_nodes, outer_edges=outer_edges,
        stack_faces=stack_faces,
        coil_data=coil_data, meta=meta,
    )


# ── Element classifier ────────────────────────────────────────────────────────

def _classify_3d(r, th, z, *, poles, R_shaft, R_mi, R_ro, R_si, R_so,
                  is_ipm, alpha_p, Qs, h_slot, b_slot, b_open, h_wedge,
                  stack_length, rotor_angle=0.0, in_stack=True) -> int:
    """Classify a 3D element by its centroid (r, th, z)."""

    # End-winding region (outside active stack)
    if not in_stack:
        # Only stator copper overhangs exist outside the stack.
        # Everything else (rotor, stator iron) doesn't extend axially.
        if r > R_si:
            return TAG_END_WINDING   # stator winding overhang region
        if r < R_shaft * 1.01:
            return _shaft_tag(poles)
        if r < R_ro * 1.001:
            return TAG_ROTOR_IRON    # rotor/shaft in end region (if we model it)
        return TAG_STATOR_IRON       # end-winding support / air classified as stator

    # ── Active stack region ───────────────────────────────────────────────
    if r < R_shaft * 1.01:
        return _shaft_tag(poles)

    if r < R_ro * 1.001:
        # Rotor / PM region
        if is_ipm:
            return TAG_ROTOR_IRON
        if r < R_mi * 1.001:
            return TAG_ROTOR_IRON
        # SPM magnet arc
        th_r    = (th - rotor_angle) % (2 * np.pi)
        pp      = 2 * np.pi / poles
        half_m  = np.pi * alpha_p / poles
        pole_ang = th_r % pp
        if abs(pole_ang - np.pi / poles) < half_m:
            return _pm_tag(int(th_r / pp) % poles)
        return TAG_ROTOR_IRON

    if r < R_si * 0.999:
        return _airgap_tag(poles)

    if r > R_so * 0.999:
        return TAG_STATOR_IRON

    # Stator slot or tooth
    sp      = 2 * np.pi / Qs
    si      = int(th / sp) % Qs
    dth     = th - (si + 0.5) * sp
    re      = max(r, R_si + 1e-6)
    a_op    = np.arcsin(min(b_open / (2 * R_si), 0.9999))
    a_sl    = np.arcsin(min(b_slot / (2 * re), 0.9999))
    in_op   = (r < R_si + h_wedge + 1e-4) and (abs(dth) < a_op)
    in_sl   = (r >= R_si + h_wedge - 1e-4) and (abs(dth) < a_sl)
    if (in_op or in_sl) and r < R_si + h_slot + 1e-4:
        lyr = 0 if r < R_si + h_slot / 2 else 1
        return _winding_tag(si, lyr, Qs)

    return TAG_STATOR_IRON


# ── Radial and axial layer helpers ────────────────────────────────────────────

def _radial_layers(R_shaft, R_mi, R_ro, R_si, R_so,
                    n_ag, is_ipm, t_m) -> list:
    """Same as py_mesh._radial_layers."""
    layers = [R_shaft]
    for k in range(1, 4):
        layers.append(R_shaft + k * (R_mi - R_shaft) / 3)
    if not is_ipm and t_m > 1e-4:
        layers.append((R_mi + R_ro) / 2)
    layers.append(R_ro)
    for k in range(1, n_ag + 1):
        layers.append(R_ro + k * (R_si - R_ro) / n_ag)
    h_stator = R_so - R_si
    for k in range(1, 7):
        layers.append(R_si + k * h_stator / 6)
    return sorted(set(round(r, 9) for r in layers))


def _axial_layers(L_stack, n_axial, n_end_winding) -> np.ndarray:
    """
    Build axial layer z-coordinates.
    Active stack:   n_axial+1 levels from z=0 to z=L_stack
    End-winding:    n_end_winding layers on each side, extending ±20% of L_stack
    """
    L_ew = 0.20 * L_stack   # end-winding extension length

    # End-winding on drive end (z < 0)
    z_neg = np.linspace(-L_ew, 0.0, n_end_winding + 1)[:-1]
    # Active stack
    z_stack = np.linspace(0.0, L_stack, n_axial + 1)
    # End-winding on non-drive end (z > L_stack)
    z_pos = np.linspace(L_stack, L_stack + L_ew, n_end_winding + 1)[1:]

    return np.concatenate([z_neg, z_stack, z_pos])


# ── Edge table builder ────────────────────────────────────────────────────────

def _build_edge_table(tets: np.ndarray):
    """
    Build the global edge table for Nédélec DOF assignment.

    Returns
    -------
    edges      : (M, 2) int32   — global edge pairs (i_lo < i_hi)
    tet_edges  : (E, 6) int32   — global edge index for each tet local edge
    edge_signs : (E, 6) int32   — +1 or -1 for orientation consistency
    """
    E = len(tets)
    # For each tet, generate its 6 edge node pairs
    # Local edges: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
    le = TET_LOCAL_EDGES   # (6,2)

    # All local edges as global node pairs: shape (E*6, 2)
    all_pairs = np.empty((E * 6, 2), dtype=np.int32)
    for k, (la, lb) in enumerate(le):
        all_pairs[k::6, 0] = tets[:, la]
        all_pairs[k::6, 1] = tets[:, lb]

    # Canonical orientation: smaller node index first
    # Store sign: +1 if local orientation matches canonical, -1 if flipped
    signs_flat = np.where(all_pairs[:, 0] < all_pairs[:, 1], 1, -1).astype(np.int32)
    canon = np.sort(all_pairs, axis=1)   # (E*6, 2) canonical edges

    # Find unique edges
    # Use structured array for lexsort
    dtype_edge = np.dtype([('a', np.int32), ('b', np.int32)])
    struct = np.empty(len(canon), dtype=dtype_edge)
    struct['a'] = canon[:, 0]
    struct['b'] = canon[:, 1]

    _, inv = np.unique(struct, return_inverse=True)
    edges_struct = np.unique(struct)
    edges = np.column_stack([edges_struct['a'], edges_struct['b']]).astype(np.int32)

    tet_edges  = inv.reshape(E, 6).astype(np.int32)
    edge_signs = signs_flat.reshape(E, 6).astype(np.int32)

    return edges, tet_edges, edge_signs


# ── Airgap surface extraction ─────────────────────────────────────────────────

def _extract_airgap_faces(tets, tags, nodes, poles, stack_length) -> np.ndarray:
    """
    Extract triangular faces on the airgap sliding surface
    (faces shared between airgap elements and rotor elements,
    within the active stack).
    """
    ag_tag  = _airgap_tag(poles)
    ag_mask = tags == ag_tag

    # For each airgap tet, check all 4 faces for adjacency with rotor
    face_triples = []

    # Build face → tet adjacency quickly using sorted face keys
    face_to_tets: dict = {}
    TET_FACES = [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]

    for ei, tet in enumerate(tets):
        for fi, (a, b, c) in enumerate(TET_FACES):
            key = tuple(sorted([tet[a], tet[b], tet[c]]))
            if key not in face_to_tets:
                face_to_tets[key] = []
            face_to_tets[key].append(ei)

    # Faces shared between airgap and rotor tets = sliding surface
    for key, tet_list in face_to_tets.items():
        if len(tet_list) == 2:
            t0, t1 = tet_list
            in_ag  = (tags[t0] == ag_tag or tags[t1] == ag_tag)
            in_rot = (tags[t0] == TAG_ROTOR_IRON or tags[t1] == TAG_ROTOR_IRON or
                      (TAG_PM_BASE <= tags[t0] <= TAG_PM_BASE + 20) or
                      (TAG_PM_BASE <= tags[t1] <= TAG_PM_BASE + 20))
            if in_ag and in_rot:
                # Only faces within stack
                z_face = np.mean(nodes[list(key), 2])
                if 0.0 <= z_face <= stack_length:
                    face_triples.append(list(key))

    if face_triples:
        return np.array(face_triples, dtype=np.int32)
    return np.empty((0, 3), dtype=np.int32)


# ── Dirichlet boundary nodes / edges ─────────────────────────────────────────

def _outer_boundary(nodes, edges, r_outer: float):
    """
    Find nodes and edges on the outer cylindrical surface (r ≈ R_so).
    These get A_tangential = 0 Dirichlet BC.
    """
    r_nodes = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
    outer_node_mask = r_nodes > r_outer
    outer_nodes = np.where(outer_node_mask)[0].astype(np.int32)

    # Edges where BOTH nodes are on the outer surface
    on_outer = outer_node_mask
    outer_edge_mask = on_outer[edges[:, 0]] & on_outer[edges[:, 1]]
    outer_edges = np.where(outer_edge_mask)[0].astype(np.int32)

    return outer_nodes, outer_edges


# ── Stack end faces ───────────────────────────────────────────────────────────

def _stack_end_faces(nodes, tets) -> dict:
    """
    Identify nodes on the z=0 and z=L_stack planes for periodic/symmetry BCs.
    """
    z = nodes[:, 2]
    z_min = z.min(); z_max = z.max()
    # Active stack boundaries (second and second-to-last z layers)
    z_vals = np.unique(np.round(z, 9))
    z0_val = z_vals[z_vals >= 0][0] if np.any(z_vals >= 0) else 0.0
    zL_val = z_vals[z_vals <= z_max][-1]

    nodes_z0 = np.where(np.abs(z - z0_val) < 1e-9)[0].astype(np.int32)
    nodes_zL = np.where(np.abs(z - zL_val) < 1e-9)[0].astype(np.int32)

    return {"z0": nodes_z0, "zL": nodes_zL,
            "z0_val": float(z0_val), "zL_val": float(zL_val)}


# ── Coil data ─────────────────────────────────────────────────────────────────

def _build_coil_data_3d(motor, tags, tets, Qs, stack_length) -> list:
    """
    Assign winding conductor elements to phases.
    Only elements within the active stack are included
    (end-winding handled separately).
    """
    coil_data = []
    winding = getattr(motor, "winding", None)
    if winding is None:
        return coil_data

    layout = []
    try:
        for ph in range(3):
            for cs in winding.coil_sides_for_phase(ph):
                layout.append((cs.slot_idx % Qs, cs.layer, cs.phase, cs.direction))
    except Exception:
        return coil_data

    for slot_idx, layer, phase, direction in layout:
        t    = _winding_tag(slot_idx % Qs, layer, Qs)
        idx  = np.where(tags == t)[0]
        if len(idx) == 0:
            continue
        coil_data.append(dict(
            slot=slot_idx % Qs, layer=layer,
            phase=phase, direction=direction,
            elem_idx=idx,
        ))
    return coil_data


# ── Summary ───────────────────────────────────────────────────────────────────

def mesh_report_3d(mesh: MotorMesh3D) -> str:
    poles = mesh.meta.get("poles", 0)
    lines = [
        f"3D Motor mesh: {mesh.n_nodes:,} nodes  "
        f"{mesh.n_tets:,} tets  {mesh.n_edges:,} edges",
        f"  Radial layers   : {mesh.meta['n_radial']}",
        f"  Angular steps   : {mesh.meta['n_theta']}  "
        f"({mesh.meta['n_theta']//mesh.meta['Qs']} per slot)",
        f"  Axial layers    : {mesh.meta['n_axial']}",
        f"  Stack z range   : {mesh.meta['z_layers'][0]*1000:.1f} mm to "
        f"{mesh.meta['z_layers'][-1]*1000:.1f} mm",
    ]
    for tag, name in [
        (TAG_STATOR_IRON,   "Stator iron"),
        (TAG_ROTOR_IRON,    "Rotor iron"),
        (_shaft_tag(poles), "Shaft"),
        (_airgap_tag(poles),"Air gap"),
        (TAG_END_WINDING,   "End-winding"),
    ]:
        c = int((mesh.tags == tag).sum())
        if c:
            lines.append(f"  {name:<18}: {c:6,} tets")
    pm_cnt = int(sum((mesh.tags == _pm_tag(p)).sum() for p in range(poles)))
    if pm_cnt:
        lines.append(f"  {'PM bodies':<18}: {pm_cnt:6,} tets")
    wc = int((mesh.tags >= 200).sum() & (mesh.tags < 500).sum())
    if wc:
        lines.append(f"  {'Winding slots':<18}: {wc:6,} tets")
    lines.append(f"  Airgap faces    : {len(mesh.airgap_faces):,} triangles")
    lines.append(f"  Outer BC edges  : {len(mesh.outer_edges):,}")
    return "\n".join(lines)
