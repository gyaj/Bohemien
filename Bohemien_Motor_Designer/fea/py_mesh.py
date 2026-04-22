"""
py_mesh.py - Structured polar mesh for 2D PMSM cross-section.

No GMSH required.  The mesh is built once at rotor_angle=0.
Call mesh.update_rotor_tags(angle) before each solve to advance rotor position.

Material tags
-------------
  1          stator iron
  2          rotor iron
  3..2+p     PM bodies (pole 0..p-1)
  3+p        shaft/bore
  3+p+1      air gap
  200+       winding conductors (slot*2 + layer)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

MU0 = 4e-7 * np.pi
_TAG_STATOR_IRON = 1
_TAG_ROTOR_IRON  = 2
_TAG_PM_BASE     = 3
_TAG_WINDING     = 200


def _pm_tag(pole):       return _TAG_PM_BASE + pole
def _airgap_tag(poles):  return 3 + poles + 1
def _shaft_tag(poles):   return 3 + poles
def _winding_tag(s, lyr, Qs): return _TAG_WINDING + (s % Qs) * 2 + lyr


# ── MotorMesh ──────────────────────────────────────────────────────────────────

@dataclass
class MotorMesh:
    nodes:           np.ndarray
    elems:           np.ndarray
    tags:            np.ndarray
    groups:          dict
    airgap_elems:    np.ndarray
    r_slide:         float
    coil_data:       list  = field(default_factory=list)
    outer_nodes:     np.ndarray = field(default_factory=lambda: np.array([], int))
    inner_nodes:     np.ndarray = field(default_factory=lambda: np.array([], int))
    rotor_elem_mask: np.ndarray = field(default_factory=lambda: np.array([], bool))
    _mesh_meta:      dict = field(default_factory=dict, repr=False)

    @property
    def n_nodes(self): return len(self.nodes)
    @property
    def n_elems(self): return len(self.elems)

    def update_rotor_tags(self, rotor_angle: float) -> None:
        """Re-classify rotor elements at new mechanical angle [rad]."""
        meta   = self._mesh_meta
        poles  = meta["poles"]
        R_mi   = meta["R_mi"]
        R_ro   = meta["R_ro"]
        R_shaft= meta["R_shaft"]
        alpha_p= meta["alpha_p"]
        is_ipm = meta["is_ipm"]

        idx = np.where(self.rotor_elem_mask)[0]
        if len(idx) == 0:
            return

        # Centroids
        e = self.elems[idx]
        cx = (self.nodes[e[:,0],0]+self.nodes[e[:,1],0]+self.nodes[e[:,2],0])/3
        cy = (self.nodes[e[:,0],1]+self.nodes[e[:,1],1]+self.nodes[e[:,2],1])/3
        r_c  = np.sqrt(cx**2 + cy**2)
        th_c = np.arctan2(cy, cx)

        # Transform to rotor frame
        th_rotor = (th_c - rotor_angle) % (2*np.pi)

        new_tags = np.full(len(idx), _TAG_ROTOR_IRON, dtype=np.int32)
        new_tags[r_c < R_shaft*1.01] = _shaft_tag(poles)

        if not is_ipm:
            in_mag_r  = (r_c > R_mi*0.999) & (r_c < R_ro*1.001)
            pp        = 2*np.pi/poles
            half_mag  = np.pi*alpha_p/poles
            pole_ang  = th_rotor % pp
            in_mag_th = np.abs(pole_ang - np.pi/poles) < half_mag
            pm        = in_mag_r & in_mag_th
            pole_idx  = (th_rotor / pp).astype(int) % poles
            new_tags[pm] = _TAG_PM_BASE + pole_idx[pm]

        self.tags[idx] = new_tags
        for p in range(poles):
            self.groups[_pm_tag(p)] = np.where(self.tags == _pm_tag(p))[0]
        self.groups[_TAG_ROTOR_IRON] = np.where(self.tags == _TAG_ROTOR_IRON)[0]


# ── Builder ────────────────────────────────────────────────────────────────────

def build_motor_mesh(motor,
                     n_radial_airgap:   int   = 4,
                     n_angular_per_slot: int  = 12,
                     rotor_angle:       float = 0.0) -> MotorMesh:
    m  = motor
    st = m.stator

    R_shaft = m.rotor_inner_radius
    R_ro    = m.rotor_outer_radius
    R_si    = st.inner_radius if st else R_ro + m.airgap
    R_so    = st.outer_radius if st else R_si * 1.6
    R_slide = (R_ro + R_si) / 2.0
    Qs      = m.slots
    p2      = m.poles

    sp      = st.slot_profile if st else None
    h_slot  = sp.depth()                  if sp else 0.020
    b_slot  = sp.area()/(h_slot+1e-9)    if sp else 0.008
    b_open  = sp.opening_width()          if sp else 0.003
    h_wedge = getattr(sp, "wedge_height", 0.0)

    from Bohemien_Motor_Designer.core.geometry.rotor import IPMRotorGeometry
    is_ipm  = isinstance(getattr(m, "rotor_geo", None), IPMRotorGeometry)
    alpha_p = getattr(m, "magnet_width_fraction", 0.80)
    t_m     = getattr(m, "magnet_thickness",      0.006)
    R_mi    = R_ro - t_m

    R_layers = _radial_layers(R_shaft, R_mi, R_ro, R_si, R_so, n_radial_airgap, is_ipm, t_m)
    nr       = len(R_layers)
    n_theta  = Qs * n_angular_per_slot
    thetas   = np.linspace(0, 2*np.pi, n_theta+1)
    nt       = n_theta

    node_idx = np.arange(nr*nt).reshape(nr, nt)
    xs = np.zeros(nr*nt)
    ys = np.zeros(nr*nt)
    for ir, r in enumerate(R_layers):
        for it in range(nt):
            i = node_idx[ir, it]
            xs[i] = r * np.cos(thetas[it])
            ys[i] = r * np.sin(thetas[it])
    nodes = np.column_stack([xs, ys])

    elems_list, tags_list = [], []
    for ir in range(nr-1):
        r_mid = (R_layers[ir]+R_layers[ir+1])/2
        for it in range(nt):
            itn  = (it+1) % nt
            n0,n1,n2,n3 = node_idx[ir,it], node_idx[ir,itn], node_idx[ir+1,itn], node_idx[ir+1,it]
            th_mid = (thetas[it]+thetas[itn])/2
            tag = _classify(r_mid, th_mid, poles=p2, R_shaft=R_shaft, R_mi=R_mi,
                            R_ro=R_ro, R_si=R_si, R_so=R_so, is_ipm=is_ipm,
                            alpha_p=alpha_p, Qs=Qs, h_slot=h_slot, b_slot=b_slot,
                            b_open=b_open, h_wedge=h_wedge, rotor_angle=rotor_angle)
            elems_list += [[n0,n1,n2],[n0,n2,n3]]
            tags_list  += [tag, tag]

    elems = np.array(elems_list, dtype=np.int32)
    tags  = np.array(tags_list,  dtype=np.int32)
    groups = {int(t): np.where(tags==t)[0] for t in np.unique(tags)}
    airgap_elems = np.where(tags == _airgap_tag(p2))[0]
    outer_nodes  = node_idx[nr-1,:].copy()
    inner_nodes  = node_idx[0,:].copy()

    # Rotor element mask (r_centroid < R_slide)
    cx = (nodes[elems[:,0],0]+nodes[elems[:,1],0]+nodes[elems[:,2],0])/3
    cy = (nodes[elems[:,0],1]+nodes[elems[:,1],1]+nodes[elems[:,2],1])/3
    rc = np.sqrt(cx**2+cy**2)
    rotor_elem_mask = rc < (R_ro - 1e-5)   # only elements fully inside rotor OD

    coil_data = _build_coil_data(motor, tags, elems, Qs)

    meta = dict(poles=p2, R_mi=R_mi, R_ro=R_ro, R_shaft=R_shaft,
                alpha_p=alpha_p, is_ipm=is_ipm)

    return MotorMesh(
        nodes=nodes, elems=elems, tags=tags, groups=groups,
        airgap_elems=airgap_elems, r_slide=R_slide,
        coil_data=coil_data, outer_nodes=outer_nodes, inner_nodes=inner_nodes,
        rotor_elem_mask=rotor_elem_mask, _mesh_meta=meta,
    )


# ── Element classifier ─────────────────────────────────────────────────────────

def _classify(r, th, *, poles, R_shaft, R_mi, R_ro, R_si, R_so,
              is_ipm, alpha_p, Qs, h_slot, b_slot, b_open, h_wedge,
              rotor_angle=0.0):
    if r < R_shaft*1.01:
        return _shaft_tag(poles)

    if r < R_ro*1.001:
        if is_ipm:
            return _TAG_ROTOR_IRON
        if r < R_mi*1.001:
            return _TAG_ROTOR_IRON
        th_r   = (th - rotor_angle) % (2*np.pi)
        pp     = 2*np.pi/poles
        if abs(th_r % pp - np.pi/poles) < np.pi*alpha_p/poles:
            return _TAG_PM_BASE + int(th_r/pp) % poles
        return _TAG_ROTOR_IRON

    if r < R_si*0.999:
        return _airgap_tag(poles)

    if r > R_so*0.999:
        return _TAG_STATOR_IRON

    # Stator slot (stator frame, no rotation)
    sp   = 2*np.pi/Qs
    si   = int(th/sp) % Qs
    dth  = th - (si+0.5)*sp
    re   = max(r, R_si+1e-6)
    a_op = np.arcsin(min(b_open/(2*R_si), 0.9999))
    a_sl = np.arcsin(min(b_slot/(2*re),   0.9999))
    in_op = (r < R_si+h_wedge+1e-4) and (abs(dth) < a_op)
    in_sl = (r >= R_si+h_wedge-1e-4) and (abs(dth) < a_sl)
    if (in_op or in_sl) and r < R_si+h_slot+1e-4:
        lyr = 0 if r < R_si+h_slot/2 else 1
        return _winding_tag(si, lyr, Qs)

    return _TAG_STATOR_IRON


# ── Radial layers ─────────────────────────────────────────────────────────────

def _radial_layers(R_shaft, R_mi, R_ro, R_si, R_so, n_ag, is_ipm, t_m):
    layers = [R_shaft]
    # Rotor iron (3 steps from shaft to magnet inner)
    for k in range(1, 4):
        layers.append(R_shaft + k*(R_mi-R_shaft)/3)
    # PM region (2 steps)
    if not is_ipm and t_m > 1e-4:
        layers.append((R_mi+R_ro)/2)
    layers.append(R_ro)
    # Air gap (n_ag steps)
    for k in range(1, n_ag+1):
        layers.append(R_ro + k*(R_si-R_ro)/n_ag)
    # Stator interior: 6 layers covering the slot height + yoke
    h_stator = R_so - R_si
    for k in range(1, 7):
        layers.append(R_si + k*h_stator/6)
    return sorted(set(round(r, 9) for r in layers))


# ── Coil data ─────────────────────────────────────────────────────────────────

def _build_coil_data(motor, tags, elems, Qs):
    coil_data = []
    winding = getattr(motor, "winding", None)
    if winding is None:
        return coil_data

    # Use the proper winding layout API
    layout = []
    try:
        for ph in range(3):
            sides = winding.coil_sides_for_phase(ph)
            for cs in sides:
                layout.append((cs.slot_idx % Qs, cs.layer, cs.phase, cs.direction))
    except Exception:
        layout = _default_layout(motor)

    for slot_idx, layer, phase, direction in layout:
        t    = _winding_tag(slot_idx % Qs, layer, Qs)
        idx  = np.where(tags == t)[0]
        if len(idx) == 0:
            continue
        coil_data.append(dict(slot=slot_idx % Qs, layer=layer,
                               phase=phase, direction=direction, elem_idx=idx))
    return coil_data


def _default_layout(motor):
    Qs, poles = motor.slots, motor.poles
    layout = []
    for s in range(Qs):
        ph  = (s * 3 // Qs) % 3
        sgn = 1 if (s // (Qs // (poles*3))) % 2 == 0 else -1
        layout.append((s, 0, ph,  sgn))
        layout.append((s, 1, ph, -sgn))
    return layout


# ── Report ────────────────────────────────────────────────────────────────────

def mesh_report(mesh: MotorMesh) -> str:
    poles = mesh._mesh_meta.get("poles", 0)
    lines = [f"Motor mesh: {mesh.n_nodes} nodes  {mesh.n_elems} elements"]
    for tag, name in [(1,"Stator iron"),(2,"Rotor iron"),
                      (_shaft_tag(poles),"Shaft"),(_airgap_tag(poles),"Air gap")]:
        c = int((mesh.tags == tag).sum())
        if c: lines.append(f"  {name:<18}: {c:5d} elems")
    pm_cnt = int(sum((mesh.tags == _pm_tag(p)).sum() for p in range(poles)))
    if pm_cnt: lines.append(f"  {'PM bodies':<18}: {pm_cnt:5d} elems")
    wc = int((mesh.tags >= _TAG_WINDING).sum())
    if wc: lines.append(f"  {'Winding slots':<18}: {wc:5d} elems")
    return "\n".join(lines)
