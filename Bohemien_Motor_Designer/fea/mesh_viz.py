"""
mesh_viz.py — Mesh visualisation for the structured polar FEM mesh.

Provides two plot functions used by the GUI Mesh Viz tab:

    plot_mesh_overview(motor, mesh, fig)
        Three-panel view:
          • Full cross-section coloured by material/region
          • 50° sector zoom into the airgap
          • Radial connectivity slice (7° wedge) with shared nodes highlighted

    These can also be called standalone:
        from Bohemien_Motor_Designer.fea.mesh_viz import plot_mesh_overview
        from Bohemien_Motor_Designer.fea.py_mesh  import build_motor_mesh
        fig = plt.figure(figsize=(18, 9))
        plot_mesh_overview(motor, mesh, fig)
        plt.show()
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from typing import Optional

# ── Colour palette ────────────────────────────────────────────────────────────
_BG       = "#0f1117"
_PANEL_FG = "#e0e0e0"
_DIM      = "#888888"

_COL = dict(
    stator_iron = "#4a6fa5",   # steel blue
    rotor_iron  = "#2d6a4f",   # dark green
    pm_n        = "#e76f51",   # warm orange   (even poles → N)
    pm_s        = "#f4a261",   # light orange  (odd  poles → S)
    shaft       = "#264653",   # near-black teal
    airgap      = "#a8d8ea",   # pale blue
    wind_A_fwd  = "#c1121f",   # deep red
    wind_A_rev  = "#ff6b6b",   # light red
    wind_B_fwd  = "#2d6a4f",   # deep green
    wind_B_rev  = "#74c69d",   # light green
    wind_C_fwd  = "#1d3557",   # deep blue
    wind_C_rev  = "#457b9d",   # light blue
    shared_Rro  = "#ffcc00",   # gold   – shared nodes at rotor surface
    shared_Rsi  = "#ff6ec7",   # pink   – shared nodes at bore
    midline     = "#ffff00",   # yellow – airgap sliding midline
)


# ── Public API ────────────────────────────────────────────────────────────────

def plot_mesh_overview(motor, mesh, fig: Figure) -> None:
    """
    Fill *fig* with three mesh visualisation panels.

    Parameters
    ----------
    motor : PMSM instance
    mesh  : MotorMesh (from build_motor_mesh)
    fig   : matplotlib Figure to draw into (cleared first)
    """
    fig.clear()
    fig.patch.set_facecolor(_BG)

    gs = fig.add_gridspec(2, 2,
                          left=0.04, right=0.97,
                          top=0.93,  bottom=0.10,
                          hspace=0.38, wspace=0.28)

    ax_full  = fig.add_subplot(gs[:, 0])   # left  — full cross-section (spans both rows)
    ax_zoom  = fig.add_subplot(gs[0, 1])   # top-right — airgap sector zoom
    ax_slice = fig.add_subplot(gs[1, 1])   # bottom-right — radial connectivity slice

    colours = _build_colour_array(motor, mesh)
    nodes   = mesh.nodes
    elems   = mesh.elems
    poles   = motor.poles

    # ── Panel 1: full cross-section ──────────────────────────────────────
    _draw_panel(ax_full, nodes, elems, colours,
                lw=0.05, alpha_edge=0.25,
                title=f"Full Cross-Section  —  {poles}p/{motor.slots}s PMSM"
                      f"  ({mesh.n_nodes:,} nodes  {mesh.n_elems:,} elems)")
    _add_reference_circles(ax_full, motor)

    # ── Panel 2: airgap sector zoom ───────────────────────────────────────
    sector_deg = 50.0
    ecx = (nodes[elems[:,0],0] + nodes[elems[:,1],0] + nodes[elems[:,2],0]) / 3
    ecy = (nodes[elems[:,0],1] + nodes[elems[:,1],1] + nodes[elems[:,2],1]) / 3
    eth = np.degrees(np.arctan2(ecy, ecx))
    mask_z = np.abs(eth) <= sector_deg / 2

    R_ro = motor.rotor_outer_radius
    R_si = (motor.stator.inner_radius
            if motor.stator else R_ro + motor.airgap)

    _draw_panel(ax_zoom, nodes, elems[mask_z], colours[mask_z],
                lw=0.30, alpha_edge=0.55,
                xlim=(-22, 122), ylim=(-52, 52),
                title=f"Airgap & Slot Detail  —  {sector_deg:.0f}° sector")

    # Airgap midline
    r_mid = (R_ro + R_si) / 2 * 1000
    mid_circ = plt.Circle((0, 0), r_mid, fill=False,
                           edgecolor=_COL["midline"], lw=1.0,
                           linestyle="-", alpha=0.8)
    ax_zoom.add_patch(mid_circ)
    ax_zoom.text(r_mid * np.cos(np.radians(44)), r_mid * np.sin(np.radians(44)),
                 "sliding\nmidline", color=_COL["midline"],
                 fontsize=6.5, ha="center", va="center")

    # Annotate R_ro / R_si
    for r_m, lbl in [(R_ro * 1000, "R_ro"), (R_si * 1000, "R_si")]:
        ax_zoom.annotate(lbl, xy=(r_m, 0), xytext=(r_m, 46),
                         color=_DIM, fontsize=7, ha="center",
                         arrowprops=dict(arrowstyle="->", color=_DIM, lw=0.7))

    # ── Panel 3: radial connectivity slice ───────────────────────────────
    _draw_radial_slice(ax_slice, motor, mesh, colours, slice_deg=7.0)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=_COL["stator_iron"], label="Stator iron"),
        mpatches.Patch(color=_COL["rotor_iron"],  label="Rotor iron"),
        mpatches.Patch(color=_COL["pm_n"],        label="PM — N pole"),
        mpatches.Patch(color=_COL["pm_s"],        label="PM — S pole"),
        mpatches.Patch(color=_COL["shaft"],       label="Shaft"),
        mpatches.Patch(color=_COL["airgap"],      label="Airgap"),
        mpatches.Patch(color=_COL["wind_A_fwd"],  label="Phase A  (+)"),
        mpatches.Patch(color=_COL["wind_A_rev"],  label="Phase A  (−)"),
        mpatches.Patch(color=_COL["wind_B_fwd"],  label="Phase B  (+)"),
        mpatches.Patch(color=_COL["wind_B_rev"],  label="Phase B  (−)"),
        mpatches.Patch(color=_COL["wind_C_fwd"],  label="Phase C  (+)"),
        mpatches.Patch(color=_COL["wind_C_rev"],  label="Phase C  (−)"),
        mpatches.Patch(color=_COL["shared_Rro"],  label="Shared nodes  R_ro"),
        mpatches.Patch(color=_COL["shared_Rsi"],  label="Shared nodes  R_si"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=7,
               framealpha=0.15, facecolor="#1a1a2e",
               edgecolor="#444455", labelcolor="#cccccc",
               fontsize=7.5, bbox_to_anchor=(0.5, 0.005))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _style_ax(ax, title=""):
    ax.set_facecolor(_BG)
    ax.set_aspect("equal")
    ax.set_title(title, color=_PANEL_FG, fontsize=9, fontweight="bold", pad=7)
    ax.tick_params(colors=_DIM, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#333333")
    ax.set_xlabel("x  [mm]", color=_DIM, fontsize=7)
    ax.set_ylabel("y  [mm]", color=_DIM, fontsize=7)


def _draw_panel(ax, nodes, elems, colours,
                lw=0.08, alpha_edge=0.35,
                xlim=None, ylim=None, title=""):
    _style_ax(ax, title)
    if len(elems) == 0:
        return

    tri_xy = nodes[elems] * 1000          # → mm

    pc = PolyCollection(tri_xy, facecolors=colours,
                        edgecolors=[[0, 0, 0, alpha_edge]],
                        linewidths=lw)
    ax.add_collection(pc)

    if xlim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        ext = float(np.max(np.abs(nodes))) * 1000 * 1.04
        ax.set_xlim(-ext, ext)
        ax.set_ylim(-ext, ext)


def _add_reference_circles(ax, motor):
    R_ro = motor.rotor_outer_radius
    R_si = (motor.stator.inner_radius
            if motor.stator else R_ro + motor.airgap)
    R_so = (motor.stator.outer_radius
            if motor.stator else R_si * 1.5)
    t_m  = getattr(motor, "magnet_thickness", 0.005)

    for r, col in [
        (motor.rotor_inner_radius, "#ffffff"),
        (R_ro - t_m,               "#aaaaaa"),   # R_mi
        (R_ro,                     "#ffcc00"),
        (R_si,                     "#ff6ec7"),
        (R_so,                     "#aaaaaa"),
    ]:
        circ = plt.Circle((0, 0), r * 1000, fill=False,
                           edgecolor=col, lw=0.7,
                           linestyle="--", alpha=0.5)
        ax.add_patch(circ)


def _draw_radial_slice(ax, motor, mesh, colours, slice_deg=7.0):
    """Unrolled radial wedge showing node sharing at material interfaces."""
    _style_ax(ax, f"Radial Connectivity  —  {slice_deg:.0f}° slice  "
                  "(● shared interface nodes)")
    ax.set_xlabel("r  [mm]", color=_DIM, fontsize=7)
    ax.set_ylabel("arc  [mm]", color=_DIM, fontsize=7)

    nodes = mesh.nodes
    elems = mesh.elems
    tags  = mesh.tags

    node_r  = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
    node_th = np.degrees(np.arctan2(nodes[:, 1], nodes[:, 0]))
    in_slice = np.abs(node_th) < slice_deg

    # Elements where ALL 3 nodes are in slice
    in_slice_e = in_slice[elems[:, 0]] & in_slice[elems[:, 1]] & in_slice[elems[:, 2]]
    slice_idx  = np.where(in_slice_e)[0]

    if len(slice_idx) == 0:
        ax.text(0.5, 0.5, "No elements in slice", transform=ax.transAxes,
                color=_PANEL_FG, ha="center")
        return

    # Draw elements as polygons in (r, arc) space  →  use centroid y in mm
    for ei in slice_idx:
        tri  = nodes[elems[ei]]                     # (3, 2) in metres
        r3   = np.sqrt(tri[:, 0]**2 + tri[:, 1]**2) * 1000   # mm
        arc3 = np.arctan2(tri[:, 1], tri[:, 0])
        arc3 = arc3 * np.mean(r3)                   # arc length in mm
        poly = plt.Polygon(np.column_stack([r3, arc3]),
                           closed=True,
                           facecolor=colours[ei],
                           edgecolor="#ffffff",
                           linewidth=0.45, alpha=0.92)
        ax.add_patch(poly)

    # All nodes in slice
    sn = np.where(in_slice)[0]
    r_sn   = node_r[sn] * 1000
    arc_sn = np.arctan2(nodes[sn, 1], nodes[sn, 0]) * r_sn
    ax.scatter(r_sn, arc_sn, s=8, c="#ffffff", zorder=10, linewidths=0)

    # Highlight shared interface nodes
    tol = 2e-4
    R_ro = motor.rotor_outer_radius
    R_si = (motor.stator.inner_radius
            if motor.stator else R_ro + motor.airgap)

    for r_iface, col, lbl in [
        (R_ro, _COL["shared_Rro"], "R_ro"),
        (R_si, _COL["shared_Rsi"], "R_si"),
    ]:
        at_r = sn[np.abs(node_r[sn] - r_iface) < tol]
        if len(at_r):
            r_pts  = node_r[at_r] * 1000
            arc_pts = np.arctan2(nodes[at_r, 1], nodes[at_r, 0]) * r_pts
            ax.scatter(r_pts, arc_pts, s=60, c=col, zorder=20,
                       linewidths=0.6, edgecolors="#ffffff")
            # Vertical dashed reference line
            ax.axvline(r_iface * 1000, color=col, lw=0.9,
                       linestyle="--", alpha=0.6)
            ax.text(r_iface * 1000, arc_pts.max() + 0.8, lbl,
                    color=col, fontsize=7, ha="center", va="bottom")

    # Region labels along top
    R_mi = R_ro - getattr(motor, "magnet_thickness", 0.005)
    region_bands = [
        (motor.rotor_inner_radius * 1000,   R_mi * 1000,   "SHAFT+\nROTOR Fe", _COL["rotor_iron"]),
        (R_mi * 1000,                        R_ro * 1000,   "PM",               _COL["pm_n"]),
        (R_ro * 1000,                        R_si * 1000,   "AIRGAP",           _COL["airgap"]),
        (R_si * 1000, (motor.stator.outer_radius if motor.stator else R_si*1.5)*1000,
                                                            "STATOR",           _COL["stator_iron"]),
    ]
    ax.autoscale_view()
    ylim = ax.get_ylim()
    ytop = ylim[1] * 0.97
    for r0, r1, lbl, col in region_bands:
        xm = (r0 + r1) / 2
        ax.text(xm, ytop, lbl, color="#dddddd", fontsize=6.5,
                ha="center", va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", facecolor=col,
                          alpha=0.45, edgecolor="none"))

    ax.set_xlim(motor.rotor_inner_radius * 1000 * 0.9,
                (motor.stator.outer_radius if motor.stator
                 else R_si * 1.5) * 1000 * 1.02)


def _build_colour_array(motor, mesh) -> np.ndarray:
    """Return per-element colour strings matching mesh.tags."""
    poles    = motor.poles
    TAG_STATOR = 1
    TAG_ROTOR  = 2
    TAG_PM_MIN = 3
    TAG_PM_MAX = 3 + poles - 1
    TAG_SHAFT  = 3 + poles
    TAG_AIRGAP = 3 + poles + 1
    TAG_WIND   = 200

    # Build slot→(phase, direction) map from winding layout
    slot_info: dict[int, tuple[int, int]] = {}
    winding = getattr(motor, "winding", None)
    if winding is not None:
        try:
            for ph in range(3):
                for cs in winding.coil_sides_for_phase(ph):
                    slot_info[cs.slot_idx % motor.slots] = (cs.phase, cs.direction)
        except Exception:
            pass

    _phase_cols = [
        (_COL["wind_A_fwd"], _COL["wind_A_rev"]),
        (_COL["wind_B_fwd"], _COL["wind_B_rev"]),
        (_COL["wind_C_fwd"], _COL["wind_C_rev"]),
    ]

    def _col(tag: int) -> str:
        if tag == TAG_STATOR:
            return _COL["stator_iron"]
        if tag == TAG_ROTOR:
            return _COL["rotor_iron"]
        if tag == TAG_SHAFT:
            return _COL["shaft"]
        if tag == TAG_AIRGAP:
            return _COL["airgap"]
        if TAG_PM_MIN <= tag <= TAG_PM_MAX:
            pole_idx = tag - TAG_PM_MIN
            return _COL["pm_n"] if pole_idx % 2 == 0 else _COL["pm_s"]
        if tag >= TAG_WIND:
            slot = (tag - TAG_WIND) // 2
            info = slot_info.get(slot % motor.slots)
            if info is None:
                return "#aaaaaa"
            ph, direction = info
            return _phase_cols[ph][0] if direction > 0 else _phase_cols[ph][1]
        return "#cccccc"

    return np.array([_col(int(t)) for t in mesh.tags])
