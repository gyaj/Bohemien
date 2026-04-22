"""
dxf_export.py — AutoCAD DXF cross-section export for PMSM designs.

Generates a fully-layered R12 ASCII DXF file from any PMSM motor object.
No external libraries required — all DXF entities are written directly.

Layers produced
---------------
  STATOR_IRON     Stator lamination body (outer ring, bore, tooth tips)
  STATOR_SLOTS    Slot outlines (walls, opening, top arc, layer separator)
  WINDING_A/B/C   Per-phase conductor arcs (forward = solid, reverse = dashed)
  ROTOR_IRON      Rotor iron (outer, inner, shaft, inter-pole arcs)
  MAGNETS_N       North-pole magnet outlines
  MAGNETS_S       South-pole magnet outlines
  AIRGAP          Dashed midline circle at airgap centre
  CENTERLINES     Centre crosshairs (dashed)
  DIMENSIONS      Radius leaders and callout text
  LABELS          Title block with all key parameters

Usage (standalone)
------------------
    from Bohemien_Motor_Designer.io.dxf_export import export_dxf
    export_dxf(motor, "PMSM_design.dxf")

Usage (from GUI)
----------------
    Called via the DXF Export tab — see gui/app.py _build_dxf_tab().
"""
from __future__ import annotations
import math
from typing import Optional


# ── Public API ────────────────────────────────────────────────────────────────

def export_dxf(motor, path: str, units: str = "mm") -> dict:
    """
    Generate an AutoCAD DXF cross-section of *motor* and write to *path*.

    Parameters
    ----------
    motor : PMSM instance
    path  : output file path (should end in .dxf)
    units : 'mm' (default) or 'm' — DXF coordinate scale

    Returns
    -------
    dict with 'entities', 'layers', 'path' keys for status display.
    """
    scale = 1000.0 if units == "mm" else 1.0

    builder = _DXFBuilder(motor, scale)
    builder.build()

    content = "\n".join(builder.lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return {
        "path":     path,
        "entities": builder.entity_count,
        "layers":   builder.layer_counts,
        "units":    units,
    }


# ── Internal DXF builder ──────────────────────────────────────────────────────

class _DXFBuilder:
    """Builds a complete R12 ASCII DXF file for a PMSM cross-section."""

    def __init__(self, motor, scale: float = 1000.0):
        self.motor  = motor
        self.scale  = scale          # multiply metres → output units
        self.lines: list[str] = []
        self.entity_count = 0
        self.layer_counts: dict[str, int] = {}

    # ── Top-level build ───────────────────────────────────────────────────────

    def build(self):
        m = self.motor
        self._extract_geometry()
        self._write_header()
        self._write_tables()
        self._write_entities()

    def _extract_geometry(self):
        """Pull all dimensions from the motor object."""
        m       = self.motor
        s       = self.scale
        st      = m.stator
        sp      = st.slot_profile if st else None

        self.R_so    = (st.outer_radius if st else 0.130) * s
        self.R_si    = (st.inner_radius if st else 0.082) * s
        self.R_ro    = m.rotor_outer_radius * s
        self.t_m     = getattr(m, "magnet_thickness", 0.006) * s
        self.R_mi    = self.R_ro - self.t_m
        self.R_shaft = m.rotor_inner_radius * s
        self.R_ag    = (self.R_ro + self.R_si) / 2.0   # airgap midline

        self.Qs      = m.slots
        self.poles   = m.poles
        self.alpha_p = getattr(m, "magnet_width_fraction", 0.83)
        self.pole_pitch  = 2 * math.pi / self.poles
        self.slot_pitch  = 2 * math.pi / self.Qs

        self.h_slot  = (sp.depth()          if sp else 0.022) * s
        self.b_slot  = (sp.area() / (sp.depth() + 1e-9) if sp else 0.008) * s
        self.b_open  = (sp.opening_width()  if sp else 0.003) * s
        self.h_wedge = (getattr(sp, "wedge_height", 0.0)) * s

        # Build slot → (phase, direction) map
        self.slot_phase: dict[int, int] = {}
        self.slot_dir:   dict[int, int] = {}
        winding = getattr(m, "winding", None)
        if winding is not None:
            try:
                for ph in range(3):
                    for cs in winding.coil_sides_for_phase(ph):
                        si = cs.slot_idx % self.Qs
                        self.slot_phase[si] = cs.phase
                        self.slot_dir[si]   = cs.direction
            except Exception:
                pass

    # ── Low-level DXF writers ─────────────────────────────────────────────────

    def _w(self, code: int, value):
        self.lines.append(f"{code:3d}")
        self.lines.append(str(value))

    def _entity(self, name: str, layer: str):
        self._w(0, name)
        self._w(8, layer)
        self.entity_count += 1
        self.layer_counts[layer] = self.layer_counts.get(layer, 0) + 1

    def _circle(self, cx: float, cy: float, r: float, layer: str):
        self._entity("CIRCLE", layer)
        self._w(10, f"{cx:.6f}"); self._w(20, f"{cy:.6f}"); self._w(30, "0.0")
        self._w(40, f"{r:.6f}")

    def _arc(self, cx: float, cy: float, r: float,
             a_start: float, a_end: float, layer: str):
        """Arc counter-clockwise from a_start to a_end (degrees)."""
        # Normalise so arc always goes CCW and doesn't cross 0→360 badly
        if a_end <= a_start:
            a_end += 360.0
        self._entity("ARC", layer)
        self._w(10, f"{cx:.6f}"); self._w(20, f"{cy:.6f}"); self._w(30, "0.0")
        self._w(40, f"{r:.6f}")
        self._w(50, f"{a_start:.6f}"); self._w(51, f"{a_end:.6f}")

    def _line(self, x1, y1, x2, y2, layer: str):
        self._entity("LINE", layer)
        self._w(10, f"{x1:.6f}"); self._w(20, f"{y1:.6f}"); self._w(30, "0.0")
        self._w(11, f"{x2:.6f}"); self._w(21, f"{y2:.6f}"); self._w(31, "0.0")

    def _text(self, x: float, y: float, height: float,
              text: str, layer: str, angle: float = 0.0):
        self._entity("TEXT", layer)
        self._w(10, f"{x:.4f}"); self._w(20, f"{y:.4f}"); self._w(30, "0.0")
        self._w(40, f"{height:.4f}")
        self._w(1,  text)
        if abs(angle) > 0.01:
            self._w(50, f"{angle:.2f}")

    def _pt(self, r: float, theta_rad: float):
        """Polar → Cartesian."""
        return r * math.cos(theta_rad), r * math.sin(theta_rad)

    # ── Header & tables ───────────────────────────────────────────────────────

    def _write_header(self):
        w = self._w
        ext = self.R_so * 1.4
        w(0, "SECTION"); w(2, "HEADER")
        w(9, "$ACADVER"); w(1, "AC1009")       # R12 — widest compatibility
        w(9, "$INSUNITS"); w(70, "4")          # 4 = mm
        w(9, "$MEASUREMENT"); w(70, "1")       # metric
        w(9, "$EXTMIN")
        w(10, f"{-ext:.3f}"); w(20, f"{-ext:.3f}"); w(30, "0.0")
        w(9, "$EXTMAX")
        w(10, f"{ext + self.R_so * 0.9:.3f}")
        w(20, f"{ext:.3f}"); w(30, "0.0")
        w(9, "$LIMMIN")
        w(10, f"{-ext:.3f}"); w(20, f"{-ext:.3f}")
        w(9, "$LIMMAX")
        w(10, f"{ext + self.R_so * 0.9:.3f}"); w(20, f"{ext:.3f}")
        w(0, "ENDSEC")

    def _write_tables(self):
        w = self._w
        w(0, "SECTION"); w(2, "TABLES")

        # Line types
        w(0, "TABLE"); w(2, "LTYPE"); w(70, "3")
        w(0, "LTYPE"); w(2, "CONTINUOUS")
        w(70, "64"); w(3, "Solid"); w(72, "65"); w(73, "0"); w(40, "0.0")
        w(0, "LTYPE"); w(2, "DASHED")
        w(70, "64"); w(3, "__ __ __"); w(72, "65"); w(73, "2")
        w(40, "0.75"); w(49, "0.5"); w(49, "-0.25")
        w(0, "LTYPE"); w(2, "CENTER")
        w(70, "64"); w(3, "___ _ ___"); w(72, "65"); w(73, "4")
        w(40, "2.0"); w(49, "1.25"); w(49, "-0.25"); w(49, "0.25"); w(49, "-0.25")
        w(0, "ENDTAB")

        # Layers
        layer_defs = [
            ("STATOR_IRON",  4,  "CONTINUOUS"),   # cyan
            ("STATOR_SLOTS", 7,  "CONTINUOUS"),   # white
            ("WINDING_A",    1,  "CONTINUOUS"),   # red
            ("WINDING_B",    3,  "CONTINUOUS"),   # green
            ("WINDING_C",    5,  "CONTINUOUS"),   # blue
            ("ROTOR_IRON",   6,  "CONTINUOUS"),   # magenta
            ("MAGNETS_N",    30, "CONTINUOUS"),   # orange
            ("MAGNETS_S",    40, "CONTINUOUS"),   # gold
            ("AIRGAP",       8,  "DASHED"),       # grey dashed
            ("CENTERLINES",  8,  "CENTER"),       # grey centre
            ("DIMENSIONS",   2,  "CONTINUOUS"),   # yellow
            ("LABELS",       7,  "CONTINUOUS"),   # white
        ]
        w(0, "TABLE"); w(2, "LAYER"); w(70, str(len(layer_defs)))
        for lname, col, ltype in layer_defs:
            w(0, "LAYER"); w(2, lname)
            w(70, "64"); w(62, str(col)); w(6, ltype)
        w(0, "ENDTAB")
        w(0, "ENDSEC")

    # ── Entity sections ───────────────────────────────────────────────────────

    def _write_entities(self):
        self._w(0, "SECTION"); self._w(2, "ENTITIES")

        self._draw_stator()
        self._draw_rotor()
        self._draw_airgap_midline()
        self._draw_centerlines()
        self._draw_dimensions()
        self._draw_title_block()

        self._w(0, "ENDSEC")
        self._w(0, "EOF")

    # ── Stator ────────────────────────────────────────────────────────────────

    def _draw_stator(self):
        # Outer and bore boundaries
        self._circle(0, 0, self.R_so, "STATOR_IRON")
        self._circle(0, 0, self.R_si, "STATOR_IRON")

        for s in range(self.Qs):
            self._draw_slot(s)

        # Tooth-tip arcs between slots (bore surface)
        for s in range(self.Qs):
            th_c    = s * self.slot_pitch + self.slot_pitch / 2
            a_open  = math.asin(min(self.b_open / (2 * self.R_si), 0.9999))
            th_next = (s + 1) * self.slot_pitch + self.slot_pitch / 2
            th_start = math.degrees(th_c + a_open)
            th_end   = math.degrees(th_next - a_open)
            if th_end > th_start:
                self._arc(0, 0, self.R_si, th_start, th_end, "STATOR_IRON")

    def _draw_slot(self, s: int):
        """Draw one slot outline + winding arcs."""
        th_c    = s * self.slot_pitch + self.slot_pitch / 2

        a_open  = math.asin(min(self.b_open / (2 * self.R_si + 1e-9), 0.9999))
        R_top   = self.R_si + self.h_slot
        a_top   = math.asin(min(self.b_slot / (2 * R_top + 1e-9), 0.9999))

        th_op_l  = th_c - a_open;  th_op_r  = th_c + a_open
        th_top_l = th_c - a_top;   th_top_r = th_c + a_top

        x_op_l,  y_op_l  = self._pt(self.R_si, th_op_l)
        x_op_r,  y_op_r  = self._pt(self.R_si, th_op_r)
        x_top_l, y_top_l = self._pt(R_top, th_top_l)
        x_top_r, y_top_r = self._pt(R_top, th_top_r)

        # Side walls
        self._line(x_op_l, y_op_l, x_top_l, y_top_l, "STATOR_SLOTS")
        self._line(x_op_r, y_op_r, x_top_r, y_top_r, "STATOR_SLOTS")
        # Top arc
        self._arc(0, 0, R_top, math.degrees(th_top_l), math.degrees(th_top_r), "STATOR_SLOTS")
        # Opening arc (slot mouth at bore)
        self._arc(0, 0, self.R_si, math.degrees(th_op_l), math.degrees(th_op_r), "STATOR_SLOTS")

        # Layer separator at mid-depth
        R_sep   = self.R_si + self.h_slot / 2
        a_sep   = math.asin(min(self.b_slot / (2 * R_sep + 1e-9), 0.9999))
        x_sl, y_sl = self._pt(R_sep, th_c - a_sep)
        x_sr, y_sr = self._pt(R_sep, th_c + a_sep)
        self._line(x_sl, y_sl, x_sr, y_sr, "STATOR_SLOTS")

        # Winding conductor arcs (one per layer, per phase)
        ph    = self.slot_phase.get(s, 0)
        dr    = self.slot_dir.get(s, 1)
        lyr_w = ["WINDING_A", "WINDING_B", "WINDING_C"][ph % 3]

        # Layer 0 centre arc
        R_l0 = self.R_si + self.h_slot * 0.25
        self._arc(0, 0, R_l0,
                  math.degrees(th_op_l), math.degrees(th_op_r), lyr_w)
        # Layer 1 centre arc
        R_l1 = self.R_si + self.h_slot * 0.75
        a_l1 = math.asin(min(self.b_slot / (2 * R_l1 + 1e-9), 0.9999))
        self._arc(0, 0, R_l1,
                  math.degrees(th_c - a_l1), math.degrees(th_c + a_l1), lyr_w)

    # ── Rotor ─────────────────────────────────────────────────────────────────

    def _draw_rotor(self):
        # Boundary circles
        self._circle(0, 0, self.R_ro,    "ROTOR_IRON")
        self._circle(0, 0, self.R_mi,    "ROTOR_IRON")
        self._circle(0, 0, self.R_shaft, "ROTOR_IRON")

        half_mag_deg = math.degrees(math.pi * self.alpha_p / self.poles)

        for p in range(self.poles):
            pc_deg   = math.degrees((p + 0.5) * self.pole_pitch)
            mag_l    = pc_deg - half_mag_deg
            mag_r    = pc_deg + half_mag_deg
            lyr_pm   = "MAGNETS_N" if p % 2 == 0 else "MAGNETS_S"

            # Magnet outer arc
            self._arc(0, 0, self.R_ro, mag_l, mag_r, lyr_pm)
            # Magnet inner arc
            self._arc(0, 0, self.R_mi, mag_l, mag_r, lyr_pm)
            # Magnet side walls
            for ang_deg in (mag_l, mag_r):
                ang = math.radians(ang_deg)
                xi, yi = self._pt(self.R_mi, ang)
                xo, yo = self._pt(self.R_ro, ang)
                self._line(xi, yi, xo, yo, lyr_pm)

            # Inter-pole rotor iron arcs (between this pole's right edge
            # and next pole's left edge)
            next_pc_deg = math.degrees((p + 1 + 0.5) * self.pole_pitch)
            inter_l = mag_r
            inter_r = next_pc_deg - half_mag_deg
            if inter_r > inter_l:
                self._arc(0, 0, self.R_ro, inter_l, inter_r, "ROTOR_IRON")
                self._arc(0, 0, self.R_mi, inter_l, inter_r, "ROTOR_IRON")

        # Close the last gap (wraps around 0°)
        last_pole_r = math.degrees((self.poles - 0.5) * self.pole_pitch) + half_mag_deg
        first_pole_l = math.degrees(0.5 * self.pole_pitch) - half_mag_deg
        # Arc from last pole right-edge to 360 and from 0 to first pole left-edge
        if last_pole_r < 360.0:
            self._arc(0, 0, self.R_ro, last_pole_r, 360.0, "ROTOR_IRON")
            self._arc(0, 0, self.R_mi, last_pole_r, 360.0, "ROTOR_IRON")
        if first_pole_l > 0.0:
            self._arc(0, 0, self.R_ro, 0.0, first_pole_l, "ROTOR_IRON")
            self._arc(0, 0, self.R_mi, 0.0, first_pole_l, "ROTOR_IRON")

    # ── Airgap midline ────────────────────────────────────────────────────────

    def _draw_airgap_midline(self):
        self._circle(0, 0, self.R_ag, "AIRGAP")

    # ── Centrelines ───────────────────────────────────────────────────────────

    def _draw_centerlines(self):
        r = self.R_shaft * 0.8
        self._line(-r, 0, r, 0, "CENTERLINES")
        self._line(0, -r, 0, r, "CENTERLINES")

    # ── Dimensions ────────────────────────────────────────────────────────────

    def _draw_dimensions(self):
        s = self.scale
        m = self.motor
        h_txt = self.R_so * 0.025   # text height proportional to motor size

        dims = [
            (self.R_shaft, f"Ø{2*self.R_shaft:.1f} shaft bore",  42),
            (self.R_mi,    f"Ø{2*self.R_mi:.1f} magnet ID",      32),
            (self.R_ro,    f"Ø{2*self.R_ro:.1f} rotor OD",       22),
            (self.R_si,    f"Ø{2*self.R_si:.1f} stator bore",    12),
            (self.R_so,    f"Ø{2*self.R_so:.1f} stator OD",       4),
        ]
        for r_mm, label, angle_deg in dims:
            ang  = math.radians(angle_deg)
            # Leader from circle to outside
            x_c, y_c = self._pt(r_mm, ang)
            r_end    = self.R_so * 1.18
            x_e, y_e = self._pt(r_end, ang)
            self._line(x_c, y_c, x_e, y_e, "DIMENSIONS")
            # Tick at circle
            pa = ang + math.pi / 2
            tk = h_txt * 0.8
            self._line(x_c - tk*math.cos(pa), y_c - tk*math.sin(pa),
                       x_c + tk*math.cos(pa), y_c + tk*math.sin(pa), "DIMENSIONS")
            self._text(x_e + h_txt, y_e, h_txt, label, "DIMENSIONS", 0)

        # Airgap thickness callout (vertical leader at 90°)
        x_ro, y_ro = self._pt(self.R_ro, math.pi / 2)
        x_si, y_si = self._pt(self.R_si, math.pi / 2)
        self._line(x_ro, y_ro, x_ro - h_txt * 6, y_ro, "DIMENSIONS")
        self._line(x_si, y_si, x_si - h_txt * 6, y_si, "DIMENSIONS")
        x_arr = x_ro - h_txt * 5
        self._line(x_arr, y_ro, x_arr, y_si, "DIMENSIONS")
        ag_mm = m.airgap * s
        self._text(x_arr - h_txt * 0.5, (y_ro + y_si) / 2,
                   h_txt, f"g={ag_mm:.1f}", "DIMENSIONS", 90)

        # Magnet thickness callout
        x_mi, y_mi = self._pt(self.R_mi, math.radians(270))
        x_ro2, y_ro2 = self._pt(self.R_ro, math.radians(270))
        self._line(x_mi,  y_mi,  x_mi  + h_txt * 6, y_mi,  "DIMENSIONS")
        self._line(x_ro2, y_ro2, x_ro2 + h_txt * 6, y_ro2, "DIMENSIONS")
        x_arr2 = x_mi + h_txt * 5
        self._line(x_arr2, y_mi, x_arr2, y_ro2, "DIMENSIONS")
        self._text(x_arr2 + h_txt * 0.5, (y_mi + y_ro2) / 2,
                   h_txt, f"tm={self.t_m:.1f}", "DIMENSIONS", 270)

    # ── Title block ───────────────────────────────────────────────────────────

    def _draw_title_block(self):
        m   = self.motor
        s   = self.scale
        h   = self.R_so * 0.025   # row height
        tx  = self.R_so * 1.22    # X start of title block
        ty  = self.R_so * 0.95    # Y start

        rows = [
            ("DRAWING",          "PMSM Cross-Section"),
            ("POLES / SLOTS",    f"{self.poles}p / {self.Qs}s"),
            ("RATED POWER",      f"{m.rated_power / 1000:.0f} kW"),
            ("RATED SPEED",      f"{m.rated_speed:.0f} rpm"),
            ("STATOR OD",        f"Ø{2 * self.R_so:.2f}"),
            ("STATOR BORE",      f"Ø{2 * self.R_si:.2f}"),
            ("ROTOR OD",         f"Ø{2 * self.R_ro:.2f}"),
            ("SHAFT BORE",       f"Ø{2 * self.R_shaft:.2f}"),
            ("AIRGAP g",         f"{m.airgap * s:.2f}"),
            ("MAGNET t_m",       f"{self.t_m:.2f}"),
            ("MAGNET arc α_p",   f"{self.alpha_p:.3f}"),
            ("STACK LENGTH",     f"{m.stack_length * s:.1f}"),
            ("WINDING",          f"N_coil={getattr(m,'turns_per_coil','-')}  layers=2"),
            ("SCALE",            f"1:1  (units={'mm' if s==1000 else 'm'})"),
        ]

        col_w = self.R_so * 0.50   # width of label column
        row_h = h * 2.2
        n     = len(rows)

        # Box
        box_x1 = tx - h * 0.5
        box_x2 = tx + col_w * 2 + h * 0.5
        box_y1 = ty + h * 1.5
        box_y2 = ty - (n) * row_h - h * 0.5
        for (xa, ya), (xb, yb) in [
            ((box_x1, box_y1), (box_x2, box_y1)),
            ((box_x2, box_y1), (box_x2, box_y2)),
            ((box_x2, box_y2), (box_x1, box_y2)),
            ((box_x1, box_y2), (box_x1, box_y1)),
            ((tx + col_w, box_y1), (tx + col_w, box_y2)),  # divider
        ]:
            self._line(xa, ya, xb, yb, "DIMENSIONS")

        for i, (label, value) in enumerate(rows):
            y = ty - i * row_h
            self._text(tx, y, h, label + ":", "LABELS")
            self._text(tx + col_w + h * 0.3, y, h, value, "LABELS")
