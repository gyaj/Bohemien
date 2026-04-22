"""
Manufacturing Report Generator for Bohemien_Motor_Designer.

Produces a complete, self-contained document covering every dimension,
specification, and material quantity a manufacturer needs to build the motor.

Sections
--------
1.  Motor overview & ratings
2.  Stator lamination dimensions
3.  Slot & insulation specification
4.  Winding specification (wire, strands, fill)
5.  Winding connection table (slot-by-slot, all 3 phases)
6.  Coil group / terminal assignment
7.  Rotor & magnet specification
8.  Bill of Materials (masses)
9.  Electrical parameters
10. Thermal & performance budget
11. Design Rule Check summary
12. Assembly tolerances & notes

Usage
-----
    from Bohemien_Motor_Designer.core.manufacturing_report import ManufacturingReport
    rpt = ManufacturingReport(motor)
    print(rpt.text())               # plain-text report
    rpt.save("motor_report.txt")    # write to file
"""
from __future__ import annotations
import math
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

MU0 = 4e-7 * math.pi
RHO_CU_20 = 1.72e-8   # Ω·m  copper at 20°C
ALPHA_CU   = 0.00393  # /°C  copper TCR
STACK_FACTOR = 0.97   # lamination stacking factor (typical punched steel)

# Standard wire diameters (IEC 60228 / AWG equivalents, mm)
# Used to find nearest standard wire to a computed diameter
_STANDARD_WIRE_MM = [
    0.200, 0.224, 0.250, 0.280, 0.315, 0.355, 0.400, 0.450, 0.500,
    0.560, 0.630, 0.710, 0.800, 0.900, 1.000, 1.060, 1.120, 1.180,
    1.250, 1.320, 1.400, 1.500, 1.600, 1.700, 1.800, 1.900, 2.000,
    2.120, 2.240, 2.360, 2.500, 2.650, 2.800, 3.000, 3.150, 3.350,
    3.550, 3.750, 4.000, 4.250, 4.500, 4.750, 5.000, 5.300, 5.600,
]


def _nearest_standard_wire(d_mm: float) -> float:
    """Return nearest standard wire diameter [mm]."""
    return min(_STANDARD_WIRE_MM, key=lambda x: abs(x - d_mm))


# ── Wire packing model ─────────────────────────────────────────────────────────

@dataclass
class WireSpec:
    """Conductor specification for one slot layer."""
    wire_diameter_mm: float        # bare wire [mm]
    insulation_build_mm: float     # total insulation build (both sides) [mm]
    n_strands: int                 # parallel strands per turn
    n_turns_per_layer: int
    n_layers: int                  # winding layers (1 or 2)
    conductor_area_mm2: float      # total copper area per turn [mm²]
    slot_fill_achieved: float      # actual fill with chosen wire
    wire_length_per_phase_m: float
    resistance_per_phase_ohm: float
    copper_mass_per_phase_kg: float

    @property
    def overall_diameter_mm(self) -> float:
        return self.wire_diameter_mm + self.insulation_build_mm

    @property
    def awg_approx(self) -> str:
        """Closest AWG designation (informational)."""
        d = self.wire_diameter_mm
        if d <= 0:
            return "N/A"
        awg_n = round(36 - 20 * math.log10(d / 0.127))  # Brown & Sharpe, d in mm
        awg_n = max(-3, min(40, awg_n))
        if awg_n <= 0:
            labels = {0: "1", -1: "2", -2: "3", -3: "4"}
            return f"AWG {labels.get(awg_n, str(awg_n))}"
        return f"AWG {awg_n}"


# ── End-winding model ──────────────────────────────────────────────────────────

def _end_winding_length(motor) -> float:
    """
    Estimate single-side end-winding overhang length [m].

    Uses the D'Angelo approximation:
        L_ew = π/2 * τ_coil  +  2 * clearance

    where τ_coil = chord of the coil span arc at mean coil radius.
    clearance ≈ 10–15 mm for form-wound, 5 mm for random-wound.
    """
    Qs      = motor.slots
    span    = motor.winding.coil_span  # slots
    R_bore  = motor.stator.inner_radius if motor.stator else (
              motor.rotor_outer_radius + motor.airgap)
    sp      = motor.stator.slot_profile if motor.stator else None
    depth   = sp.depth() if sp else 0.02
    R_mean  = R_bore + depth / 2.0         # mean coil radius

    slot_pitch_rad = 2 * math.pi / Qs
    arc_angle      = span * slot_pitch_rad
    chord          = 2 * R_mean * math.sin(arc_angle / 2)

    # Semicircular end turn approximation + straight clearance section
    L_ew = math.pi / 2 * chord + 0.010    # 10 mm clearance for random-wound
    return float(L_ew)


# ── Wire spec calculation ──────────────────────────────────────────────────────

def _compute_wire_spec(motor, operating_temp_C: float = 120.0) -> WireSpec:
    """
    Size the conductor wire from slot geometry and current requirements.

    Strategy:
    1. Compute required conductor area from slot fill target.
    2. If single wire > 3.55 mm, split into multiple strands.
    3. Select nearest standard wire diameter.
    4. Compute achieved fill, resistance, wire length, mass.
    """
    sp          = motor.stator.slot_profile if motor.stator else None
    A_slot      = sp.area() if sp else 1.5e-4
    ff_target   = getattr(motor, "slot_fill_factor", 0.45)
    N_coil      = motor.turns_per_coil        # turns per coil-side
    n_lay       = motor.winding.layers        # 1 or 2
    turns_slot  = N_coil * n_lay              # turns in one slot
    N_series    = motor.winding.total_series_turns_per_phase

    # Required conductor area per turn
    A_cond_req = A_slot * ff_target / turns_slot   # m²

    # Insulation build: IEC 60317 heavy build approx: 0.06 mm per side for
    # d < 1.5 mm, 0.08 mm for d < 3 mm, 0.10 mm for d >= 3 mm
    def ins_build(d_mm):
        if d_mm < 1.5:   return 0.12
        if d_mm < 3.0:   return 0.16
        return 0.20

    # Stranding threshold: single wire > 3.55 mm is impractical
    MAX_SINGLE_D = 3.55   # mm

    d_single = math.sqrt(4 * A_cond_req / math.pi) * 1000  # mm
    if d_single > MAX_SINGLE_D:
        # Split into parallel strands
        n_strands = math.ceil((d_single / MAX_SINGLE_D) ** 2)
        d_strand  = math.sqrt(4 * A_cond_req / (n_strands * math.pi)) * 1000
    else:
        n_strands = 1
        d_strand  = d_single

    d_std = _nearest_standard_wire(d_strand)
    d_ins = ins_build(d_std)

    # ── Packing check: strands must physically fit across the slot width ──
    # Slot width at bore (conservative — narrowest point for parallel-tooth slot)
    sp = motor.stator.slot_profile if motor.stator else None
    slot_w_mm = (sp.area() / sp.depth() * 1000) if sp else 999.0
    d_overall_trial = d_std + d_ins  # mm overall per strand

    # Max strands that fit side-by-side in one row across slot width
    max_across = max(1, int(slot_w_mm / d_overall_trial))
    # Required rows to fit all turns × strands
    n_needed_rows = math.ceil((turns_slot * n_strands) / max_across)

    # If strands don't fit (more rows than we have depth for), increase strand count
    # to allow smaller wire that fits across
    if d_overall_trial * n_strands > slot_w_mm:
        # Recompute: find minimum n_strands such that d_overall ≤ slot_w / n_strands
        for ns in range(n_strands, 50):
            d_s = math.sqrt(4 * A_cond_req / (ns * math.pi)) * 1000
            d_s_std = _nearest_standard_wire(d_s)
            d_overall_ns = d_s_std + ins_build(d_s_std)
            if d_overall_ns * ns <= slot_w_mm:
                n_strands = ns
                d_std = d_s_std
                d_ins = ins_build(d_std)
                break

    A_strand   = math.pi * (d_std / 1000) ** 2 / 4   # m²
    A_cond_act = A_strand * n_strands

    # Packing factor: for stranded conductors use 0.785 (hexagonal packing ≈ π/4)
    # Achieved fill = turns × strands × (d_std + d_ins)² × π/4 / A_slot
    d_overall  = (d_std + d_ins) / 1000   # m
    ff_act     = turns_slot * n_strands * (math.pi * d_overall**2 / 4) / A_slot
    ff_act     = min(ff_act, 0.99)        # cap at physical max

    # Wire length per turn
    L_stack  = motor.stack_length
    L_ew     = _end_winding_length(motor)
    L_turn   = 2 * L_stack + 2 * L_ew        # both active sides + both ends

    # Wire length per phase (N_series turns in series)
    # Account for parallel paths
    paths    = getattr(motor.winding, "parallel_paths", 1)
    L_phase  = N_series * L_turn / paths     # total wire per parallel path
    # But we have n_strands parallel wires per turn, all the same length
    L_strand_per_phase = L_phase             # each strand travels same path

    # Resistance per phase at operating temperature
    rho = RHO_CU_20 * (1 + ALPHA_CU * (operating_temp_C - 20))
    R_phase = rho * L_strand_per_phase / (A_strand * n_strands)   # Ω

    # Copper mass per phase
    rho_cu    = 8960   # kg/m³
    vol_phase = A_strand * n_strands * L_strand_per_phase
    mass_phase = vol_phase * rho_cu

    return WireSpec(
        wire_diameter_mm        = float(d_std),
        insulation_build_mm     = float(d_ins),
        n_strands               = int(n_strands),
        n_turns_per_layer       = int(N_coil),
        n_layers                = int(n_lay),
        conductor_area_mm2      = float(A_cond_act * 1e6),
        slot_fill_achieved      = float(ff_act),
        wire_length_per_phase_m = float(L_strand_per_phase),
        resistance_per_phase_ohm= float(R_phase),
        copper_mass_per_phase_kg= float(mass_phase),
    )


# ── BOM calculation ────────────────────────────────────────────────────────────

@dataclass
class BillOfMaterials:
    # Copper
    copper_mass_stator_kg: float   # total 3-phase copper
    copper_volume_m3: float

    # Stator steel
    stator_iron_mass_kg: float
    n_laminations: int
    lamination_thickness_mm: float

    # Rotor steel
    rotor_iron_mass_kg: float

    # Magnets
    n_magnets: int
    magnet_mass_per_pole_kg: float
    magnet_mass_total_kg: float
    magnet_volume_per_pole_m3: float
    magnet_material: str

    # Summary
    total_active_mass_kg: float

    def to_rows(self) -> list[tuple]:
        """Return [(item, qty, unit, mass_kg, notes)] rows."""
        return [
            ("Stator laminations",   self.n_laminations, "pcs",
             self.stator_iron_mass_kg, f"{self.lamination_thickness_mm:.2f} mm each"),
            ("Rotor laminations",    "—",                "lot",
             self.rotor_iron_mass_kg, "same stack length"),
            ("Magnets",              self.n_magnets,     "pcs",
             self.magnet_mass_total_kg,
             f"{self.magnet_mass_per_pole_kg*1000:.1f} g each — {self.magnet_material}"),
            ("Stator copper (3-ph)", "—",                "lot",
             self.copper_mass_stator_kg, "includes end-windings"),
            ("TOTAL active material","—",                "—",
             self.total_active_mass_kg, ""),
        ]


def _compute_bom(motor, wire_spec: WireSpec,
                 material_lib=None) -> BillOfMaterials:
    from Bohemien_Motor_Designer.materials.library import MaterialLibrary
    mlib = material_lib or MaterialLibrary()

    # ── Steel ──
    st     = motor.stator
    lam_t  = getattr(st, "lamination_thickness", 0.00035) if st else 0.00035
    lam_name = getattr(st, "lamination_material", "M270-35A") if st else "M270-35A"
    try:
        lam_mat = mlib.lamination(lam_name)
        steel_density = lam_mat.density
    except Exception:
        steel_density = 7650

    L      = motor.stack_length
    n_lam  = max(1, round(L * STACK_FACTOR / lam_t))

    # Stator cross-section area (annular ring minus slots)
    if st:
        R_so    = st.outer_radius
        R_si    = st.inner_radius
        sp      = st.slot_profile
        A_slot  = sp.area() if sp else 1.5e-4
        A_stator_gross = math.pi * (R_so**2 - R_si**2)
        A_slots_total  = motor.slots * A_slot
        A_stator_net   = max(A_stator_gross - A_slots_total, 0.01)
    else:
        A_stator_net = 0.05

    stator_vol  = A_stator_net * L * STACK_FACTOR
    stator_mass = stator_vol * steel_density

    # Rotor cross-section (solid iron, minus shaft bore and magnets)
    R_ro = motor.rotor_outer_radius
    R_ri = motor.rotor_inner_radius
    t_m  = getattr(motor, "magnet_thickness", 0.006)
    alpha= getattr(motor, "magnet_width_fraction", 0.85)
    poles= motor.poles

    mag_arc  = alpha * 2 * math.pi / poles   # radians
    A_mag    = mag_arc * R_ro * t_m           # arc × thickness (per pole)
    A_rotor_gross = math.pi * ((R_ro - t_m)**2 - R_ri**2)
    A_rotor_net   = max(A_rotor_gross - poles * A_mag / L, A_rotor_gross * 0.5)

    rotor_vol  = A_rotor_net * L * STACK_FACTOR
    rotor_mass = rotor_vol * steel_density

    # ── Magnets ──
    try:
        mag_mat  = mlib.magnet(getattr(motor, "magnet_material", "N42SH"))
        mag_density = mag_mat.density
    except Exception:
        mag_density = 7500

    # Magnet volume per pole: arc-segment approximation
    R_mag_mid = R_ro - t_m / 2
    vol_per_pole = mag_arc * R_mag_mid * t_m * L
    mass_per_pole = vol_per_pole * mag_density
    n_mags = poles

    # ── Copper ──
    cu_mass_total = wire_spec.copper_mass_per_phase_kg * 3

    total = stator_mass + rotor_mass + mass_per_pole * poles + cu_mass_total

    return BillOfMaterials(
        copper_mass_stator_kg    = float(cu_mass_total),
        copper_volume_m3         = float(wire_spec.copper_mass_per_phase_kg * 3 / 8960),
        stator_iron_mass_kg      = float(stator_mass),
        n_laminations            = int(n_lam),
        lamination_thickness_mm  = float(lam_t * 1000),
        rotor_iron_mass_kg       = float(rotor_mass),
        n_magnets                = int(n_mags),
        magnet_mass_per_pole_kg  = float(mass_per_pole),
        magnet_mass_total_kg     = float(mass_per_pole * poles),
        magnet_volume_per_pole_m3= float(vol_per_pole),
        magnet_material          = getattr(motor, "magnet_material", "N42SH"),
        total_active_mass_kg     = float(total),
    )


# ── Winding connection table ───────────────────────────────────────────────────

def _winding_table(motor) -> str:
    """
    Full 96-entry winding connection table (48 slots × 2 layers).

    Columns: Slot | Layer | Phase | Direction | Coil# | Connected to
    """
    winding = motor.winding
    Qs      = motor.slots
    phases  = "ABC"

    # Build a flat list, sort by slot then layer
    rows = []
    coil_counters = {ph: 0 for ph in range(3)}
    for ph in range(3):
        sides = winding.coil_sides_for_phase(ph)
        for cs in sides:
            coil_counters[cs.phase] += 1
            rows.append({
                "slot":    cs.slot_idx,
                "layer":   cs.layer,
                "phase":   cs.phase,
                "dir":     cs.direction,
                "coil_n":  coil_counters[cs.phase],
            })
    rows.sort(key=lambda r: (r["slot"], r["layer"]))

    # Assign coil numbers correctly (sequential per phase)
    ph_seq = {0: 0, 1: 0, 2: 0}
    for r in rows:
        ph_seq[r["phase"]] += 1
        r["coil_n"] = ph_seq[r["phase"]]

    lines = []
    hdr = (f"{'Slot':>4}  {'Lay':>3}  {'Ph':>2}  {'Dir':>4}  "
           f"{'Coil#':>6}  {'Conductor label':<20}")
    lines.append(hdr)
    lines.append("─" * len(hdr))

    for r in rows:
        ph_ch = phases[r["phase"]]
        dir_s = "GO " if r["dir"] > 0 else "RET"
        label = f"{ph_ch}{r['coil_n']:02d}-{'U' if r['layer']==0 else 'L'}"
        lines.append(
            f"{r['slot']+1:>4}  {r['layer']:>3}  {ph_ch:>2}  {dir_s:>4}  "
            f"{r['coil_n']:>6}  {label:<20}"
        )

    return "\n".join(lines)


def _coil_groups(motor) -> str:
    """
    Coil group summary: groups by phase showing series/parallel connections
    and terminal labels (U1, U2, V1, V2, W1, W2).
    """
    winding = motor.winding
    N_series = winding.total_series_turns_per_phase
    paths    = getattr(winding, "parallel_paths", 1)
    coils_per_phase = motor.slots // motor.phases  # total coil sides / 2
    lines = []

    terminal_labels = {0: ("U1", "U2"), 1: ("V1", "V2"), 2: ("W1", "W2")}

    for ph in range(3):
        t_in, t_out = terminal_labels[ph]
        ph_ch = "ABC"[ph]
        sides = winding.coil_sides_for_phase(ph)
        go_slots  = sorted([cs.slot_idx + 1 for cs in sides if cs.direction > 0])
        ret_slots = sorted([cs.slot_idx + 1 for cs in sides if cs.direction < 0])
        lines.append(
            f"  Phase {ph_ch}  ({t_in} → {t_out})  |  "
            f"N_series={N_series}  paths={paths}  "
            f"coil span={winding.coil_span} slots"
        )
        lines.append(
            f"    GO  slots: {go_slots}"
        )
        lines.append(
            f"    RET slots: {ret_slots}"
        )

    # Star point / delta note
    conn = getattr(motor, "connection", "star")
    if conn == "star":
        lines.append("")
        lines.append(
            "  Connection: STAR (Y)  —  join U2, V2, W2 as neutral point."
        )
    else:
        lines.append("")
        lines.append(
            "  Connection: DELTA (Δ)  —  U2→V1, V2→W1, W2→U1."
        )

    return "\n".join(lines)


# ── Slot insulation ────────────────────────────────────────────────────────────

def _slot_insulation(motor) -> str:
    """
    Slot liner and wedge specification.
    Based on IEC 60085 thermal class and rated voltage.
    """
    Vdc = 400 * math.sqrt(2)  # conservative: 400 V bus → ~566 V pk
    try:
        from Bohemien_Motor_Designer.core.specs import DesignSpec
        if hasattr(motor, "spec") and motor.spec:
            Vdc = motor.spec.drive.dc_bus_voltage
    except Exception:
        pass

    # Select insulation class
    ins_class = "F (155°C)"   # default for traction/EV
    liner_t   = 0.30           # mm  — standard slot liner
    wedge_t   = 1.0            # mm  — slot closing wedge
    inter_layer_t = 0.20       # mm  — inter-layer separator

    if Vdc > 800:
        liner_t = 0.40
        ins_class = "H (180°C) — required for SiC drive > 800 V"
    elif Vdc > 1000:
        liner_t = 0.50
        ins_class = "H (180°C) — partial discharge testing mandatory"

    sp   = motor.stator.slot_profile if motor.stator else None
    A_slot  = sp.area() * 1e6 if sp else 150.0   # mm²
    depth   = sp.depth() * 1000 if sp else 22.0
    width_m = sp.area() / sp.depth() if sp else 0.008
    perim   = 2 * depth + width_m * 1000  # mm — approx

    liner_area_loss = perim * liner_t   # mm²

    lines = [
        f"  Thermal class         : {ins_class}",
        f"  Slot liner thickness  : {liner_t:.2f} mm  (both sides + bottom)",
        f"  Inter-layer separator : {inter_layer_t:.2f} mm",
        f"  Slot closing wedge    : {wedge_t:.1f} mm thick, nonmagnetic (glass-epoxy G10/FR4)",
        f"  Liner material        : Nomex 410 or Kapton 500HN",
        f"  Liner area loss/slot  : {liner_area_loss:.1f} mm²  "
          f"({liner_area_loss/A_slot*100:.1f}% of gross slot area)",
        f"  Impregnation          : Vacuum Pressure Impregnation (VPI) — Class F resin",
        f"  Varnish               : Solventless epoxy, 130°C min rating",
    ]
    return "\n".join(lines)


# ── Rotor / magnet section ─────────────────────────────────────────────────────

def _magnet_spec(motor, bom: BillOfMaterials) -> str:
    poles  = motor.poles
    R_ro   = motor.rotor_outer_radius
    t_m    = getattr(motor, "magnet_thickness", 0.006)
    alpha  = getattr(motor, "magnet_width_fraction", 0.85)
    L      = motor.stack_length

    mag_arc_deg  = alpha * 360 / poles
    half_ang     = math.radians(mag_arc_deg / 2)
    chord_outer  = 2 * R_ro * math.sin(half_ang)
    chord_inner  = 2 * (R_ro - t_m) * math.sin(half_ang)
    arc_length   = R_ro * math.radians(mag_arc_deg)

    try:
        from Bohemien_Motor_Designer.materials.library import MaterialLibrary
        mlib = MaterialLibrary()
        mag  = mlib.magnet(getattr(motor, "magnet_material", "N42SH"))
        Br    = mag.remanence_Br
        Hcj   = mag.HcJ
        BHmax = mag.energy_product
        Br_hot  = mag.Br_at(120.0)
        Hcj_hot = mag.Hcj_at(120.0)
        mat_line = (f"  Grade                 : {mag.name}  "
                    f"(Br={Br:.2f}T@20°C / {Br_hot:.2f}T@120°C, "
                    f"HcJ={Hcj:.0f}kA/m@20°C / {Hcj_hot:.0f}kA/m@120°C, "
                    f"BHmax={BHmax:.0f}kJ/m3)")
    except Exception:
        mat_line = f"  Grade                 : {getattr(motor, 'magnet_material', 'N42SH')}"

    lines = [
        f"  Number of poles       : {poles}",
        f"  Magnet type           : SPM arc-segment",
        mat_line,
        f"  Magnet thickness      : {t_m*1000:.2f} mm",
        f"  Arc fraction α_p      : {alpha:.3f}  ({mag_arc_deg:.2f}°/pole)",
        f"  Arc length (outer)    : {arc_length*1000:.2f} mm",
        f"  Chord width (outer ID): {chord_outer*1000:.2f} mm",
        f"  Chord width (inner ID): {chord_inner*1000:.2f} mm",
        f"  Stack length          : {L*1000:.1f} mm",
        f"  Volume per magnet     : {bom.magnet_volume_per_pole_m3*1e6:.1f} cm³",
        f"  Mass per magnet       : {bom.magnet_mass_per_pole_kg*1000:.1f} g",
        f"  Total magnet mass     : {bom.magnet_mass_total_kg*1000:.0f} g",
        f"  Coating               : Ni-Cu-Ni electroplating (standard) or epoxy",
        f"  Magnetisation         : Radial, alternating N-S per pole",
        f"  Retention method      : Stainless steel sleeve OR structural adhesive",
        f"    Adhesive spec       : Loctite AA 332 or equiv., 0.05–0.10 mm bondline",
    ]
    return "\n".join(lines)


# ── Assembly tolerances ────────────────────────────────────────────────────────

def _iso286_H7(d_mm: float) -> float:
    """ISO 286-1 IT7 fundamental tolerance [mm] for bore diameter d_mm."""
    # ISO Table: ranges (nominal size, IT7 µm)
    table = [(3,10),(6,12),(10,15),(18,18),(30,21),(50,25),
             (80,30),(120,35),(180,40),(250,46),(315,52),(400,57),(500,63)]
    for upper, it7 in table:
        if d_mm <= upper:
            return it7 / 1000
    return 63 / 1000

def _iso286_h6(d_mm: float) -> float:
    """ISO 286-1 IT6 fundamental tolerance [mm] for shaft diameter d_mm."""
    table = [(3,6),(6,8),(10,9),(18,11),(30,13),(50,16),
             (80,19),(120,22),(180,25),(250,29),(315,32),(400,36),(500,40)]
    for upper, it6 in table:
        if d_mm <= upper:
            return it6 / 1000
    return 40 / 1000

def _tolerances(motor) -> str:
    g     = motor.airgap * 1000   # mm
    g_tol = max(0.05, g * 0.10)  # 10% of airgap, min 0.05 mm
    R_si  = motor.stator.inner_radius * 1000 if motor.stator else 82.0
    R_ro  = motor.rotor_outer_radius * 1000
    bore_dia  = R_si * 2
    rotor_dia = R_ro * 2
    h7_tol = _iso286_H7(bore_dia)
    h6_tol = _iso286_h6(rotor_dia)

    lines = [
        f"  Stator bore ID        : {bore_dia:.3f} mm  H7 fit  "
          f"(+0.000 / +{h7_tol:.3f} mm  — ISO 286-1)",
        f"  Rotor OD              : {rotor_dia:.3f} mm  h6 fit  "
          f"(-{h6_tol:.3f} / +0.000 mm  — ISO 286-1)",
        f"  Target air gap        : {g:.3f} mm",
        f"  Air gap tolerance     : ± {g_tol:.3f} mm  (min gap ≥ {g-g_tol:.3f} mm)",
        f"  Stator bore roundness : ≤ {g_tol/2:.3f} mm TIR",
        f"  Rotor OD runout       : ≤ {g_tol/2:.3f} mm TIR",
        f"  Stack length          : {motor.stack_length*1000:.1f} mm  ± 0.5 mm",
        f"  Lamination burr ht.   : ≤ 0.05 mm (IEC 60404-8-4)",
        f"  Shaft bearing seat    : k6 interference fit (NN, NNU series bearings)",
        f"  Housing bearing seat  : H6 (press-fit outer race)",
        f"  Rotor balance grade   : G1.0 per ISO 21940-11 (motor class S1)",
    ]
    return "\n".join(lines)


# ── Main class ─────────────────────────────────────────────────────────────────

class ManufacturingReport:
    """
    Compile and render a complete manufacturing specification for a PMSM.
    """

    def __init__(self, motor, spec=None, material_lib=None,
                 operating_temp_C: float = 120.0):
        self.motor   = motor
        self.spec    = spec or getattr(motor, "spec", None)
        self.mlib    = material_lib
        self.op_temp = operating_temp_C

        # Pre-compute all derived quantities once
        self._wire    = _compute_wire_spec(motor, operating_temp_C)
        self._bom     = _compute_bom(motor, self._wire, material_lib)
        self._L_ew    = _end_winding_length(motor)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _separator(self, char="═", width=72) -> str:
        return char * width

    def _section(self, title: str) -> str:
        return f"\n{self._separator()}\n  {title}\n{self._separator()}"

    def _run_drc(self) -> str:
        try:
            from Bohemien_Motor_Designer.utils.validation import DesignRuleChecker
            checker = DesignRuleChecker(self.motor, spec=self.spec, material_lib=self.mlib)
            return checker.report()
        except Exception as e:
            return f"  (DRC unavailable: {e})"

    # ── Electrical parameters section ──────────────────────────────────────

    def _electrical_params(self) -> str:
        m   = self.motor
        w   = self._wire
        Ke  = m.back_emf_constant()
        pp  = m.pole_pairs
        omega_e = m.rated_speed * 2 * math.pi / 60 * pp
        E_pk    = Ke * omega_e
        T_rated = m.rated_torque
        Iq_rated = T_rated / (1.5 * pp * Ke)
        I_rms    = Iq_rated / math.sqrt(2)
        J_rms    = I_rms / (w.conductor_area_mm2)

        Vph_rms = getattr(m, "rated_voltage", 230.0)
        Vph_pk  = Vph_rms * math.sqrt(2)

        Ld = getattr(m, "Ld", m._compute_inductances()[0] if hasattr(m, "_compute_inductances") else 0)
        Lq = getattr(m, "Lq", m._compute_inductances()[1] if hasattr(m, "_compute_inductances") else 0)
        Vd = w.resistance_per_phase_ohm * 0 - omega_e * Lq * Iq_rated
        Vq = w.resistance_per_phase_ohm * Iq_rated + omega_e * (Ld * 0 + Ke)
        V_total = math.sqrt(Vd**2 + Vq**2)
        MI_ratio = E_pk / (Vph_pk + 1e-9)

        lines = [
            f"  Back-EMF constant  Ke   : {Ke:.4f}  Wb  ({Ke:.4f} Vs/rad)",
            f"  Peak back-EMF @ rated n : {E_pk:.1f}  V pk  "
              f"({E_pk/math.sqrt(2):.1f} V rms)",
            f"  Modulation index @ rated: {MI_ratio:.2f}  "
              f"({MI_ratio*100:.0f}% of Vpk_max)",
            f"  Rated torque            : {T_rated:.2f}  N·m",
            f"  Rated Iq (peak / rms)   : {Iq_rated:.1f}  /  {I_rms:.1f}  A",
            f"  Current density J (rms) : {J_rms:.2f}  A/mm²",
            f"  Phase resistance Rs     : "
              f"{w.resistance_per_phase_ohm / (1 + ALPHA_CU*(self.op_temp-20)) * 1000:.2f}  mΩ @ 20°C  /  "
              f"{w.resistance_per_phase_ohm*1000:.2f}  mΩ @ {self.op_temp:.0f}°C",
            f"  Ld / Lq (analytical)    : {Ld*1e3:.3f}  /  {Lq*1e3:.3f}  mH",
            f"  Rated electrical freq.  : {m.electrical_frequency:.1f}  Hz",
            f"  Pole pairs              : {pp}",
        ]
        return "\n".join(lines)

    # ── Master text generator ──────────────────────────────────────────────

    def text(self) -> str:
        m  = self.motor
        w  = self._wire
        bom= self._bom
        st = m.stator

        lines = []

        # ── Cover ──
        lines.append(self._separator("═"))
        lines.append("  PMSM MANUFACTURING SPECIFICATION")
        lines.append(f"  Generated by Bohemien_Motor_Designer")
        lines.append(self._separator("─"))
        lines.append(f"  {m.poles}p / {m.slots}s  |  "
                      f"{m.rated_power/1000:.0f} kW  |  "
                      f"{m.rated_speed:.0f} rpm  |  "
                      f"{getattr(m,'rated_voltage',230):.0f} V (phase RMS)")
        lines.append(self._separator("═"))

        # ── 1. Ratings ──
        lines.append(self._section("1. RATINGS & TOPOLOGY"))
        lines.append(f"  Rated power              : {m.rated_power/1000:.2f}  kW")
        lines.append(f"  Rated speed              : {m.rated_speed:.0f}  rpm")
        lines.append(f"  Rated torque             : {m.rated_torque:.2f}  N·m")
        lines.append(f"  Phase voltage (RMS)      : {getattr(m,'rated_voltage',230.0):.1f}  V")
        lines.append(f"  Connection               : {getattr(m,'connection','star').upper()}")
        lines.append(f"  Phases                   : {m.phases}")
        lines.append(f"  Poles / Slots            : {m.poles}p / {m.slots}s")
        lines.append(f"  Slots/pole/phase (q)     : {m.slots_per_pole_per_phase:.3f}")
        lines.append(f"  Winding factor kw        : {m.winding_factor():.4f}")
        lines.append(f"  Magnet type              : SPM (Surface Permanent Magnet)")

        # ── 2. Stator dimensions ──
        lines.append(self._section("2. STATOR LAMINATION DIMENSIONS"))
        if st:
            sp = st.slot_profile
            lines.append(f"  Stator OD                : {st.outer_radius*1000:.3f}  mm")
            lines.append(f"  Stator bore ID           : {st.inner_radius*1000:.3f}  mm")
            lines.append(f"  Yoke thickness           : {st.yoke_thickness*1000:.2f}  mm")
            lines.append(f"  Tooth width              : {st.tooth_width*1000:.3f}  mm")
            lines.append(f"  Slot pitch (at bore)     : {2*math.pi*st.inner_radius/m.slots*1000:.3f}  mm")
            lines.append(f"  Stack length             : {m.stack_length*1000:.1f}  mm")
            lines.append(f"  Lamination material      : {getattr(st,'lamination_material','M270-35A')}")
            lines.append(f"  Lamination thickness     : {bom.lamination_thickness_mm:.2f}  mm")
            lines.append(f"  Number of laminations    : {bom.n_laminations}")
            lines.append(f"  Stacking factor          : {STACK_FACTOR:.2f}")
            lines.append(f"  Stator iron mass         : {bom.stator_iron_mass_kg:.2f}  kg")

        # ── 3. Slot dimensions ──
        lines.append(self._section("3. SLOT & INSULATION SPECIFICATION"))
        if st and st.slot_profile:
            sp = st.slot_profile
            lines.append(f"  Slot profile type        : {sp.__class__.__name__}")
            lines.append(f"  Slot gross area          : {sp.area()*1e6:.2f}  mm²")
            lines.append(f"  Slot depth               : {sp.depth()*1000:.3f}  mm")
            lines.append(f"  Slot width (mean)        : {sp.area()/sp.depth()*1000:.3f}  mm")
            lines.append(f"  Slot opening             : {sp.opening_width()*1000:.3f}  mm")
            if hasattr(sp, "wedge_height") and sp.wedge_height > 0:
                lines.append(f"  Wedge height             : {sp.wedge_height*1000:.2f}  mm")
        lines.append("")
        lines.append("  --- Insulation ---")
        lines.append(_slot_insulation(m))

        # ── 4. Winding specification ──
        lines.append(self._section("4. WINDING SPECIFICATION"))
        N_series = m.winding.total_series_turns_per_phase
        paths    = getattr(m.winding, "parallel_paths", 1)
        lines.append(f"  Winding type             : Distributed, 2-layer")
        lines.append(f"  Coil span                : {m.winding.coil_span} slots")
        lines.append(f"  Turns per coil (each side): {m.turns_per_coil}")
        lines.append(f"  Series turns / phase     : {N_series}")
        lines.append(f"  Parallel paths / phase   : {paths}")
        lines.append(f"  Winding pitch factor     : {m.winding.pitch_factor:.4f}")
        lines.append(f"  Distribution factor      : {m.winding.distribution_factor:.4f}")
        lines.append(f"  Winding factor kw        : {m.winding_factor():.4f}")
        lines.append("")
        lines.append("  --- Conductor ---")
        lines.append(f"  Required cond. area/turn : {w.conductor_area_mm2:.3f}  mm²")
        lines.append(f"  Wire diameter (bare)     : {w.wire_diameter_mm:.3f}  mm  "
                      f"({w.awg_approx})")
        lines.append(f"  Insulation build (total) : {w.insulation_build_mm:.3f}  mm  "
                      f"(IEC 60317 heavy build)")
        lines.append(f"  Overall diameter         : {w.overall_diameter_mm:.3f}  mm")
        lines.append(f"  Parallel strands/turn    : {w.n_strands}")
        if w.n_strands > 1:
            lines.append(f"  NOTE: Strands must be transposed (Roebel or hand-twist)")
        lines.append(f"  Achieved slot fill       : {w.slot_fill_achieved:.3f}  "
                      f"({w.slot_fill_achieved*100:.1f}%)")
        lines.append(f"  Target slot fill         : {getattr(m,'slot_fill_factor',0.45):.3f}")
        lines.append(f"  End-winding overhang     : {self._L_ew*1000:.1f}  mm  (each side)")
        lines.append(f"  Wire length / phase      : {w.wire_length_per_phase_m:.2f}  m")
        lines.append(f"  Wire length total (3-ph) : {w.wire_length_per_phase_m*3:.2f}  m")
        lines.append(f"  Phase resistance @ {self.op_temp:.0f}°C  : "
                      f"{w.resistance_per_phase_ohm*1000:.2f}  mΩ")
        lines.append(f"  Copper mass / phase      : {w.copper_mass_per_phase_kg*1000:.0f}  g")
        lines.append(f"  Copper mass total        : {bom.copper_mass_stator_kg*1000:.0f}  g")

        # ── 5. Winding connection table ──
        lines.append(self._section("5. WINDING CONNECTION TABLE"))
        lines.append("  (Slot numbering 1-based; Layer 0 = near bore, Layer 1 = near yoke)")
        lines.append("  (Direction: GO = current into page at rated operation, "
                      "RET = out of page)")
        lines.append("")
        table = _winding_table(m)
        for row in table.splitlines():
            lines.append("  " + row)

        # ── 6. Coil groups & terminals ──
        lines.append(self._section("6. COIL GROUPS & TERMINAL CONNECTIONS"))
        cg = _coil_groups(m)
        for row in cg.splitlines():
            lines.append(row)

        # ── 7. Rotor & magnets ──
        lines.append(self._section("7. ROTOR & MAGNET SPECIFICATION"))
        lines.append(f"  Rotor OD                 : {m.rotor_outer_radius*1000:.3f}  mm")
        lines.append(f"  Shaft bore ID            : {m.rotor_inner_radius*1000:.3f}  mm")
        lines.append(f"  Air gap                  : {m.airgap*1000:.3f}  mm")
        lines.append(f"  Rotor iron mass (est.)   : {bom.rotor_iron_mass_kg:.2f}  kg")
        lines.append("")
        lines.append("  --- Magnets ---")
        lines.append(_magnet_spec(m, bom))

        # ── 8. BOM ──
        lines.append(self._section("8. BILL OF MATERIALS (ACTIVE PARTS)"))
        header = f"  {'Item':<28} {'Qty':>5}  {'Unit':<5}  {'Mass (kg)':>9}  Notes"
        lines.append(header)
        lines.append("  " + "─" * 68)
        for item, qty, unit, mass, note in bom.to_rows():
            qty_s = f"{qty}" if isinstance(qty, int) else qty
            lines.append(
                f"  {item:<28} {qty_s:>5}  {unit:<5}  {mass:>9.2f}  {note}"
            )

        # ── 9. Electrical parameters ──
        lines.append(self._section("9. ELECTROMAGNETIC PARAMETERS"))
        lines.append(self._electrical_params())

        # ── 10. Loss budget ──
        lines.append(self._section("10. LOSS BUDGET & EFFICIENCY"))
        try:
            from Bohemien_Motor_Designer.analysis.losses import LossCalculator
            lc = LossCalculator(m)
            lb = lc.loss_budget()
            lines.append(f"  @ {m.rated_speed:.0f} rpm / {m.rated_torque:.1f} N·m:")
            lines.append(f"  Copper loss              : {lb.copper_loss_W:.1f}  W")
            lines.append(f"  Stator iron loss         : {lb.stator_iron_W:.1f}  W")
            lines.append(f"  Rotor iron loss          : {lb.rotor_iron_W:.1f}  W")
            lines.append(f"  Mechanical (fr+wind)     : {lb.friction_W+lb.windage_W:.1f}  W")
            lines.append(f"  Stray losses             : {lb.stray_W:.1f}  W")
            lines.append(f"  Total losses             : {lb.total_loss_W:.1f}  W")
            lines.append(f"  EFFICIENCY               : {float(lb.efficiency)*100:.2f}  %")
        except Exception as e:
            lines.append(f"  (Loss calculator unavailable: {e})")

        # Cogging
        try:
            from Bohemien_Motor_Designer.analysis.losses import cogging_torque_Nm
            cog = cogging_torque_Nm(m)
            lines.append(f"  Cogging torque Tpp       : {cog['Tcog_pp_Nm']:.3f}  N·m  "
                          f"({cog['Tcog_pp_pct']:.2f}% of rated)")
            lines.append(f"  Cogging period           : {cog['cogging_period_deg']:.2f}°  "
                          f"(LCM={cog['lcm_slots_poles']})")
        except Exception:
            pass

        # ── 11. DRC ──
        lines.append(self._section("11. DESIGN RULE CHECK"))
        lines.append(self._run_drc())

        # ── 12. Tolerances ──
        lines.append(self._section("12. ASSEMBLY TOLERANCES & NOTES"))
        lines.append(_tolerances(m))
        lines.append("")
        lines.append("  General notes:")
        lines.append("    • All dimensions in mm unless stated otherwise.")
        lines.append("    • Slot liner to be pre-formed and inserted before winding.")
        lines.append("    • Magnets to be bonded with rotor at ambient temperature,")
        lines.append("      then cured per adhesive datasheet before sleeve installation.")
        lines.append("    • Winding resistance to be measured at 20°C after VPI and curing;")
        lines.append("      compare to calculated value ± 10%.")
        lines.append("    • Final air gap to be verified with feeler gauge at 4 quadrants.")

        lines.append("")
        lines.append(self._separator("═"))
        lines.append("  END OF MANUFACTURING SPECIFICATION")
        lines.append(self._separator("═"))

        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Write the report to a plain-text file."""
        from pathlib import Path
        Path(path).write_text(self.text(), encoding="utf-8")
        print(f"Manufacturing report saved to: {path}")
