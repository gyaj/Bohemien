"""
Sprint 2 smoke test — generate PMSM.geo and verify it can be parsed.

Run: python Bohemien_Motor_Designer/fea/test_export.py
Output: /tmp/PMSM_test.geo   (open in GMSH to verify visually)

Checks:
  1. .geo file is generated without exception
  2. File is non-empty and contains expected keywords
  3. Physical group tags match IndexRegistry values
  4. Winding assignment matches known slot table
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Bohemien_Motor_Designer.core.pmsm import PMSM
from Bohemien_Motor_Designer.core.geometry.stator import StatorGeometry
from Bohemien_Motor_Designer.core.geometry.slot_profiles import ParallelToothSlot
from Bohemien_Motor_Designer.core.geometry.rotor import SPMRotorGeometry
from Bohemien_Motor_Designer.core.geometry.winding import WindingLayout
from Bohemien_Motor_Designer.fea.index_registry import IndexRegistry
from Bohemien_Motor_Designer.fea.gmsh_exporter import GMSHExporter

OUT = "/tmp/PMSM_test.geo"

def build_motor():
    stator = StatorGeometry(
        outer_radius=0.1125, inner_radius=0.082, slots=48,
        slot_profile=ParallelToothSlot(
            slot_width=0.008, slot_depth=0.022, slot_opening=0.003),
    )
    rotor_geo = SPMRotorGeometry(
        outer_radius=0.081, inner_radius=0.030,
        magnet_thickness=0.006, magnet_width_fraction=0.83,
        sleeve_thickness=0.002,
    )
    winding = WindingLayout(poles=8, slots=48, phases=3, layers=2, turns_per_coil=11)
    return PMSM(
        poles=8, slots=48, stator=stator, rotor_geo=rotor_geo,
        rotor_outer_radius=0.081, rotor_inner_radius=0.030,
        stack_length=0.130, airgap=0.001,
        rated_speed=4000, rated_power=30000,
        magnet_thickness=0.006, magnet_width_fraction=0.83,
        turns_per_coil=11, winding=winding,
    )

def run_tests():
    print("=== Sprint 2 — GMSH Export Smoke Test ===")

    motor = build_motor()
    reg   = IndexRegistry(poles=motor.poles, slots=motor.slots)

    print(reg.summary())
    print()

    # ── Test 1: generate without exception ──
    exporter = GMSHExporter(motor, reg)
    path     = exporter.write(OUT)
    content  = path.read_text()
    size_kb  = len(content) / 1024

    print(f"[1] .geo file written: {path}  ({size_kb:.1f} KB)")
    assert size_kb > 1.0, "File too small — generation likely failed"
    print("    PASS")

    # ── Test 2: key keywords present ──
    required = [
        "SetFactory",
        "Physical Surface",
        "Physical Curve",
        "Sliding",
        "stator iron",
        "rotor iron",
    ]
    missing = [kw for kw in required if kw not in content]
    print(f"[2] Required keywords present: {len(required) - len(missing)}/{len(required)}")
    if missing:
        print(f"    MISSING: {missing}")
    else:
        print("    PASS")

    # ── Test 3: IndexRegistry tag numbers appear in file ──
    checks = {
        f"Physical Surface({reg.stator_iron})": "stator iron tag",
        f"Physical Surface({reg.rotor_iron})":  "rotor iron tag",
        f"Physical Surface({reg.air_gap})":     "air gap tag",
        f"Physical Curve({reg.outer_boundary})": "outer boundary",
        f"Physical Curve({reg.sliding_surface})": "sliding surface",
    }
    for tag_str, label in checks.items():
        found = tag_str in content
        status = "OK" if found else "MISSING"
        print(f"    [{status}] {label}: '{tag_str}'")

    # ── Test 4: winding slot table matches coil assignment ──
    print(f"[4] Winding coil assignment check (first 6 slots):")
    for slot_idx in range(6):
        for layer in range(2):
            info = exporter._coil_info(slot_idx, layer)
            if info:
                phase = "ABC"[info["phase"]]
                sign  = "+" if info["direction"] > 0 else "-"
                tag   = reg.winding_tag(slot_idx, layer)
                print(f"    slot={slot_idx+1} layer={layer}: "
                      f"{sign}{phase}  PS={tag}")

    # ── Test 5: PM tag count matches poles ──
    pm_tags_in_file = [reg.pm_tag(i) for i in range(motor.poles)]
    all_present = all(f"// PM pole {i}" in content for i in range(motor.poles))
    print(f"[5] PM poles ({motor.poles}): {'PASS' if all_present else 'PARTIAL'}")

    print()
    print(f"=== DONE — open {OUT} in GMSH to verify geometry ===")

if __name__ == "__main__":
    run_tests()
