"""
FEA Index Registry — shared numbering for GMSH physical groups and Elmer body IDs.

This module is the single source of truth for all physical surface / body
index assignments.  Both gmsh_exporter.py and sif_generator.py import from
here.  The indices must be kept consistent between the two writers — this is
the main design decision that prevents silent mismatches.

Physical Surface conventions (GMSH Physical Surface tags):
  1         — stator iron (all lamination iron)
  2         — rotor iron
  3..3+p-1  — PM bodies, one per pole (p = number of poles)
  3+p       — shaft (rotor bore)
  3+p+1     — air gap (all air gap elements: inner + outer half)
  3+p+2     — winding slot air (coil bodies, coloured by phase/direction)
              Each slot layer gets its own tag for phase current assignment.

Physical Line (boundary) tags:
  100       — outer stator boundary (Dirichlet A_z = 0)
  101       — sliding surface (Mortar BC, rotor air / stator air interface)
  102       — periodic boundary left  (symmetry, optional)
  103       — periodic boundary right (symmetry, optional)

Winding physical surface tags are generated dynamically:
  200 + slot_idx * 2 + layer  for slot_idx in 0..Qs-1, layer in {0,1}
  (tag 200..200+2*Qs-1)
  Sign and phase are carried in metadata, not in the tag number.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ── Fixed tags ────────────────────────────────────────────────────────────────

STATOR_IRON   = 1
ROTOR_IRON    = 2
# PM tags: 3 .. 2 + poles  (one per pole)
SHAFT         = None   # set by registry after poles known
AIR_GAP       = None   # set by registry after poles known
# Winding tags start at WINDING_BASE
WINDING_BASE  = 200

# Physical line tags
OUTER_BOUNDARY   = 100
SLIDING_SURFACE  = 101
PERIODIC_LEFT    = 102
PERIODIC_RIGHT   = 103


@dataclass
class IndexRegistry:
    """
    Per-motor index registry.

    Instantiate once with the motor's pole and slot count.
    Pass the same instance to both GMSHExporter and SIFGenerator.
    """
    poles: int
    slots: int
    layers: int = 2

    # Derived in __post_init__
    _pm_base:    int = field(init=False)
    _shaft_tag:  int = field(init=False)
    _airgap_tag: int = field(init=False)

    def __post_init__(self):
        self._pm_base    = 3                          # PM_0 = 3, PM_1 = 4, ...
        self._shaft_tag  = 3 + self.poles             # after last PM
        self._airgap_tag = 3 + self.poles + 1

    # ── Fixed body tags ───────────────────────────────────────────────────

    @property
    def stator_iron(self) -> int:
        return STATOR_IRON

    @property
    def rotor_iron(self) -> int:
        return ROTOR_IRON

    @property
    def shaft(self) -> int:
        return self._shaft_tag

    @property
    def air_gap(self) -> int:
        return self._airgap_tag

    def pm_tag(self, pole_idx: int) -> int:
        """Physical surface tag for PM body pole_idx (0-based)."""
        if not 0 <= pole_idx < self.poles:
            raise ValueError(f"pole_idx {pole_idx} out of range 0..{self.poles-1}")
        return self._pm_base + pole_idx

    # ── Winding tags ──────────────────────────────────────────────────────

    def winding_tag(self, slot_idx: int, layer: int) -> int:
        """Physical surface tag for a specific slot + layer conductor body."""
        return WINDING_BASE + slot_idx * self.layers + layer

    # ── Boundary line tags ────────────────────────────────────────────────

    @property
    def outer_boundary(self) -> int:
        return OUTER_BOUNDARY

    @property
    def sliding_surface(self) -> int:
        return SLIDING_SURFACE

    @property
    def periodic_left(self) -> int:
        return PERIODIC_LEFT

    @property
    def periodic_right(self) -> int:
        return PERIODIC_RIGHT

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"IndexRegistry ({self.poles}p / {self.slots}s / {self.layers}-layer)",
            f"  Stator iron     : PS {self.stator_iron}",
            f"  Rotor iron      : PS {self.rotor_iron}",
            f"  PM bodies       : PS {self._pm_base} .. {self._pm_base + self.poles - 1}",
            f"  Shaft           : PS {self.shaft}",
            f"  Air gap         : PS {self.air_gap}",
            f"  Winding slots   : PS {WINDING_BASE} .. "
            f"{WINDING_BASE + self.slots * self.layers - 1}",
            f"  Outer boundary  : PL {self.outer_boundary}",
            f"  Sliding surface : PL {self.sliding_surface}",
        ]
        return "\n".join(lines)
