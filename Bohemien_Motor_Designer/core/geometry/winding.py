"""
WindingLayout — Universal winding table generator.

Uses the Star-of-Slots method (also called EMF phasor method) which correctly
handles integer-q, fractional-q, and concentrated windings for any pole/slot
combination. This replaces all hardcoded slot assignment logic.

Reference: Bianchi & Dai Pré, "Use of the star of slots in designing
           fractional-slot single-layer windings," IEE-EPA 2006.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CoilSide:
    """One conductor group in one slot layer."""
    slot_idx:    int
    phase:       int    # 0 = A, 1 = B, 2 = C
    layer:       int    # 0 = bottom / inner, 1 = top / outer
    direction:   int    # +1 = going, -1 = return
    n_turns:     int    = 1


class WindingLayout:
    """
    Complete winding table for any motor topology.

    Supports:
      - Integer-q distributed windings (classic 3-phase)
      - Fractional-slot distributed windings
      - Fractional-slot concentrated windings (FSCW, coil_span=1)
      - Single-layer and double-layer
      - Arbitrary number of phases

    Parameters
    ----------
    poles       : Number of magnetic poles.
    slots       : Number of stator slots.
    phases      : Number of electrical phases (default 3).
    layers      : Winding layers per slot (1 or 2).
    coil_span   : Coil span in slots. None = full pitch (slots // poles).
                  Set to 1 for concentrated windings.
    parallel_paths: Number of parallel circuits per phase.
    turns_per_coil: Number of conductor turns per coil.
    """

    def __init__(
        self,
        poles:          int,
        slots:          int,
        phases:         int   = 3,
        layers:         int   = 2,
        coil_span:      Optional[int] = None,
        parallel_paths: int   = 1,
        turns_per_coil: int   = 1,
    ):
        self.poles          = poles
        self.slots          = slots
        self.phases         = phases
        self.layers         = layers
        self.parallel_paths = parallel_paths
        self.turns_per_coil = turns_per_coil

        # Full pitch in slots
        self._full_pitch = slots // poles
        self.coil_span   = coil_span if coil_span is not None else self._full_pitch

        # Derived winding parameters
        self.pole_pairs = poles // 2
        self.q = slots / (poles * phases)   # slots per pole per phase

        self._table: list[CoilSide] = []
        self._phase_assignment: np.ndarray  = np.zeros(slots, dtype=int)
        self._phase_sign:       np.ndarray  = np.zeros(slots, dtype=int)
        self._build()

    # ── Build ─────────────────────────────────────────────────────────────

    def _build(self):
        """Star-of-slots assignment."""
        slots  = self.slots
        phases = self.phases
        p      = self.pole_pairs

        # Electrical angle between adjacent slots [rad]
        alpha_e = 2 * np.pi * p / slots

        # For each slot, compute its position in the phasor star and assign
        # to the nearest phase belt.
        phase_belt_width = np.pi / phases   # half the electrical cycle / phases

        for slot_idx in range(slots):
            # Phasor angle of this slot's EMF [rad, modulo 2π]
            angle = (slot_idx * alpha_e) % (2 * np.pi)

            # Find nearest phase — choose phase with minimum angular distance
            best_phase = 0
            best_dir   = 1
            best_dist  = np.inf
            for ph in range(phases):
                for sign, base in [(+1, 0), (-1, np.pi)]:
                    centre = (base + ph * 2 * np.pi / phases) % (2 * np.pi)
                    dist   = abs(angle - centre)
                    dist   = min(dist, 2 * np.pi - dist)   # wrap
                    if dist < best_dist:
                        best_dist  = dist
                        best_phase = ph
                        best_dir   = sign

            self._phase_assignment[slot_idx] = best_phase
            self._phase_sign[slot_idx]       = best_dir

            # Bottom layer (going side)
            self._table.append(CoilSide(
                slot_idx  = slot_idx,
                phase     = best_phase,
                layer     = 0,
                direction = best_dir,
                n_turns   = self.turns_per_coil,
            ))

            if self.layers == 2:
                # Top layer = return side of the coil, placed in the slot
                # that is coil_span away.  This is the RETURN slot, not the
                # same slot as the go side.
                ret_slot = (slot_idx + self.coil_span) % slots
                self._table.append(CoilSide(
                    slot_idx  = ret_slot,         # ← return side lives here
                    phase     = best_phase,
                    layer     = 1,
                    direction = -best_dir,         # return current
                    n_turns   = self.turns_per_coil,
                ))

    # ── Winding function ──────────────────────────────────────────────────

    def winding_function(self, n_theta: int = 1440) -> np.ndarray:
        """
        Winding function W_k(θ) for each phase k.

        Returns array [phases × n_theta] where each row is the
        winding function (cumulative turns) around the air-gap.
        The mean of each row is subtracted (MMF constraint).

        Parameters
        ----------
        n_theta : angular resolution points around the air gap.
        """
        theta  = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        W      = np.zeros((self.phases, n_theta))
        dtheta = 2 * np.pi / n_theta
        slot_pitch_rad = 2 * np.pi / self.slots

        for cs in self._table:
            # Angular span of this coil side
            slot_centre = cs.slot_idx * slot_pitch_rad
            # Each slot covers ±half a slot pitch
            half = slot_pitch_rad * 0.45
            mask = np.abs(theta - slot_centre) < half
            # Handle wrap-around
            if slot_centre < half:
                mask |= theta > (2 * np.pi - half + slot_centre)
            elif slot_centre > (2 * np.pi - half):
                mask |= theta < (slot_centre - 2 * np.pi + half)

            W[cs.phase, mask] += cs.direction * cs.n_turns

        # Remove DC (MMF must close around the air gap)
        for ph in range(self.phases):
            W[ph] -= W[ph].mean()

        return W

    # ── Fourier analysis ──────────────────────────────────────────────────

    def winding_harmonics(self, n_theta: int = 1440) -> np.ndarray:
        """
        Complex Fourier coefficients of winding function.

        Returns array [phases × (n_theta//2 + 1)] — positive frequencies only.
        Key index: W_harm[:, pole_pairs] is the fundamental working harmonic.
        """
        W    = self.winding_function(n_theta)
        return np.fft.rfft(W, axis=1) / n_theta

    def winding_factor(self, harmonic: int = 1) -> float:
        """
        Winding factor kw = kd x kp (distribution x pitch factors).
        Uses standard analytical formula for integer-q windings.
        """
        alpha = np.pi * self.pole_pairs / self.slots
        q     = self.q
        if abs(q - round(q)) < 0.05 and round(q) >= 1:
            q_int = max(1, int(round(q)))
            kd    = (np.sin(harmonic * q_int * alpha / 2) /
                     (q_int * np.sin(harmonic * alpha / 2) + 1e-12))
        else:
            kd = 0.9   # fractional-slot approximation
        kp = np.sin(harmonic * np.pi * self.coil_span /
                    (2 * self._full_pitch + 1e-12))
        return abs(float(kd * kp))

    @property
    def pitch_factor(self) -> float:
        """Pitch (chording) factor kp."""
        return float(np.sin(np.pi * self.coil_span /
                            (2 * self._full_pitch + 1e-12)))

    @property
    def distribution_factor(self) -> float:
        """Distribution factor kd for fundamental."""
        q   = self.q
        alpha = np.pi * self.pole_pairs / self.slots
        q_int = int(round(q)) if abs(q - round(q)) < 0.1 else 1
        if q_int >= 1:
            kd = np.sin(q_int * alpha / 2) / (q_int * np.sin(alpha / 2) + 1e-12)
        else:
            kd = 1.0
        return abs(kd)

    @property
    def total_series_turns_per_phase(self) -> int:
        """Total series turns per phase [turns].
        Each slot has one going coil side per phase. Coils are series/parallel_paths.
        For 2-layer: each slot contributes 1 coil side (going), return is in another slot.
        """
        # Number of coils per phase = slots / phases  (one coil-side per slot per layer,
        # but each coil occupies 2 slots so coils_per_phase = slots/phases/layers * layers = slots/phases)
        coils_per_phase = self.slots // self.phases
        return (coils_per_phase * self.turns_per_coil) // (self.parallel_paths * 2)

    # ── Diagnostics ──────────────────────────────────────────────────────

    def coil_sides_for_phase(self, phase: int) -> list[CoilSide]:
        return [c for c in self._table if c.phase == phase]

    def slot_table(self) -> str:
        """Human-readable slot assignment table."""
        phase_names = "ABCDEFGH"
        lines = ["Slot | Layer 0       | Layer 1"]
        lines.append("-" * 40)
        for s in range(self.slots):
            bot = [c for c in self._table if c.slot_idx == s and c.layer == 0]
            top = [c for c in self._table if c.slot_idx == s and c.layer == 1]
            def _fmt(c_list):
                if not c_list: return "  —  "
                c = c_list[0]
                sign = "+" if c.direction > 0 else "−"
                return f" {sign}{phase_names[c.phase]}  "
            lines.append(f" {s+1:3d} | {_fmt(bot)}       | {_fmt(top)}")
        return "\n".join(lines)

    def summary(self) -> str:
        kw = self.winding_factor()
        return (
            f"WindingLayout: {self.poles}p / {self.slots}s, "
            f"q={self.q:.3f}, layers={self.layers}, "
            f"span={self.coil_span}/{self._full_pitch}, "
            f"kw={kw:.4f}, "
            f"series_turns/ph={self.total_series_turns_per_phase}"
        )
