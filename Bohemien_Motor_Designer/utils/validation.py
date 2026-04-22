"""
Design Rule Checker — physics-based sanity checks before FEA.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CheckResult:
    name:     str
    level:    str    # "OK" | "WARN" | "ERROR"
    message:  str
    value:    Optional[float] = None
    limit:    Optional[float] = None


class DesignRuleChecker:
    """
    Validate a motor design against physics and manufacturing constraints.

    Checks geometry, winding, electromagnetics, thermal, and voltage stress.
    Each rule generates an OK / WARN / ERROR result.

    Usage::
        checker = DesignRuleChecker(motor, spec=spec, inverter=inverter)
        results = checker.check_all()
        print(checker.report())
    """

    def __init__(self, motor, spec=None, inverter=None, material_lib=None):
        self.motor    = motor
        self.spec     = spec
        self.inverter = inverter
        from Bohemien_Motor_Designer.materials.library import MaterialLibrary
        self.mlib = material_lib or MaterialLibrary()
        self._results: list[CheckResult] = []

    def _add(self, name: str, level: str, message: str,
             value=None, limit=None):
        self._results.append(CheckResult(name, level, message, value, limit))

    # ── Geometry checks ────────────────────────────────────────────────────

    def _check_geometry(self):
        m = self.motor

        # Air gap
        g = m.airgap * 1e3   # mm
        if g < 0.3:
            self._add("airgap_min", "ERROR",
                      f"Air gap {g:.2f}mm < 0.3mm — manufacturing risk", g, 0.3)
        elif g < 0.5:
            self._add("airgap_min", "WARN",
                      f"Air gap {g:.2f}mm — tight tolerance, requires precision machining", g, 0.5)
        else:
            self._add("airgap_min", "OK", f"Air gap {g:.2f}mm — acceptable")

        if g > 5.0:
            self._add("airgap_max", "WARN",
                      f"Air gap {g:.2f}mm > 5mm — large airgap reduces power factor", g, 5.0)

        # Slot geometry
        if m.stator is not None:
            st = m.stator
            issues = st.validate()
            for iss in issues:
                self._add("stator_geometry", "WARN", iss)
            if not issues:
                self._add("stator_geometry", "OK", "Stator geometry dimensions consistent")

            # Slot fill factor
            ff = getattr(m, "slot_fill_factor", 0.45)
            if ff > 0.75:
                self._add("fill_factor", "ERROR",
                          f"Slot fill {ff:.2f} > 0.75 — not manufacturable", ff, 0.75)
            elif ff < 0.25:
                self._add("fill_factor", "WARN",
                          f"Slot fill {ff:.2f} < 0.25 — poor copper utilisation", ff, 0.25)
            else:
                self._add("fill_factor", "OK", f"Slot fill {ff:.2f} — acceptable")

        # Rotor tip speed
        if hasattr(m, "rotor_geo") and m.rotor_geo is not None:
            rg = m.rotor_geo
            tip = rg.tip_speed(m.rated_speed)
            limit = 200.0 if getattr(rg, "sleeve_thickness", 0) > 0 else 100.0
            if tip > limit:
                self._add("tip_speed", "ERROR",
                          f"Tip speed {tip:.1f} m/s > {limit:.0f} m/s limit", tip, limit)
            elif tip > limit * 0.85:
                self._add("tip_speed", "WARN",
                          f"Tip speed {tip:.1f} m/s — approaching {limit:.0f} m/s limit", tip, limit)
            else:
                self._add("tip_speed", "OK", f"Tip speed {tip:.1f} m/s — acceptable")

    # ── Winding checks ─────────────────────────────────────────────────────

    def _check_winding(self):
        m = self.motor
        q = m.slots_per_pole_per_phase

        if q < 0.4:
            self._add("q_factor", "ERROR",
                      f"q = {q:.3f} < 0.4 — too few slots, high harmonic content", q, 0.4)
        elif q < 0.5:
            self._add("q_factor", "WARN",
                      f"q = {q:.3f} — concentrated winding, higher losses", q, 0.5)
        elif q > 6:
            self._add("q_factor", "WARN",
                      f"q = {q:.1f} > 6 — diminishing returns, high leakage inductance")
        else:
            self._add("q_factor", "OK", f"q = {q:.3f} — good slot/pole combination")

        kw = m.winding_factor()
        if kw < 0.85:
            self._add("winding_factor", "WARN",
                      f"Winding factor kw = {kw:.3f} < 0.85 — poor harmonic utilisation", kw, 0.85)
        else:
            self._add("winding_factor", "OK", f"Winding factor kw = {kw:.4f}")

    # ── Electromagnetic checks ─────────────────────────────────────────────

    def _check_electromagnetics(self):
        m = self.motor

        if not hasattr(m, "back_emf_constant"):
            return

        Ke      = m.back_emf_constant()
        omega_e = m.rated_speed * 2 * np.pi / 60 * m.pole_pairs
        E_pk    = Ke * omega_e

        # Back-EMF vs voltage
        if self.inverter:
            V_max_pk = self.inverter.max_phase_voltage_peak()
        elif m.rated_voltage > 0:
            V_max_pk = m.rated_voltage * np.sqrt(2)
        else:
            V_max_pk = 325.0

        ratio = E_pk / (V_max_pk + 1e-9)
        if ratio > 1.05:
            self._add("back_emf", "ERROR",
                      f"Back-EMF {E_pk:.0f}V > Vpk {V_max_pk:.0f}V — motor cannot start at rated speed",
                      E_pk, V_max_pk)
        elif ratio > 0.85:
            self._add("back_emf", "OK",
                      f"Back-EMF {E_pk:.0f}V ({ratio*100:.0f}% of Vpk) — good utilisation", ratio)
        elif ratio < 0.40:
            self._add("back_emf", "WARN",
                      f"Back-EMF {E_pk:.0f}V ({ratio*100:.0f}% of Vpk) — high current required",
                      E_pk, V_max_pk)
        else:
            self._add("back_emf", "OK",
                      f"Back-EMF {E_pk:.0f}V ({ratio*100:.0f}% of Vpk)")

        # Flux density in steel — use magnet-circuit B_gap (more accurate than EMF backtrack)
        if m.stator is not None:
            try:
                from Bohemien_Motor_Designer.materials.library import MaterialLibrary
                _mlib = MaterialLibrary()
                _mag  = _mlib.magnet(getattr(m, "magnet_material", "N42SH"))
                _t_m  = getattr(m, "magnet_thickness", 0.005)
                _mu_r = _mag.mu_r
                B_g   = _mag.remanence_Br * _t_m / (_t_m + _mu_r * m.airgap)
            except Exception:
                # Fallback: back-calculate from Ke
                kw = m.winding_factor()
                N  = m.winding.total_series_turns_per_phase
                tau_p = m.pole_pitch
                B_g = Ke * np.pi / (2 * N * kw * tau_p * m.stack_length + 1e-9)
            B_g = np.clip(B_g, 0.05, 2.0)
            alpha_p = getattr(m, "magnet_width_fraction", 1.0)
            B_yoke  = m.stator.yoke_flux_density(B_g, m.poles, m.stack_length,
                                                  magnet_arc_fraction=alpha_p)
            B_tooth = m.stator.tooth_flux_density(B_g)

            if B_tooth > 1.85:
                self._add("tooth_flux", "ERROR",
                          f"Tooth flux {B_tooth:.2f}T > 1.85T — heavy saturation", B_tooth, 1.85)
            elif B_tooth > 1.65:
                self._add("tooth_flux", "WARN",
                          f"Tooth flux {B_tooth:.2f}T > 1.65T — approaching saturation", B_tooth, 1.65)
            else:
                self._add("tooth_flux", "OK", f"Tooth flux density {B_tooth:.2f}T")

            if B_yoke > 1.6:
                self._add("yoke_flux", "WARN",
                          f"Yoke flux {B_yoke:.2f}T > 1.6T — elevated iron losses", B_yoke, 1.6)
            else:
                self._add("yoke_flux", "OK", f"Yoke flux density {B_yoke:.2f}T")

        # Magnet demagnetisation check (basic)
        if hasattr(m, "magnet_material"):
            try:
                mag = self.mlib.magnet(m.magnet_material)
                T_max_C = 100.0
                Hcj = mag.Hcj_at(T_max_C)
                if Hcj < 800:
                    self._add("magnet_Hcj", "WARN",
                              f"HcJ = {Hcj:.0f} kA/m at {T_max_C}°C — risk of demagnetisation under fault current")
                else:
                    self._add("magnet_Hcj", "OK",
                              f"HcJ = {Hcj:.0f} kA/m at {T_max_C}°C — adequate coercivity")
            except Exception:
                pass

    # ── Thermal checks ─────────────────────────────────────────────────────

    def _check_thermal(self):
        m = self.motor
        if not hasattr(m, "back_emf_constant"):
            return

        # Current density from slot fill — correct for stranded/multi-wire conductors
        psi_m = m.back_emf_constant()
        Iq    = m.rated_torque / (1.5 * m.pole_pairs * psi_m + 1e-9)  # peak A
        I_rms = Iq / np.sqrt(2)
        if m.stator is not None:
            A_slot = m.stator.slot_profile.area()
        else:
            A_slot = 150e-6   # 150 mm² fallback
        fill     = getattr(m, "slot_fill_factor", 0.45)
        t_c      = max(getattr(m, "turns_per_coil", 1), 1)
        n_layers = m.winding.layers if m.winding else 2
        # conductor area per turn = slot area × fill / (turns_per_coil × layers)
        A_c   = A_slot * fill / (t_c * n_layers)   # m² per conductor turn
        J     = I_rms / (A_c * 1e6 + 1e-9)        # A/mm²

        cooling = self.spec.cooling.cooling_type if self.spec else "air"
        from Bohemien_Motor_Designer.scaling.similarity import CURRENT_DENSITY_LIMIT
        J_lim = CURRENT_DENSITY_LIMIT.get(cooling, 5.0)

        if J > J_lim:
            # Compute actionable fix options
            scale  = J / J_lim               # factor to reduce J by
            # Option A: enlarge slot (increase slot_depth by sqrt(scale))
            if m.stator is not None:
                cur_depth = m.stator.slot_profile.depth() * 1e3
                new_depth = cur_depth * scale
            else:
                cur_depth = 0; new_depth = 0
            # Option B: raise cooling limit
            from Bohemien_Motor_Designer.scaling.similarity import CURRENT_DENSITY_LIMIT
            better_cooling = next(
                (k for k, v in CURRENT_DENSITY_LIMIT.items() if v >= J * 1.05),
                "direct-water")
            msg = (f"J = {J:.1f} A/mm² exceeds {J_lim:.0f} A/mm² limit for {cooling}. "
                   f"Fix options: "
                   f"(A) increase slot depth from {cur_depth:.0f}→{new_depth:.0f}mm, "
                   f"(B) switch cooling to '{better_cooling}' (limit {CURRENT_DENSITY_LIMIT[better_cooling]:.0f} A/mm²), "
                   f"(C) add parallel paths to share current across conductors")
            self._add("current_density", "ERROR", msg, J, J_lim)
        elif J > J_lim * 0.85:
            self._add("current_density", "WARN",
                      f"J = {J:.1f} A/mm² is near {J_lim:.0f} A/mm² thermal limit for {cooling} — "
                      f"consider larger slots or additional parallel paths", J, J_lim)
        else:
            self._add("current_density", "OK",
                      f"J = {J:.1f} A/mm² for {cooling}  (limit {J_lim:.0f} A/mm²)")

    # ── Insulation / high voltage checks ──────────────────────────────────

    def _check_insulation(self):
        if not self.spec:
            return

        Vdc = self.spec.drive.dc_bus_voltage
        ins = self.spec.insulation

        if Vdc > 800 and ins.insulation_class in ("A", "E", "B"):
            self._add("insulation_class", "ERROR",
                      f"DC bus {Vdc:.0f}V with class {ins.insulation_class} insulation — "
                      f"upgrade to class F or H minimum")
        elif Vdc > 800:
            self._add("insulation_class", "OK",
                      f"DC bus {Vdc:.0f}V with class {ins.insulation_class} — adequate")

        if Vdc > 1000:
            self._add("partial_discharge", "WARN",
                      f"DC bus {Vdc:.0f}V — partial discharge testing required. "
                      f"Ensure PDIV > {ins.partial_discharge_inception_V:.0f}V")

        if Vdc > 1200:
            self._add("clearance", "WARN",
                      f"DC bus {Vdc:.0f}V — increased creepage/clearance distances required "
                      f"(IEC 60664-1 OVC III)")

    # ── Main entry point ────────────────────────────────────────────────────

    def check_all(self) -> list[CheckResult]:
        self._results = []
        self._check_geometry()
        self._check_winding()
        self._check_electromagnetics()
        self._check_thermal()
        self._check_insulation()
        return self._results

    def has_errors(self) -> bool:
        return any(r.level == "ERROR" for r in self._results)

    def has_warnings(self) -> bool:
        return any(r.level == "WARN" for r in self._results)

    def report(self) -> str:
        if not self._results:
            self.check_all()
        lines = ["\nDesign Rule Check Report", "=" * 45]
        errors   = [r for r in self._results if r.level == "ERROR"]
        warnings = [r for r in self._results if r.level == "WARN"]
        ok_items = [r for r in self._results if r.level == "OK"]

        for r in errors:
            lines.append(f"  [ERROR]  {r.message}")
        for r in warnings:
            lines.append(f"  [WARN]   {r.message}")
        for r in ok_items:
            lines.append(f"  [OK]     {r.message}")

        lines.append("=" * 45)
        status = "PASS" if not errors else "FAIL"
        lines.append(f"  Result: {status}  "
                     f"({len(errors)} errors, {len(warnings)} warnings, {len(ok_items)} OK)")
        return "\n".join(lines)
