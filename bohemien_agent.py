"""
Bohemien Motor Designer — Design Optimisation Agent
====================================================
Uses the Anthropic API with tool use to autonomously find the best
PMSM design that meets a set of engineering requirements.

The agent calls the actual Bohemien solver (analytical models) as
tools and iterates until all constraints are satisfied.

Usage
-----
    python bohemien_agent.py

Requirements
------------
    pip install anthropic
    Set ANTHROPIC_API_KEY environment variable, or enter when prompted.

Place this file in the same directory that CONTAINS Bohemien_Motor_Designer\
(i.e. the repo root), NOT inside the package folder.

    your-repo/
        Bohemien_Motor_Designer/   <- the package
        bohemien_agent.py          <- this file
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

# ── Make Bohemien package importable ─────────────────────────────────────────
# Searches for the package folder (Bohemien_Motor_Designer or motordesign2),
# adds it to sys.path, then imports using whatever name the folder has.
# This works regardless of whether you have renamed the package or not.

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))   # so "import Bohemien_Motor_Designer" works

# Auto-detect the package folder name
_CANDIDATES = [
    "Bohemien_Motor_Designer",
    "motordesign2",
    "bohemien",
    "bohemien_motor_designer",
]
_PKG = None
for _name in _CANDIDATES:
    if (ROOT / _name / "__init__.py").exists():
        _PKG = _name
        print(f"✓  Package loaded from: {ROOT / _name}")
        break

if _PKG is None:
    print(f"ERROR: Could not find Bohemien package in {ROOT}")
    print(f"Expected one of: {_CANDIDATES}")
    sys.exit(1)

# Dynamic imports — use the actual folder name so internal cross-imports work
import importlib

def _from(submodule: str, name: str):
    """Import 'name' from '_PKG.submodule'."""
    mod = importlib.import_module(f"{_PKG}.{submodule}")
    return getattr(mod, name)

PMSM              = _from("core.pmsm",                    "PMSM")
Motor             = _from("core.motor",                   "Motor")
StatorGeometry    = _from("core.geometry.stator",         "StatorGeometry")
ParallelToothSlot = _from("core.geometry.slot_profiles",  "ParallelToothSlot")
WindingLayout     = _from("core.geometry.winding",        "WindingLayout")
LossCalculator    = _from("analysis.losses",              "LossCalculator")
ThermalNetwork    = _from("thermal.lumped_model",         "ThermalNetwork")
ThermalResult     = _from("thermal.lumped_model",         "ThermalResult")
WaterJacketCooling= _from("thermal.cooling",              "WaterJacketCooling")
DesignRuleChecker = _from("utils.validation",             "DesignRuleChecker")
MaterialLibrary   = _from("materials.library",            "MaterialLibrary")


# ══════════════════════════════════════════════════════════════════════════════
# Tool implementations — these call the actual Bohemien solver
# ══════════════════════════════════════════════════════════════════════════════

def _build_motor(p: dict) -> PMSM:
    """Construct a PMSM object from a flat parameter dict."""
    # Geometry in metres (inputs are in mm)
    R_so = p["stator_od_mm"]   / 2000.0
    R_si = p["bore_mm"]        / 2000.0
    g    = p["airgap_mm"]      / 1000.0
    L    = p["stack_length_mm"]/ 1000.0
    t_m  = p["magnet_thickness_mm"] / 1000.0
    R_ro = R_si - g            # rotor outer radius = bore - airgap
    R_ri = R_ro * 0.35         # shaft ≈ 35% of rotor radius (typical)

    slot_depth   = (R_so - R_si) * 0.65
    slot_opening = 0.003
    slot_width   = (2 * 3.14159 * R_si / p["slots"]) * 0.50

    stator = StatorGeometry(
        outer_radius=R_so,
        inner_radius=R_si,
        slots=p["slots"],
        slot_profile=ParallelToothSlot(
            slot_width=slot_width,
            slot_depth=slot_depth,
            slot_opening=slot_opening,
        ),
    )

    motor = PMSM(
        poles=p["poles"],
        slots=p["slots"],
        stator=stator,
        rotor_outer_radius=R_ro,
        rotor_inner_radius=R_ri,
        stack_length=L,
        airgap=g,
        rated_power=p["rated_power_W"],
        rated_speed=p["rated_speed_rpm"],
        magnet_thickness=t_m,
        magnet_width_fraction=p["magnet_width_fraction"],
        turns_per_coil=p["turns_per_coil"],
        magnet_material=p.get("magnet_grade", "N42SH"),
    )
    return motor


def tool_run_analytical(p: dict) -> dict:
    """
    Build a PMSM and return all key analytical performance metrics.
    Wraps the actual Bohemien PMSM.back_emf_constant(), LossCalculator,
    and ThermalNetwork.
    """
    try:
        motor = _build_motor(p)
        mat   = MaterialLibrary()

        # ── Electromagnetic ───────────────────────────────────────────────
        Ke        = motor.back_emf_constant()
        T_rated   = motor.rated_torque
        import numpy as np
        omega_e   = motor.electrical_frequency * 2 * np.pi
        I_rated   = motor.rated_current
        Ld, Lq    = motor.Ld, motor.Lq

        # ── Losses ────────────────────────────────────────────────────────
        lc = LossCalculator(motor, material_lib=mat)
        budget = lc.loss_budget(
            speed_rpm=motor.rated_speed,
            torque_Nm=T_rated,
        )
        P_cu   = budget.copper_W
        P_fe   = budget.iron_W
        P_mech = budget.friction_W + budget.windage_W
        eta    = budget.efficiency * 100

        # ── Flux densities ────────────────────────────────────────────────
        B_gap = lc._airgap_B_fundamental()
        B_tooth = motor.stator.tooth_flux_density(B_gap)
        B_yoke  = motor.stator.yoke_flux_density(
            B_gap,
            motor.poles,
            motor.stack_length,
            motor.magnet_width_fraction,
        )

        # ── Current density ────────────────────────────────────────────────
        A_slot   = motor.stator.slot_profile.area()
        ff       = 0.45
        A_cond   = (A_slot * ff) / (motor.turns_per_coil * 2)
        J_Amm2   = (I_rated / np.sqrt(2)) / (A_cond * 1e6)

        # ── Thermal ────────────────────────────────────────────────────────
        cooling = WaterJacketCooling(flow_lpm=10.0)
        cooling._inlet_temp_C = 40.0   # coolant inlet temperature
        thm     = ThermalNetwork(motor, cooling=cooling, material_lib=mat)
        thermal = thm.steady_state({
            "copper_W": P_cu,
            "iron_W":   P_fe,
            "friction_W": P_mech,
            "windage_W": 0.0,
        })
        T_winding = thermal.T_winding_C
        T_magnet  = thermal.T_rotor_C

        return {
            "Ke_Wb":              round(Ke, 5),
            "rated_torque_Nm":    round(T_rated, 2),
            "rated_current_Arms": round(I_rated / 1.414, 2),
            "Ld_mH":              round(Ld * 1000, 4),
            "Lq_mH":              round(Lq * 1000, 4),
            "efficiency_pct":     round(eta, 2),
            "P_copper_W":         round(P_cu, 1),
            "P_iron_W":           round(P_fe, 1),
            "P_mech_W":           round(P_mech, 1),
            "winding_temp_C":     round(T_winding, 1),
            "magnet_temp_C":      round(T_magnet, 1),
            "tooth_flux_T":       round(B_tooth, 3),
            "yoke_flux_T":        round(B_yoke, 3),
            "airgap_flux_T":      round(B_gap, 3),
            "current_density_Amm2": round(J_Amm2, 2),
            "N_series":           motor.total_series_turns,
            "kw":                 round(motor.winding_factor(), 4),
        }
    except Exception as e:
        return {"error": str(e)}


def tool_check_design_rules(p: dict) -> dict:
    """
    Run the Bohemien DesignRuleChecker on the given motor parameters.
    Returns pass/fail for every engineering constraint.
    """
    try:
        motor = _build_motor(p)
        drc   = DesignRuleChecker(motor)
        results = drc.check_all()

        checks = {}
        for r in results:
            checks[r.name] = {
                "level":   r.level,       # "OK" | "WARN" | "ERROR"
                "message": r.message,
                "pass":    r.level == "OK",
            }

        errors   = [r for r in results if r.level == "ERROR"]
        warnings = [r for r in results if r.level == "WARN"]

        return {
            "all_pass":    len(errors) == 0,
            "n_errors":    len(errors),
            "n_warnings":  len(warnings),
            "checks":      checks,
            "report":      drc.report(),
        }
    except Exception as e:
        return {"error": str(e)}


def tool_compare_candidates(candidates: list[dict]) -> dict:
    """
    Run analytical evaluation on multiple candidate designs and
    rank them by efficiency. Returns the top design.
    """
    results = []
    for i, c in enumerate(candidates):
        try:
            res = tool_run_analytical(c)
            if "error" not in res:
                results.append({
                    "candidate_index": i,
                    "params": c,
                    "results": res,
                })
        except Exception:
            pass

    if not results:
        return {"error": "All candidates failed to evaluate"}

    # Rank by efficiency (descending)
    results.sort(key=lambda x: x["results"]["efficiency_pct"], reverse=True)

    return {
        "n_evaluated": len(results),
        "ranked": results[:5],   # top 5
        "best": results[0],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tool schemas — tell Claude exactly what each tool accepts
# ══════════════════════════════════════════════════════════════════════════════

MOTOR_PARAMS_SCHEMA = {
    "type": "object",
    "properties": {
        "poles":                  {"type": "integer",  "description": "Number of rotor poles. Must be even. Common: 4, 6, 8, 10, 12"},
        "slots":                  {"type": "integer",  "description": "Number of stator slots. Must give integer q=slots/(poles*phases). Common pairs: 8p/48s, 8p/36s, 6p/36s, 10p/12s"},
        "turns_per_coil":         {"type": "integer",  "description": "Conductor turns per coil side (1–20). Lower = lower Ke, higher = higher Ke"},
        "magnet_thickness_mm":    {"type": "number",   "description": "PM radial thickness in mm (3–12). Increases Ke and airgap flux density"},
        "magnet_width_fraction":  {"type": "number",   "description": "Fraction of pole pitch covered by magnet, alpha_p (0.60–0.95). 0.83 is typical"},
        "magnet_grade":           {"type": "string",   "description": "Magnet material key: N35, N42, N42SH, N52, N52H. Higher number = higher Br"},
        "stator_od_mm":           {"type": "number",   "description": "Stator outer diameter in mm. Hard constraint from packaging"},
        "bore_mm":                {"type": "number",   "description": "Stator bore diameter in mm. Sets airgap radius and rotor volume"},
        "stack_length_mm":        {"type": "number",   "description": "Active axial stack length in mm"},
        "airgap_mm":              {"type": "number",   "description": "Radial airgap clearance in mm (0.5–2.0). Smaller = higher Ke but tighter tolerance"},
        "rated_power_W":          {"type": "number",   "description": "Target rated output power in Watts"},
        "rated_speed_rpm":        {"type": "number",   "description": "Rated shaft speed in RPM"},
    },
    "required": [
        "poles", "slots", "turns_per_coil",
        "magnet_thickness_mm", "magnet_width_fraction",
        "stator_od_mm", "bore_mm", "stack_length_mm", "airgap_mm",
        "rated_power_W", "rated_speed_rpm",
    ],
}

TOOLS = [
    {
        "name": "run_analytical",
        "description": (
            "Evaluate a PMSM design analytically using the Bohemien solver. "
            "Returns Ke, torque, efficiency, losses, flux densities, winding temperature, "
            "current density, and inductances. Fast (< 1 second). "
            "Call this first to evaluate any candidate design."
        ),
        "input_schema": MOTOR_PARAMS_SCHEMA,
    },
    {
        "name": "check_design_rules",
        "description": (
            "Run the Bohemien DesignRuleChecker on a motor design. "
            "Checks tooth flux density, yoke flux density, slot fill, current density, "
            "tip speed, insulation, and winding temperature against engineering limits. "
            "Returns pass/fail for each rule and a full report. "
            "Call this after run_analytical converges to verify the final design."
        ),
        "input_schema": MOTOR_PARAMS_SCHEMA,
    },
    {
        "name": "compare_candidates",
        "description": (
            "Evaluate multiple candidate designs in one call and rank them by efficiency. "
            "Use this to rapidly explore the design space — e.g. sweep pole counts or "
            "turns while keeping other parameters fixed. "
            "Returns the top 5 ranked designs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "description": "List of motor parameter dicts to evaluate and compare",
                    "items": MOTOR_PARAMS_SCHEMA,
                }
            },
            "required": ["candidates"],
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# Rate-aware API client
# ══════════════════════════════════════════════════════════════════════════════

class RateAwareClient:
    """
    Wraps the Anthropic client. After every API call it reads the
    rate-limit headers and sleeps automatically if the token budget
    is nearly exhausted. This lets long optimisation runs proceed
    overnight without supervision.
    """

    def __init__(self, api_key: str, reserve_tokens: int = 5000):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.reserve = reserve_tokens
        self.calls   = 0

    def call(self, messages: list, system: str) -> anthropic.types.Message:
        while True:
            try:
                self.calls += 1
                response = self.client.messages.create(
                    model      = "claude-sonnet-4-20250514",
                    max_tokens = 4096,
                    system     = system,
                    tools      = TOOLS,
                    messages   = messages,
                )

                # ── Check rate limit headers ──────────────────────────────
                hdrs = getattr(response, "headers", {}) or {}
                remaining = int(hdrs.get("anthropic-ratelimit-tokens-remaining", 999999))
                reset_str = hdrs.get("anthropic-ratelimit-tokens-reset")

                if remaining < self.reserve and reset_str:
                    try:
                        reset_dt = datetime.fromisoformat(
                            reset_str.replace("Z", "+00:00")
                        )
                        wait_s = (reset_dt - datetime.now(timezone.utc)).total_seconds()
                        if wait_s > 0:
                            print(f"\n  ⏳ Token budget low ({remaining} remaining)."
                                  f" Waiting {wait_s:.0f}s for reset...")
                            time.sleep(wait_s + 2)
                    except Exception:
                        pass

                return response

            except anthropic.RateLimitError:
                print("\n  ⏳ Rate limit hit. Waiting 60s...")
                time.sleep(60)

            except anthropic.APIStatusError as e:
                if e.status_code == 529:
                    print("\n  ⏳ API overloaded. Waiting 30s...")
                    time.sleep(30)
                else:
                    raise


# ══════════════════════════════════════════════════════════════════════════════
# Tool dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def dispatch_tool(name: str, inputs: dict) -> Any:
    """Route a tool call to the correct Python function."""
    if name == "run_analytical":
        return tool_run_analytical(inputs)
    elif name == "check_design_rules":
        return tool_check_design_rules(inputs)
    elif name == "compare_candidates":
        return tool_compare_candidates(inputs["candidates"])
    else:
        return {"error": f"Unknown tool: {name}"}


# ══════════════════════════════════════════════════════════════════════════════
# Agent system prompt
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert PMSM motor design engineer with deep knowledge of 
electromagnetic design, thermal management, and manufacturing constraints.

You have access to the Bohemien Motor Designer analytical solver. Your job is to find 
the optimal PMSM design that satisfies all of the user's requirements.

## Optimisation strategy

1. **Understand requirements**: Parse the user's power, speed, voltage, size, and efficiency targets.

2. **Generate initial candidates**: Use compare_candidates to sweep a few pole/slot combinations 
   (e.g. 8p/48s, 8p/36s, 6p/36s) with rough initial geometry. Identify the most promising.

3. **Refine systematically**: Focus on the best candidate. Adjust one or two parameters at a time:
   - If efficiency is low: check where losses are (copper vs iron)
   - If winding temp is high: reduce current density (larger slot or fewer turns)
   - If Ke is wrong: adjust turns_per_coil (linear scaling) or magnet_thickness
   - If tooth flux > 1.8T: widen tooth (reduce slot_width or increase bore)

4. **Validate**: Once all analytical targets are met, call check_design_rules for the final verdict.

5. **Report**: Present the final design with a clear table of all parameters and results.

## Parameter relationships (key rules)
- Ke scales linearly with turns_per_coil and with N_series = turns_per_coil × slots/phases
- Copper loss scales as turns² (more turns → higher Rs → more loss)
- Iron loss scales as B²×f — reducing bore or increasing turns raises B at same power
- Winding temp directly follows copper loss — keep J < 5 A/mm² for continuous duty
- Tooth flux density = airgap_B × slot_pitch / tooth_width — keep < 1.8 T
- Valid slot/pole combinations: slots must be divisible by (poles × phases / gcd)

## Always show your reasoning between tool calls. Be concise but precise."""


# ══════════════════════════════════════════════════════════════════════════════
# Main agent loop
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(requirements: str, api_key: str, max_iterations: int = 20) -> None:
    """
    Main agentic loop. Runs until the agent signals end_turn,
    a design passes all rules, or max_iterations is reached.
    """

    client   = RateAwareClient(api_key)
    messages = [{"role": "user", "content": requirements}]

    print("\n" + "═" * 60)
    print("  BOHEMIEN DESIGN OPTIMISATION AGENT")
    print("═" * 60)
    print(f"\nRequirements:\n{requirements}\n")
    print("─" * 60)

    iteration   = 0
    final_found = False

    while iteration < max_iterations:
        iteration += 1
        print(f"\n[Iteration {iteration}]  API call #{client.calls + 1}")

        response = client.call(messages, SYSTEM_PROMPT)
        content  = response.content

        # ── Print text blocks ──────────────────────────────────────────────
        for block in content:
            if block.type == "text" and block.text.strip():
                print(f"\n{'─'*60}")
                print(block.text.strip())

        # ── Check stop condition ───────────────────────────────────────────
        if response.stop_reason == "end_turn":
            print(f"\n{'═'*60}")
            print("  AGENT COMPLETE")
            print("═" * 60)
            break

        # ── Process tool calls ─────────────────────────────────────────────
        tool_blocks = [b for b in content if b.type == "tool_use"]
        if not tool_blocks:
            break

        # Add the assistant turn to history
        messages.append({"role": "assistant", "content": content})

        tool_results = []
        for tb in tool_blocks:
            print(f"\n  🔧 Calling: {tb.name}")
            if tb.name == "compare_candidates":
                n = len(tb.input.get("candidates", []))
                print(f"     Comparing {n} candidates...")
            else:
                params = tb.input
                print(f"     poles={params.get('poles')}  slots={params.get('slots')}  "
                      f"turns={params.get('turns_per_coil')}  "
                      f"bore={params.get('bore_mm')}mm  "
                      f"OD={params.get('stator_od_mm')}mm  "
                      f"L={params.get('stack_length_mm')}mm")

            result = dispatch_tool(tb.name, tb.input)

            # Pretty-print the result summary
            if "error" in result:
                print(f"     ✗ ERROR: {result['error']}")
            elif tb.name == "run_analytical":
                print(f"     Ke={result.get('Ke_Wb')}Wb  "
                      f"T={result.get('rated_torque_Nm')}Nm  "
                      f"η={result.get('efficiency_pct')}%  "
                      f"T_wind={result.get('winding_temp_C')}°C  "
                      f"J={result.get('current_density_Amm2')}A/mm²")
            elif tb.name == "check_design_rules":
                status = "✓ ALL PASS" if result.get("all_pass") else \
                         f"✗ {result.get('n_errors')} errors, {result.get('n_warnings')} warnings"
                print(f"     {status}")
                if result.get("all_pass"):
                    final_found = True
            elif tb.name == "compare_candidates":
                best = result.get("best", {})
                if best:
                    r = best.get("results", {})
                    print(f"     Best: η={r.get('efficiency_pct')}%  "
                          f"T_wind={r.get('winding_temp_C')}°C")

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": tb.id,
                "content":     json.dumps(result),
            })

        # Add tool results to history
        messages.append({"role": "user", "content": tool_results})

        if final_found:
            # Let the agent produce its final summary
            print(f"\n[Final summary pass]")
            response = client.call(messages, SYSTEM_PROMPT)
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"\n{'═'*60}")
                    print(block.text.strip())
            break

    if iteration >= max_iterations:
        print(f"\n⚠  Reached maximum iterations ({max_iterations}). "
              "Review results above and tighten requirements if needed.")

    print(f"\n  Total API calls: {client.calls}")
    print(f"  Total iterations: {iteration}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_REQUIREMENTS = """
Design a surface PM synchronous motor (PMSM / BLAC) with these requirements:

Performance targets:
  - Rated power:      15 kW continuous
  - Rated speed:      3000 rpm
  - DC bus voltage:   400 V
  - Minimum efficiency: 93%

Packaging constraints:
  - Stator outer diameter: 200 mm maximum
  - Axial stack length: 100–160 mm

Thermal constraints:
  - Max winding temperature: 120°C (with 40°C coolant inlet)
  - Cooling: water jacket, 10 L/min

Additional:
  - 3-phase, 2-layer distributed winding
  - SPM rotor (surface magnets)
  - Magnet grade: N42SH or better
  - Target continuous current density: < 5 A/mm²

Please find the optimised pole/slot combination and geometry.
"""


def main():
    print("Bohemien Motor Designer — Design Optimisation Agent")
    print("=" * 50)

    # ── API key ───────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        api_key = input("\nEnter your Anthropic API key (or set ANTHROPIC_API_KEY env var): ").strip()
    if not api_key:
        print("ERROR: API key is required.")
        sys.exit(1)

    # ── Requirements ──────────────────────────────────────────────────────
    print("\nUsing default requirements. Edit DEFAULT_REQUIREMENTS in this file to customise.")
    print("\nRequirements preview:")
    print(DEFAULT_REQUIREMENTS[:300] + "...")

    use_default = input("\nUse default requirements? [Y/n]: ").strip().lower()
    if use_default in ("n", "no"):
        print("Enter requirements (press Enter twice when done):")
        lines, last = [], ""
        while True:
            line = input()
            if line == "" and last == "":
                break
            lines.append(line)
            last = line
        requirements = "\n".join(lines)
    else:
        requirements = DEFAULT_REQUIREMENTS

    # ── Run ───────────────────────────────────────────────────────────────
    run_agent(requirements, api_key)


if __name__ == "__main__":
    main()