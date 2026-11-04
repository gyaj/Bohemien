# Bohemien Motor Designer — Development Log

> **How to use this file**
> Paste the entire contents of this file as your **first message** when starting
> a new Claude chat. Claude will read it and have full context to continue
> development immediately. Update this file after each session by asking:
> *"Update DEVELOPMENT_LOG.md with what we did today."*

---

## Project identity

| Item | Value |
|---|---|
| Repository | github.com/gyaj/Bohemien |
| Package folder | `Bohemien_Motor_Designer/` |
| Python package name (internal imports) | `Bohemien_Motor_Designer` |
| Original name (all internal code was renamed from this) | `Bohemien_Motor_Designer` |
| License | MIT |
| CLA | In place via GitHub Actions (`license/cla` status check) |
| Python version target | 3.11 (user's machine: Python 3.11.8 on Windows 10) |
| GUI framework | tkinter + matplotlib TkAgg backend |
| Key dependencies | numpy, scipy, matplotlib, anthropic (for agent) |
| Exe build | PyInstaller 6.x, `build_exe.bat` → `dist/Bohemien_Motor_Designer.exe` |

---

## Reference motor (used for all validation)

| Parameter | Value |
|---|---|
| Topology | 8-pole / 48-slot SPMSM |
| Rated power | 30 kW |
| Rated speed | 4000 rpm |
| DC bus voltage | 400 V |
| Stator OD | 225 mm |
| Stator bore | 164 mm |
| Airgap | 1 mm |
| Stack length | 130 mm |
| Magnet thickness | 6 mm |
| Magnet width fraction α_p | 0.83 |
| Magnet grade | N42SH (Br = 1.22 T, μ_r = 1.05) |
| Turns per coil | **1** (was listed as 2 under the pre-Nov-2026 turns bug) |
| N_series per phase | 16 (= coils_per_phase 16 × 1) |
| Ke analytical | 0.1133 Wb (FEM 2D 0.1134, FEM 3D 0.1139) |
| Rated torque | 71.6 Nm |
| Efficiency | 94.7% (after turns fix; was 96.67% with half the real copper) |
| Ld / Lq (analytical total) | 0.0911 mH (L_2d 0.0833 + L_ew 0.0078) |
| Ld / Lq (2D FEM raw) | 0.0852 / 0.0784 mH |
| Ld / Lq (3D FEM raw) | 0.0857 / 0.0790 mH |

---

## Package structure (63 Python files)

```
Bohemien_Motor_Designer/
  __init__.py                    version = "2.1.0", lazy (no eager imports)
  run_gui.py                     entry point for script and PyInstaller exe
  Bohemien_Motor_Designer.spec   PyInstaller build recipe
  build_exe.bat                  Windows double-click build script (cd .. then pyinstaller)
  setup.py

  core/
    motor.py                     Motor base dataclass
    pmsm.py                      PMSM: Ke, dq-model, MTPA, field weakening, analytical Ld/Lq
    geometry/
      stator.py                  StatorGeometry dataclass
      slot_profiles.py           ParallelToothSlot, TrapezoidalSlot, OpenSlot, RoundBottomSlot
      winding.py                 WindingLayout, winding_factor(), coil_sides_for_phase()
      rotor.py
    induction.py
    synrel.py
    manufacturing_report.py      12-section manufacturing specification generator
    specs.py

  analysis/
    losses.py                    LossCalculator: Bertotti iron loss, Dowell copper loss
    performance.py

  fea/
    py_mesh.py                   2D structured polar mesh — build_motor_mesh()
    py_solver.py                 2D magnetostatic FEM — solve_magnetostatic(), compute_B_field()
    py_torque.py                 Arkkio torque, compute_flux_linkage()
    py_runner.py                 2D FEM runner — Ke, Ld, Lq, cogging sweep
    mesh3d.py                    3D tetrahedral mesh — Freudenthal hex→5-tet split, edge table
    solver3d.py                  3D Nédélec edge-element FEM — curl-curl, tree-cotree gauge
    runner3d.py                  FEMRunner3D — pure-extrusion 3D check of 2D (+ analytical L_ew); energy cross-check
    mesh_viz.py                  plot_mesh_overview() — 3-panel mesh visualisation
    [legacy files]               fem_mesh.py, fem_solver.py, gmsh_exporter.py, runner.py etc.

  gui/
    __init__.py                  lazy — no module-level imports
    app.py                       MotorDesignApp (tk.Tk) — 12 tabs, 2252 lines

  io/
    dxf_export.py                AutoCAD R12 ASCII DXF — export_dxf()
    json_spec.py

  thermal/
    lumped_model.py              ThermalNetwork class (NOT LumpedThermalModel)
    cooling.py                   WaterJacketCooling, AirCooling, OilSprayCooling

  materials/
    library.py                   MaterialLibrary, LaminationMaterial, MagnetMaterial

  scaling/
    similarity.py

  utils/
    validation.py                DesignRuleChecker — check_all() → list[CheckResult]

  tests/
    test_inductance_reconciliation.py   7 tests, 17 s: analytical ↔ 2D ↔ 3D (run: python -m pytest Bohemien_Motor_Designer/tests)

  drive/
    field_weakening.py
    inverter.py
```

---

## GUI tabs

| Tab | Name | Key widget |
|---|---|---|
| 1 | Summary | scrolledtext |
| 2 | Losses | scrolledtext |
| 3 | Thermal | scrolledtext |
| 4 | DRC | scrolledtext |
| 5 | Efficiency Map | matplotlib figure |
| 6 | Scaling | scrolledtext |
| 7 | Mfg Report | scrolledtext + Save button |
| 8 | FEA | 2D FEM controls + figure |
| 9 | Mesh Viz | 3-panel mesh figure |
| 10 | DXF Export | DXF preview + export button |
| 11 | 3D FEM | 3D FEM controls + results text |
| 12 | Flux Viz | **NEW** — 3-panel flux visualisation |

---

## Complete bug fix history

### Session 1–3 (2026-03-01) — Original motordesign package
| # | File | Bug | Fix |
|---|---|---|---|
| 1 | solver.py | 25 T airgap field, 800 kV back-EMF, zero torque | Six-bug fix: Carter reluctance formula, PM source term, mesh element tagging |
| 2 | solver.py | Non-120° back-EMF phase separation | Winding phase assignment algorithm rewritten |
| 3 | solver.py | Trapezoidal back-EMF waveform | Staircase winding function replaced with sinusoidal |
| 4 | solver.py | Efficiency map showing >100% | Flux density calculation corrected |
| 5 | Various | 4-pole motors broken | Hardcoded 12-slot pattern generalised |
| 6 | back_emf_constant() | Formula 3.5× too high | Wrong reluctance fraction g/g_eff → t_m/(t_m+μ_r·g) |
| 7 | torque | Static torque sign error | Sign convention fixed |
| 8 | FEM stack | PM remanence never applied to mesh | Entire FEM replaced with validated Fourier solver |

### Session 4–7 (2026-03-05 to 03-10) — Bohemien_Motor_Designer rebuild
| # | File | Bug | Fix |
|---|---|---|---|
| 9 | winding.py | Winding factor calculation wrong | Distribution factor formula corrected |
| 10 | materials/library.py | Steinmetz coefficients missing | BH curve tables added for M19/M270/M400/M800/Arnon5 |
| 11 | thermal/lumped_model.py | Thermal ODE unstable | Implicit Euler integrator added |
| 12 | utils/validation.py | Current density J formula used slot area not conductor area | Fixed to A_conductor = A_slot × ff / (N_coil × layers) |
| 13 | py_solver.py | PM polarity sign wrong → all poles same direction | Sign alternation added per pole index |
| 14 | py_runner.py | Arkkio torque giving wrong result | Replaced with Park torque from flux linkage derivatives |
| 15 | gui/app.py | LabeledEntry missing delete/insert methods | Methods added |
| 16 | gmsh_exporter.py | Slot geometry split incorrectly | Split into 2 surfaces per layer |
| 17 | manufacturing_report.py | DRC J formula, B_gap formula, ISO tolerances, strand packing | 8 fixes across 4 files |

### Session 8 (2026-04-02) — 3D FEM Phase 1 + PyInstaller
| # | File | Bug | Fix |
|---|---|---|---|
| 18 | fea/solver3d.py | `np.stack(axis=1)` → wrong barycentric gradients → B field 50–100× too small | Changed to `axis=2` (edge vectors must be COLUMNS of T) |
| 19 | fea/solver3d.py | Flux linkage had extra L² factor | `dpsi = N × sum(Az×dV) / (A_layer × L)` |
| 20 | gui/app.py | `import tkinter` at module level → PyInstaller dropped module | All tkinter/matplotlib imports moved inside `main()` |
| 21 | gui/__init__.py | `from .app import main` at module level | Deferred to wrapper function |
| 22 | Bohemien_Motor_Designer/__init__.py | Eager imports chain → tkinter at analysis time | Lazy `__init__` (version string only) |
| 23 | Bohemien_Motor_Designer.spec | `__file__` undefined in spec exec context | Use `SPECPATH` instead |
| 24 | Bohemien_Motor_Designer.spec | `pathex=['.']` wrong — inside package dir | `pathex=[Path(SPECPATH).parent.resolve()]` |
| 25 | build_exe.bat | Ran from inside Bohemien_Motor_Designer\ | Added `cd /d "%~dp0.."` |
| 26 | gui/app.py | `\uXXXX` backslash escapes inside f-string `{}` expressions → SyntaxError Python 3.11 | Replaced all 15 occurrences with actual Unicode characters |
| 27 | Bohemien_Motor_Designer.spec | `PIL` in excludes → matplotlib.colors imports PIL → ModuleNotFoundError at runtime | Removed PIL from excludes, added to hiddenimports |
| 28 | fea/runner3d.py | Ld/Lq perturbation used Iq_rated background → Park projection error → Ld 1000× too large | Zero-background perturbation, reuse psi_d0=Ke_3d |

### Session 9 (current session) — Packaging, CLA, agents, flux viz
| # | File | Bug | Fix |
|---|---|---|---|
| 29 | gui/app.py | `import tkinter` still at module-level in old zip | Restructured: only stdlib at module level, tkinter inside `main()` |
| 30 | Bohemien_Motor_Designer.spec | `__file__` in spec (again, different user's zip) | `SPECPATH` used throughout |
| 31 | Bohemien_Motor_Designer.spec | `pathex` pointed to package folder not parent | `Path(SPECPATH).parent` |
| 32 | gui/app.py | `\uXXXX` escapes in f-strings → Python 3.11 SyntaxError (reported from exe) | All 15 occurrences replaced with literal Unicode |
| 33 | spec | `PIL` excluded → matplotlib crash at runtime | PIL removed from excludes, added to hiddenimports |
| 34 | fea/py_runner.py | 2D Ld/Lq: Iq_rated background (172 A) contaminates δψ_d via Park projection error | Zero background current, dI=20 A fixed |
| 35 | fea/runner3d.py | 3D dI = 0.05×Ke/Ld_est = 123 A → iron saturation | Fixed dI = 20 A |
| 36 | bohemien_agent.py | Class names: LumpedThermalModel → ThermalNetwork, CoolingSystem → WaterJacketCooling | All class names corrected to match renamed package |


### Session 10 (2026-11-03) — 2D/3D inductance reconciliation (the ">10×" discrepancy)
Root cause was five stacked bugs, not end-winding leakage.  All verified on the reference motor
with a stored-energy cross-check (L = 4W/3I², gauge-invariant, independent of flux-linkage bookkeeping).

| # | File | Bug | Fix |
|---|---|---|---|
| 37 | core/geometry/winding.py | `total_series_turns_per_phase` divided by an extra 2 for double-layer windings → every analytical quantity used half the real turns (Ke 2× low, L 4× low). FEM summed the 32 coil sides correctly. | `N_series = coils_per_phase × N_coil / a`, `coils_per_phase = slots·layers/(2·phases)`. New `coils_per_phase` property. |
| 38 | fea/py_solver.py, fea/solver3d.py | Current source `J = N_coil·I/(A_layer·ff)` — J is applied over the whole meshed slot area, so dividing by fill factor injected 1/ff = 2.08× too many ampere-turns | `J = N_coil·I/A_layer`. (Caught because L_energy/L_fluxlink = 1/ff exactly.) |
| 39 | fea/py_mesh.py (×2), fea/mesh3d.py | Rotor inter-pole gaps (R_mi<r<R_ro outside magnet arc) tagged ROTOR_IRON → inset-magnet rotor with q-axis iron → Lq/Ld = 1.6–1.9 on an SPM | New `_rotor_air_tag` (= 3+poles+2), kept out of the Arkkio ring. Lq/Ld now 0.92. |
| 40 | core/pmsm.py | Analytical `L_ag = (3/π²)μ0(Nkw)²τ_pL/g_c` — missing 1/p (right only for p=2; 2× high here). Slot leakage used 2m/Q not 4m/Q. Tooth-tip and harmonic leakage absent. | `L_ag=(6/π²)μ0(Nkw)²τ_pL/(p·g_c)`, `L_slot=(4m/Q)μ0LN²λ_slot`, `L_tt=(4m/Q)μ0LN²λ_tt`, `L_h=σ_δ·L_ag` (Pyrhönen 3.110/3.127/4.33/4.43). New `inductance_breakdown()`, `end_winding_leakage_inductance()` (Richter/Pyrhönen eq. 4.64, λ_lew=0.324, λ_W=0.243). |
| 41 | fea/mesh3d.py | Single Freudenthal 5-tet template on every hex → **75 % of interior faces unshared** (non-conforming mesh): field leaks across cracks, current-driven L 25–30 % high and growing with refinement | `HEX_TO_TET_5_MIRROR` + checkerboard parity `(ir+it+iz)&1`. Zero cracks; ~25 % fewer edges. |
| 42 | fea/mesh3d.py | End region outside the stack tagged rotor iron / stator iron ("iron block"); current source terminated at stack faces (not div-free) | End region = air; slot conductors continued straight to the axial boundary (`_end_coil_tag`, `elem_idx_ext`); A_t=0 Dirichlet added on z-faces. Default `n_end_winding=0` = pure extrusion. |
| 43 | fea/solver3d.py | Spanning tree built from node 0 ignoring the Dirichlet boundary → interior tree paths between boundary nodes pinned to zero → spurious zero-flux constraints (field collapsed to 0 with z-face Dirichlet) | Boundary-first BFS tree (Manges & Cendes 1995): span the boundary graph with boundary edges first, then grow inward. |
| 44 | fea/mesh3d.py `_outer_boundary` | "Both endpoints on boundary" also caught interior tet diagonals between a z-face node and a cylinder node | Edge is on the boundary only if both endpoints are on the **same** piece (cylinder, z0 face, or zL face). |
| 45 | fea/solver3d.py `compute_flux_linkage_3d` | Divided by L_stack (returned Wb/m) and integrated A_z over the open in-stack segment — **gauge-dependent** under tree-cotree | ψ = N_coil·Σ(A_z·dV)/A_layer over the extended conductor (loop closes on the A_t=0 faces at zero cost). Verified = energy method to 4 digits. |
| 46 | fea/runner3d.py | dI sized from Ke/Ld_est → 123 A (saturation) [from S9]; no cross-check | dI = 20 A; built-in energy-method Ld → `Ld_energy_check` (expect 1.000); reports `L_ew_mH`, `Ld_total_mH`, `L_2d_anal_mH`, `end_region_model`. |
| 47 | fea/py_runner.py | No convergence information; no end-winding term; default 4 airgap layers (Ke 0.80×) | `_convergence_check()` (fine vs half-density mesh, flag ratio > 1.10); `L_ew_mH`, `Ld_total_mH`, analytical breakdown fields; default `n_radial_airgap=8`. |
| 48 | gui/app.py | 3D tab claimed "end-winding effects included"; showed raw numbers only | Tab 8 and Tab 11 now show raw FEM, +L_ew, totals, analytical L_2d/L_total, 2D-vs-3D ratios with ✓/~/⚠ tags, energy check, mesh convergence. 3D defaults 8/2/8/0. 2D runner 12 ang × 8 AG. |
| 49 | examples/full_design_example.py | 11 turns/coil "→ 88 series turns" — physically 176 turns, 2 kV back-EMF (DRC now flags it) | 1 turn/coil → 16 series turns, Ke 0.113, 190 V pk. |

**Reconciliation result (reference motor, 12 ang/slot, 8 airgap layers):**

| | Ke/Ke_anal | Ld (mH) | Lq (mH) | Ld+L_ew (mH) |
|---|---|---|---|---|
| 2D FEM | 1.001 | 0.0852 | 0.0784 | 0.0930 |
| 3D FEM (pure extrusion) | 1.005 | 0.0857 | 0.0790 | 0.0935 |
| Analytical (corrected) | 1.000 | 0.0833 | 0.0833 | 0.0911 |

3D/2D on Ld = **1.006** (was > 10×). 3D solve now ~10 s at 170k edges (was 20 min).

**What the literature says and what Bohemien now does with it:** 2D cannot see end-winding leakage
(Chalmers ICEM 2024 / IECON 2025, Calvano COMPEL 2013: typically 5–30 % of L_sync, more for hairpin /
short stacks). The 3D solver here is a pure extrusion, so it cannot see it either — it is a *check* on 2D,
not an end-winding model. L_ew is added analytically (Richter/Pyrhönen, 8.5 % of total here, ±30–50 %)
to both. Real end-turn geometry (circumferential arcs, loop closure) is a roadmap item; the
straight-conductor `n_end_winding>0` option is labelled experimental because it adds the inductance of
straight conductors in air (0.014 mH here), not end-turn leakage.

---

## Critical architecture decisions

### PyInstaller — why it kept failing
The failure chain that caused "No module named Bohemien_Motor_Designer.gui.app" three times:

1. `app.py` had `import tkinter` at module level → PyInstaller's Linux build environment has no tkinter → module silently dropped from bundle
2. `pathex=['.']` inside the package folder → `import Bohemien_Motor_Designer` looked for `Bohemien_Motor_Designer/Bohemien_Motor_Designer/` → not found
3. `__file__` undefined in spec exec context → use `SPECPATH`
4. `\uXXXX` inside f-string expressions → Python 3.11 SyntaxError → file silently dropped
5. `PIL` in excludes list → matplotlib.colors imports PIL → runtime crash

**The definitive build rule:** Always run `pyinstaller` from the **parent** of the package folder. `build_exe.bat` does this with `cd /d "%~dp0.."`.

### 3D FEM — Nédélec edge elements
- DOFs live on mesh edges, not nodes
- `axis=2` in `np.stack()` — edge vectors must be COLUMNS of T (not rows)
- Tree-cotree gauge removes curl-null space — the tree MUST be boundary-first (span the Dirichlet boundary with boundary edges, then grow inward) or Dirichlet+tree pin spurious interior loops
- Hex→tet split MUST alternate template parity (checkerboard) or the mesh is non-conforming
- Flux linkage: `ψ = N_coil × direction × Σ(A_z_cent × vol) / A_layer` over the EXTENDED conductor (gauge-invariant only as a closed loop; no /L_stack)
- Always cross-check Ld with the stored-energy method L = 4W/3I² (gauge-invariant); `Ld_energy_check` in runner3d results
- Zero-background perturbation for Ld/Lq — never use rated current as background

### Inductance perturbation — universal rule
Always use (and never divide the current source by the fill factor — J acts on the whole meshed slot area):
- Background current: `Id=0, Iq=0` (no-load)
- Baseline `ψ_d0 = Ke_fea` (reuse from no-load solve, saves one FEM call)
- Perturbation: `dI = 20 A` fixed (well above noise floor ~0.001 mWb, well below saturation)
- Solve `(dI, 0)` for Ld and `(0, dI)` for Lq

### Class names after rename (Bohemien_Motor_Designer)
| Wrong name (old) | Correct name (current) |
|---|---|
| `LumpedThermalModel` | `ThermalNetwork` |
| `CoolingSystem(...)` | `WaterJacketCooling(...)` |
| `thermal.winding_C` | `thermal.T_winding_C` |
| `thermal.rotor_C` | `thermal.T_rotor_C` |
| `LossBudget` | `LossBudget` ← unchanged |
| `DesignRuleChecker` | `DesignRuleChecker` ← unchanged |

---

## FEM validated results (Session 10, reference motor, turns_per_coil = 1)

### 2D convergence — airgap radial layers are the controlling parameter
| n_ang | n_ag | nodes | Ke/Ke_anal | Ld_2d (mH) |
|---|---|---|---|---|
| 8 | 4 | 6 144 | 0.761 | 0.0606 |
| 16 | 4 | 12 288 | 0.805 | 0.0681 |
| 16 | 8 | 15 360 | 1.007 | 0.0845 |
| 12 | 8–24 | 11–21 k | 0.98–1.00 | 0.0840–0.0852 |
| 24 | 16 | 32 256 | 0.977 | 0.0832 |

### 3D convergence (pure extrusion, n_axial = 2)
| n_ang | n_ag | edges | time | Ke/Ke_anal | Ld_3d (mH) | energy check |
|---|---|---|---|---|---|---|
| 4 | 8 | 74 k | 4 s | 0.738 | 0.0653 | 1.0000 |
| 8 | 8 | 112 k | 6 s | 0.961 | 0.0759 | 1.0000 |
| 12 | 8 | 168 k | 9 s | 1.005 | 0.0857 | 1.0000 |
| 12 | 16 | 237 k | 19 s | 0.981 | 0.0845 | 1.0000 |

n_axial 1 → 4 changes Ld_3d by < 2 % (extrusion is axially consistent).

### Analytical breakdown (corrected formulas)
L_ag 0.0590 · L_h 0.0016 · L_slot 0.0144 · L_tt 0.0084 → **L_2d 0.0833 mH** · L_ew 0.0078 → **L_total 0.0911 mH**

---

## Licensing and CLA setup

| Item | Status |
|---|---|
| License | MIT (in repo root as `LICENSE`) |
| CLA.md | In repo root |
| CONTRIBUTING.md | In repo root |
| CLA enforcement | GitHub Actions `license/cla` status check (required to merge) |
| Branch protection | main branch requires `license/cla` to pass |
| CLA signer storage | `signatures/cla.json` in repo |
| Jurisdiction | State of California, Contra Costa County |
| Privacy Policy | Gist at gist.github.com |

---

## Agent (bohemien_agent.py)

**Location:** repo root (same level as `Bohemien_Motor_Designer/`)

**Run:**
```powershell
cd C:\Bohemien\https-github.com-gyaj-Bohemien
python bohemien_agent.py
```

**How it imports the package:**
Scans for `Bohemien_Motor_Designer/`, `Bohemien_Motor_Designer/`, `bohemien/` in the repo root. Uses `importlib.import_module(f"{_PKG}.submodule")` so it works regardless of what the folder is named.

**Three tools exposed to Claude API:**
1. `run_analytical` — calls PMSM.back_emf_constant(), LossCalculator, ThermalNetwork
2. `check_design_rules` — calls DesignRuleChecker.check_all()
3. `compare_candidates` — batch-evaluates multiple designs and ranks by efficiency

**Rate-aware wrapper:** `RateAwareClient` reads `anthropic-ratelimit-tokens-remaining` header and sleeps until reset if budget is low. Safe for overnight runs.

**Dependencies:** `pip install anthropic` (one new package; numpy/scipy already installed)

---

## Tab 12 — Flux Visualisation (Phase 1, implemented)

**What was added to `gui/app.py`:**
- Tab registration in `_build_ui()`
- `_build_flux_tab(parent)` — builds controls strip + 3-panel figure
- `_flux_run_async()` — launches background thread
- `_flux_worker()` — mesh build + magnetostatic solve
- `_flux_render(mesh, A_z, Bx, By, B_mag, motor, angle_deg)` — draws 3 panels
- `_flux_animate_step()` — animation loop (advances rotor angle, re-solves)
- `_flux_save()` — saves PNG/SVG/PDF

**Three panels:**
1. `tricontourf` + `tricontour` of A_z — exact flux lines (A_z isolines = flux lines in 2D)
2. `tripcolor` of B_mag (gouraud) + geometry circles — saturation map
3. B magnitude histogram per material region + saturation reference line at 1.8 T

**Physics note:** A_z contours ARE the flux lines because B = curl(A_z ẑ), so B is tangent to A_z isolines. No interpolation needed — direct tricontour on nodal A_z values.

---

## Pending items (not yet implemented)

| Item | Priority | Notes |
|---|---|---|
| Flux Viz Phase 2 | High | Streamplot via scipy.griddata + rotor angle slider live update |
| Flux Viz Phase 3 | Medium | PyVista 3D streamlines + VTK export button |
| IPM rotor topologies | Medium | V-type magnet barriers, GMSH Boolean subtract |
| Induction motor | Medium | IM equivalent circuit + torque-speed curve + GUI tab |
| 3D end-turn geometry | High | Circumferential end-turn arcs with loop closure so the 3D solver can actually resolve L_ew; validate against Richter/Pyrhönen and one commercial 3D FEM run |
| Winding factor check | Low | `winding_factor()` returns 0.9914 for q=2 (textbook k_d = 0.966); Ke still matches FEM to 0.1 %, so investigate before changing |
| 3D FEM iterative solver | Medium | pyamg for meshes >500k DOF (less urgent now: 170k-edge conforming mesh solves in ~10 s) |
| 3D FEM sliding interface | Low | Cogging torque in 3D |
| 3D FEM Phase 2 (IM) | Low | Time-harmonic A-V formulation, end-ring coupling |
| Agent: FEM Interpreter | Medium | Runs FEM, diagnoses Ke/Ld discrepancies automatically |
| Agent: Design Review | Medium | Checks DRC + produces written engineering report |
| Agent: Rate-aware overnight | Low | Already in bohemien_agent.py as RateAwareClient |
| Update design document | Low | Add flux viz and 3D FEM sections |
| Have IP lawyer review CLA | High (before revenue) | California jurisdiction, Contra Costa County |
| Register trademark | Medium | Before commercial launch |

---

## How to continue development in a new chat

**Paste this entire file** as your first message, then say what you want to work on. For example:

> *"[paste this file] I want to implement Phase 2 of the flux visualisation tab — streamplot with live rotor angle slider."*

Claude will read the log, understand the codebase state, and continue from exactly where we left off.

**After each session**, ask Claude:
> *"Update DEVELOPMENT_LOG.md with what we changed today."*

Claude will output the updated sections to paste back into this file.

---

*Last updated: Session 10 — 3 Nov 2026*
*Total bugs fixed across all sessions: 49*
*Total Python files: 63 across 11 subpackages (incl. tests/)*
*GUI lines of code: 2,252*
*Test suite: `python -m pytest Bohemien_Motor_Designer/tests` — 7 passed in ~17 s*
