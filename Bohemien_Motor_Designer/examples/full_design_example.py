"""
Bohemien_Motor_Designer — Full Design Workflow Example
============================================
Demonstrates the complete top-down design flow:

  1.  Define requirements via DesignSpec (power, speed, DC bus, cooling)
  2.  Feasibility check using scaling laws before any geometry
  3.  Construct motor with parametric geometry
  4.  Run Design Rule Checker
  5.  Loss budget at rated point
  6.  Efficiency map
  7.  Thermal steady-state analysis
  8.  Drive cycle transient temperature
  9.  Save/load design to JSON

Motor: 30 kW, 4000 rpm, 400V DC bus, water-jacket cooled, 8p/48s PMSM
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from Bohemien_Motor_Designer.core import (DesignSpec, DriveSpec, CoolingSpec,
                                InsulationSpec, EnvelopeConstraints, PMSM,
                                StatorGeometry, ParallelToothSlot,
                                SPMRotorGeometry, WindingLayout)
from Bohemien_Motor_Designer.materials import MaterialLibrary
from Bohemien_Motor_Designer.analysis import LossCalculator, PerformanceAnalyzer
from Bohemien_Motor_Designer.thermal import ThermalNetwork, WaterJacketCooling
from Bohemien_Motor_Designer.drive import Inverter
from Bohemien_Motor_Designer.scaling import MotorScalingLaws
from Bohemien_Motor_Designer.utils import DesignRuleChecker
from Bohemien_Motor_Designer.io import save_design


def main():
    print("=" * 65)
    print("  Bohemien_Motor_Designer — Full Design Workflow")
    print("=" * 65)

    # ──────────────────────────────────────────────────────────────────
    # STEP 1 — Define requirements
    # ──────────────────────────────────────────────────────────────────
    print("\n[1] Defining design requirements (DesignSpec)...")

    spec = DesignSpec(
        power_kW    = 30.0,
        speed_rpm   = 4000.0,
        speed_range = (500, 12000),
        drive = DriveSpec(
            dc_bus_voltage  = 400.0,
            topology        = "2L-VSI",
            device          = "SiC-MOSFET",
            switching_freq  = 20e3,
            modulation      = "SVPWM",
        ),
        cooling = CoolingSpec(
            cooling_type    = "water-jacket",
            coolant_temp_C  = 65.0,
            coolant_flow_lpm= 12.0,
        ),
        insulation = InsulationSpec(
            insulation_class = "H",
        ),
        envelope = EnvelopeConstraints(
            max_outer_diameter_mm = 260.0,
            max_length_mm         = 220.0,
            max_mass_kg           = 25.0,
        ),
        overload_factor    = 2.5,
        efficiency_target  = 0.95,
        duty_cycle         = "S1",
    )
    print(spec.summary())

    # ──────────────────────────────────────────────────────────────────
    # STEP 2 — Scaling law feasibility check
    # ──────────────────────────────────────────────────────────────────
    print("\n[2] Scaling law feasibility analysis...")

    est = MotorScalingLaws.size_estimate(
        spec.power_kW, spec.speed_rpm, "water-jacket")
    print(est.summary())

    feasibility = MotorScalingLaws.feasibility_check(
        spec.power_kW, spec.speed_rpm,
        spec.envelope.max_outer_diameter_mm,
        spec.envelope.max_length_mm,
        "water-jacket",
    )
    print(f"\n  Fits in envelope: {'YES' if feasibility['feasible'] else 'NO'}")
    print(f"  OD margin        : {feasibility['OD_margin_pct']:.1f}%")
    print(f"  Length margin    : {feasibility['L_margin_pct']:.1f}%")
    print(f"  Note             : {feasibility['recommendation']}")

    # Compare cooling options
    print("\n  Cooling comparison (smallest first):")
    for e in MotorScalingLaws.compare_cooling(spec.power_kW, spec.speed_rpm):
        print(f"    {e.cooling:25s}: OD={e.outer_diameter_mm:.0f}mm  "
              f"L={e.stack_length_mm:.0f}mm  V={e.active_volume_L:.2f}L")

    # ──────────────────────────────────────────────────────────────────
    # STEP 3 — Construct motor geometry
    # ──────────────────────────────────────────────────────────────────
    print("\n[3] Constructing motor geometry...")

    # Parametric stator slot
    slot = ParallelToothSlot(
        slot_width   = 0.0080,
        slot_depth   = 0.0240,
        slot_opening = 0.0030,
        wedge_height = 0.0010,
    )
    stator = StatorGeometry(
        outer_radius = 0.125,
        inner_radius = 0.082,
        slots        = 48,
        slot_profile = slot,
        lamination   = "M270-35A",
    )

    # SPM rotor with retention sleeve (for 4000 rpm)
    rotor_geo = SPMRotorGeometry(
        outer_radius         = 0.081,
        inner_radius         = 0.030,
        magnet_thickness     = 0.006,
        magnet_width_fraction= 0.83,
        magnet_material      = "N42SH",
        sleeve_thickness     = 0.002,
        sleeve_material      = "carbon_fibre",
    )

    # Winding: 8p/48s, double-layer, 11 turns/coil, 1 parallel path
    # → 88 series turns/phase → Ke = 0.109 V·s/rad → back-EMF 183V @ 4000rpm
    winding = WindingLayout(
        poles          = 8,
        slots          = 48,
        phases         = 3,
        layers         = 2,
        turns_per_coil = 11,
        parallel_paths = 1,
    )

    motor = PMSM(
        poles              = 8,
        slots              = 48,
        stator             = stator,
        rotor_geo          = rotor_geo,
        rotor_outer_radius = 0.081,
        rotor_inner_radius = 0.030,
        stack_length       = 0.130,
        airgap             = 0.001,
        rated_speed        = 4000.0,
        rated_power        = 30000.0,
        magnet_material    = "N42SH",
        magnet_thickness   = 0.006,
        magnet_width_fraction = 0.83,
        turns_per_coil     = 11,
        conductor_diameter = 0.0009,   # 0.9mm wire, ~77A rated
        slot_fill_factor   = 0.48,
        parallel_paths     = 1,
        winding            = winding,
        spec               = spec,
    )

    print(motor.summary())
    print(f"\n  Stator: {stator.summary()}")
    print(f"\n  Winding: {winding.summary()}")
    print(f"\n  Back-EMF constant Ke = {motor.back_emf_constant():.4f} V·s/rad")
    omega_e = motor.rated_speed * 2*np.pi/60 * motor.pole_pairs
    print(f"  Peak back-EMF @ {motor.rated_speed:.0f} rpm = {motor.back_emf_constant()*omega_e:.1f} V")
    print(f"  Max available phase Vpk = {spec.drive.max_phase_voltage_peak():.1f} V")

    # ──────────────────────────────────────────────────────────────────
    # STEP 4 — Design Rule Checker
    # ──────────────────────────────────────────────────────────────────
    print("\n[4] Running Design Rule Checker...")

    lib      = MaterialLibrary()
    inverter = Inverter(
        dc_bus_V       = spec.drive.dc_bus_voltage,
        switching_freq = spec.drive.switching_freq,
        topology       = spec.drive.topology,
        device         = spec.drive.device,
    )
    print(inverter.summary())

    checker = DesignRuleChecker(motor, spec=spec, inverter=inverter,
                                 material_lib=lib)
    checker.check_all()
    print(checker.report())

    # ──────────────────────────────────────────────────────────────────
    # STEP 5 — Loss budget at rated point
    # ──────────────────────────────────────────────────────────────────
    print("\n[5] Loss budget at rated operating point...")

    loss_calc = LossCalculator(motor, lib, temperature=120.0, inverter=inverter)
    lb = loss_calc.loss_budget(motor.rated_speed, motor.rated_torque)
    lb.print_summary()

    # ──────────────────────────────────────────────────────────────────
    # STEP 6 — Efficiency map
    # ──────────────────────────────────────────────────────────────────
    print("\n[6] Computing efficiency map...")

    perf     = PerformanceAnalyzer(motor, lib, inverter=inverter)
    eff_map  = perf.pmsm_efficiency_map(
        speed_range=(200, 12000),
        n_speed=35,
        n_torque=25,
    )
    print(f"  Peak efficiency        : {eff_map['peak_eff']*100:.2f}%")
    print(f"  Efficiency at rated pt : {perf.efficiency_at(motor.rated_speed, motor.rated_torque)*100:.2f}%")

    # Torque-speed envelope
    envelope = perf.torque_speed_envelope(n_points=80, I_max=150.0)
    print(f"  Max torque at base speed: {envelope['T_max_Nm'][0]:.1f} N·m")
    print(f"  Speed range modelled    : {envelope['speed_rpm'][0]:.0f} – {envelope['speed_rpm'][-1]:.0f} rpm")

    # ──────────────────────────────────────────────────────────────────
    # STEP 7 — Thermal analysis (steady state)
    # ──────────────────────────────────────────────────────────────────
    print("\n[7] Thermal analysis — steady state...")

    cooling = WaterJacketCooling(
        flow_lpm      = spec.cooling.coolant_flow_lpm,
        _inlet_temp_C = spec.cooling.coolant_temp_C,
        jacket_width  = 0.008,
        n_channels    = 12,
    )
    print(f"  {cooling.summary()}")
    print(f"  Max heat rejection : {cooling.max_heat_rejection_W()/1e3:.1f} kW")

    therm        = ThermalNetwork(motor, cooling, lib)
    loss_dict    = lb.to_dict()
    thermal_result = therm.steady_state(loss_dict)
    print(thermal_result.summary())
    print(f"\n  Insulation limit : {spec.insulation.max_winding_temp_C:.0f} °C")
    margin = spec.insulation.max_winding_temp_C - thermal_result.T_winding_C
    print(f"  Thermal margin   : {margin:.1f} °C  ({'OK' if margin > 20 else 'MARGINAL'})")

    # ──────────────────────────────────────────────────────────────────
    # STEP 8 — Drive cycle transient temperature
    # ──────────────────────────────────────────────────────────────────
    print("\n[8] Transient thermal — example drive cycle...")

    duty_cycle = [
        {"torque_Nm": motor.rated_torque * 2.5, "speed_rpm": 2000, "duration_s": 30},
        {"torque_Nm": motor.rated_torque,        "speed_rpm": 4000, "duration_s": 120},
        {"torque_Nm": motor.rated_torque * 0.3,  "speed_rpm": 6000, "duration_s": 60},
        {"torque_Nm": motor.rated_torque * 0.1,  "speed_rpm": 500,  "duration_s": 30},
        {"torque_Nm": motor.rated_torque,        "speed_rpm": 4000, "duration_s": 120},
    ]
    total_dur = sum(s["duration_s"] for s in duty_cycle)
    print(f"  Drive cycle duration: {total_dur}s  ({len(duty_cycle)} steps)")

    transient = therm.transient(duty_cycle, dt_s=5.0, loss_calculator=loss_calc)
    print(f"  Peak winding temp  : {np.max(transient['T_winding_C']):.1f} °C")
    print(f"  Final winding temp : {transient['T_winding_C'][-1]:.1f} °C")
    print(f"  Final rotor temp   : {transient['T_rotor_C'][-1]:.1f} °C")

    # ──────────────────────────────────────────────────────────────────
    # STEP 9 — Generate plots
    # ──────────────────────────────────────────────────────────────────
    print("\n[9] Generating plots...")

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Bohemien_Motor_Designer — 30 kW / 4000 rpm PMSM Design Summary", fontsize=14)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # A — Efficiency map
    ax1 = fig.add_subplot(gs[0, :2])
    S, T = np.meshgrid(eff_map["speed_rpm"], eff_map["torque_Nm"])
    eff_pct = eff_map["efficiency"] * 100
    c1 = ax1.contourf(S, T, eff_pct, levels=np.linspace(70, 98, 20), cmap="RdYlGn")
    plt.colorbar(c1, ax=ax1, label="Efficiency [%]")
    ax1.contour(S, T, eff_pct, levels=[90, 93, 95, 96, 97], colors="k", linewidths=0.7)
    ax1.axvline(motor.rated_speed, color="w", ls="--", lw=1.5, label=f"Rated {motor.rated_speed:.0f}rpm")
    ax1.axhline(motor.rated_torque, color="c", ls="--", lw=1.5, label=f"Rated {motor.rated_torque:.0f}Nm")
    ax1.plot(envelope["speed_rpm"], envelope["T_max_Nm"], "w-", lw=2, label="Max torque envelope")
    ax1.set_xlabel("Speed [rpm]"); ax1.set_ylabel("Torque [N·m]")
    ax1.set_title("Efficiency Map")
    ax1.legend(loc="upper right", fontsize=8)

    # B — Loss breakdown pie
    ax2 = fig.add_subplot(gs[0, 2])
    loss_labels = ["Copper", "Stator Fe", "Rotor Fe", "Friction", "Inverter"]
    loss_vals   = [lb.copper_loss_W, lb.stator_iron_W, lb.rotor_iron_W,
                   lb.friction_W, lb.inverter_loss_W]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    wedges, texts, auts = ax2.pie(
        [max(v,0.1) for v in loss_vals], labels=loss_labels,
        colors=colors, autopct="%1.0f%%", startangle=140,
        textprops={"fontsize": 8})
    ax2.set_title(f"Loss Breakdown @ Rated\n(Total: {lb.total_loss_W:.0f}W, η={lb.efficiency*100:.1f}%)")

    # C — Transient temperature
    ax3 = fig.add_subplot(gs[1, :2])
    t  = transient["time_s"]
    ax3.plot(t, transient["T_winding_C"], "r-", lw=2, label="Winding")
    ax3.plot(t, transient["T_rotor_C"],   "b-", lw=1.5, label="Rotor/Magnet")
    ax3.axhline(spec.insulation.max_winding_temp_C, color="r",
                ls="--", lw=1.5, label=f"Limit ({spec.insulation.max_winding_temp_C:.0f}°C)")
    ax3.axhline(spec.cooling.coolant_temp_C, color="b",
                ls=":", lw=1.5, label=f"Coolant inlet ({spec.cooling.coolant_temp_C:.0f}°C)")
    # shade duty steps
    t_start = 0
    for k, step in enumerate(duty_cycle):
        ax3.axvspan(t_start, t_start + step["duration_s"],
                    alpha=0.06, color="gray" if k % 2 else "white")
        t_start += step["duration_s"]
    ax3.set_xlabel("Time [s]"); ax3.set_ylabel("Temperature [°C]")
    ax3.set_title("Transient Thermal — Drive Cycle")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # D — Scaling law comparison
    ax4 = fig.add_subplot(gs[1, 2])
    cooling_types = list(MotorScalingLaws.ESSON_COEFFICIENT if
                         hasattr(MotorScalingLaws, "ESSON_COEFFICIENT") else
                         {"natural_convection":8,"TEFC_air":15,"forced_air":25,
                          "water-jacket":50,"oil-spray":80,"direct-water":150}.keys())
    from Bohemien_Motor_Designer.scaling.similarity import ESSON_COEFFICIENT
    c_names = list(ESSON_COEFFICIENT.keys())
    c_vols  = []
    for cn in c_names:
        e = MotorScalingLaws.size_estimate(30, 4000, cn)
        c_vols.append(e.active_volume_L)
    bars = ax4.barh(range(len(c_names)), c_vols,
                    color=["#3498db"]*len(c_names))
    bars[3].set_color("#e74c3c")  # highlight water-jacket
    ax4.set_yticks(range(len(c_names)))
    ax4.set_yticklabels([n.replace("_"," ") for n in c_names], fontsize=8)
    ax4.set_xlabel("Active Volume [L]")
    ax4.set_title("Volume vs Cooling Type\n(30kW / 4000rpm)")
    ax4.axvline(c_vols[3], color="r", ls="--", lw=1.5)
    ax4.grid(True, alpha=0.3, axis="x")

    plt.savefig("/tmp/Bohemien_Motor_Designer_full_example.png", dpi=150, bbox_inches="tight")
    print("  Saved: /tmp/Bohemien_Motor_Designer_full_example.png")

    # ──────────────────────────────────────────────────────────────────
    # STEP 10 — Save design
    # ──────────────────────────────────────────────────────────────────
    print("\n[10] Saving design to JSON...")
    save_design(motor, "/tmp/motor_30kW_design.json", spec=spec,
                results={"peak_efficiency": eff_map["peak_eff"],
                         "T_winding_rated_C": thermal_result.T_winding_C})

    print("\n" + "=" * 65)
    print("  Design workflow complete!")
    print("=" * 65)
    return motor, spec, lb, eff_map

if __name__ == "__main__":
    main()
