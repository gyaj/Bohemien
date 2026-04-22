"""
JSON serialisation / deserialisation for motor designs.

Supports save/load of complete design configurations including
DesignSpec, motor geometry, winding, and simulation results.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


def save_design(motor, path: str, spec=None, results: dict = None):
    """
    Save a complete motor design to a JSON file.

    Parameters
    ----------
    motor  : Motor instance.
    path   : Output file path (.json).
    spec   : Optional DesignSpec.
    results: Optional dict of simulation results to embed.
    """
    import dataclasses

    data = {
        "Bohemien_Motor_Designer_version": "2.0",
        "motor_class": motor.__class__.__name__,
        "geometry": {
            "poles":              motor.poles,
            "slots":              motor.slots,
            "rotor_outer_radius": motor.rotor_outer_radius,
            "rotor_inner_radius": motor.rotor_inner_radius,
            "stack_length":       motor.stack_length,
            "airgap":             motor.airgap,
            "stator_outer_radius": motor.stator_outer_radius,
            "stator_inner_radius": motor.stator_inner_radius,
        },
        "electrical": {
            "rated_speed":   motor.rated_speed,
            "rated_power":   motor.rated_power,
            "rated_voltage": motor.rated_voltage,
            "phases":        motor.phases,
            "connection":    motor.connection,
        },
        "winding": {
            "turns_per_coil":       getattr(motor, "turns_per_coil", None),
            "conductor_diameter":   getattr(motor, "conductor_diameter", None),
            "slot_fill_factor":     getattr(motor, "slot_fill_factor", None),
            "parallel_paths":       getattr(motor, "parallel_paths", 1),
            "total_series_turns":   motor.winding.total_series_turns_per_phase,
            "winding_factor":       motor.winding_factor(),
            "coil_span":            motor.winding.coil_span,
            "layers":               motor.winding.layers,
        },
    }

    # PMSM-specific
    if hasattr(motor, "magnet_material"):
        data["magnets"] = {
            "magnet_type":          motor.magnet_type,
            "magnet_material":      motor.magnet_material,
            "magnet_thickness":     motor.magnet_thickness,
            "magnet_width_fraction":motor.magnet_width_fraction,
            "back_emf_constant":    motor.back_emf_constant(),
        }
        if motor.Ld > 0:
            data["circuit_params"] = {
                "Ld": motor.Ld, "Lq": motor.Lq, "Rs": motor.Rs}

    if spec is not None:
        data["spec"] = spec.to_dict()

    if results is not None:
        data["results"] = results

    Path(path).write_text(
        json.dumps(data, indent=2, cls=NumpyEncoder), encoding="utf-8")
    print(f"Design saved to: {path}")


def load_spec(path: str):
    """Load a DesignSpec from a JSON file."""
    from Bohemien_Motor_Designer.core.specs import DesignSpec
    d = json.loads(Path(path).read_text())
    if "spec" in d:
        return DesignSpec.from_dict(d["spec"])
    return DesignSpec.from_dict(d)


def load_design(path: str):
    """
    Load a motor design from a JSON file.
    Returns (motor, spec) tuple.
    """
    from Bohemien_Motor_Designer.core.specs import DesignSpec
    from Bohemien_Motor_Designer.core.pmsm import PMSM

    d = json.loads(Path(path).read_text())
    spec = DesignSpec.from_dict(d["spec"]) if "spec" in d else None

    geom  = d.get("geometry", {})
    elec  = d.get("electrical", {})
    wind  = d.get("winding", {})
    mags  = d.get("magnets", {})
    circ  = d.get("circuit_params", {})

    cls_name = d.get("motor_class", "PMSM")
    if cls_name == "PMSM":
        motor = PMSM(
            poles=geom["poles"],
            slots=geom["slots"],
            rotor_outer_radius=geom["rotor_outer_radius"],
            rotor_inner_radius=geom["rotor_inner_radius"],
            stack_length=geom["stack_length"],
            airgap=geom["airgap"],
            rated_speed=elec["rated_speed"],
            rated_power=elec["rated_power"],
            rated_voltage=elec.get("rated_voltage", 230.0),
            phases=elec.get("phases", 3),
            magnet_material=mags.get("magnet_material", "N42SH"),
            magnet_thickness=mags.get("magnet_thickness", 0.005),
            magnet_width_fraction=mags.get("magnet_width_fraction", 0.85),
            turns_per_coil=wind.get("turns_per_coil", 8),
            conductor_diameter=wind.get("conductor_diameter", 0.0012),
            slot_fill_factor=wind.get("slot_fill_factor", 0.45),
            spec=spec,
        )
        if circ:
            motor.Ld = circ.get("Ld", 0.0)
            motor.Lq = circ.get("Lq", 0.0)
            motor.Rs = circ.get("Rs", 0.0)
    else:
        raise NotImplementedError(f"Load not implemented for {cls_name}")

    return motor, spec
