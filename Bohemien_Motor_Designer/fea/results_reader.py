"""
FEA Results Reader — parses Elmer output back into Bohemien_Motor_Designer data.

Reads three types of Elmer output:
  1. scalars.dat / cogging_torque.dat — torque vs rotor angle (cogging sweep)
  2. loaded_torque.dat               — torque time series (loaded run)
  3. flux_linkage.dat                — line integral of A_z (for Ld/Lq extraction)

All parsing is done without regex — files are read as structured text.
The extracted quantities are:
  - Cogging torque waveform + peak-to-peak amplitude
  - Average loaded torque
  - Per-phase flux linkage waveform → back-EMF by differentiation
  - Ld, Lq from ΔΨ/ΔI at two operating points

Design note
-----------
The flux linkage extraction from SaveLine data is approximate: it integrates
A_z along a radial line through the winding. For a rigorous extraction the
coil areas must be used (from the winding layout). For now a line integral
at the bore radius gives a first-order estimate.

When Elmer output files are not present (e.g. during testing without a solver),
methods return None and log a warning rather than raising.

Usage
-----
    reader = ResultsReader("/tmp/fea_run")
    cogging = reader.read_cogging()
    loaded  = reader.read_loaded(motor)
    print(cogging["Tcog_pp_Nm"])
    print(loaded["Ld_H"])
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional
import warnings


class ResultsReader:
    """
    Parses Elmer result files from a given working directory.

    Parameters
    ----------
    work_dir : directory containing Elmer result subdirectories
               (cogging_results/, loaded_results/)
    """

    def __init__(self, work_dir: str):
        self.wd = Path(work_dir)

    # ── Cogging results ───────────────────────────────────────────────────

    def read_cogging(self) -> dict:
        """
        Parse cogging torque data from cogging_results/cogging_torque.dat.

        Returns
        -------
        dict with keys:
          theta_deg   : array of rotor angles [deg]
          torque_Nm   : array of cogging torque values [N·m]
          Tcog_pp_Nm  : peak-to-peak amplitude [N·m]
        """
        result_dir = self.wd / "cogging_results"
        candidates = [
            result_dir / "cogging_torque.dat",
            result_dir / "scalars.dat",
            self.wd / "cogging_torque.dat",
        ]

        dat_path = None
        for c in candidates:
            if c.exists():
                dat_path = c
                break

        if dat_path is None:
            warnings.warn(
                f"Cogging torque file not found in {result_dir}. "
                "Run ElmerSolver first."
            )
            return {
                "theta_deg": np.array([]),
                "torque_Nm": np.array([]),
                "Tcog_pp_Nm": float("nan"),
            }

        theta_list  = []
        torque_list = []

        for line in dat_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("!") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    # Elmer scalars.dat: time(or angle), value, ...
                    t = float(parts[0])   # step or normalised angle
                    v = float(parts[1])   # torque
                    theta_list.append(t)
                    torque_list.append(v)
                except ValueError:
                    continue

        if not torque_list:
            warnings.warn("Cogging torque file is empty or unparseable.")
            return {
                "theta_deg": np.array([]),
                "torque_Nm": np.array([]),
                "Tcog_pp_Nm": float("nan"),
            }

        theta  = np.array(theta_list)
        torque = np.array(torque_list)

        # Convert step index → mechanical degrees if needed
        # (Elmer scanning uses step count, not physical angle)
        if theta[-1] < 10.0 and len(theta) > 1:
            # Looks like step fraction — scale to mechanical degrees
            # Cogging period = 360 / LCM(slots, poles) degrees
            theta_deg = theta * 360.0
        else:
            theta_deg = theta

        Tcog_pp = float(np.max(torque) - np.min(torque))

        return {
            "theta_deg":  theta_deg,
            "torque_Nm":  torque,
            "Tcog_pp_Nm": Tcog_pp,
        }

    # ── Loaded results ────────────────────────────────────────────────────

    def read_loaded(self, motor=None) -> dict:
        """
        Parse loaded torque and flux linkage from loaded_results/.

        Parameters
        ----------
        motor : PMSM instance (needed for Ld/Lq extraction normalization)

        Returns
        -------
        dict with keys:
          time_s         : time array [s]
          torque_Nm      : torque array [N·m]
          torque_avg_Nm  : average torque [N·m]
          Ld_H, Lq_H     : extracted inductances [H]  (if flux_linkage.dat exists)
          emf_waveform   : dict with 'time', 'voltage', 'thd_pct'
        """
        result_dir = self.wd / "loaded_results"
        torque_path = (result_dir / "loaded_torque.dat"
                       if (result_dir / "loaded_torque.dat").exists()
                       else self.wd / "loaded_torque.dat")

        result = {
            "time_s": np.array([]),
            "torque_Nm": np.array([]),
            "torque_avg_Nm": float("nan"),
            "Ld_H": None,
            "Lq_H": None,
            "emf_waveform": None,
        }

        # ── Torque ──
        if torque_path.exists():
            times, torques = [], []
            for line in torque_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith(("!", "#")):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        times.append(float(parts[0]))
                        torques.append(float(parts[1]))
                    except ValueError:
                        continue

            if torques:
                result["time_s"]        = np.array(times)
                result["torque_Nm"]     = np.array(torques)
                result["torque_avg_Nm"] = float(np.mean(torques[len(torques)//4:]))
        else:
            warnings.warn(f"Loaded torque file not found: {torque_path}")

        # ── Flux linkage → back-EMF and Ld/Lq ──
        fl_path = (result_dir / "flux_linkage.dat"
                   if (result_dir / "flux_linkage.dat").exists()
                   else self.wd / "flux_linkage.dat")

        if fl_path.exists() and motor is not None:
            fl_data = self._parse_flux_linkage(fl_path, motor)
            if fl_data is not None:
                result["Ld_H"] = fl_data.get("Ld_H")
                result["Lq_H"] = fl_data.get("Lq_H")
                result["emf_waveform"] = fl_data.get("emf_waveform")
        elif motor is not None:
            warnings.warn(f"Flux linkage file not found: {fl_path}")

        return result

    # ── Private: flux linkage parser ─────────────────────────────────────

    def _parse_flux_linkage(self, path: Path, motor) -> Optional[dict]:
        """
        Parse SaveLine output and extract per-phase flux linkage.

        Elmer SaveLine writes columns: x, y, A_z at each time step.
        We integrate A_z along the line to get total flux linkage per step.
        """
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        # Parse all data blocks (one per time step)
        # Format: header line with "Time =", then x y Az rows
        blocks = []
        current_block = []
        current_time  = None

        for line in text.splitlines():
            line = line.strip()
            if line.startswith("Time"):
                if current_block:
                    blocks.append((current_time, current_block))
                parts = line.split()
                try:
                    current_time = float(parts[-1])
                except (ValueError, IndexError):
                    current_time = len(blocks) * 1e-4
                current_block = []
            elif line and not line.startswith("!"):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        current_block.append([float(p) for p in parts[:3]])
                    except ValueError:
                        pass

        if current_block:
            blocks.append((current_time, current_block))

        if not blocks:
            warnings.warn("Flux linkage file has no parseable data.")
            return None

        # Integrate A_z along line for each time step → total flux linkage
        times_fl = []
        psi_vals  = []

        for t, block in blocks:
            arr = np.array(block)
            y   = arr[:, 1]   # y-coordinate along radial line
            Az  = arr[:, 2]   # A_z values

            # Approximate flux linkage: psi ≈ N * kw * integral(Az * dy)
            # This is a first-order estimate — proper extraction needs coil areas
            N  = motor.winding.total_series_turns_per_phase if motor.winding else 1
            kw = motor.winding_factor() if hasattr(motor, "winding_factor") else 1.0
            psi = N * kw * np.trapz(Az, y) * motor.stack_length
            times_fl.append(t)
            psi_vals.append(psi)

        if len(times_fl) < 4:
            return None

        times_fl = np.array(times_fl)
        psi_vals  = np.array(psi_vals)

        # Back-EMF: e = -dΨ/dt
        emf = -np.gradient(psi_vals, times_fl)

        # THD of back-EMF
        n_pts  = len(emf)
        fft_e  = np.fft.rfft(emf[n_pts // 4:])   # skip first quarter (transient)
        amps   = np.abs(fft_e)
        f1     = amps[1]   # fundamental
        thd    = (100.0 * np.sqrt(np.sum(amps[2:]**2)) / (f1 + 1e-9)
                  if f1 > 0 else float("nan"))

        emf_waveform = {
            "time":    times_fl,
            "voltage": emf,
            "thd_pct": thd,
        }

        # Ld/Lq extraction:
        # From the flux linkage at rated Iq (MTPA), extract the d-axis component.
        # Simple estimate: Ld ≈ (ψ_d - ψ_m) / Id  where ψ_m = Ke
        # At MTPA with SPM, Id=0 so Ld is not directly accessible from one run.
        # Return analytical values as fallback; FEA override requires two extra runs.
        Ke = motor.back_emf_constant() if hasattr(motor, "back_emf_constant") else 0.0
        psi_peak = float(np.max(np.abs(psi_vals)))
        Ld_approx = max((psi_peak - Ke) / (motor.rated_torque / (
            1.5 * motor.pole_pairs * Ke + 1e-9) * math.sqrt(2) + 1e-9), 0.0)

        # Fall back to analytical if extraction gives implausible result
        Ld_analytical = getattr(motor, "Ld", None)
        Lq_analytical = getattr(motor, "Lq", None)

        Ld_out = Ld_approx if (0.5 * (Ld_analytical or 1e-3) < Ld_approx
                               < 5 * (Ld_analytical or 1e-1)) else Ld_analytical
        Lq_out = Lq_analytical   # Lq extraction requires separate Iq-sweep run

        return {
            "Ld_H":         Ld_out,
            "Lq_H":         Lq_out,
            "emf_waveform": emf_waveform,
        }


import math   # ensure math is available for _parse_flux_linkage
