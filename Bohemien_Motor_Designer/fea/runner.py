"""
FEA Orchestrator — chains GMSH → ElmerGrid → ElmerSolver.

Manages the working directory, log tailing, process control, and
result path tracking. Designed to be called from both CLI and GUI.

Usage
-----
    from Bohemien_Motor_Designer.fea.runner import FEARunner, FEARuntimeError

    runner = FEARunner(motor, work_dir="/tmp/fea_run")
    runner.prepare()          # writes .geo, BH file, SIF files

    # Run cogging sweep (Id=Iq=0, rotor position sweep)
    torque_data = runner.run_cogging(progress_cb=print)

    # Run loaded transient (rated MTPA, one electrical period)
    results = runner.run_loaded(progress_cb=print)

    print(f"Cogging Tpp  = {results['Tcog_pp_Nm']:.3f} Nm")
    print(f"Ld (FEA)     = {results['Ld_H']*1000:.3f} mH")

Progress callback
-----------------
    progress_cb(message: str, fraction: float)
      message  — human-readable status line
      fraction — 0.0 to 1.0 completion estimate
"""
from __future__ import annotations
import subprocess
import threading
import time
import os
import shutil
from pathlib import Path
from typing import Callable, Optional
import numpy as np

from Bohemien_Motor_Designer.fea.index_registry import IndexRegistry
from Bohemien_Motor_Designer.fea.gmsh_exporter import GMSHExporter
from Bohemien_Motor_Designer.fea.bh_writer import write_bh_files
from Bohemien_Motor_Designer.fea.sif_generator import SIFGenerator
from Bohemien_Motor_Designer.fea.results_reader import ResultsReader
from Bohemien_Motor_Designer.materials.library import MaterialLibrary


class FEARuntimeError(RuntimeError):
    """Raised when GMSH, ElmerGrid, or ElmerSolver exits non-zero."""
    def __init__(self, tool: str, returncode: int, last_lines: str):
        self.tool        = tool
        self.returncode  = returncode
        self.last_lines  = last_lines
        super().__init__(
            f"{tool} failed (rc={returncode}).\nLast output:\n{last_lines}"
        )


class FEARunner:
    """
    Orchestrates the GMSH → ElmerGrid → ElmerSolver pipeline.

    Parameters
    ----------
    motor    : PMSM instance
    work_dir : working directory for all FEA files
    n_cog    : rotor positions for cogging sweep (default 31)
    n_steps  : time steps per electrical period for loaded run (default 60)
    gmsh_exe : path to gmsh executable (default: 'gmsh', assumes on PATH)
    elmer_grid_exe : path to ElmerGrid (default: 'ElmerGrid')
    elmer_solver_exe : path to ElmerSolver (default: 'ElmerSolver')
    """

    def __init__(
        self,
        motor,
        work_dir:  str = None,
        n_cog:     int = 31,
        n_steps:   int = 60,
        gmsh_exe:          str = "gmsh",
        elmer_grid_exe:    str = "ElmerGrid",
        elmer_solver_exe:  str = "ElmerSolver",
    ):
        self.m    = motor
        self.wd   = Path(work_dir) if work_dir else Path.home() / "Bohemien_Motor_Designer_fea"
        self.wd.mkdir(parents=True, exist_ok=True)

        self.n_cog   = n_cog
        self.n_steps = n_steps

        self.gmsh_exe   = gmsh_exe
        self.elmer_grid = elmer_grid_exe
        self.elmer_solver = elmer_solver_exe

        self.reg = IndexRegistry(poles=motor.poles, slots=motor.slots)

        self._prepared = False
        self._geo_path: Optional[Path] = None
        self._bh_path:  Optional[Path] = None

    # ── Prepare ───────────────────────────────────────────────────────────

    def prepare(self, progress_cb: Callable = None) -> None:
        """
        Write all input files (geo, BH, SIFs) to the working directory.
        Does NOT run any external tools.
        """
        self._cb(progress_cb, "Writing GMSH geometry...", 0.0)

        # Geometry
        geo_exp = GMSHExporter(self.m, self.reg)
        self._geo_path = geo_exp.write(str(self.wd / "PMSM.geo"))
        self._cb(progress_cb, f"Written: {self._geo_path.name}", 0.15)

        # BH file
        lam = getattr(self.m.stator, "lamination", "M270-35A") if self.m.stator else "M270-35A"
        try:
            lib = MaterialLibrary()
            self._bh_path = write_bh_files(lam, str(self.wd), lib)
            self._cb(progress_cb, f"Written: {self._bh_path.name}", 0.25)
        except Exception as e:
            self._cb(progress_cb, f"WARNING: BH file failed ({e}) — using linear material", 0.25)

        # SIF files
        bh = str(self._bh_path) if self._bh_path else None
        gen = SIFGenerator(
            self.m, self.reg,
            work_dir=str(self.wd),
            bh_file=bh,
            n_positions_cogging=self.n_cog,
            n_steps_loaded=self.n_steps,
        )
        p_cog = gen.write_cogging_sif()
        p_lod = gen.write_loaded_sif()
        self._cb(progress_cb, f"Written: {p_cog.name} + {p_lod.name}", 0.35)

        self._prepared = True

    # ── Mesh generation ───────────────────────────────────────────────────

    def generate_mesh(self, progress_cb: Callable = None) -> Path:
        """
        Run GMSH on PMSM.geo → PMSM.msh (second-order triangles).
        Returns path to .msh file.
        """
        if not self._prepared:
            self.prepare(progress_cb)

        self._cb(progress_cb, "Running GMSH mesher...", 0.40)
        geo  = str(self._geo_path)
        msh  = str(self.wd / "PMSM.msh")

        cmd = [self.gmsh_exe, geo, "-2", "-order", "2", "-o", msh]
        self._run_tool("GMSH", cmd, self.wd, progress_cb, frac_start=0.40, frac_end=0.60)

        # ElmerGrid converts GMSH mesh → Elmer format
        self._cb(progress_cb, "Running ElmerGrid...", 0.60)
        cmd_eg = [self.elmer_grid, "14", "2", msh, "-autoclean"]
        self._run_tool("ElmerGrid", cmd_eg, self.wd, progress_cb, frac_start=0.60, frac_end=0.70)

        return Path(msh)

    # ── Cogging run ───────────────────────────────────────────────────────

    def run_cogging(self, progress_cb: Callable = None) -> dict:
        """
        Full cogging sweep pipeline: prepare → mesh → ElmerSolver → read results.

        Returns
        -------
        dict with keys:
          theta_deg      : rotor angle array [deg]
          torque_Nm      : cogging torque array [N·m]
          Tcog_pp_Nm     : peak-to-peak cogging [N·m]
          Tcog_pp_pct    : as % of rated torque
        """
        self.generate_mesh(progress_cb)

        self._cb(progress_cb, "Running ElmerSolver (cogging)...", 0.70)
        sif = str(self.wd / "case_cogging.sif")
        self._run_tool("ElmerSolver (cogging)", [self.elmer_solver, sif],
                       self.wd, progress_cb, frac_start=0.70, frac_end=0.95)

        self._cb(progress_cb, "Reading cogging results...", 0.95)
        reader = ResultsReader(self.wd)
        data   = reader.read_cogging()

        T_rated = self.m.rated_torque
        data["Tcog_pp_pct"] = 100.0 * data["Tcog_pp_Nm"] / (T_rated + 1e-9)

        self._cb(progress_cb,
                 f"Cogging Tpp={data['Tcog_pp_Nm']:.3f}Nm ({data['Tcog_pp_pct']:.1f}%)", 1.0)
        return data

    # ── Loaded run ────────────────────────────────────────────────────────

    def run_loaded(self, progress_cb: Callable = None) -> dict:
        """
        Full loaded pipeline: prepare → mesh → ElmerSolver → read results.
        Extracts Ld, Lq, back-EMF waveform, average torque.

        Results are written back to motor.Ld and motor.Lq.

        Returns
        -------
        dict with keys:
          torque_avg_Nm    : average electromagnetic torque [N·m]
          Ld_H, Lq_H      : extracted inductances [H]
          emf_waveform     : dict with 'time', 'voltage', 'thd_pct'
        """
        self.generate_mesh(progress_cb)

        self._cb(progress_cb, "Running ElmerSolver (loaded)...", 0.70)
        sif = str(self.wd / "case_loaded.sif")
        self._run_tool("ElmerSolver (loaded)", [self.elmer_solver, sif],
                       self.wd, progress_cb, frac_start=0.70, frac_end=0.95)

        self._cb(progress_cb, "Reading loaded results...", 0.95)
        reader = ResultsReader(self.wd)
        data   = reader.read_loaded(self.m)

        # Write back to motor object
        if "Ld_H" in data:
            self.m.Ld = data["Ld_H"]
        if "Lq_H" in data:
            self.m.Lq = data["Lq_H"]

        self._cb(progress_cb,
                 f"T_avg={data.get('torque_avg_Nm', 0):.1f}Nm  "
                 f"Ld={data.get('Ld_H', 0)*1000:.2f}mH  "
                 f"Lq={data.get('Lq_H', 0)*1000:.2f}mH", 1.0)
        return data

    # ── Helpers ───────────────────────────────────────────────────────────

    def _run_tool(self, name: str, cmd: list, cwd: Path,
                  progress_cb: Callable, frac_start=0.0, frac_end=1.0):
        """
        Run external tool in subprocess, tailing its log line-by-line.
        Raises FEARuntimeError if return code != 0.
        """
        last_lines = []

        try:
            proc = subprocess.Popen(
                cmd, cwd=str(cwd),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace",
            )
        except FileNotFoundError:
            raise FEARuntimeError(
                name, -1,
                f"Executable not found: {cmd[0]}\n"
                f"Install {name} and ensure it is on your PATH."
            )

        # Read output line by line, forward to callback
        for line in proc.stdout:
            line = line.rstrip()
            last_lines.append(line)
            if len(last_lines) > 40:
                last_lines.pop(0)
            if progress_cb:
                # Estimate fraction from output patterns
                frac = frac_start
                if "Mesh" in line or "mesh" in line:
                    frac = frac_start + (frac_end - frac_start) * 0.3
                elif "Iteration" in line or "iteration" in line:
                    frac = frac_start + (frac_end - frac_start) * 0.7
                elif "SOLVER TOTAL" in line or "ElmerSolver finished" in line:
                    frac = frac_end
                self._cb(progress_cb, f"{name}: {line[:80]}", frac)

        proc.wait()
        if proc.returncode != 0:
            raise FEARuntimeError(name, proc.returncode, "\n".join(last_lines[-20:]))

    @staticmethod
    def _cb(cb: Callable, msg: str, frac: float):
        if cb is not None:
            try:
                cb(msg, frac)
            except TypeError:
                cb(msg)   # fallback: single-arg callback (e.g. print)

    # ── Dependency checks ─────────────────────────────────────────────────

    def check_dependencies(self) -> dict[str, bool]:
        """
        Check that external tools are available.
        Returns dict {tool_name: available}.
        """
        results = {}
        for name, exe in [
            ("GMSH",       self.gmsh_exe),
            ("ElmerGrid",  self.elmer_grid),
            ("ElmerSolver", self.elmer_solver),
        ]:
            results[name] = shutil.which(exe) is not None
        return results

    def dependency_report(self) -> str:
        deps = self.check_dependencies()
        lines = ["FEA Dependency Check:"]
        for tool, ok in deps.items():
            status = "✓ found" if ok else "✗ NOT FOUND"
            lines.append(f"  {tool:15s}: {status}")
        if not all(deps.values()):
            lines.append("\nInstallation:")
            lines.append("  GMSH:  https://gmsh.info/   (or: apt install gmsh)")
            lines.append("  Elmer: https://elmerfem.org/ (or: apt install elmer)")
        return "\n".join(lines)
