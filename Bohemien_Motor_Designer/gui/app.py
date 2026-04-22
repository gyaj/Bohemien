"""
Bohemien_Motor_Designer GUI — tkinter + matplotlib desktop application.
Run: python -m Bohemien_Motor_Designer.gui.app
"""
# Only stdlib at module level — lets PyInstaller compile this file
# even in environments without tkinter/matplotlib (e.g. Linux build servers).
import sys
import os
import threading
import traceback


def main():
    """Launch the Bohemien_Motor_Designer GUI. All heavy imports happen here."""
    # Make Bohemien_Motor_Designer importable regardless of working directory
    _gui  = os.path.dirname(os.path.abspath(__file__))   # .../gui/
    _pkg  = os.path.dirname(_gui)                         # .../Bohemien_Motor_Designer/
    _root = os.path.dirname(_pkg)                         # parent of package
    for _d in (_root, _pkg):
        if _d not in sys.path:
            sys.path.insert(0, _d)

    # ── GUI / scientific imports ──────────────────────────────────────────────
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext
    import numpy as np
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    import matplotlib.gridspec as gridspec

    # ── Colour palette ────────────────────────────────────────────────────────────
    BG       = "#1a1d2e"
    PANEL    = "#242740"
    ACCENT   = "#4f8ef7"
    ACCENT2  = "#f7c04f"
    SUCCESS  = "#4fcf82"
    WARNING  = "#f7804f"
    TEXT     = "#e8eaf6"
    TEXT_DIM = "#7c82a8"
    ENTRY_BG = "#2e3252"
    BORDER   = "#3a3f6b"

    FONT_H1  = ("Segoe UI", 13, "bold")
    FONT_H2  = ("Segoe UI", 10, "bold")
    FONT_BODY= ("Segoe UI", 9)
    FONT_MONO= ("Consolas", 9)


    # ─────────────────────────────────────────────────────────────────────────────
    class LabeledEntry(tk.Frame):
        """Label + Entry with unit annotation."""
        def __init__(self, parent, label, default, unit="", width=10, **kw):
            super().__init__(parent, bg=PANEL, **kw)
            tk.Label(self, text=label, bg=PANEL, fg=TEXT_DIM,
                     font=FONT_BODY, anchor="w", width=22).pack(side="left")
            self.var = tk.StringVar(value=str(default))
            tk.Entry(self, textvariable=self.var, width=width,
                     bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
                     relief="flat", font=FONT_BODY,
                     highlightthickness=1, highlightbackground=BORDER,
                     highlightcolor=ACCENT).pack(side="left", padx=(0, 4))
            if unit:
                tk.Label(self, text=unit, bg=PANEL, fg=TEXT_DIM,
                         font=FONT_BODY).pack(side="left")

        def get(self):  return self.var.get()
        def set(self, v): self.var.set(str(v))
        def delete(self, *args): self.var.set("")          # satisfy tk.Entry-style callers
        def insert(self, idx, v): self.var.set(str(v))     # satisfy tk.Entry-style callers


    class LabeledCombo(tk.Frame):
        """Label + Combobox."""
        def __init__(self, parent, label, values, default, **kw):
            super().__init__(parent, bg=PANEL, **kw)
            tk.Label(self, text=label, bg=PANEL, fg=TEXT_DIM,
                     font=FONT_BODY, anchor="w", width=22).pack(side="left")
            self.var = tk.StringVar(value=default)
            cb = ttk.Combobox(self, textvariable=self.var, values=values,
                              width=16, state="readonly", font=FONT_BODY)
            cb.pack(side="left")

        def get(self): return self.var.get()


    # ─────────────────────────────────────────────────────────────────────────────
    class MotorDesignApp(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Bohemien_Motor_Designer — Electric Motor Design Suite")
            self.geometry("1280x820")
            self.minsize(1100, 700)
            self.configure(bg=BG)
            self._motor = None
            self._results = {}
            self._build_ui()

        # ── UI construction ────────────────────────────────────────────────────

        def _build_ui(self):
            self._style()
            self._build_header()
            main = tk.Frame(self, bg=BG)
            main.pack(fill="both", expand=True, padx=12, pady=(0, 8))
            main.columnconfigure(1, weight=1)
            main.rowconfigure(0, weight=1)
            self._build_inputs(main)
            self._build_outputs(main)
            self._build_status()

        def _style(self):
            s = ttk.Style(self)
            s.theme_use("clam")
            s.configure("TCombobox", fieldbackground=ENTRY_BG, background=ENTRY_BG,
                        foreground=TEXT, selectbackground=ACCENT,
                        arrowcolor=TEXT_DIM, borderwidth=0)
            s.configure("TNotebook", background=BG, borderwidth=0, tabmargins=0)
            s.configure("TNotebook.Tab", background=PANEL, foreground=TEXT_DIM,
                        padding=[12, 5], font=FONT_BODY)
            s.map("TNotebook.Tab",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "#ffffff")])
            s.configure("TProgressbar", troughcolor=ENTRY_BG, background=ACCENT,
                        thickness=4)

        def _build_header(self):
            hdr = tk.Frame(self, bg=PANEL, height=52)
            hdr.pack(fill="x")
            hdr.pack_propagate(False)
            tk.Label(hdr, text="⚙  Bohemien_Motor_Designer", font=("Segoe UI", 15, "bold"),
                     bg=PANEL, fg=ACCENT).pack(side="left", padx=18, pady=10)
            tk.Label(hdr, text="Electric Motor Design Suite  v2.0",
                     font=FONT_BODY, bg=PANEL, fg=TEXT_DIM).pack(side="left")
            tk.Label(hdr, text="1 kW – 1 MW  ·  0–1500 V DC bus  ·  Thermal + Drive + FEA-ready",
                     font=FONT_BODY, bg=PANEL, fg=TEXT_DIM).pack(side="right", padx=18)

        def _build_inputs(self, parent):
            # Scrollable left panel — all inputs fit regardless of window height
            outer = tk.Frame(parent, bg=PANEL, width=320)
            outer.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
            outer.pack_propagate(False)
            outer.grid_propagate(False)

            canvas = tk.Canvas(outer, bg=PANEL, highlightthickness=0, width=300)
            vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=vsb.set)
            vsb.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)

            left = tk.Frame(canvas, bg=PANEL)
            win_id = canvas.create_window((0, 0), window=left, anchor="nw")

            def _on_frame_configure(e):
                canvas.configure(scrollregion=canvas.bbox("all"))
            def _on_canvas_configure(e):
                canvas.itemconfig(win_id, width=e.width)
            def _on_mousewheel(e):
                canvas.yview_scroll(int(-1*(e.delta/120)), "units")

            left.bind("<Configure>", _on_frame_configure)
            canvas.bind("<Configure>", _on_canvas_configure)
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

            tk.Label(left, text="DESIGN REQUIREMENTS", font=FONT_H2,
                     bg=PANEL, fg=ACCENT).pack(anchor="w", padx=14, pady=(14, 4))
            ttk.Separator(left).pack(fill="x", padx=14, pady=2)

            def section(title):
                tk.Label(left, text=title, font=("Segoe UI", 8, "bold"),
                         bg=PANEL, fg=TEXT_DIM).pack(anchor="w", padx=14, pady=(10, 2))

            # Power & Speed
            section("POWER & SPEED")
            self.e_power   = LabeledEntry(left, "Output power",   30,   "kW")
            self.e_speed   = LabeledEntry(left, "Rated speed",  4000,  "rpm")
            self.e_speed_max=LabeledEntry(left, "Max speed",   12000,  "rpm")
            for w in (self.e_power, self.e_speed, self.e_speed_max):
                w.pack(anchor="w", padx=14, pady=1)

            # Topology
            section("TOPOLOGY")
            self.e_poles      = LabeledEntry(left, "Poles",           8,   "p")
            self.e_slots      = LabeledEntry(left, "Slots",          48,   "s")
            self.e_stack      = LabeledEntry(left, "Stack length",  130,   "mm")
            self.e_airgap     = LabeledEntry(left, "Air gap",        1.0,  "mm")
            self.e_slot_depth = LabeledEntry(left, "Slot depth",     0,    "mm  (0=auto)")
            self.e_slot_width = LabeledEntry(left, "Slot width",     0,    "mm  (0=auto)")
            for w in (self.e_poles, self.e_slots, self.e_stack, self.e_airgap,
                      self.e_slot_depth, self.e_slot_width):
                w.pack(anchor="w", padx=14, pady=1)

            # Magnet
            section("MAGNETS & WINDING")
            self.c_magnet  = LabeledCombo(left, "Magnet grade",
                                ["N42SH","N42","N48","N35","N52","SmCo26","Ferrite-Y30"],
                                "N42SH")
            self.e_tmag    = LabeledEntry(left, "Magnet thickness", 6, "mm")
            self.e_alpha   = LabeledEntry(left, "Magnet arc fraction", 0.83, "")
            self.e_turns   = LabeledEntry(left, "Turns per coil",    0, "t")
            self.e_paths   = LabeledEntry(left, "Parallel paths",     1, "")
            self.c_laminate= LabeledCombo(left, "Lamination",
                                ["M270-35A","M19","M400-50A","M800-65A","Arnon5"],
                                "M270-35A")
            for w in (self.c_magnet, self.e_tmag, self.e_alpha,
                      self.e_turns, self.e_paths, self.c_laminate):
                w.pack(anchor="w", padx=14, pady=1)

            # Drive
            section("DRIVE")
            self.e_vbus    = LabeledEntry(left, "DC bus voltage",    400, "V")
            self.c_device  = LabeledCombo(left, "Switch device",
                                ["SiC-MOSFET","Si-IGBT","GaN"], "SiC-MOSFET")
            self.e_fsw     = LabeledEntry(left, "Switching freq",     20, "kHz")
            for w in (self.e_vbus, self.c_device, self.e_fsw):
                w.pack(anchor="w", padx=14, pady=1)

            # Thermal
            section("THERMAL")
            self.c_cooling = LabeledCombo(left, "Cooling type",
                                ["water-jacket","air","oil-spray","direct-water"],
                                "water-jacket")
            self.e_tcoolant= LabeledEntry(left, "Coolant temp",      65, "°C")
            self.e_flow    = LabeledEntry(left, "Flow rate",         12, "lpm")
            self.c_ins     = LabeledCombo(left, "Insulation class",
                                ["H","F","B","C"], "H")
            for w in (self.c_cooling, self.e_tcoolant, self.e_flow, self.c_ins):
                w.pack(anchor="w", padx=14, pady=1)

            # Run button
            tk.Frame(left, bg=PANEL).pack(fill="y", expand=True)
            self.btn_run = tk.Button(
                left, text="▶  RUN DESIGN",
                font=("Segoe UI", 11, "bold"),
                bg=ACCENT, fg="#ffffff", activebackground="#3a7adf",
                relief="flat", cursor="hand2", pady=10,
                command=self._run_async)
            self.btn_run.pack(fill="x", padx=14, pady=(8, 14))

        def _build_outputs(self, parent):
            right = tk.Frame(parent, bg=BG)
            right.grid(row=0, column=1, sticky="nsew")

            nb = ttk.Notebook(right)
            nb.pack(fill="both", expand=True)

            # Tab 1 — Summary
            t1 = tk.Frame(nb, bg=PANEL)
            nb.add(t1, text="  Summary  ")
            self.txt_summary = scrolledtext.ScrolledText(
                t1, bg=PANEL, fg=TEXT, font=FONT_MONO,
                relief="flat", wrap="word", state="disabled",
                insertbackground=TEXT)
            self.txt_summary.pack(fill="both", expand=True, padx=8, pady=8)

            # Tab 2 — Loss Budget
            t2 = tk.Frame(nb, bg=PANEL)
            nb.add(t2, text="  Losses  ")
            self.txt_losses = scrolledtext.ScrolledText(
                t2, bg=PANEL, fg=TEXT, font=FONT_MONO,
                relief="flat", wrap="word", state="disabled")
            self.txt_losses.pack(fill="both", expand=True, padx=8, pady=8)

            # Tab 3 — Thermal
            t3 = tk.Frame(nb, bg=PANEL)
            nb.add(t3, text="  Thermal  ")
            self.txt_thermal = scrolledtext.ScrolledText(
                t3, bg=PANEL, fg=TEXT, font=FONT_MONO,
                relief="flat", wrap="word", state="disabled")
            self.txt_thermal.pack(fill="both", expand=True, padx=8, pady=8)

            # Tab 4 — DRC
            t4 = tk.Frame(nb, bg=PANEL)
            nb.add(t4, text="  DRC  ")
            self.txt_drc = scrolledtext.ScrolledText(
                t4, bg=PANEL, fg=TEXT, font=FONT_MONO,
                relief="flat", wrap="word", state="disabled")
            self.txt_drc.pack(fill="both", expand=True, padx=8, pady=8)

            # Tab 5 — Efficiency Map
            t5 = tk.Frame(nb, bg=BG)
            nb.add(t5, text="  Efficiency Map  ")
            self._fig = Figure(figsize=(7, 4.5), facecolor=BG)
            self._canvas = FigureCanvasTkAgg(self._fig, master=t5)
            self._canvas.get_tk_widget().pack(fill="both", expand=True)
            toolbar = NavigationToolbar2Tk(self._canvas, t5,
                                            pack_toolbar=False)
            toolbar.configure(bg=BG)
            toolbar.pack(side="bottom", fill="x")
            self._canvas.draw()

            # Tab 6 — Scaling
            t6 = tk.Frame(nb, bg=PANEL)
            nb.add(t6, text="  Scaling  ")
            self.txt_scaling = scrolledtext.ScrolledText(
                t6, bg=PANEL, fg=TEXT, font=FONT_MONO,
                relief="flat", wrap="word", state="disabled")
            self.txt_scaling.pack(fill="both", expand=True, padx=8, pady=8)

            # Tab 7 — Manufacturing Report
            t7 = tk.Frame(nb, bg=PANEL)
            nb.add(t7, text="  Mfg Report  ")
            self._build_mfg_tab(t7)

            # Tab 8 — FEA
            t8 = tk.Frame(nb, bg=PANEL)
            nb.add(t8, text="  FEA  ")
            self._build_fea_tab(t8)

            # Tab 9 — Mesh Visualisation
            t9 = tk.Frame(nb, bg=PANEL)
            nb.add(t9, text="  Mesh Viz  ")
            self._build_mesh_viz_tab(t9)

            # Tab 10 — DXF Export
            t10 = tk.Frame(nb, bg=PANEL)
            nb.add(t10, text="  DXF Export  ")
            self._build_dxf_tab(t10)

            # Tab 11 — 3D FEM
            t11 = tk.Frame(nb, bg=PANEL)
            nb.add(t11, text="  3D FEM  ")
            self._build_3dfem_tab(t11)

            self._notebook = nb

        def _build_mfg_tab(self, parent):
            """Manufacturing report tab — text view + Save button."""
            import tkinter.scrolledtext as scrolledtext
            top = tk.Frame(parent, bg=PANEL)
            top.pack(fill="x", padx=10, pady=(6, 2))

            tk.Label(top, text="Manufacturing Specification",
                     bg=PANEL, fg=TEXT, font=FONT_BODY).pack(side="left")

            tk.Button(top, text="💾  Save .txt",
                      bg=PANEL, fg=TEXT, activebackground=ACCENT,
                      relief="flat", cursor="hand2",
                      command=self._mfg_save).pack(side="right", padx=4)

            tk.Button(top, text="⟳  Refresh",
                      bg=PANEL, fg=TEXT, activebackground=ACCENT,
                      relief="flat", cursor="hand2",
                      command=self._mfg_refresh).pack(side="right", padx=4)

            self.txt_mfg = scrolledtext.ScrolledText(
                parent, bg="#0d0d1a", fg=TEXT, font=("Courier New", 8),
                relief="flat", wrap="none", state="disabled")
            self.txt_mfg.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        def _mfg_refresh(self):
            """Generate and display manufacturing report for current design."""
            if not self._results or "motor" not in self._results:
                self._set_text(self.txt_mfg,
                               "Run a design first (▶ RUN DESIGN), then click Refresh.")
                return
            try:
                from Bohemien_Motor_Designer.core.manufacturing_report import ManufacturingReport
                rpt = ManufacturingReport(self._results["motor"],
                                          spec=self._results.get("spec"))
                self._set_text(self.txt_mfg, rpt.text())
                self._mfg_report = rpt
            except Exception as e:
                import traceback
                self._set_text(self.txt_mfg,
                               f"Error generating report:\n{traceback.format_exc()}")

        def _mfg_save(self):
            """Save manufacturing report to a text file."""
            if not hasattr(self, "_mfg_report") or self._mfg_report is None:
                self._mfg_refresh()
            if not hasattr(self, "_mfg_report") or self._mfg_report is None:
                return
            from tkinter import filedialog
            path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialfile="PMSM_manufacturing_spec.txt",
                title="Save Manufacturing Report")
            if path:
                self._mfg_report.save(path)
                self._set_status(f"Report saved: {path}", SUCCESS)

        def _build_fea_tab(self, parent):
            """Build the FEA tab: controls, log, results panels."""
            # ── Top: controls ──
            ctrl = tk.Frame(parent, bg=PANEL)
            ctrl.pack(fill="x", padx=10, pady=(8, 4))

            tk.Label(ctrl, text="Work dir:", bg=PANEL, fg=TEXT_DIM,
                     font=FONT_BODY).grid(row=0, column=0, sticky="w", padx=(0, 4))
            self._fea_workdir_var = tk.StringVar(value=str(
                __import__("pathlib").Path.home() / "Bohemien_Motor_Designer_fea"))
            tk.Entry(ctrl, textvariable=self._fea_workdir_var,
                     bg=ENTRY_BG, fg=TEXT, font=FONT_BODY,
                     relief="flat", insertbackground=TEXT,
                     width=38).grid(row=0, column=1, sticky="ew", padx=4)

            self._btn_fea_cog = tk.Button(
                ctrl, text="▶ Run FEA (cogging)",
                bg=ACCENT, fg="#ffffff", font=FONT_BODY,
                relief="flat", cursor="hand2",
                command=self._fea_run_cogging_async)
            self._btn_fea_cog.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(6, 0))

            self._btn_fea_loaded = tk.Button(
                ctrl, text="▶ Run FEA (loaded)",
                bg=PANEL, fg=TEXT_DIM, font=FONT_BODY,
                relief="flat", cursor="hand2",
                command=self._fea_run_loaded_async)
            self._btn_fea_loaded.grid(row=1, column=1, sticky="ew", padx=4, pady=(6, 0))

            self._btn_fea_prep = tk.Button(
                ctrl, text="📄 Prepare files only",
                bg=PANEL, fg=TEXT_DIM, font=FONT_BODY,
                relief="flat", cursor="hand2",
                command=self._fea_prepare_async)
            self._btn_fea_prep.grid(row=1, column=2, sticky="ew", padx=4, pady=(6, 0))

            ctrl.columnconfigure(1, weight=1)

            # ── Dependency check label ──
            self._lbl_fea_deps = tk.Label(parent, text="", bg=PANEL, fg=TEXT_DIM,
                                           font=FONT_BODY, anchor="w")
            self._lbl_fea_deps.pack(fill="x", padx=10)

            # ── Log output ──
            tk.Label(parent, text="Log:", bg=PANEL, fg=TEXT_DIM,
                     font=FONT_BODY, anchor="w").pack(fill="x", padx=10, pady=(4, 0))
            self.txt_fea_log = scrolledtext.ScrolledText(
                parent, bg="#1a1a2e", fg="#90ee90", font=FONT_MONO,
                relief="flat", wrap="word", state="disabled", height=10)
            self.txt_fea_log.pack(fill="x", padx=10)

            # ── FEA progress bar ──
            self._fea_progress = ttk.Progressbar(parent, mode="determinate",
                                                  length=200, maximum=100)
            self._fea_progress.pack(fill="x", padx=10, pady=(4, 0))

            # ── Results: cogging + EMF panels ──
            results_frame = tk.Frame(parent, bg=PANEL)
            results_frame.pack(fill="both", expand=True, padx=10, pady=(8, 6))

            # Cogging waveform plot
            self._fea_fig = Figure(figsize=(7, 3.2), facecolor=PANEL)
            self._fea_fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.18,
                                           wspace=0.35)
            self._ax_cog = self._fea_fig.add_subplot(1, 2, 1)
            self._ax_emf = self._fea_fig.add_subplot(1, 2, 2)
            for ax in (self._ax_cog, self._ax_emf):
                ax.set_facecolor("#1c1c2e")
                for spine in ax.spines.values():
                    spine.set_color("#444455")
                ax.tick_params(colors=TEXT_DIM, labelsize=7)
                ax.xaxis.label.set_color(TEXT_DIM)
                ax.yaxis.label.set_color(TEXT_DIM)
                ax.title.set_color(TEXT)
            self._ax_cog.set_title("Cogging Torque", fontsize=9, color=TEXT)
            self._ax_emf.set_title("Back-EMF Harmonics", fontsize=9, color=TEXT)

            self._fea_canvas = FigureCanvasTkAgg(self._fea_fig, master=results_frame)
            self._fea_canvas.get_tk_widget().pack(fill="both", expand=True)
            self._fea_canvas.draw()

            # ── Results text ──
            self.txt_fea_results = scrolledtext.ScrolledText(
                parent, bg=PANEL, fg=TEXT, font=FONT_MONO,
                relief="flat", wrap="word", state="disabled", height=6)
            self.txt_fea_results.pack(fill="x", padx=10, pady=(4, 8))

            # Show analytical results immediately (populated after each design run)
            self._fea_show_analytical()

        def _build_status(self):
            bar = tk.Frame(self, bg=PANEL, height=28)
            bar.pack(fill="x", side="bottom")
            bar.pack_propagate(False)
            self.lbl_status = tk.Label(bar, text="Ready", font=FONT_BODY,
                                        bg=PANEL, fg=TEXT_DIM, anchor="w")
            self.lbl_status.pack(side="left", padx=12)
            self.progress = ttk.Progressbar(bar, mode="indeterminate", length=140)
            self.progress.pack(side="right", padx=12, pady=6)

        # ── Input helpers ──────────────────────────────────────────────────────

        def _flt(self, widget, default=0.0):
            try:    return float(widget.get())
            except: return default

        def _int(self, widget, default=0):
            try:    return int(widget.get())
            except: return default

        # ── Run workflow ───────────────────────────────────────────────────────

        def _run_async(self):
            self.btn_run.config(state="disabled", text="Running…")
            self.progress.start(12)
            self._set_status("Running design workflow…", ACCENT)
            t = threading.Thread(target=self._run_design, daemon=True)
            t.start()

        def _run_design(self):
            try:
                self._do_run()
            except Exception:
                err = traceback.format_exc()
                self.after(0, lambda: self._on_error(err))
            finally:
                self.after(0, self._run_done)

        def _do_run(self):
            from Bohemien_Motor_Designer.core import (DesignSpec, DriveSpec, CoolingSpec,
                                            InsulationSpec, PMSM, StatorGeometry,
                                            ParallelToothSlot, SPMRotorGeometry,
                                            WindingLayout)
            from Bohemien_Motor_Designer.materials import MaterialLibrary
            from Bohemien_Motor_Designer.analysis import LossCalculator, PerformanceAnalyzer
            from Bohemien_Motor_Designer.thermal import ThermalNetwork, WaterJacketCooling, AirCooling
            from Bohemien_Motor_Designer.drive import Inverter
            from Bohemien_Motor_Designer.scaling import MotorScalingLaws
            from Bohemien_Motor_Designer.utils import DesignRuleChecker

            self._set_status("Building DesignSpec…", ACCENT)

            power   = self._flt(self.e_power, 30)
            speed   = self._flt(self.e_speed, 4000)
            s_max   = self._flt(self.e_speed_max, 12000)
            poles   = self._int(self.e_poles, 8)
            slots   = self._int(self.e_slots, 48)
            stack   = self._flt(self.e_stack, 130) / 1000
            airgap  = self._flt(self.e_airgap, 1.0) / 1000
            t_mag   = self._flt(self.e_tmag, 6) / 1000
            alpha   = self._flt(self.e_alpha, 0.83)
            turns   = self._int(self.e_turns, 11)
            paths   = self._int(self.e_paths, 1)
            vbus    = self._flt(self.e_vbus, 400)
            fsw     = self._flt(self.e_fsw, 20) * 1e3
            t_cool  = self._flt(self.e_tcoolant, 65)
            flow    = self._flt(self.e_flow, 12)

            spec = DesignSpec(
                power_kW=power, speed_rpm=speed,
                speed_range=(speed * 0.1, s_max),
                drive=DriveSpec(dc_bus_voltage=vbus, device=self.c_device.get(),
                                switching_freq=fsw, modulation="SVPWM"),
                cooling=CoolingSpec(cooling_type=self.c_cooling.get(),
                                    coolant_temp_C=t_cool, coolant_flow_lpm=flow),
                insulation=InsulationSpec(insulation_class=self.c_ins.get()),
                efficiency_target=0.93, overload_factor=2.5,
            )

            self._set_status("Constructing motor geometry…", ACCENT)

            # Auto-size stator from scaling estimate
            from Bohemien_Motor_Designer.scaling.similarity import ESSON_COEFFICIENT
            C = ESSON_COEFFICIENT.get(self.c_cooling.get(), 50) * 1e3
            omega = speed * 2 * np.pi / 60
            torque = power * 1e3 / omega
            D_bore = (8 * torque / (np.pi * C * 1.2)) ** (1/3)
            D_OD   = D_bore * 1.55
            slot_pitch = np.pi * D_bore / slots
            _sd_input = self._flt(self.e_slot_depth, 0.0)
            _sw_input = self._flt(self.e_slot_width, 0.0)
            slot_depth = (_sd_input / 1000) if _sd_input > 0 else min(0.38 * (D_OD - D_bore) / 2, slot_pitch * 1.8)
            slot_width = (_sw_input / 1000) if _sw_input > 0 else slot_pitch * 0.52

            # ── Iterative slot sizing: enlarge until J ≤ J_limit ──────────
            from Bohemien_Motor_Designer.scaling.similarity import CURRENT_DENSITY_LIMIT
            J_lim = CURRENT_DENSITY_LIMIT.get(self.c_cooling.get(), 12.0)

            lam_key  = self.c_laminate.get()
            rotor_or = D_bore / 2 - airgap
            rotor_ir = rotor_or * 0.38
            rotor_geo = SPMRotorGeometry(
                outer_radius=rotor_or, inner_radius=rotor_ir,
                magnet_thickness=t_mag, magnet_width_fraction=alpha,
                magnet_material=self.c_magnet.get())

            winding = WindingLayout(poles=poles, slots=slots, phases=3,
                                     layers=2, turns_per_coil=max(turns, 1),
                                     parallel_paths=paths)

            # ── Auto-compute N_coil if user left field at 0 ────────────────
            # Target: E_peak = 0.70 × Vpk_max at rated speed
            # Vpk_max = Vbus / sqrt(3)  (SVPWM limit)
            if turns <= 0:
                MU0 = 4e-7 * np.pi
                _mu_r = 1.05
                _Br   = 1.22   # conservative NdFeB estimate
                _tm   = t_mag
                _g    = airgap
                _B_gap = _Br * _tm / (_tm + _mu_r * _g)
                _B_rv  = (4 / 3.14159) * _B_gap * np.sin(3.14159 * alpha / 2)
                _kw    = 0.966  # typical for q=2 distributed winding
                _tau_p = np.pi * (D_bore/2) / (poles//2)
                _L     = stack
                _omega_e = speed * 2 * np.pi / 60 * (poles // 2)
                _Vpk   = vbus / (3 ** 0.5)          # SVPWM phase peak limit
                _E_target = 0.70 * _Vpk
                # Ke = (2/pi) * N_series * kw * B_rv * tau_p * L
                # N_series = slots/phases * N_coil / parallel_paths  (2-layer, both sides)
                _slots_per_phase = slots // 3
                _N_series_per_Ncoil = _slots_per_phase // paths   # series turns per unit N_coil
                # Ke_target = _E_target / _omega_e
                _Ke_target = _E_target / (_omega_e + 1e-9)
                _N_series_target = _Ke_target / ((2 / np.pi) * _kw * _B_rv * _tau_p * _L + 1e-9)
                turns = max(1, round(_N_series_target / _N_series_per_Ncoil))
                # Clamp to reasonable range
                turns = int(np.clip(turns, 1, 30))
                # Write back to GUI field so user can see what was chosen
                self.e_turns.delete(0, "end")
                self.e_turns.insert(0, str(turns))
                self._set_status(f"Auto-sized turns per coil → {turns}", ACCENT)

                winding = WindingLayout(poles=poles, slots=slots, phases=3,
                                         layers=2, turns_per_coil=turns,
                                         parallel_paths=paths)

            # Start with initial slot dimensions, expand depth up to 60% of
            # available radial build until J is within limit
            max_depth = (D_OD/2 - D_bore/2) * 0.60
            for scale in np.linspace(1.0, 2.5, 12):
                sd = min(slot_depth * scale, max_depth)
                sw = slot_width  # keep width fixed — depth easier to grow
                slot = ParallelToothSlot(
                    slot_width=sw, slot_depth=sd,
                    slot_opening=max(0.002, sw * 0.35))
                stator = StatorGeometry(outer_radius=D_OD/2, inner_radius=D_bore/2,
                                         slots=slots, slot_profile=slot,
                                         lamination=lam_key)
                # Quick J estimate using correct Ke formula
                _Br_est = 1.22  # conservative until material is loaded
                _B_gap_est = _Br_est * t_mag / (t_mag + 1.05 * airgap)
                _B_rv_est  = (4/np.pi) * _B_gap_est * np.sin(np.pi * alpha / 2)
                _N_s_est   = winding.total_series_turns_per_phase
                _kw_est    = 0.966
                _tau_p_est = np.pi * (D_bore/2) / (poles//2)
                _Ke_est    = (2/np.pi) * _N_s_est * _kw_est * _B_rv_est * _tau_p_est * sd
                _Ke_est    = max(_Ke_est, 0.001)  # guard against zero
                A_c   = slot.area() * 0.45 / max(turns, 1)  # m²
                Iq    = (power*1e3/(speed*2*np.pi/60)) / (1.5*(poles//2)*_Ke_est + 1e-9)
                I_rms = Iq / np.sqrt(2)
                J_est = I_rms / (A_c * 1e6)
                if J_est <= J_lim * 1.05:   # within 5% tolerance
                    break
            # Use final stator
            motor = PMSM(
                poles=poles, slots=slots,
                stator=stator, rotor_geo=rotor_geo,
                rotor_outer_radius=rotor_or, rotor_inner_radius=rotor_ir,
                stack_length=stack, airgap=airgap,
                rated_speed=speed, rated_power=power * 1e3,
                magnet_material=self.c_magnet.get(),
                magnet_thickness=t_mag, magnet_width_fraction=alpha,
                turns_per_coil=turns, parallel_paths=paths,
                winding=winding, spec=spec)

            # Write back computed slot dimensions ONLY when field was left at 0 (auto mode).
            # If the user typed a value, preserve it exactly — never overwrite manual input.
            sd_used = motor.stator.slot_profile.depth() * 1e3
            sw_used = motor.stator.slot_profile.slot_width * 1e3
            if self._flt(self.e_slot_depth, 0.0) == 0.0:
                self.after(0, lambda sd=sd_used: self.e_slot_depth.set(f"{sd:.1f}"))
            if self._flt(self.e_slot_width, 0.0) == 0.0:
                self.after(0, lambda sw=sw_used: self.e_slot_width.set(f"{sw:.1f}"))

            self._set_status("Running DRC…", ACCENT)
            lib = MaterialLibrary()
            inverter = Inverter(dc_bus_V=vbus, switching_freq=fsw,
                                device=self.c_device.get())
            checker = DesignRuleChecker(motor, spec=spec, inverter=inverter,
                                         material_lib=lib)
            checker.check_all()

            self._set_status("Computing losses…", ACCENT)
            loss_calc = LossCalculator(motor, lib, temperature=120.0,
                                        inverter=inverter)
            lb = loss_calc.loss_budget(speed, motor.rated_torque)

            self._set_status("Computing efficiency map…", ACCENT)
            perf = PerformanceAnalyzer(motor, lib, inverter=inverter)
            eff_map = perf.pmsm_efficiency_map(
                speed_range=(speed * 0.05, s_max), n_speed=30, n_torque=22)

            self._set_status("Thermal analysis…", ACCENT)
            cooling_type = self.c_cooling.get()
            if "water" in cooling_type or "oil" in cooling_type:
                cooling = WaterJacketCooling(flow_lpm=flow, _inlet_temp_C=t_cool)
            else:
                cooling = AirCooling(_inlet_temp_C=t_cool)
            therm = ThermalNetwork(motor, cooling, lib)
            thermal_result = therm.steady_state(lb.to_dict())

            self._set_status("Scaling analysis…", ACCENT)
            scaling_comparison = MotorScalingLaws.compare_cooling(power, speed)
            feasibility = MotorScalingLaws.feasibility_check(
                power, speed, D_OD * 1000 * 1.3, stack * 1000 * 1.5, cooling_type)

            self._results = dict(motor=motor, spec=spec, lb=lb, eff_map=eff_map,
                                  thermal=thermal_result, checker=checker,
                                  scaling=scaling_comparison,
                                  feasibility=feasibility)

            self.after(0, self._update_ui)

        def _update_ui(self):
            r = self._results
            m  = r["motor"]
            lb = r["lb"]
            th = r["thermal"]
            chk= r["checker"]

            # ── Summary tab ────────────────────────────────────────────────
            Ke = m.back_emf_constant()
            omega_e = m.rated_speed * 2*np.pi/60 * m.pole_pairs
            lines = [
                "═" * 54,
                f"  Motor Summary",
                "═" * 54,
                f"  Topology        : {m.poles}p / {m.slots}s SPM",
                f"  Stator OD / Bore: {m.stator_outer_radius*1e3:.1f} / {m.stator_inner_radius*1e3:.1f} mm",
                f"  Stack length    : {m.stack_length*1e3:.1f} mm",
                f"  Air gap         : {m.airgap*1e3:.2f} mm",
                f"  Electrical freq : {m.electrical_frequency:.1f} Hz",
                f"  Slots/pole/phase: {m.slots_per_pole_per_phase:.3f}",
                "",
                f"  Winding factor  : {m.winding_factor():.4f}",
                f"  Series turns/ph : {m.winding.total_series_turns_per_phase}",
                f"  Back-EMF const  : {Ke:.4f} V·s/rad",
                f"  Peak back-EMF   : {Ke*omega_e:.1f} V  @ {m.rated_speed:.0f} rpm",
                f"  Rated torque    : {m.rated_torque:.2f} N·m",
                f"  Rated current   : {m.rated_current:.1f} A RMS (est.)",
                "",
                f"  DC bus voltage  : {m.spec.drive.dc_bus_voltage:.0f} V",
                f"  Max phase Vpk   : {m.spec.drive.max_phase_voltage_peak():.1f} V",
                f"  Back-EMF / Vpk  : {Ke*omega_e / m.spec.drive.max_phase_voltage_peak() * 100:.1f} %",
                "",
                f"  EFFICIENCY      : {lb.efficiency*100:.2f} %",
                f"  Total losses    : {lb.total_loss_W:.1f} W",
                "═" * 54,
            ]
            self._set_text(self.txt_summary, "\n".join(lines))

            # ── Loss tab ──────────────────────────────────────────────────
            loss_lines = [
                "═" * 54,
                f"  Loss Budget  @  {lb.speed_rpm:.0f} rpm | {lb.torque_Nm:.1f} N·m",
                "═" * 54,
                f"  Output power    : {lb.output_power_W/1e3:.3f} kW",
                f"  Input power     : {lb.input_power_W/1e3:.3f} kW",
                "─" * 54,
                f"  Copper loss     : {lb.copper_loss_W:.1f} W",
                f"  Stator iron     : {lb.stator_iron_W:.1f} W",
                f"  Rotor iron      : {lb.rotor_iron_W:.1f} W",
                f"  Friction        : {lb.friction_W:.1f} W",
                f"  Windage         : {lb.windage_W:.1f} W",
                f"  Stray           : {lb.stray_W:.1f} W",
                f"  Inverter        : {lb.inverter_loss_W:.1f} W",
                "─" * 54,
                f"  TOTAL LOSSES    : {lb.total_loss_W:.1f} W",
                f"  EFFICIENCY      : {lb.efficiency*100:.2f} %",
                "═" * 54,
            ]
            self._set_text(self.txt_losses, "\n".join(loss_lines))

            # ── Thermal tab ──────────────────────────────────────────────
            ins_lim = m.spec.insulation.max_winding_temp_C
            margin  = ins_lim - th.T_winding_C
            status  = "✓ OK" if margin > 10 else ("⚠ MARGINAL" if margin > 0 else "✗ OVER LIMIT")
            th_lines = [
                "═" * 54,
                f"  Thermal Analysis (Steady State)",
                "═" * 54,
                f"  Winding temp    : {th.T_winding_C:.1f} °C",
                f"  Stator teeth    : {th.T_teeth_C:.1f} °C",
                f"  Stator yoke     : {th.T_yoke_C:.1f} °C",
                f"  Cooling jacket  : {th.T_jacket_C:.1f} °C",
                f"  Rotor / magnet  : {th.T_rotor_C:.1f} °C",
                f"  Coolant outlet  : {th.T_coolant_out_C:.1f} °C",
                "─" * 54,
                f"  Insulation limit: {ins_lim:.0f} °C (class {m.spec.insulation.insulation_class})",
                f"  Thermal margin  : {margin:.1f} °C  {status}",
                "═" * 54,
            ]
            self._set_text(self.txt_thermal, "\n".join(th_lines))

            # ── DRC tab ───────────────────────────────────────────────────
            self._set_text(self.txt_drc, r["checker"].report())
            # Colour errors/warnings
            self._colour_drc()

            # ── Efficiency Map ────────────────────────────────────────────
            self._plot_eff_map(r["eff_map"], m)

            # ── Scaling tab ───────────────────────────────────────────────
            sc_lines = [
                "═" * 58,
                f"  Scaling Analysis  —  {lb.output_power_W/1e3:.0f} kW / {m.rated_speed:.0f} rpm",
                "═" * 58,
                f"  {'Cooling':<24}  {'OD mm':>7}  {'L mm':>7}  {'Vol L':>6}",
                "─" * 58,
            ]
            for e in r["scaling"]:
                sc_lines.append(
                    f"  {e.cooling:<24}  {e.outer_diameter_mm:>7.0f}  "
                    f"{e.stack_length_mm:>7.0f}  {e.active_volume_L:>6.2f}")
            sc_lines += [
                "─" * 58,
                f"  Selected cooling : {m.spec.cooling.cooling_type}",
                f"  Fits in envelope : {'YES' if r['feasibility']['feasible'] else 'NO'}",
                f"  OD margin        : {r['feasibility']['OD_margin_pct']:.1f} %",
                f"  Note             : {r['feasibility']['recommendation']}",
                "═" * 58,
            ]
            self._set_text(self.txt_scaling, "\n".join(sc_lines))

            # Switch to summary tab
            self._notebook.select(0)

        def _plot_eff_map(self, eff_map, motor):
            self._fig.clear()
            gs = gridspec.GridSpec(1, 2, figure=self._fig,
                                    width_ratios=[3, 1], wspace=0.35)
            ax1 = self._fig.add_subplot(gs[0])
            ax2 = self._fig.add_subplot(gs[1])

            S, T   = np.meshgrid(eff_map["speed_rpm"], eff_map["torque_Nm"])
            ep_pct = eff_map["efficiency"] * 100

            cf = ax1.contourf(S, T, ep_pct, levels=np.linspace(60, 99, 25),
                              cmap="RdYlGn")
            cb = self._fig.colorbar(cf, ax=ax1)
            cb.set_label("Efficiency [%]", color=TEXT, fontsize=8)
            cb.ax.yaxis.set_tick_params(color=TEXT)
            plt_ticks = [60, 70, 80, 90, 93, 95, 97]
            ax1.contour(S, T, ep_pct, levels=plt_ticks,
                        colors="black", linewidths=0.6, alpha=0.6)
            ax1.axvline(motor.rated_speed, color="white", ls="--",
                        lw=1.5, label=f"Rated {motor.rated_speed:.0f} rpm")
            ax1.axhline(motor.rated_torque, color="cyan", ls="--",
                        lw=1.2, label=f"Rated {motor.rated_torque:.1f} N·m")
            ax1.set_facecolor(BG)
            ax1.tick_params(colors=TEXT_DIM, labelsize=8)
            ax1.set_xlabel("Speed [rpm]", color=TEXT_DIM, fontsize=9)
            ax1.set_ylabel("Torque [N·m]", color=TEXT_DIM, fontsize=9)
            ax1.set_title("Efficiency Map", color=TEXT, fontsize=10)
            ax1.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)
            for sp in ax1.spines.values(): sp.set_color(BORDER)

            # Loss pie
            lb_vals = [
                self._results["lb"].copper_loss_W,
                self._results["lb"].stator_iron_W,
                self._results["lb"].rotor_iron_W,
                self._results["lb"].friction_W + self._results["lb"].windage_W,
                self._results["lb"].inverter_loss_W,
            ]
            lb_labels = ["Copper", "Fe stator", "Fe rotor", "Mech", "Inverter"]
            colors    = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6"]
            non_zero  = [(v, l, c) for v, l, c in zip(lb_vals, lb_labels, colors) if v > 0.1]
            if non_zero:
                vals, labs, cols = zip(*non_zero)
                ax2.pie(vals, labels=labs, colors=cols, autopct="%1.0f%%",
                        startangle=140, textprops={"color": TEXT, "fontsize": 7})
            ax2.set_facecolor(BG)
            eta = self._results["lb"].efficiency
            ax2.set_title(f"η = {eta*100:.1f}%\nTotal {self._results['lb'].total_loss_W:.0f} W",
                          color=TEXT, fontsize=9)

            self._fig.patch.set_facecolor(BG)
            self._canvas.draw()

        # ── Helpers ────────────────────────────────────────────────────────────

        def _set_text(self, widget, text):
            widget.config(state="normal")
            widget.delete("1.0", "end")
            widget.insert("end", text)
            widget.config(state="disabled")

        def _colour_drc(self):
            w = self.txt_drc
            w.config(state="normal")
            for tag, colour in [("[ERROR]", WARNING), ("[WARN]", ACCENT2),
                                 ("[OK]", SUCCESS)]:
                start = "1.0"
                while True:
                    pos = w.search(tag, start, stopindex="end")
                    if not pos: break
                    end = f"{pos}+{len(tag)}c"
                    w.tag_add(tag, pos, end)
                    w.tag_config(tag, foreground=colour)
                    start = end
            w.config(state="disabled")

        def _set_status(self, msg, colour=TEXT_DIM):
            self.after(0, lambda: self.lbl_status.config(text=msg, fg=colour))

        def _on_error(self, err):
            self._set_text(self.txt_summary,
                           "ERROR during design run:\n\n" + err)
            messagebox.showerror("Design Error", err[:300])

        def _run_done(self):
            self.progress.stop()
            self.btn_run.config(state="normal", text="▶  RUN DESIGN")
            has_errors = self._results.get("checker") and \
                         self._results["checker"].has_errors()
            if has_errors:
                self._set_status("Done — DRC errors found", WARNING)
            else:
                self._set_status("Done — Design complete", SUCCESS)

            # Refresh FEA + manufacturing tabs whenever a design run completes
            self.after(100, self._fea_show_analytical)
            self.after(200, self._mfg_refresh)

        # ── FEA Tab methods ────────────────────────────────────────────────────

        def _fea_log(self, msg: str, frac: float = None):
            """Append a line to the FEA log widget (thread-safe via after())."""
            def _update():
                self.txt_fea_log.config(state="normal")
                self.txt_fea_log.insert("end", msg + "\n")
                self.txt_fea_log.see("end")
                self.txt_fea_log.config(state="disabled")
                if frac is not None:
                    self._fea_progress["value"] = max(0, min(100, int(frac * 100)))
            self.after(0, _update)

        def _fea_show_analytical(self):
            """Populate FEA tab with analytical results from last design run."""
            if not self._results or "motor" not in self._results:
                return
            m = self._results["motor"]
            try:
                from Bohemien_Motor_Designer.analysis.losses import cogging_torque_Nm
                cog   = cogging_torque_Nm(m)
                h     = m.back_emf_harmonics(n_harmonics=15)
                b_gap_note = (f"B_gap={cog['B_gap_used']:.3f}T magnet-circuit"
                              if cog.get("B_gap_used") else "")
                lines = [
                    "─── Analytical Results ───",
                    f"  Ld (analytical)  : {m.Ld*1000:.3f} mH",
                    f"  Lq (analytical)  : {m.Lq*1000:.3f} mH",
                    f"  Cogging Tpp      : {cog['Tcog_pp_Nm']:.3f} N·m  "
                    f"({cog['Tcog_pp_pct']:.2f}% rated)  [{b_gap_note}]",
                    f"  Cogging period   : {cog['cogging_period_deg']:.2f}°  "
                    f"(LCM={cog['lcm_slots_poles']})",
                    f"  Back-EMF fund    : {h['fundamental']:.1f} V pk",
                    f"  Back-EMF THD     : {h['thd']:.2f}%  (excl. triplens)",
                    "─── Run FEA above for FEM-assisted cogging waveform ───",
                ]
                self._set_text(self.txt_fea_results, "\n".join(lines))

                # Plot back-EMF harmonic bar chart
                self._ax_emf.clear()
                self._ax_emf.set_facecolor("#1c1c2e")
                for sp in self._ax_emf.spines.values():
                    sp.set_color("#444455")

                orders = h["harmonics"][:13]
                amps   = [a / h["fundamental"] * 100 for a in h["amplitudes"][:13]]
                colors = ["#e74c3c" if n % 3 == 0 else "#3498db" for n in orders]
                self._ax_emf.bar(range(len(orders)), amps, color=colors, width=0.7)
                self._ax_emf.set_xticks(range(len(orders)))
                self._ax_emf.set_xticklabels([str(n) for n in orders],
                                              fontsize=6, color=TEXT_DIM)
                self._ax_emf.set_ylabel("%  of  fund.", color=TEXT_DIM, fontsize=7)
                self._ax_emf.set_title(
                    f"Back-EMF Harmonics  THD={h['thd']:.1f}%",
                    fontsize=8, color=TEXT)
                self._ax_emf.tick_params(colors=TEXT_DIM, labelsize=7)
                self._ax_emf.axhline(y=0, color="#444455", lw=0.5)

                # Show dependency check
                try:
                    from Bohemien_Motor_Designer.fea.runner import FEARunner
                    runner = FEARunner(m, work_dir=self._fea_workdir_var.get())
                    deps = runner.check_dependencies()
                    dep_txt = "  ".join(
                        f"{'✓' if ok else '✗'} {tool}"
                        for tool, ok in deps.items()
                    )
                    self._lbl_fea_deps.config(
                        text=dep_txt,
                        fg=SUCCESS if all(deps.values()) else WARNING)
                except Exception:
                    pass

                self._fea_canvas.draw()
            except Exception as e:
                self._fea_log(f"Analytical FEA results failed: {e}")

        def _fea_prepare_async(self):
            """Write all FEA input files without running solver."""
            if not self._results or "motor" not in self._results:
                self._fea_log("Run a design first (▶ RUN DESIGN).")
                return
            self._fea_set_buttons(False)
            t = threading.Thread(target=self._fea_prepare_worker, daemon=True)
            t.start()

        def _fea_prepare_worker(self):
            try:
                from Bohemien_Motor_Designer.fea.runner import FEARunner
                m      = self._results["motor"]
                runner = FEARunner(m, work_dir=self._fea_workdir_var.get())
                runner.prepare(progress_cb=self._fea_log)
                self._fea_log(f"Files written to: {runner.wd}", 1.0)
            except Exception as e:
                self._fea_log(f"ERROR: {e}")
            finally:
                self.after(0, lambda: self._fea_set_buttons(True))

        def _fea_run_cogging_async(self):
            """Run cogging sweep in background thread."""
            if not self._results or "motor" not in self._results:
                self._fea_log("Run a design first (▶ RUN DESIGN).")
                return
            self._fea_set_buttons(False)
            t = threading.Thread(target=self._fea_cogging_worker, daemon=True)
            t.start()

        def _fea_cogging_worker(self):
            try:
                m = self._results["motor"]
                runner = self._make_fea_runner(m)
                data   = runner.run_cogging(progress_cb=self._fea_log)
                self.after(0, lambda: self._fea_plot_cogging(data))
            except Exception as e:
                self._fea_log(f"ERROR: {e}")
                import traceback
                self._fea_log(traceback.format_exc())
            finally:
                self.after(0, lambda: self._fea_set_buttons(True))

        def _fea_run_loaded_async(self):
            """Run loaded transient in background thread."""
            if not self._results or "motor" not in self._results:
                self._fea_log("Run a design first (> RUN DESIGN).")
                return
            self._fea_set_buttons(False)
            t = threading.Thread(target=self._fea_loaded_worker, daemon=True)
            t.start()

        def _fea_loaded_worker(self):
            try:
                m = self._results["motor"]
                runner = self._make_fea_runner(m)
                data   = runner.run_loaded(progress_cb=self._fea_log)
                self.after(0, lambda: self._fea_update_loaded(data))
            except Exception as e:
                self._fea_log(f"ERROR: {e}")
                import traceback
                self._fea_log(traceback.format_exc())
            finally:
                self.after(0, lambda: self._fea_set_buttons(True))

        def _make_fea_runner(self, motor):
            """
            Return the best available FEA runner.

            Priority:
              1. PythonFEARunner  -- always available (numpy/scipy only)
              2. FEARunner (Elmer) -- used only if GMSH + Elmer are both on PATH

            The Python runner is preferred because it needs no installation.
            The user can force Elmer by setting Bohemien_Motor_Designer_USE_ELMER=1 in
            the environment.
            """
            import os, shutil
            use_elmer = os.environ.get("Bohemien_Motor_Designer_USE_ELMER", "0") == "1"

            if use_elmer:
                from Bohemien_Motor_Designer.fea.runner import FEARunner
                self._fea_log("Using Elmer FEA pipeline (Bohemien_Motor_Designer_USE_ELMER=1)", 0.0)
                return FEARunner(motor, work_dir=self._fea_workdir_var.get())

            # Default: py_runner (validated numpy/scipy FEM, no external tools)
            from Bohemien_Motor_Designer.fea.py_runner import PythonFEARunner as _PyRunner
            self._fea_log("Using built-in Python FEM solver (numpy/scipy only)", 0.0)
            runner = _PyRunner(motor, n_radial_airgap=4, n_ang_per_slot=8)
            runner.build_mesh()
            self._fea_log("FEM mesh built — ready to solve.", 0.05)
            return runner

        def _fea_plot_cogging(self, data):
            """Update cogging torque plot from FEA results."""
            theta = data.get("theta_deg", [])
            torq  = data.get("torque_Nm", [])
            if len(theta) == 0:
                return

            self._ax_cog.clear()
            self._ax_cog.set_facecolor("#1c1c2e")
            for sp in self._ax_cog.spines.values():
                sp.set_color("#444455")
            self._ax_cog.plot(theta, torq, color="#3498db", lw=1.5)
            self._ax_cog.axhline(0, color="#444455", lw=0.7)
            Tpp = data.get("Tcog_pp_Nm", 0)
            pct = data.get("Tcog_pp_pct", 0)
            self._ax_cog.set_title(
                f"Cogging  Tpp={Tpp:.2f}Nm ({pct:.1f}%)",
                fontsize=8, color=TEXT)
            self._ax_cog.set_xlabel("θ [deg]", fontsize=7, color=TEXT_DIM)
            self._ax_cog.set_ylabel("Torque [N·m]", fontsize=7, color=TEXT_DIM)
            self._ax_cog.tick_params(colors=TEXT_DIM, labelsize=7)
            self._fea_canvas.draw()

            # Update results text with FEA + analytical comparison
            m = self._results["motor"]
            from Bohemien_Motor_Designer.analysis.losses import cogging_torque_Nm
            cog_anal = cogging_torque_Nm(m)
            b_note = (f"  B_gap={cog_anal['B_gap_used']:.3f}T"
                      if cog_anal.get("B_gap_used") else "")
            lines = [
                "─── Cogging FEA (Zhu-Howe spectral) ───",
                f"  T_pp : {Tpp:.3f} N·m  ({pct:.2f}%)",
                f"  Period : {data.get('cog_period_deg', 0):.2f}°  "
                f"LCM={data.get('lcm_val', 0)}" + b_note,
            ]
            self._set_text(self.txt_fea_results, "\n".join(lines))

        def _fea_update_loaded(self, data):
            """Update GUI with loaded FEA results: Ld/Lq comparison + EMF."""
            m = self._results["motor"]
            lines = [
                "─── Loaded FEA Results ───",
                f"  T_avg (FEA)  : {data.get('torque_avg_Nm', float('nan')):.2f} N·m",
                f"  T_rated      : {m.rated_torque:.2f} N·m",
            ]
            if data.get("Ld_H"):
                lines += [
                    f"  Ld (FEA)     : {data['Ld_H']*1000:.3f} mH",
                    f"  Lq (FEA)     : {(data.get('Lq_H') or m.Lq)*1000:.3f} mH",
                ]
            if data.get("emf_waveform") and data["emf_waveform"].get("thd_pct") is not None:
                lines.append(f"  EMF THD (FEA): {data['emf_waveform']['thd_pct']:.1f}%")

            self._set_text(self.txt_fea_results, "\n".join(lines))

            # Plot EMF waveform if available
            emf = data.get("emf_waveform")
            if emf and len(emf.get("time", [])) > 4:
                self._ax_emf.clear()
                self._ax_emf.set_facecolor("#1c1c2e")
                self._ax_emf.plot(emf["time"] * 1000, emf["voltage"],
                                  color="#3498db", lw=1.5)
                self._ax_emf.set_xlabel("t [ms]", fontsize=7, color=TEXT_DIM)
                self._ax_emf.set_ylabel("E [V]", fontsize=7, color=TEXT_DIM)
                self._ax_emf.set_title(
                    f"Back-EMF (FEA)  THD={emf['thd_pct']:.1f}%",
                    fontsize=8, color=TEXT)
                self._ax_emf.tick_params(colors=TEXT_DIM, labelsize=7)
            self._fea_canvas.draw()

        def _fea_set_buttons(self, enabled: bool):
            """Enable/disable all FEA buttons."""
            state = "normal" if enabled else "disabled"
            for btn in (self._btn_fea_cog, self._btn_fea_loaded, self._btn_fea_prep):
                btn.config(state=state)

        # ── Mesh Visualisation tab ────────────────────────────────────────────────

        def _build_mesh_viz_tab(self, parent):
            """Mesh Viz tab: control strip + three-panel matplotlib figure."""
            # ── Controls ──────────────────────────────────────────────────────
            ctrl = tk.Frame(parent, bg=PANEL)
            ctrl.pack(fill="x", padx=10, pady=(8, 4))

            # n_angular_per_slot spinner
            tk.Label(ctrl, text="Angular els/slot:", bg=PANEL, fg=TEXT_DIM,
                     font=FONT_BODY).grid(row=0, column=0, sticky="w", padx=(0, 4))
            self._mv_nang_var = tk.IntVar(value=8)
            tk.Spinbox(ctrl, from_=4, to=32, increment=4,
                       textvariable=self._mv_nang_var, width=5,
                       bg=PANEL, fg=TEXT, font=FONT_BODY,
                       buttonbackground=PANEL, relief="flat"
                       ).grid(row=0, column=1, sticky="w", padx=(0, 12))

            # n_radial_airgap spinner
            tk.Label(ctrl, text="Airgap radial layers:", bg=PANEL, fg=TEXT_DIM,
                     font=FONT_BODY).grid(row=0, column=2, sticky="w", padx=(0, 4))
            self._mv_nag_var = tk.IntVar(value=4)
            tk.Spinbox(ctrl, from_=2, to=12, increment=1,
                       textvariable=self._mv_nag_var, width=5,
                       bg=PANEL, fg=TEXT, font=FONT_BODY,
                       buttonbackground=PANEL, relief="flat"
                       ).grid(row=0, column=3, sticky="w", padx=(0, 16))

            # Generate button
            self._btn_mv = tk.Button(
                ctrl, text="▶ Generate Mesh Viz",
                bg=ACCENT, fg="#ffffff", font=FONT_BODY,
                relief="flat", cursor="hand2",
                command=self._mesh_viz_run_async)
            self._btn_mv.grid(row=0, column=4, sticky="ew", padx=(0, 8))

            # Export button
            self._btn_mv_export = tk.Button(
                ctrl, text="💾 Save PNG",
                bg=PANEL, fg=TEXT, font=FONT_BODY,
                relief="flat", cursor="hand2",
                command=self._mesh_viz_export)
            self._btn_mv_export.grid(row=0, column=5, sticky="ew")

            # Status label
            self._mv_status = tk.Label(parent, text="Run a design first, then click Generate.",
                                       bg=PANEL, fg=TEXT_DIM, font=FONT_BODY, anchor="w")
            self._mv_status.pack(fill="x", padx=10, pady=(0, 4))

            # ── Figure ────────────────────────────────────────────────────────
            self._mv_fig = Figure(figsize=(14, 7), facecolor="#0f1117")
            self._mv_canvas = FigureCanvasTkAgg(self._mv_fig, master=parent)
            self._mv_canvas.get_tk_widget().pack(fill="both", expand=True,
                                                 padx=6, pady=(0, 4))
            toolbar = NavigationToolbar2Tk(self._mv_canvas, parent, pack_toolbar=False)
            toolbar.configure(bg=BG)
            toolbar.pack(side="bottom", fill="x")
            self._mv_canvas.draw()

            # Cache for export
            self._mv_mesh   = None
            self._mv_motor  = None

        def _mesh_viz_run_async(self):
            """Build mesh + render visualisation in a background thread."""
            if not self._results or "motor" not in self._results:
                self._mv_status.config(text="⚠  Run a design first (▶ RUN DESIGN).")
                return
            self._btn_mv.config(state="disabled")
            self._mv_status.config(text="Building mesh…")
            t = threading.Thread(target=self._mesh_viz_worker, daemon=True)
            t.start()

        def _mesh_viz_worker(self):
            """Background: build mesh, draw, update canvas on main thread."""
            try:
                motor   = self._results["motor"]
                n_ang   = self._mv_nang_var.get()
                n_ag    = self._mv_nag_var.get()

                self.after(0, lambda: self._mv_status.config(
                    text=f"Building mesh  (n_ang={n_ang}, n_ag={n_ag})…"))

                from Bohemien_Motor_Designer.fea.py_mesh import build_motor_mesh
                mesh = build_motor_mesh(motor,
                                        n_radial_airgap=n_ag,
                                        n_angular_per_slot=n_ang)

                self.after(0, lambda: self._mv_status.config(
                    text=f"Mesh built — {mesh.n_nodes:,} nodes  {mesh.n_elems:,} elems  "
                         f"· Rendering…"))

                from Bohemien_Motor_Designer.fea.mesh_viz import plot_mesh_overview
                plot_mesh_overview(motor, mesh, self._mv_fig)

                # Store for export
                self._mv_mesh  = mesh
                self._mv_motor = motor

                self.after(0, self._mesh_viz_done)

            except Exception as exc:
                import traceback as _tb
                msg = f"Mesh Viz error: {exc}\n{_tb.format_exc()}"
                self.after(0, lambda: self._mv_status.config(text=f"⚠  {exc}"))
            finally:
                self.after(0, lambda: self._btn_mv.config(state="normal"))

        def _mesh_viz_done(self):
            """Refresh canvas after render completes."""
            self._mv_canvas.draw()
            mesh = self._mv_mesh
            if mesh:
                self._mv_status.config(
                    text=f"✔  {mesh.n_nodes:,} nodes  ·  {mesh.n_elems:,} elements  "
                         f"·  {len(set(mesh.tags))} material regions")

        def _mesh_viz_export(self):
            """Save the current mesh figure as PNG."""
            if self._mv_motor is None:
                self._mv_status.config(text="⚠  Generate a mesh first.")
                return
            from tkinter.filedialog import asksaveasfilename
            path = asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
                initialfile="mesh_visualization.png",
                title="Save mesh visualisation")
            if path:
                self._mv_fig.savefig(path, dpi=200,
                                     bbox_inches="tight",
                                     facecolor="#0f1117")
                self._mv_status.config(text=f"✔  Saved → {path}")




        # ── DXF Export tab ────────────────────────────────────────────────────────

        def _build_dxf_tab(self, parent):
            """DXF Export tab — preview canvas + export controls."""
            import matplotlib.patches as mpatches

            # ── Controls ──────────────────────────────────────────────────────
            ctrl = tk.Frame(parent, bg=PANEL)
            ctrl.pack(fill="x", padx=10, pady=(8, 4))

            tk.Label(ctrl, text="Units:", bg=PANEL, fg=TEXT_DIM,
                     font=FONT_BODY).grid(row=0, column=0, sticky="w", padx=(0, 4))
            self._dxf_units_var = tk.StringVar(value="mm")
            for col_i, unit in enumerate(["mm", "m"], start=1):
                tk.Radiobutton(ctrl, text=unit, variable=self._dxf_units_var,
                               value=unit, bg=PANEL, fg=TEXT,
                               selectcolor=BG, activebackground=PANEL,
                               font=FONT_BODY).grid(row=0, column=col_i, sticky="w", padx=2)

            self._btn_dxf_preview = tk.Button(
                ctrl, text="\u25b6 Preview",
                bg=ACCENT, fg="#ffffff", font=FONT_BODY,
                relief="flat", cursor="hand2",
                command=self._dxf_preview_async)
            self._btn_dxf_preview.grid(row=0, column=4, sticky="ew",
                                       padx=(16, 6), pady=(0, 2))

            self._btn_dxf_export = tk.Button(
                ctrl, text="\U0001f4be Export DXF",
                bg=PANEL, fg=TEXT, font=FONT_BODY,
                relief="flat", cursor="hand2",
                command=self._dxf_export)
            self._btn_dxf_export.grid(row=0, column=5, sticky="ew")

            self._dxf_status = tk.Label(
                parent,
                text="Run a design first, then click Preview or Export DXF.",
                bg=PANEL, fg=TEXT_DIM, font=FONT_BODY, anchor="w")
            self._dxf_status.pack(fill="x", padx=10, pady=(0, 4))

            # ── Layer legend ──────────────────────────────────────────────────
            legend_frame = tk.Frame(parent, bg=PANEL)
            legend_frame.pack(fill="x", padx=10, pady=(0, 4))
            legend_data = [
                ("STATOR_IRON",  "cyan",    "Stator lamination"),
                ("STATOR_SLOTS", "white",   "Slot outlines"),
                ("WINDING_A/B/C","red",     "Phase conductors"),
                ("ROTOR_IRON",   "magenta", "Rotor iron + shaft"),
                ("MAGNETS_N/S",  "#e76f51", "PM magnets N/S"),
                ("AIRGAP",       "grey",    "Midline (dashed)"),
                ("DIMENSIONS",   "yellow",  "Diameter leaders"),
                ("LABELS",       "white",   "Title block"),
            ]
            for idx, (lname, lcol, ldesc) in enumerate(legend_data):
                r, c = divmod(idx, 4)
                cell = tk.Frame(legend_frame, bg=PANEL)
                cell.grid(row=r, column=c, sticky="w", padx=(0, 18), pady=1)
                try:
                    tk.Label(cell, text="\u25a0", fg=lcol,
                             bg=PANEL, font=FONT_BODY).pack(side="left")
                except Exception:
                    tk.Label(cell, text="\u25a0", bg=PANEL,
                             font=FONT_BODY).pack(side="left")
                tk.Label(cell, text=f" {lname}: {ldesc}",
                         bg=PANEL, fg=TEXT_DIM, font=FONT_BODY).pack(side="left")

            # ── Preview figure ────────────────────────────────────────────────
            self._dxf_fig = Figure(figsize=(10, 7), facecolor="#0d0d14")
            self._dxf_canvas = FigureCanvasTkAgg(self._dxf_fig, master=parent)
            self._dxf_canvas.get_tk_widget().pack(fill="both", expand=True,
                                                  padx=6, pady=(0, 4))
            toolbar = NavigationToolbar2Tk(self._dxf_canvas, parent,
                                           pack_toolbar=False)
            toolbar.configure(bg=BG)
            toolbar.pack(side="bottom", fill="x")
            self._dxf_canvas.draw()
            self._dxf_motor_cache = None

        def _dxf_preview_async(self):
            if not self._results or "motor" not in self._results:
                self._dxf_status.config(text="\u26a0  Run a design first.")
                return
            self._btn_dxf_preview.config(state="disabled")
            self._btn_dxf_export.config(state="disabled")
            self._dxf_status.config(text="Rendering preview\u2026")
            t = threading.Thread(target=self._dxf_preview_worker, daemon=True)
            t.start()

        def _dxf_preview_worker(self):
            try:
                motor = self._results["motor"]
                self._dxf_motor_cache = motor
                self._dxf_draw_preview(motor)
                self.after(0, self._dxf_preview_done)
            except Exception as exc:
                self.after(0, lambda: self._dxf_status.config(
                    text=f"\u26a0  Preview error: {exc}"))
            finally:
                self.after(0, lambda: self._btn_dxf_preview.config(state="normal"))
                self.after(0, lambda: self._btn_dxf_export.config(state="normal"))

        def _dxf_draw_preview(self, motor):
            """Matplotlib preview matching DXF geometry exactly."""
            import math
            import matplotlib.patches as mpatches
            from matplotlib.patches import Wedge

            fig = self._dxf_fig
            fig.clear()
            fig.patch.set_facecolor("#0d0d14")
            ax = fig.add_subplot(111)
            ax.set_facecolor("#0d0d14")
            ax.set_aspect("equal")
            ax.tick_params(colors="#666666", labelsize=7)
            for sp in ax.spines.values():
                sp.set_color("#333333")
            ax.set_xlabel("mm", color="#666666", fontsize=7)
            ax.set_ylabel("mm", color="#666666", fontsize=7)

            st   = motor.stator
            spr  = st.slot_profile if st else None
            R_so = (st.outer_radius if st else 0.130) * 1000
            R_si = (st.inner_radius if st else 0.082) * 1000
            R_ro = motor.rotor_outer_radius * 1000
            t_m  = getattr(motor, "magnet_thickness", 0.006) * 1000
            R_mi = R_ro - t_m
            R_sh = motor.rotor_inner_radius * 1000
            Qs   = motor.slots
            poles = motor.poles
            alpha_p = getattr(motor, "magnet_width_fraction", 0.83)
            pole_pitch = 2 * math.pi / poles
            slot_pitch = 2 * math.pi / Qs
            h_slot = (spr.depth() if spr else 0.022) * 1000
            b_slot = (spr.area() / (spr.depth() + 1e-9) if spr else 0.008) * 1000
            b_open = (spr.opening_width() if spr else 0.003) * 1000

            slot_phase = {}; slot_dir = {}
            winding = getattr(motor, "winding", None)
            if winding:
                try:
                    for ph in range(3):
                        for cs in winding.coil_sides_for_phase(ph):
                            si = cs.slot_idx % Qs
                            slot_phase[si] = cs.phase
                            slot_dir[si]   = cs.direction
                except Exception:
                    pass

            WCOL = [("#c1121f","#ff6b6b"),("#2d6a4f","#74c69d"),("#1d3557","#457b9d")]
            PCOL = ["#e76f51","#f4a261"]

            def wedge(r0,r1,a0,a1,col,ec="none",lw=0,z=2):
                ax.add_patch(Wedge((0,0),r1,a0,a1,width=r1-r0,
                                   facecolor=col,edgecolor=ec,linewidth=lw,zorder=z))

            wedge(R_si,R_so,0,360,"#1e2d4a",z=1)
            wedge(R_sh,R_mi,0,360,"#112018",z=1)
            wedge(0,R_sh,  0,360,"#080e10",z=1)
            wedge(R_ro,R_si,0,360,"#070d18",z=2)

            hm = math.degrees(math.pi * alpha_p / poles)
            for p in range(poles):
                pc = math.degrees((p+0.5)*pole_pitch)
                wedge(R_mi,R_ro,pc-hm,pc+hm,PCOL[p%2],ec="#111",lw=0.4,z=3)
                ang = math.radians(pc); tr=(R_mi+R_ro)/2
                ax.text(tr*math.cos(ang),tr*math.sin(ang),
                        "N" if p%2==0 else "S",
                        ha="center",va="center",color="white",fontsize=6,fontweight="bold",zorder=10)

            for s in range(Qs):
                th_c   = s*slot_pitch+slot_pitch/2
                a_open = math.asin(min(b_open/(2*R_si),0.9999))
                th_l   = math.degrees(th_c-a_open); th_r = math.degrees(th_c+a_open)
                R_top  = R_si+h_slot
                a_top  = math.asin(min(b_slot/(2*R_top),0.9999))
                th_tl  = math.degrees(th_c-a_top); th_tr = math.degrees(th_c+a_top)
                ph = slot_phase.get(s,0)%3; dr = slot_dir.get(s,1)
                c0 = WCOL[ph][0] if dr>0 else WCOL[ph][1]
                c1 = WCOL[ph][1] if dr>0 else WCOL[ph][0]
                wedge(R_si-0.2,R_si+0.3,th_l,th_r,"#070d18",z=4)
                wedge(R_si+0.3,R_si+h_slot*0.48,th_l,th_r,c0,ec="#000",lw=0.15,z=5)
                wedge(R_si+h_slot*0.52,R_top-0.3,th_tl,th_tr,c1,ec="#000",lw=0.15,z=5)

            for r,col,lw,ls in [(R_so,"#4a7fc1",1.4,"-"),(R_si,"#4a7fc1",0.7,"-"),
                                 (R_ro,"#3a8a5a",0.7,"-"),(R_mi,"#667",0.4,"--"),
                                 (R_sh,"#3a8a5a",0.9,"-"),((R_ro+R_si)/2,"#223",0.4,"--")]:
                ax.add_patch(plt.Circle((0,0),r,fill=False,edgecolor=col,
                                        linewidth=lw,linestyle=ls,zorder=8))

            dc="#aaaaaa"
            for r,lbl,ang in [(R_sh,f"Ø{2*R_sh:.0f}",44),(R_mi,f"Ø{2*R_mi:.0f}",34),
                              (R_ro,f"Ø{2*R_ro:.0f}",24),(R_si,f"Ø{2*R_si:.0f}",14),
                              (R_so,f"Ø{2*R_so:.0f}",5)]:
                ang_r=math.radians(ang)
                xc,yc=r*math.cos(ang_r),r*math.sin(ang_r)
                xe,ye=(R_so*1.15)*math.cos(ang_r),(R_so*1.15)*math.sin(ang_r)
                ax.plot([xc,xe],[yc,ye],color=dc,lw=0.6,zorder=12)
                ax.text(xe+1,ye,lbl,ha="left",va="center",color=dc,fontsize=6.5,zorder=12)

            ax.annotate("",xy=(R_ro,0),xytext=(R_si,0),
                        arrowprops=dict(arrowstyle="<->",color="#ffff66",lw=0.9))
            ax.text((R_ro+R_si)/2,1.5,f"g={motor.airgap*1000:.0f}mm",
                    ha="center",va="bottom",color="#ffff66",fontsize=6)

            ax.set_xlim(-R_so*1.05, R_so*1.32); ax.set_ylim(-R_so*1.05, R_so*1.05)
            ax.set_title(f"{poles}p/{Qs}s PMSM  —  DXF Preview  "
                         f"(Ø{2*R_so:.0f}mm stator, "
                         f"{motor.stack_length*1000:.0f}mm stack)",
                         color="#e0e0e0",fontsize=10,fontweight="bold")

            items=[mpatches.Patch(color=c,label=l) for c,l in [
                ("#1e2d4a","Stator iron"),("#112018","Rotor iron"),
                ("#e76f51","PM N"),("#f4a261","PM S"),
                ("#c1121f","Ph A"),("#2d6a4f","Ph B"),("#1d3557","Ph C")]]
            ax.legend(handles=items,loc="lower right",framealpha=0.2,
                      facecolor="#1a1a2e",edgecolor="#444455",
                      labelcolor="#cccccc",fontsize=7,ncol=4)

        def _dxf_preview_done(self):
            self._dxf_canvas.draw()
            m = self._dxf_motor_cache
            if m and m.stator:
                self._dxf_status.config(
                    text=f"✔  Preview ready  —  {m.poles}p/{m.slots}s  "
                         f"Ø{m.stator.outer_radius*2000:.0f}mm stator  "
                         f"·  Click 'Export DXF' to save.")

        def _dxf_export(self):
            if not self._results or "motor" not in self._results:
                self._dxf_status.config(text="\u26a0  Run a design first.")
                return
            from tkinter.filedialog import asksaveasfilename
            motor = self._results["motor"]
            default = (f"PMSM_{motor.poles}p{motor.slots}s_"
                       f"{int(motor.rated_power/1000)}kW.dxf")
            path = asksaveasfilename(
                defaultextension=".dxf",
                filetypes=[("AutoCAD DXF", "*.dxf"), ("All files", "*.*")],
                initialfile=default,
                title="Export AutoCAD DXF")
            if not path:
                return
            try:
                self._dxf_status.config(text="Writing DXF\u2026")
                self.update_idletasks()
                from Bohemien_Motor_Designer.io.dxf_export import export_dxf
                info = export_dxf(motor, path, units=self._dxf_units_var.get())
                layer_summary = "  ".join(
                    f"{k}:{v}" for k, v in sorted(info["layers"].items()) if v > 0)
                self._dxf_status.config(
                    text=f"✔  Saved → {path}  "
                         f"({info['entities']} entities  ·  {layer_summary})")
            except Exception as exc:
                self._dxf_status.config(text=f"\u26a0  Export failed: {exc}")


        # ── 3D FEM tab ────────────────────────────────────────────────────────────

        def _build_3dfem_tab(self, parent):
            """3D FEM tab: mesh controls + result panels."""
            # ── Controls ──────────────────────────────────────────────────────
            ctrl = tk.Frame(parent, bg=PANEL)
            ctrl.pack(fill="x", padx=10, pady=(8, 4))

            def _spin(parent, label, var, lo, hi, step, col):
                tk.Label(ctrl, text=label, bg=PANEL, fg=TEXT_DIM,
                         font=FONT_BODY).grid(row=0, column=col, sticky="w", padx=(0, 3))
                tk.Spinbox(ctrl, from_=lo, to=hi, increment=step,
                           textvariable=var, width=4,
                           bg=PANEL, fg=TEXT, font=FONT_BODY,
                           buttonbackground=PANEL, relief="flat"
                           ).grid(row=0, column=col+1, sticky="w", padx=(0, 10))

            self._3d_nang  = tk.IntVar(value=4)
            self._3d_nax   = tk.IntVar(value=6)
            self._3d_nag   = tk.IntVar(value=3)
            self._3d_new   = tk.IntVar(value=2)

            _spin(ctrl, "Ang/slot:",  self._3d_nang, 4,  16, 2, 0)
            _spin(ctrl, "Axial:",     self._3d_nax,  4,  16, 2, 2)
            _spin(ctrl, "AG layers:", self._3d_nag,  2,   8, 1, 4)
            _spin(ctrl, "End-wind:",  self._3d_new,  1,   4, 1, 6)

            self._btn_3d = tk.Button(
                ctrl, text="\u25b6 Run 3D FEM",
                bg=ACCENT, fg="#ffffff", font=FONT_BODY,
                relief="flat", cursor="hand2",
                command=self._3dfem_run_async)
            self._btn_3d.grid(row=0, column=8, sticky="ew", padx=(12, 0))

            self._3dfem_status = tk.Label(
                parent,
                text="Run a design first, then click Run 3D FEM.",
                bg=PANEL, fg=TEXT_DIM, font=FONT_BODY, anchor="w")
            self._3dfem_status.pack(fill="x", padx=10, pady=(0, 4))

            # ── Progress bar ──────────────────────────────────────────────────
            self._3d_progress = ttk.Progressbar(parent, mode="indeterminate",
                                                 length=200)
            self._3d_progress.pack(fill="x", padx=10, pady=(0, 4))

            # ── Results panels ────────────────────────────────────────────────
            res_frame = tk.Frame(parent, bg=PANEL)
            res_frame.pack(fill="both", expand=True, padx=10, pady=(4, 6))

            # Left: numerical results
            left = tk.Frame(res_frame, bg=PANEL)
            left.pack(side="left", fill="both", expand=True, padx=(0, 6))
            tk.Label(left, text="3D FEM Results", bg=PANEL, fg=TEXT,
                     font=FONT_BODY).pack(anchor="w")
            self.txt_3d_results = scrolledtext.ScrolledText(
                left, bg=PANEL, fg=TEXT, font=FONT_MONO,
                relief="flat", wrap="word", state="disabled", height=14)
            self.txt_3d_results.pack(fill="both", expand=True)

            # Right: log
            right = tk.Frame(res_frame, bg=PANEL)
            right.pack(side="right", fill="both", expand=True)
            tk.Label(right, text="Solver Log", bg=PANEL, fg=TEXT,
                     font=FONT_BODY).pack(anchor="w")
            self.txt_3d_log = scrolledtext.ScrolledText(
                right, bg="#1a1a2e", fg="#90ee90", font=FONT_MONO,
                relief="flat", wrap="word", state="disabled", height=14)
            self.txt_3d_log.pack(fill="both", expand=True)

            # Note about expected run time
            note = (
                "Note: 3D FEM is compute-intensive. "
                "ang/slot=4, axial=6 takes ~60-120s per solve "
                "(3 solves for Ke+Ld+Lq). Run overnight for finer meshes."
            )
            tk.Label(parent, text=note, bg=PANEL, fg=TEXT_DIM,
                     font=FONT_BODY, wraplength=700, justify="left",
                     anchor="w").pack(fill="x", padx=10, pady=(0, 6))

            self._3d_runner = None

        def _3dfem_run_async(self):
            if not self._results or "motor" not in self._results:
                self._3dfem_status.config(text="\u26a0  Run a design first.")
                return
            self._btn_3d.config(state="disabled")
            self._3d_progress.start(12)
            self._3dfem_log_clear()
            t = threading.Thread(target=self._3dfem_worker, daemon=True)
            t.start()

        def _3dfem_worker(self):
            try:
                motor  = self._results["motor"]
                n_ang  = self._3d_nang.get()
                n_ax   = self._3d_nax.get()
                n_ag   = self._3d_nag.get()
                n_ew   = self._3d_new.get()

                self.after(0, lambda: self._3dfem_status.config(
                    text=f"Building 3D mesh (ang={n_ang}, ax={n_ax}, ag={n_ag})..."))

                from Bohemien_Motor_Designer.fea.runner3d import FEMRunner3D
                runner = FEMRunner3D(motor,
                                     n_angular_per_slot=n_ang,
                                     n_axial=n_ax,
                                     n_end_winding=n_ew,
                                     n_radial_airgap=n_ag)
                runner.build_mesh(progress_cb=self._3dfem_log)
                self._3d_runner = runner

                result = runner.run_static(progress_cb=self._3dfem_log)
                self.after(0, lambda: self._3dfem_done(result))

            except Exception as exc:
                import traceback as _tb
                msg = f"3D FEM error: {exc}"
                self.after(0, lambda: self._3dfem_status.config(text=f"\u26a0  {msg}"))
                self._3dfem_log(f"ERROR: {exc}")
                self._3dfem_log(_tb.format_exc())
            finally:
                self.after(0, lambda: self._btn_3d.config(state="normal"))
                self.after(0, lambda: self._3d_progress.stop())

        def _3dfem_log(self, msg, frac=None):
            """Append a message to the 3D FEM log box (thread-safe)."""
            def _do():
                self.txt_3d_log.config(state="normal")
                self.txt_3d_log.insert("end", msg + "\n")
                self.txt_3d_log.see("end")
                self.txt_3d_log.config(state="disabled")
                if frac is not None:
                    pct = int(frac * 100)
                    self._3dfem_status.config(
                        text=f"Solving...  {pct}%  |  {msg[:70]}")
            self.after(0, _do)

        def _3dfem_log_clear(self):
            self.txt_3d_log.config(state="normal")
            self.txt_3d_log.delete("1.0", "end")
            self.txt_3d_log.config(state="disabled")

        def _3dfem_done(self, result: dict):
            """Populate result panel after solve completes."""
            self._3d_progress.stop()

            r = result
            Ke_3d   = r.get("Ke_3d_Wb", 0)
            Ke_an   = r.get("Ke_anal_Wb", 0)
            Ld_3d   = r.get("Ld_mH", 0)
            Lq_3d   = r.get("Lq_mH", 0)
            Ld_an   = r.get("Ld_anal_mH", 0)
            Lq_an   = r.get("Lq_anal_mH", 0)
            B_mean  = r.get("B_gap_mean_T", 0)
            B_max   = r.get("B_gap_max_T", 0)
            B_an    = r.get("B_gap_anal_T", 0)
            nodes   = r.get("n_nodes", 0)
            tets    = r.get("n_tets", 0)
            edges   = r.get("n_edges", 0)
            dt      = r.get("solve_time_s", 0)
            theta_w = r.get("theta_w_deg", 0)

            ke_ratio = Ke_3d / Ke_an if Ke_an > 1e-9 else 0

            lines = [
                "\u2500\u2500\u2500 3D N\u00e9d\u00e9lec FEM Results \u2500\u2500\u2500",
                f"  Mesh:    {nodes:,} nodes  {tets:,} tets  {edges:,} edge DOFs",
                f"  Solve:   {dt:.0f}s total (mesh + 3 solves)",
                "",
                "  — Flux Linkage (Ke) —",
                f"  Ke 3D      : {Ke_3d:.5f} Wb",
                f"  Ke analyt  : {Ke_an:.5f} Wb",
                f"  Ratio      : {ke_ratio:.3f}  "
                f"({'↑ end-winding' if ke_ratio < 0.95 else 'converged'})",
                f"  \u03b8_w offset : {theta_w:.1f}\u00b0 electrical",
                "",
                "  — Inductances (Park perturbation) —",
                f"  Ld 3D      : {Ld_3d:.3f} mH    analyt: {Ld_an:.3f} mH",
                f"  Lq 3D      : {Lq_3d:.3f} mH    analyt: {Lq_an:.3f} mH",
                f"  Lq/Ld 3D   : {Lq_3d/(Ld_3d+1e-9):.3f}  "
                f"(analyt: {Lq_an/(Ld_an+1e-9):.3f})",
                "",
                "  — Airgap B-field (mid-stack) —",
                f"  |B| mean   : {B_mean:.3f} T    analyt: {B_an:.3f} T",
                f"  |B| max    : {B_max:.3f} T",
                "",
                "  Tip: increase Ang/slot for better Ke accuracy.",
                "  End-winding effects reduce Ke vs 2D by 5-15%.",
            ]

            self._set_text(self.txt_3d_results, "\n".join(lines))
            self._3dfem_status.config(
                text=f"✔  3D FEM done  —  "
                     f"Ke={Ke_3d:.4f}Wb (ratio {ke_ratio:.2f})  "
                     f"Ld={Ld_3d:.2f}mH  Lq={Lq_3d:.2f}mH  "
                     f"[{dt:.0f}s]")





    # ── Run ──────────────────────────────────────────────────────────────────
    app = MotorDesignApp()
    app.mainloop()


if __name__ == "__main__":
    main()
