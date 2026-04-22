"""
Material library — laminates, magnets, conductors, coolants.

All properties are temperature-dependent where significant.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ── Lamination steel ──────────────────────────────────────────────────────────

@dataclass
class LaminationMaterial:
    """
    Electrical steel lamination.

    Loss model: modified Bertotti three-term
      p(B, f) = kh * f * B^n  +  ke * (f*B)²  +  ka * (f*B)^1.5   [W/kg]
    """
    name:             str
    density:          float   # kg/m³
    resistivity:      float   # Ω·m  at 20°C
    kh:               float   # hysteresis coefficient [W/(kg·Hz·T^n)]
    ke:               float   # eddy current coefficient [W/(kg·(Hz·T)²)]
    ka:               float   # excess loss coefficient
    steinmetz_n:      float   # hysteresis exponent (typically 1.6–2.0)
    B_sat:            float   # saturation flux density [T]
    mu_r_initial:     float   # initial relative permeability
    thickness:        float   # standard lamination thickness [m]

    # B-H curve as (B [T], mu_r) pairs — for nonlinear FEA
    bh_table: Optional[list[tuple[float, float]]] = field(default=None, repr=False)

    def loss_density(self, B: float, f: float,
                     harmonics: Optional[list[tuple]] = None) -> float:
        """
        Iron loss density [W/kg] at flux density B [T] and frequency f [Hz].

        If harmonics is provided as [(order, B_amplitude)], the loss from
        each harmonic is summed (important for PWM-fed motors).
        """
        if harmonics:
            return sum(
                self.kh * (h * f) * Bh**self.steinmetz_n +
                self.ke * (h * f * Bh)**2 +
                self.ka * (h * f * Bh)**1.5
                for h, Bh in harmonics
            )
        B  = min(B, self.B_sat)
        Ph = self.kh * f * B**self.steinmetz_n
        Pe = self.ke * (f * B)**2
        Pa = self.ka * (f * B)**1.5
        return Ph + Pe + Pa

    def mu_r_at(self, B: float) -> float:
        """Relative permeability at operating flux density.

        When a BH table is present the table stores (B [T], H [A/m]) pairs.
        mu_r is derived as  mu_r = B / (mu_0 * H), clamped to mu_r_initial
        at very low B to avoid division-by-zero.
        """
        MU0 = 4e-7 * np.pi
        if self.bh_table:
            Bs, Hs = zip(*self.bh_table)
            H_interp = float(np.interp(B, Bs, Hs))
            if H_interp < 1.0:          # avoid /0 near origin
                return float(self.mu_r_initial)
            return B / (MU0 * H_interp)
        # Simplified Froehlich model (fallback when no BH table)
        return self.mu_r_initial / (1 + B / self.B_sat * self.mu_r_initial / 2000)


@dataclass
class MagnetMaterial:
    """
    Permanent magnet material.

    Properties at reference temperature (usually 20°C).
    Temperature coefficients allow correction to operating temperature.
    """
    name:            str
    remanence_Br:    float   # T at T_ref
    coercivity_Hc:   float   # kA/m at T_ref  (BHc, not HcJ)
    HcJ:             float   # kA/m intrinsic coercivity at T_ref
    energy_product:  float   # kJ/m³
    mu_r:            float   # relative permeability (1.03–1.10)
    density:         float   # kg/m³
    T_ref:           float   # reference temperature [°C] (usually 20)
    alpha_Br:        float   # reversible temp coeff of Br [%/°C] (neg)
    alpha_Hcj:       float   # reversible temp coeff of HcJ [%/°C] (neg)
    T_max:           float   # max operating temperature [°C]
    grade:           str     = ""  # e.g. "N42SH", "SmCo26"

    def Br_at(self, T_C: float) -> float:
        """Remanence at temperature T_C [°C]."""
        dT = T_C - self.T_ref
        return self.remanence_Br * (1 + self.alpha_Br / 100 * dT)

    def Hcj_at(self, T_C: float) -> float:
        """Intrinsic coercivity at temperature T_C [°C]."""
        dT = T_C - self.T_ref
        return self.HcJ * (1 + self.alpha_Hcj / 100 * dT)

    def is_demagnetised(self, B_min: float, T_C: float) -> bool:
        """
        Check if magnet is irreversibly demagnetised.
        B_min = minimum field seen by magnet (may be negative under fault).
        """
        Hcj = self.Hcj_at(T_C)
        Br  = self.Br_at(T_C)
        # Knee point: B_knee = Br - mu0*mu_r*HcJ (simplified)
        B_knee = Br - 4e-7 * np.pi * self.mu_r * Hcj * 1e3
        return B_min < B_knee


@dataclass
class ConductorMaterial:
    """Electrical conductor (copper, aluminium, or Litz wire)."""
    name:              str
    resistivity_20C:   float   # Ω·m at 20°C
    temp_coeff:        float   # 1/°C (alpha)
    density:           float   # kg/m³
    specific_heat:     float   # J/(kg·K)
    thermal_cond:      float   # W/(m·K)
    conductor_type:    str = "solid"   # "solid" | "litz" | "hairpin"

    def resistivity_at(self, T_C: float) -> float:
        return self.resistivity_20C * (1 + self.temp_coeff * (T_C - 20))

    def ac_factor_dowell(self, height: float, freq: float,
                          n_layers: int = 1) -> float:
        """
        AC resistance factor Rac/Rdc using Dowell's method.
        height : conductor height in slot [m].
        freq   : fundamental or harmonic frequency [Hz].
        """
        rho   = self.resistivity_20C   # approximate
        delta = np.sqrt(rho / (np.pi * freq * 4e-7 * np.pi))   # skin depth
        xi    = height / (delta + 1e-12)
        if xi < 0.01:
            return 1.0
        F  = xi * (np.sinh(2*xi) + np.sin(2*xi)) / (np.cosh(2*xi) - np.cos(2*xi) + 1e-12)
        G  = 2*xi * (np.sinh(xi) - np.sin(xi)) / (np.cosh(xi) + np.cos(xi) + 1e-12)
        return F + (n_layers**2 - 1) / 3.0 * G


@dataclass
class CoolantMaterial:
    """Fluid coolant properties for thermal modelling."""
    name:          str
    density:       float   # kg/m³
    specific_heat: float   # J/(kg·K)
    viscosity:     float   # Pa·s (dynamic)
    thermal_cond:  float   # W/(m·K)
    prandtl:       float   # dimensionless

    def nusselt_turbulent(self, reynolds: float) -> float:
        """Dittus-Boelter correlation for turbulent internal flow."""
        if reynolds < 2300:
            return 3.66   # laminar
        return 0.023 * reynolds**0.8 * self.prandtl**0.4

    def heat_transfer_coeff(self, velocity_m_s: float,
                             hydraulic_diameter: float) -> float:
        """Convection coefficient h [W/m²·K] for internal flow."""
        Re = self.density * velocity_m_s * hydraulic_diameter / (self.viscosity + 1e-12)
        Nu = self.nusselt_turbulent(Re)
        return Nu * self.thermal_cond / (hydraulic_diameter + 1e-12)


# ── Material Library ─────────────────────────────────────────────────────────

class MaterialLibrary:
    """
    Central material database.

    Access via:
        lib = MaterialLibrary()
        steel  = lib.lamination("M270-35A")
        magnet = lib.magnet("N42SH")
        copper = lib.conductor("copper")
        water  = lib.coolant("water-glycol-50")
    """

    _LAMINATES: dict[str, LaminationMaterial] = {}
    _MAGNETS:   dict[str, MagnetMaterial]     = {}
    _CONDUCTORS:dict[str, ConductorMaterial]  = {}
    _COOLANTS:  dict[str, CoolantMaterial]    = {}

    def __init__(self):
        if not self._LAMINATES:
            self._populate()

    @classmethod
    def _populate(cls):
        # ── Lamination steels ──────────────────────────────────────────
        # Steinmetz coefficients fitted to IEC 60404 loss data:
        # P [W/kg] = kh * f * B^n  +  ke * (f*B)^2  +  ka * (f*B)^1.5
        # Derived from P(1T,50Hz) and P(1T,400Hz) datasheet targets.
        # BH tables: list of (B [T], H [A/m]) pairs.
        # Data from IEC 60404-2 / manufacturer datasheets, validated against
        # published loss curves.  Format matches Elmer's two-column BH file.
        # Each table runs from (0,0) through deep saturation to ~4 T / 10 MA/m
        # so that FEM solvers never extrapolate outside the table.
        _BH_M19 = [
            (0.0,    0.0),   (0.1,   28.0),  (0.2,   36.0),  (0.3,   43.0),
            (0.4,   48.0),   (0.5,   53.0),  (0.6,   59.0),  (0.7,   67.0),
            (0.8,   78.0),   (0.9,   93.0),  (1.0,  115.0),  (1.1,  152.0),
            (1.2,  220.0),   (1.3,  400.0),  (1.4, 1020.0),  (1.5, 2650.0),
            (1.59, 5200.0),  (1.64, 6800.0), (1.67, 7900.0), (1.72, 9600.0),
            (1.86,19500.0),  (1.92,29000.0), (1.99,48000.0), (2.09,97000.0),
            (2.23,195000.0), (2.37,295000.0),(4.0, 10000000.0),
        ]
        _BH_M270 = [
            (0.0,    0.0),   (0.1,   35.0),  (0.2,   44.0),  (0.3,   53.0),
            (0.4,   59.0),   (0.5,   65.0),  (0.6,   72.0),  (0.7,   82.0),
            (0.8,   95.0),   (0.9,  113.0),  (1.0,  140.0),  (1.1,  183.0),
            (1.2,  265.0),   (1.3,  485.0),  (1.4, 1230.0),  (1.5, 3200.0),
            (1.6, 7800.0),   (1.7,16000.0),  (1.8,30000.0),  (1.9,60000.0),
            (2.0,120000.0),  (2.1,200000.0), (4.0,10000000.0),
        ]
        _BH_M400 = [
            (0.0,    0.0),   (0.1,   42.0),  (0.2,   54.0),  (0.3,   64.0),
            (0.4,   72.0),   (0.5,   80.0),  (0.6,   89.0),  (0.7,  102.0),
            (0.8,  118.0),   (0.9,  141.0),  (1.0,  174.0),  (1.1,  227.0),
            (1.2,  328.0),   (1.3,  600.0),  (1.4, 1520.0),  (1.5, 3900.0),
            (1.6, 9500.0),   (1.7,19500.0),  (1.8,36000.0),  (1.9,70000.0),
            (2.0,130000.0),  (4.0,10000000.0),
        ]
        _BH_M800 = [
            (0.0,    0.0),   (0.1,   55.0),  (0.2,   70.0),  (0.3,   83.0),
            (0.4,   94.0),   (0.5,  105.0),  (0.6,  118.0),  (0.7,  134.0),
            (0.8,  156.0),   (0.9,  186.0),  (1.0,  230.0),  (1.1,  300.0),
            (1.2,  433.0),   (1.3,  790.0),  (1.4, 2000.0),  (1.5, 5100.0),
            (1.6,12500.0),   (1.7,25500.0),  (1.8,47000.0),  (1.9,90000.0),
            (2.05,180000.0), (4.0,10000000.0),
        ]
        _BH_ARNON5 = [
            (0.0,    0.0),   (0.1,   22.0),  (0.2,   28.0),  (0.3,   34.0),
            (0.4,   39.0),   (0.5,   44.0),  (0.6,   50.0),  (0.7,   57.0),
            (0.8,   67.0),   (0.9,   80.0),  (1.0,   99.0),  (1.1,  130.0),
            (1.2,  188.0),   (1.3,  342.0),  (1.4,  870.0),  (1.5, 2260.0),
            (1.6, 5540.0),   (1.7,11300.0),  (1.8,21000.0),  (1.85,32000.0),
            (4.0,10000000.0),
        ]

        cls._LAMINATES = {
            "M19": LaminationMaterial(
                name="M19", density=7650, resistivity=52e-8,
                kh=0.03414, ke=7.71e-5, ka=1.54e-4, steinmetz_n=1.70,
                B_sat=2.05, mu_r_initial=8000, thickness=0.00047,
                bh_table=_BH_M19),
            "M270-35A": LaminationMaterial(
                name="M270-35A", density=7650, resistivity=52e-8,
                kh=0.05029, ke=7.43e-5, ka=1.49e-4, steinmetz_n=1.75,
                B_sat=2.0, mu_r_initial=7000, thickness=0.00035,
                bh_table=_BH_M270),
            "M400-50A": LaminationMaterial(
                name="M400-50A", density=7700, resistivity=48e-8,
                kh=0.07286, ke=1.43e-4, ka=2.86e-4, steinmetz_n=1.80,
                B_sat=2.0, mu_r_initial=6000, thickness=0.00050,
                bh_table=_BH_M400),
            "M800-65A": LaminationMaterial(
                name="M800-65A", density=7700, resistivity=45e-8,
                kh=0.14893, ke=2.21e-4, ka=4.43e-4, steinmetz_n=1.85,
                B_sat=2.05, mu_r_initial=5000, thickness=0.00065,
                bh_table=_BH_M800),
            "Arnon5": LaminationMaterial(
                name="Arnon5", density=7650, resistivity=65e-8,
                kh=0.01543, ke=1.14e-5, ka=2.29e-5, steinmetz_n=1.65,
                B_sat=1.85, mu_r_initial=12000, thickness=0.00005,
                bh_table=_BH_ARNON5),
        }

        # ── Permanent magnets ──────────────────────────────────────────
        cls._MAGNETS = {
            "N35": MagnetMaterial(
                name="N35", grade="N35",
                remanence_Br=1.17, coercivity_Hc=880, HcJ=955,
                energy_product=263, mu_r=1.05, density=7500,
                T_ref=20, alpha_Br=-0.12, alpha_Hcj=-0.60, T_max=80),
            "N42": MagnetMaterial(
                name="N42", grade="N42",
                remanence_Br=1.29, coercivity_Hc=980, HcJ=1114,
                energy_product=318, mu_r=1.05, density=7500,
                T_ref=20, alpha_Br=-0.12, alpha_Hcj=-0.60, T_max=80),
            "N42SH": MagnetMaterial(
                name="N42SH", grade="N42SH",
                remanence_Br=1.29, coercivity_Hc=980, HcJ=2388,
                energy_product=318, mu_r=1.05, density=7500,
                T_ref=20, alpha_Br=-0.12, alpha_Hcj=-0.30, T_max=150),
            "N48": MagnetMaterial(
                name="N48", grade="N48",
                remanence_Br=1.38, coercivity_Hc=1040, HcJ=1114,
                energy_product=367, mu_r=1.05, density=7500,
                T_ref=20, alpha_Br=-0.12, alpha_Hcj=-0.60, T_max=80),
            "N52": MagnetMaterial(
                name="N52", grade="N52",
                remanence_Br=1.45, coercivity_Hc=1060, HcJ=1114,
                energy_product=398, mu_r=1.05, density=7500,
                T_ref=20, alpha_Br=-0.12, alpha_Hcj=-0.60, T_max=60),
            "SmCo26": MagnetMaterial(
                name="SmCo26", grade="SmCo26",
                remanence_Br=1.05, coercivity_Hc=796, HcJ=1990,
                energy_product=207, mu_r=1.03, density=8300,
                T_ref=20, alpha_Br=-0.03, alpha_Hcj=-0.15, T_max=350),
            "Ferrite-Y30": MagnetMaterial(
                name="Ferrite-Y30", grade="Ferrite-Y30",
                remanence_Br=0.39, coercivity_Hc=200, HcJ=225,
                energy_product=27, mu_r=1.05, density=4900,
                T_ref=20, alpha_Br=-0.20, alpha_Hcj=+0.27, T_max=250),
        }

        # ── Conductors ─────────────────────────────────────────────────
        cls._CONDUCTORS = {
            "copper": ConductorMaterial(
                name="copper", resistivity_20C=1.72e-8,
                temp_coeff=0.00393, density=8960,
                specific_heat=385, thermal_cond=400),
            "copper-75C": ConductorMaterial(
                name="copper-75C", resistivity_20C=2.09e-8,
                temp_coeff=0.00393, density=8960,
                specific_heat=385, thermal_cond=400),
            "aluminium": ConductorMaterial(
                name="aluminium", resistivity_20C=2.82e-8,
                temp_coeff=0.00429, density=2700,
                specific_heat=900, thermal_cond=237),
        }

        # ── Coolants ────────────────────────────────────────────────────
        cls._COOLANTS = {
            "water-glycol-50": CoolantMaterial(
                name="water-glycol-50 (50/50)",
                density=1065, specific_heat=3480,
                viscosity=2.0e-3, thermal_cond=0.42, prandtl=14.0),
            "water": CoolantMaterial(
                name="water", density=998,
                specific_heat=4182, viscosity=1.0e-3,
                thermal_cond=0.60, prandtl=7.0),
            "oil-ATF": CoolantMaterial(
                name="ATF oil", density=870,
                specific_heat=1900, viscosity=20e-3,
                thermal_cond=0.14, prandtl=250.0),
            "air": CoolantMaterial(
                name="air", density=1.20,
                specific_heat=1005, viscosity=1.8e-5,
                thermal_cond=0.026, prandtl=0.71),
        }

    # ── Access methods ────────────────────────────────────────────────────

    def lamination(self, key: str) -> LaminationMaterial:
        if key not in self._LAMINATES:
            raise KeyError(f"Unknown lamination '{key}'. "
                           f"Available: {list(self._LAMINATES)}")
        return self._LAMINATES[key]

    def magnet(self, key: str) -> MagnetMaterial:
        if key not in self._MAGNETS:
            raise KeyError(f"Unknown magnet '{key}'. "
                           f"Available: {list(self._MAGNETS)}")
        return self._MAGNETS[key]

    def conductor(self, key: str = "copper") -> ConductorMaterial:
        if key not in self._CONDUCTORS:
            raise KeyError(f"Unknown conductor '{key}'. "
                           f"Available: {list(self._CONDUCTORS)}")
        return self._CONDUCTORS[key]

    def coolant(self, key: str = "water-glycol-50") -> CoolantMaterial:
        if key not in self._COOLANTS:
            raise KeyError(f"Unknown coolant '{key}'. "
                           f"Available: {list(self._COOLANTS)}")
        return self._COOLANTS[key]

    def summary(self) -> str:
        return (
            f"MaterialLibrary\n"
            f"  Laminates  : {sorted(self._LAMINATES)}\n"
            f"  Magnets    : {sorted(self._MAGNETS)}\n"
            f"  Conductors : {sorted(self._CONDUCTORS)}\n"
            f"  Coolants   : {sorted(self._COOLANTS)}"
        )
