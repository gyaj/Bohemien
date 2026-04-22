"""
Elmer BH curve file writer.

Writes LaminationMaterial.bh_table to the two-column ASCII format
expected by Elmer's NonLinear Materials (H_B_Curve keyword).

File format::

    # Material: M270-35A
    # H [A/m]    B [T]
    0.0           0.0
    35.0          0.1
    ...

Usage::

    from Bohemien_Motor_Designer.fea.bh_writer import write_bh_file, write_bh_files
    write_bh_file(lib.lamination("M270-35A"), "/tmp/M270-35A_BH.dat")
"""
from __future__ import annotations
from pathlib import Path
from Bohemien_Motor_Designer.materials.library import MaterialLibrary


def write_bh_file(material, path: str) -> Path:
    """
    Write a single lamination's BH curve to an Elmer-format .dat file.

    Parameters
    ----------
    material : LaminationMaterial instance (must have .bh_table set)
    path     : output file path

    Returns
    -------
    Path of written file.

    Raises
    ------
    ValueError if bh_table is None or empty.
    """
    if not material.bh_table:
        raise ValueError(
            f"Material '{material.name}' has no BH table. "
            "Add (B [T], H [A/m]) pairs to bh_table in library.py."
        )

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"! Material: {material.name}",
        f"! Steinmetz: kh={material.kh:.5g}  ke={material.ke:.5g}  n={material.steinmetz_n:.3f}",
        f"! B_sat={material.B_sat:.2f} T  mu_r_initial={material.mu_r_initial}",
        "! Format: H [A/m]  B [T]",
        f"{len(material.bh_table)}",   # Elmer expects point count on first data line
    ]

    for B, H in material.bh_table:
        lines.append(f"{H:.4f}  {B:.6f}")

    p.write_text("\n".join(lines) + "\n")
    return p


def write_bh_files(grade: str, directory: str,
                   lib: MaterialLibrary = None) -> Path:
    """
    Convenience: look up material by grade name and write BH file.

    Parameters
    ----------
    grade     : material key, e.g. 'M270-35A'
    directory : directory for output file
    lib       : MaterialLibrary instance (creates new one if None)

    Returns
    -------
    Path of written file (directory / grade + '_BH.dat').
    """
    if lib is None:
        lib = MaterialLibrary()
    mat  = lib.lamination(grade)
    path = Path(directory) / f"{grade}_BH.dat"
    return write_bh_file(mat, str(path))


def bh_table_for_elmer(material) -> list[tuple[float, float]]:
    """
    Return BH table as list of (H, B) tuples in Elmer convention.
    (Elmer's NonLinear material expects H first, then B.)
    """
    if not material.bh_table:
        raise ValueError(f"Material '{material.name}' has no BH table.")
    return [(H, B) for B, H in material.bh_table]


if __name__ == "__main__":
    import sys
    lib    = MaterialLibrary()
    grades = ["M19", "M270-35A", "M400-50A", "M800-65A", "Arnon5"]
    outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    for g in grades:
        p = write_bh_files(g, outdir, lib)
        mat = lib.lamination(g)
        print(f"Written {p}  ({len(mat.bh_table)} points)")
