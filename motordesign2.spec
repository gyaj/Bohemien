# -*- mode: python ; coding: utf-8 -*-
"""
MotorDesign2 PyInstaller spec.

HOW TO BUILD
============
Option A — double-click  build_exe.bat  (recommended)

Option B — manual:
  1. Extract the zip. You get a folder called  motordesign2\
  2. Open a terminal in the directory that CONTAINS motordesign2\
     (the parent, NOT inside it)
  3. Run:
         pyinstaller motordesign2\motordesign2.spec --clean --noconfirm
  4. Exe at:  dist\MotorDesign2.exe

NOTE: SPECPATH is a PyInstaller built-in that gives the directory
containing the spec file (i.e. motordesign2\).  Its parent is the
directory that contains motordesign2\ as a package — that is what
must be on sys.path for  import motordesign2  to work.
"""
from pathlib import Path

block_cipher = None

# SPECPATH is injected by PyInstaller — it is the directory containing
# this spec file, which is the motordesign2\ package folder itself.
# Its parent is the directory that holds motordesign2\ as a sub-folder,
# which is what Python needs on sys.path to resolve  import motordesign2.
_PARENT = str(Path(SPECPATH).parent.resolve())

_MD2 = [
    'motordesign2',
    'motordesign2.analysis',
    'motordesign2.analysis.losses',
    'motordesign2.analysis.performance',
    'motordesign2.core',
    'motordesign2.core.geometry',
    'motordesign2.core.geometry.rotor',
    'motordesign2.core.geometry.slot_profiles',
    'motordesign2.core.geometry.stator',
    'motordesign2.core.geometry.winding',
    'motordesign2.core.induction',
    'motordesign2.core.manufacturing_report',
    'motordesign2.core.motor',
    'motordesign2.core.pmsm',
    'motordesign2.core.specs',
    'motordesign2.core.synrel',
    'motordesign2.drive',
    'motordesign2.drive.field_weakening',
    'motordesign2.drive.inverter',
    'motordesign2.examples',
    'motordesign2.examples.full_design_example',
    'motordesign2.fea',
    'motordesign2.fea.bh_writer',
    'motordesign2.fea.fem_mesh',
    'motordesign2.fea.fem_solver',
    'motordesign2.fea.fem_torque',
    'motordesign2.fea.gmsh_exporter',
    'motordesign2.fea.index_registry',
    'motordesign2.fea.mesh3d',
    'motordesign2.fea.mesh_reader',
    'motordesign2.fea.mesh_viz',
    'motordesign2.fea.py_mesh',
    'motordesign2.fea.py_runner',
    'motordesign2.fea.py_solver',
    'motordesign2.fea.py_torque',
    'motordesign2.fea.python_runner',
    'motordesign2.fea.results_reader',
    'motordesign2.fea.rotor_rotation',
    'motordesign2.fea.runner',
    'motordesign2.fea.runner3d',
    'motordesign2.fea.sif_generator',
    'motordesign2.fea.solver',
    'motordesign2.fea.solver3d',
    'motordesign2.fea.test_export',
    'motordesign2.fea.torque',
    'motordesign2.gui',
    'motordesign2.gui.app',
    'motordesign2.io',
    'motordesign2.io.dxf_export',
    'motordesign2.io.json_spec',
    'motordesign2.materials',
    'motordesign2.materials.library',
    'motordesign2.run_gui',
    'motordesign2.scaling',
    'motordesign2.scaling.similarity',
    'motordesign2.setup',
    'motordesign2.thermal',
    'motordesign2.thermal.cooling',
    'motordesign2.thermal.lumped_model',
    'motordesign2.utils',
    'motordesign2.utils.validation',
]

_SCI = [
    'scipy.sparse', 'scipy.sparse.linalg',
    'scipy.sparse.linalg._dsolve', 'scipy.sparse.linalg._dsolve.SuperLU',
    'scipy.sparse._compressed', 'scipy.sparse._csr', 'scipy.sparse._csc',
    'scipy.sparse._coo', 'scipy.linalg.lapack', 'scipy.linalg.blas',
    'scipy.optimize',
    'matplotlib', 'matplotlib.pyplot', 'matplotlib.figure',
    'matplotlib.backends.backend_tkagg', 'matplotlib.backends._backend_tk',
    'matplotlib.collections', 'matplotlib.patches',
    'tkinter', 'tkinter.ttk', 'tkinter.scrolledtext',
    'tkinter.messagebox', 'tkinter.filedialog', '_tkinter',
    'PIL', 'PIL.Image', 'PIL.ImageDraw', 'PIL.ImageFilter',
]

a = Analysis(
    [str(Path(SPECPATH) / 'run_gui.py')],
    pathex=[_PARENT],
    binaries=[],
    datas=[],
    hiddenimports=_MD2 + _SCI,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest', 'jupyter', 'IPython', 'pandas', 'cv2'],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],
    name='MotorDesign2',
    debug=False, strip=False, upx=True, upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    target_arch=None,
)
