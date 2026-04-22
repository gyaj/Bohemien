# -*- mode: python ; coding: utf-8 -*-
"""
Bohemien_Motor_Designer PyInstaller spec.

HOW TO BUILD
============
Option A — double-click  build_exe.bat  (recommended)

Option B — manual:
  1. Extract the zip. You get a folder called  Bohemien_Motor_Designer\
  2. Open a terminal in the directory that CONTAINS Bohemien_Motor_Designer\
     (the parent, NOT inside it)
  3. Run:
         pyinstaller Bohemien_Motor_Designer\Bohemien_Motor_Designer.spec --clean --noconfirm
  4. Exe at:  dist\Bohemien_Motor_Designer.exe

NOTE: SPECPATH is a PyInstaller built-in that gives the directory
containing the spec file (i.e. Bohemien_Motor_Designer\).  Its parent is the
directory that contains Bohemien_Motor_Designer\ as a package — that is what
must be on sys.path for  import Bohemien_Motor_Designer  to work.
"""
from pathlib import Path

block_cipher = None

# SPECPATH is injected by PyInstaller — it is the directory containing
# this spec file, which is the Bohemien_Motor_Designer\ package folder itself.
# Its parent is the directory that holds Bohemien_Motor_Designer\ as a sub-folder,
# which is what Python needs on sys.path to resolve  import Bohemien_Motor_Designer.
_PARENT = str(Path(SPECPATH).parent.resolve())

_MD2 = [
    'Bohemien_Motor_Designer',
    'Bohemien_Motor_Designer.analysis',
    'Bohemien_Motor_Designer.analysis.losses',
    'Bohemien_Motor_Designer.analysis.performance',
    'Bohemien_Motor_Designer.core',
    'Bohemien_Motor_Designer.core.geometry',
    'Bohemien_Motor_Designer.core.geometry.rotor',
    'Bohemien_Motor_Designer.core.geometry.slot_profiles',
    'Bohemien_Motor_Designer.core.geometry.stator',
    'Bohemien_Motor_Designer.core.geometry.winding',
    'Bohemien_Motor_Designer.core.induction',
    'Bohemien_Motor_Designer.core.manufacturing_report',
    'Bohemien_Motor_Designer.core.motor',
    'Bohemien_Motor_Designer.core.pmsm',
    'Bohemien_Motor_Designer.core.specs',
    'Bohemien_Motor_Designer.core.synrel',
    'Bohemien_Motor_Designer.drive',
    'Bohemien_Motor_Designer.drive.field_weakening',
    'Bohemien_Motor_Designer.drive.inverter',
    'Bohemien_Motor_Designer.examples',
    'Bohemien_Motor_Designer.examples.full_design_example',
    'Bohemien_Motor_Designer.fea',
    'Bohemien_Motor_Designer.fea.bh_writer',
    'Bohemien_Motor_Designer.fea.fem_mesh',
    'Bohemien_Motor_Designer.fea.fem_solver',
    'Bohemien_Motor_Designer.fea.fem_torque',
    'Bohemien_Motor_Designer.fea.gmsh_exporter',
    'Bohemien_Motor_Designer.fea.index_registry',
    'Bohemien_Motor_Designer.fea.mesh3d',
    'Bohemien_Motor_Designer.fea.mesh_reader',
    'Bohemien_Motor_Designer.fea.mesh_viz',
    'Bohemien_Motor_Designer.fea.py_mesh',
    'Bohemien_Motor_Designer.fea.py_runner',
    'Bohemien_Motor_Designer.fea.py_solver',
    'Bohemien_Motor_Designer.fea.py_torque',
    'Bohemien_Motor_Designer.fea.python_runner',
    'Bohemien_Motor_Designer.fea.results_reader',
    'Bohemien_Motor_Designer.fea.rotor_rotation',
    'Bohemien_Motor_Designer.fea.runner',
    'Bohemien_Motor_Designer.fea.runner3d',
    'Bohemien_Motor_Designer.fea.sif_generator',
    'Bohemien_Motor_Designer.fea.solver',
    'Bohemien_Motor_Designer.fea.solver3d',
    'Bohemien_Motor_Designer.fea.test_export',
    'Bohemien_Motor_Designer.fea.torque',
    'Bohemien_Motor_Designer.gui',
    'Bohemien_Motor_Designer.gui.app',
    'Bohemien_Motor_Designer.io',
    'Bohemien_Motor_Designer.io.dxf_export',
    'Bohemien_Motor_Designer.io.json_spec',
    'Bohemien_Motor_Designer.materials',
    'Bohemien_Motor_Designer.materials.library',
    'Bohemien_Motor_Designer.run_gui',
    'Bohemien_Motor_Designer.scaling',
    'Bohemien_Motor_Designer.scaling.similarity',
    'Bohemien_Motor_Designer.setup',
    'Bohemien_Motor_Designer.thermal',
    'Bohemien_Motor_Designer.thermal.cooling',
    'Bohemien_Motor_Designer.thermal.lumped_model',
    'Bohemien_Motor_Designer.utils',
    'Bohemien_Motor_Designer.utils.validation',
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
    name='Bohemien_Motor_Designer',
    debug=False, strip=False, upx=True, upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    target_arch=None,
)
