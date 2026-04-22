"""Entry point for Bohemien_Motor_Designer — script and PyInstaller exe."""
import sys, os

if hasattr(sys, "_MEIPASS"):
    sys.path.insert(0, sys._MEIPASS)
else:
    _here = os.path.dirname(os.path.abspath(__file__))  # Bohemien_Motor_Designer/
    sys.path.insert(0, os.path.dirname(_here))           # parent
    sys.path.insert(0, _here)

from Bohemien_Motor_Designer.gui.app import main
main()
