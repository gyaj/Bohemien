# No imports at module level — keeps this importable without tkinter.
__all__ = ["main", "MotorDesignApp"]

def main():
    from Bohemien_Motor_Designer.gui.app import main as _m
    _m()
