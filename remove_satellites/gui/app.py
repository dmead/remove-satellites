"""GUI entry point — `remove-satellites gui`."""

from __future__ import annotations

import sys


def run(output: str | None = None) -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("PySide6 is not installed — install the GUI extra:\n"
              "  uv tool install 'remove-satellites[gui]'", file=sys.stderr)
        return 2

    if sys.platform == "win32":
        # own AppUserModelID so the taskbar doesn't group us under python.exe
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "remove-satellites.gui")

    from .window import MainWindow

    qt_app = QApplication.instance() or QApplication(sys.argv)
    qt_app.setApplicationName("remove-satellites")
    win = MainWindow(output)
    win.show()
    return qt_app.exec()
