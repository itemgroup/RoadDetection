"""
道路障碍检测系统 - 入口点
用法: python -m main   (在 road_detection_app 目录下)
"""

import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
