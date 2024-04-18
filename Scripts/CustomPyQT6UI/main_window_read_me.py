import sys

from PyQt6.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class ReadMeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.show()

    def init_ui(self):
        self.setWindowTitle("ASL Recognition")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.resize(640, 640)

        # create the needed UI elements
        title_label = QLabel("ASL Recognition")
        title_font = QFont()
        # create layouts
        main_layout = QVBoxLayout()

        # add widgets to layouts
        main_layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignTop)
        # create layout hierarchy

        # set main layout
        self.setLayout(main_layout)

        # change non-widget settings and attributes
        title_font.setBold(True)
        title_font.setPointSize(16)

        # change widget settings and attributes
        title_label.setFont(title_font)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = ReadMeWindow()
    sys.exit(app.exec())
