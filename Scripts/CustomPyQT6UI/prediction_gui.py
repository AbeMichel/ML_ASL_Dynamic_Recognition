import sys

from PyQt6.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt6.QtCore import Qt


class PredictionApp(QWidget):
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
        load_model_btn = QPushButton("Load Model")
        prediction_btn = QPushButton("Predict")

        # create layouts
        main_layout = QVBoxLayout()

        # add widgets to layouts
        main_layout.addWidget(prediction_btn, stretch=1)
        # create layout hierarchy

        # set main layout
        self.setLayout(main_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = PredictionApp()
    sys.exit(app.exec())
