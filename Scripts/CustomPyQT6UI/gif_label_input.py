from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QPushButton
from PyQt6.QtCore import pyqtSignal


class ActionLabelInput(QWidget):
    label_updated_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        main_layout = QHBoxLayout()
        self.input_box = QLineEdit()
        self.update_label_btn = QPushButton("Set Label")
        self.update_label_btn.clicked.connect(self.update_label)

        main_layout.addWidget(self.input_box)
        main_layout.addWidget(self.update_label_btn)

        self.setLayout(main_layout)

    def update_label(self):
        new_label = self.input_box.text()
        self.label_updated_signal.emit(new_label)
