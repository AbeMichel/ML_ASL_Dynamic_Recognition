import sys
from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout


class ModelSummaryDialog(QDialog):
    def __init__(self,  summary_string: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Summary")
        main_layout = QVBoxLayout()

        main_label = QLabel(summary_string)

        main_layout.addWidget(main_label)

        self.setLayout(main_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = ModelSummaryDialog("Hello")
    dialog.exec()

