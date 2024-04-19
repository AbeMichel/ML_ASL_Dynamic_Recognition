from PyQt6.QtWidgets import QWidget, QFileDialog, QLineEdit, QPushButton, QHBoxLayout
from PyQt6.QtCore import pyqtSignal


class DirectorySelect(QWidget):
    directory_selected_signal = pyqtSignal(str)

    def __init__(self, btn_label: str):
        super().__init__()
        self.init_ui(btn_label)
        self.last_accessed_dir = ''

    def init_ui(self, btn_label: str):
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        file_line = QLineEdit()
        select_file_btn = QPushButton(btn_label)

        main_layout.addWidget(file_line, stretch=2)
        main_layout.addWidget(select_file_btn, stretch=1)

        select_file_btn.clicked.connect(self.select_directory)
        self.directory_selected_signal.connect(lambda: file_line.setText(self.last_accessed_dir))

    def select_directory(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        if file_dialog.exec():
            self.last_accessed_dir = file_dialog.selectedFiles()[0]
            self.directory_selected_signal.emit(self.last_accessed_dir)
