from PyQt6.QtWidgets import QWidget, QFileDialog, QLineEdit, QPushButton, QHBoxLayout
from PyQt6.QtCore import pyqtSignal


class FileSelect(QWidget):
    file_selected_signal = pyqtSignal(str)

    def __init__(self, btn_label: str, file_filter='', title='Open file'):
        super().__init__()
        self.init_ui(btn_label)
        self.file_path = ''
        self.file_filter = file_filter
        self.title = title

    def init_ui(self, btn_label: str):
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        file_line = QLineEdit()
        select_file_btn = QPushButton(btn_label)

        main_layout.addWidget(file_line, stretch=2)
        main_layout.addWidget(select_file_btn, stretch=1)

        select_file_btn.clicked.connect(self.select_file)
        self.file_selected_signal.connect(lambda: file_line.setText(self.file_path))

    def select_file(self):
        file_dialog = QFileDialog(self)
        path, ok = file_dialog.getOpenFileName(self,
                                               self.title,
                                               directory='./',
                                               filter=self.file_filter)
        if ok:
            self.file_path = path
            self.file_selected_signal.emit(self.file_path)
