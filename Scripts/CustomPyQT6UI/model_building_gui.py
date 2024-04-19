import sys
from PyQt6.QtWidgets import QWidget, QApplication, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QSpinBox, \
    QDoubleSpinBox
from PyQt6.QtCore import Qt, pyqtSlot
from Scripts.CustomPyQT6UI.directory_select_widget import DirectorySelect
from Scripts.CustomPyQT6UI.file_select_widget import FileSelect
from Scripts.machine_learning_model import create_model_from_json_path, save_model_and_labels


class ModelBuildApp(QWidget):
    def __init__(self):
        super().__init__()
        self.epoch_spinbox: QSpinBox = QSpinBox()
        self.batch_size_spinbox: QSpinBox = QSpinBox()
        self.val_split_spinbox: QDoubleSpinBox = QDoubleSpinBox()
        self.save_model_status_label = None
        self.build_model_status_label = None
        self.current_model = None
        self.current_encoder = None
        self.current_json_file_path = ''
        self.current_save_dir = ''
        self.init_ui()

        self.show()

    def init_ui(self):
        self.setWindowTitle("ASL Recognition")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.resize(640, 640)

        # create the needed UI elements
        json_select = FileSelect("Select JSON File", "JSON (*.json)", "Open JSON Data")
        self.build_model_status_label = QLabel()
        build_model_btn = QPushButton("Build Model")
        save_directory_select = DirectorySelect("Select Save Directory")
        self.save_model_status_label = QLabel()
        save_model_btn = QPushButton("Save Model")

        batch_label = QLabel("Batch Size:")
        epoch_label = QLabel("Number of Epochs:")
        val_split_label = QLabel("Val Split:")

        # create layouts
        main_layout = QVBoxLayout()
        model_settings_layout = QHBoxLayout()

        # set main layout
        self.setLayout(main_layout)

        # add to layouts

        main_layout.addWidget(json_select, stretch=1)
        main_layout.addLayout(model_settings_layout)
        main_layout.addWidget(self.build_model_status_label, stretch=1)
        main_layout.addWidget(build_model_btn, stretch=1)

        main_layout.addWidget(save_directory_select, stretch=1)
        main_layout.addWidget(self.save_model_status_label, stretch=1)
        main_layout.addWidget(save_model_btn, stretch=1)
        main_layout.addWidget(QLabel(), stretch=10)

        model_settings_layout.addWidget(batch_label, alignment=Qt.AlignmentFlag.AlignLeft)
        model_settings_layout.addWidget(self.batch_size_spinbox, alignment=Qt.AlignmentFlag.AlignLeft)
        model_settings_layout.addWidget(epoch_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        model_settings_layout.addWidget(self.epoch_spinbox, alignment=Qt.AlignmentFlag.AlignHCenter)
        model_settings_layout.addWidget(val_split_label, alignment=Qt.AlignmentFlag.AlignRight)
        model_settings_layout.addWidget(self.val_split_spinbox, alignment=Qt.AlignmentFlag.AlignRight)

        # set widget attributes
        build_model_btn.clicked.connect(self.build_curr_model)
        save_model_btn.clicked.connect(self.save_curr_model)

        json_select.file_selected_signal.connect(self.set_json_path)
        save_directory_select.directory_selected_signal.connect(self.set_save_dir)

        self.batch_size_spinbox.setValue(15)
        self.epoch_spinbox.setValue(20)
        self.val_split_spinbox.setValue(0.4)

        self.batch_size_spinbox.setMinimum(1)
        self.epoch_spinbox.setMinimum(1)
        self.val_split_spinbox.setRange(0.01, 0.99)

    @pyqtSlot(str)
    def set_json_path(self, json_path: str):
        self.current_json_file_path = json_path
        self.build_model_status_label.setText("")

    @pyqtSlot(str)
    def set_save_dir(self, dir_path: str):
        self.current_save_dir = dir_path
        self.save_model_status_label.setText("")

    def build_curr_model(self):
        if not self.current_json_file_path.endswith('.json'):
            self.build_model_status_label.setText("No json file selected!")
            return
        self.build_model_status_label.setText("Building model... please wait")
        self.current_model, self.current_encoder, metrics = create_model_from_json_path(self.current_json_file_path,
                                                                                        self.batch_size_spinbox.value(),
                                                                                        self.epoch_spinbox.value(),
                                                                                        self.val_split_spinbox.value())
        num_decimal_places = 3
        self.build_model_status_label.setText(f"Model built successfully:"
                                              f"\n\tAccuracy = {round(metrics.history['accuracy'][-1], num_decimal_places)}"
                                              f"\n\tLoss = {round(metrics.history['loss'][-1], num_decimal_places)}"
                                              f"\n\tValidation Accuracy = {round(metrics.history['val_accuracy'][-1], num_decimal_places)}"
                                              f"\n\tValidation Loss = {round(metrics.history['val_loss'][-1], num_decimal_places)}")

    def save_curr_model(self):
        if self.current_model is None:
            self.save_model_status_label.setText("No model to save!")
            return
        elif self.current_save_dir == '':
            self.save_model_status_label.setText("No save directory selected!")
            return
        self.save_model_status_label.setText("")
        model_folder = save_model_and_labels(self.current_model, self.current_encoder, self.current_save_dir)
        self.save_model_status_label.setText(f"Model saved successfully to: {model_folder}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = ModelBuildApp()
    sys.exit(app.exec())
