import sys
import time
import numpy as np
from contextlib import redirect_stdout
from io import StringIO
from PyQt6.QtWidgets import QWidget, QApplication, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QProgressBar, \
    QFileDialog
from PyQt6.QtCore import Qt, pyqtSlot, QTimer
from PyQt6.QtGui import QHideEvent, QShowEvent

from Scripts.gif import GIFCV, convert_cv_frame_to_qt, FRAMES_PER_GIF
from Scripts.machine_learning_model import predict_gif, load_model_labels_and_metrics, plot_model_metrics
from Scripts.CustomPyQT6UI.video_thread import VideoThread
from Scripts.CustomPyQT6UI.gif_preview_widget import GIFPreviewWidget
from Scripts.CustomPyQT6UI.tf_summary_dialog import ModelSummaryDialog

class PredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.metrics: dict = None
        self.thread: VideoThread = None
        self.model = None
        self.encoder = None
        self.curr_gif: GIFCV = GIFCV()
        self.frame_time_milli = int(1500 / FRAMES_PER_GIF)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.capture_frame_recording)
        self.progress_bar = QProgressBar()
        self.image_label = QLabel()
        self.model_loading_status = QLabel()
        self.prediction_status = QLabel()
        self.metrics_status = QLabel()
        self.summary_status = QLabel()
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
        view_metrics_btn = QPushButton("View Model Metrics")
        view_summary_btn = QPushButton("View Model Summary")
        record_btn = QPushButton("Record GIF [R]")
        prediction_btn = QPushButton("Predict [Space]")

        # create layouts
        main_layout = QVBoxLayout()

        # set main layout
        self.setLayout(main_layout)

        # add widgets to layouts
        main_layout.addWidget(self.model_loading_status)
        main_layout.addWidget(load_model_btn, stretch=1)
        main_layout.addWidget(self.metrics_status)
        main_layout.addWidget(view_metrics_btn)
        main_layout.addWidget(self.summary_status)
        main_layout.addWidget(view_summary_btn)
        main_layout.addWidget(record_btn)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.prediction_status)
        main_layout.addWidget(prediction_btn, stretch=1)

        # set widget attributes
        load_model_btn.clicked.connect(self.load_model_from_folder)
        record_btn.clicked.connect(self.start_recording)
        prediction_btn.clicked.connect(self.predict_on_new)
        view_metrics_btn.clicked.connect(self.view_model_metrics)
        view_summary_btn.clicked.connect(self.view_model_summary)

        self.progress_bar.setRange(0, FRAMES_PER_GIF)
        self.progress_bar.setValue(0)

        # create layout hierarchy

    def load_model_from_folder(self):
        dialog = QFileDialog(self)
        self.metrics_status.setText("")
        self.summary_status.setText("")
        path = dialog.getExistingDirectory(self,
                                           caption="Select model folder",
                                           directory="./")
        if path:
            self.model, self.encoder, self.metrics = load_model_labels_and_metrics(path)
            if self.model is not None:
                self.model_loading_status.setText(f"Model successfully loaded with classes: {self.encoder.classes_}")
            else:
                self.model_loading_status.setText("Folder does not contain a keras model.")
        else:
            self.model_loading_status.setText("No valid path selected.")

    def predict_on_new(self):
        if self.curr_gif is None or self.curr_gif.frame_count() == 0:
            self.prediction_status.setText("No GIF to predict on")
            return
        elif self.model is None or self.encoder is None:
            self.prediction_status.setText("Model or encoder has not been loaded")
            return

        prediction = predict_gif(self.curr_gif.to_simple(), self.model, self.encoder)
        self.prediction_status.setText(f"You most likely signed '{prediction}'")

    def view_model_metrics(self):
        if self.model is None:
            self.metrics_status.setText("No model loaded.")
            return
        elif self.metrics is None:
            self.metrics_status.setText("No metrics saved with model.")
            return
        self.metrics_status.setText("")
        plot_model_metrics(self.metrics)

    def view_model_summary(self):
        if self.model is None:
            self.metrics_status.setText("No model loaded.")
            return
        self.metrics_status.setText("")
        summary_output = StringIO()
        with redirect_stdout(summary_output):
            self.model.summary()
        summary_string = summary_output.getvalue()
        dialog = ModelSummaryDialog(summary_string, self)
        dialog.exec()

    def hideEvent(self, a0: QHideEvent):
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
        super().hideEvent(a0)

    def showEvent(self, a0: QShowEvent):
        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()
        super().showEvent(a0)

    def keyPressEvent(self, a0) -> None:
        key = a0.key()
        if key == Qt.Key.Key_R:
            self.start_recording()
        elif key == Qt.Key.Key_P:
            self.preview_current_gif()
        elif key == Qt.Key.Key_Space:
            self.predict_on_new()

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
        event.accept()

    def start_recording(self):
        if self.timer.isActive():
            self.timer.stop()
        self.clear_current()

        # do a countdown
        self.progress_bar.setValue(self.progress_bar.maximum())
        while self.progress_bar.value() > 0:
            time.sleep(self.frame_time_milli / 2000)
            self.progress_bar.setValue(self.progress_bar.value() - 1)

        self.timer.start(self.frame_time_milli)

    def preview_current_gif(self):
        if self.curr_gif.frame_count() > 0:
            dialog = GIFPreviewWidget(self.curr_gif, self.frame_time_milli, "GIF Preview")  # TODO
            dialog.exec()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = convert_cv_frame_to_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def save_img(self):
        if self.curr_gif.frame_count() > 0:
            gif_save_path = self.save_path + '\\' + str(self.curr_img_index) + '.gif'
            self.curr_gif.save_gif(gif_save_path)
            self.curr_img_index += 1
            print(f"\tCurrent Data Count: {self.curr_img_index}")
            self.clear_current()

    def capture_frame_recording(self):
        if self.curr_gif.frame_count() < FRAMES_PER_GIF:
            self.capture_frame()
            self.progress_bar.setValue(self.curr_gif.frame_count())
        else:
            self.timer.stop()

    def capture_frame(self):
        img = self.thread.get_last_img()
        if img is not None:
            self.curr_gif.add_frame(img)

    def clear_current(self):
        self.curr_gif = GIFCV()
        self.progress_bar.setValue(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = PredictionApp()
    sys.exit(app.exec())
