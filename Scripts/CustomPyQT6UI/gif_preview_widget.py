from PyQt6.QtWidgets import QLabel, QVBoxLayout, QDialog
from PyQt6.QtCore import QTimer, Qt
from Scripts.gif import GIFCV


class GIFPreviewWidget(QDialog):
    def __init__(self, gif_instance: GIFCV, frame_duration, gif_title):
        super().__init__()
        self.setWindowTitle(gif_title)
        self.gif = gif_instance
        self.current_frame = 0
        self.cached_frames = []
        self.use_cached = False

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.update_frame()
        self.timer.start(frame_duration)  # Update frame every 100 milliseconds

    def closeEvent(self, a0):
        self.cached_frames = []

    def keyPressEvent(self, a0) -> None:
        key = a0.key()
        if key == Qt.Key.Key_Escape:
            self.done(QDialog.DialogCode.Accepted)

    def update_frame(self):
        if len(self.cached_frames) == 0:
            self.cached_frames = self.gif.get_cached_pixmap()

        pixmap = self.cached_frames[self.current_frame]
        self.label.setPixmap(pixmap)

        self.current_frame += 1
        if self.current_frame >= len(self.cached_frames):
            self.current_frame = 0
