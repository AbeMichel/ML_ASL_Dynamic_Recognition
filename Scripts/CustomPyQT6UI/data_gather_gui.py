import os.path
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageQt
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton
from PyQt6.QtWidgets import QProgressBar, QDialog
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QPixelFormat, QHideEvent, QShowEvent

from Scripts.gif import FRAMES_PER_GIF, ACTION_DIRECTORY, GIFCV, convert_cv_frame_to_qt
from Scripts.CustomPyQT6UI.video_thread import VideoThread
from Scripts.CustomPyQT6UI.gif_label_input import ActionLabelInput
from Scripts.CustomPyQT6UI.gif_preview_widget import GIFPreviewWidget

SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480


class DataGatherApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Create Training Data")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.curr_gif: GIFCV = GIFCV()
        self.curr_img_index = 0
        self.curr_action_label = ''

        self.save_dir = ''

        self.total_action_time = 1.5
        # self.frame_time_milli = 100
        self.frame_time_milli = int(self.total_action_time * 1000 / FRAMES_PER_GIF)
        # print(self.frame_time_milli)
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(SCREEN_WIDTH, SCREEN_HEIGHT)

        # create the side bar to do the training actions
        main_layout = QHBoxLayout()  # Main layout
        video_layout = QVBoxLayout()  # Layout with video and label selection
        action_layout = QVBoxLayout()  # Layout with all the action buttons

        train_btn = QPushButton("Train New Model")
        create_btn = QPushButton("Record New GIF [Space]")
        preview_btn = QPushButton("Preview Current GIF [P]")
        save_btn = QPushButton("Save Current GIF [S]")
        delete_btn = QPushButton("Delete Current GIF [Delete]")

        self.progress_bar = QProgressBar()
        self.trainingLabelInput = ActionLabelInput()

        train_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # TODO

        create_btn.clicked.connect(self.start_recording)
        create_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        preview_btn.clicked.connect(self.preview_current_gif)
        preview_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        save_btn.clicked.connect(self.save_img)
        save_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        delete_btn.clicked.connect(self.clear_current)
        delete_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.progress_bar.setRange(0, FRAMES_PER_GIF)
        self.progress_bar.setValue(0)

        self.trainingLabelInput.label_updated_signal.connect(self.update_current_label)
        self.trainingLabelInput.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # create a vertical box layout and add the two labels
        video_layout.addWidget(self.progress_bar)
        video_layout.addWidget(self.image_label)
        video_layout.addWidget(self.trainingLabelInput)

        action_layout.addWidget(train_btn)
        action_layout.addWidget(create_btn)
        action_layout.addWidget(preview_btn)
        action_layout.addWidget(save_btn)
        action_layout.addWidget(delete_btn)

        main_layout.addLayout(video_layout)
        main_layout.addLayout(action_layout)

        # set the vbox layout as the widgets layout
        self.setLayout(main_layout)

        # create a timer for capturing actions
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.capture_frame_recording)

        self.thread = None

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
        if key == Qt.Key.Key_Space:
            self.start_recording()
        elif key == Qt.Key.Key_S:
            self.save_img()
        elif key == Qt.Key.Key_Delete:
            self.clear_current()
        elif key == Qt.Key.Key_P:
            self.preview_current_gif()

    def closeEvent(self, event):
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
            dialog = GIFPreviewWidget(self.curr_gif, self.frame_time_milli, self.curr_action_label)  # TODO
            dialog.exec()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = convert_cv_frame_to_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def save_img(self):
        if self.curr_gif.frame_count() > 0:
            gif_save_path = self.save_dir + '\\' + str(self.curr_img_index) + '.gif'
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

    @pyqtSlot(str)
    def update_current_label(self, new_label: str):
        self.save_dir = os.path.abspath('.\\') + '\\' + ACTION_DIRECTORY + '\\' + new_label.split(".gif", 1)[0] + '\\'
        self.curr_img_index = check_action_directory(self.save_dir)
        print(f"New save directory:\n\t{self.save_dir}")
        print(f"\tCurrent Data Count: {self.curr_img_index}")
        self.clear_current()


def check_action_directory(dir_name: str) -> int:
    if not os.path.exists(dir_name):
        if not os.path.exists(os.path.dirname(dir_name)):
            os.mkdir(os.path.dirname(dir_name))
        os.mkdir(dir_name)
        return 0
    i = 0
    for file in os.listdir(dir_name):
        # print(file)
        if os.path.isfile(dir_name + file):
            i += 1
    return i


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = DataGatherApp()
    a.show()
    sys.exit(app.exec())
