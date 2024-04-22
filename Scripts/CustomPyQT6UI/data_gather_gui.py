import os.path
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageQt
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton
from PyQt6.QtWidgets import QProgressBar, QDialog, QFileDialog
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QPixelFormat, QHideEvent, QShowEvent

from Scripts.gif import FRAMES_PER_GIF, GIFCV, convert_cv_frame_to_qt, convert_all_gifs_to_simple_json
from Scripts.CustomPyQT6UI.video_thread import VideoThread
from Scripts.CustomPyQT6UI.gif_label_input import ActionLabelInput
from Scripts.CustomPyQT6UI.gif_preview_widget import GIFPreviewWidget
from Scripts.CustomPyQT6UI.directory_select_widget import DirectorySelect

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

        self.save_dir = './'
        self.save_path = ''

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

        create_btn = QPushButton("Record New GIF [R]")
        preview_btn = QPushButton("Preview Current GIF [P]")
        save_btn = QPushButton("Save Current GIF [S]")
        delete_btn = QPushButton("Delete Current GIF [Delete]")
        save_dir_select = DirectorySelect("Select Save Directory")
        save_to_json_btn = QPushButton("Create JSON Data From Actions")

        self.progress_bar = QProgressBar()
        self.trainingLabelInput = ActionLabelInput()

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

        save_dir_select.directory_selected_signal.connect(self.update_save_dir)

        save_to_json_btn.clicked.connect(self.create_json_from_dir)

        # create a vertical box layout and add the two labels
        video_layout.addWidget(self.progress_bar)
        video_layout.addWidget(self.image_label)
        video_layout.addWidget(save_dir_select)
        video_layout.addWidget(self.trainingLabelInput)
        video_layout.addWidget(save_to_json_btn)

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
        if key == Qt.Key.Key_R:
            self.start_recording()
        elif key == Qt.Key.Key_S:
            self.save_img()
        elif key == Qt.Key.Key_Delete:
            self.clear_current()
        elif key == Qt.Key.Key_P:
            self.preview_current_gif()

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
            dialog = GIFPreviewWidget(self.curr_gif, self.frame_time_milli, self.curr_action_label)  # TODO
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

    @pyqtSlot(str)
    def update_save_dir(self, dir_path: str):
        self.save_dir = dir_path
        self.update_current_label(self.curr_action_label)

    @pyqtSlot(str)
    def update_current_label(self, new_label: str):
        self.curr_action_label = new_label.split(".gif", 1)[0]
        self.save_path = self.save_dir + '\\' + self.curr_action_label + '\\'
        self.curr_img_index = check_action_directory(self.save_path)
        print(f"New save directory:\n\t{self.save_path}")
        print(f"\tCurrent Data Count: {self.curr_img_index}")
        self.clear_current()

    def create_json_from_dir(self):
        working_dir = os.path.dirname(self.save_dir)
        print(get_num_gifs_in_dir(self.save_dir))
        if get_num_gifs_in_dir(self.save_dir) == 0:
            print("Current action directory is empty!")
            return
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        file_dialog.setNameFilter("*.json")
        file_name, ok = file_dialog.getSaveFileName(self,
                                                    caption="JSON Save DATA",
                                                    directory=working_dir,
                                                    filter='JSON (*.json)'
                                                    )

        if ok:
            convert_all_gifs_to_simple_json(self.save_dir, file_name)


def get_num_gifs_in_dir(dir_path: str) -> int:
    count = 0
    full_dir_path = os.path.abspath(dir_path)
    for path in os.listdir(full_dir_path):
        full_path = full_dir_path + '\\' + path
        if os.path.isdir(full_path):
            count += get_num_gifs_in_dir(full_path)
        elif full_path.endswith('.gif'):
            count += 1
    return count


def check_action_directory(dir_name: str) -> int:
    if not os.path.exists(dir_name):
        if not os.path.exists(os.path.dirname(dir_name)):
            os.mkdir(os.path.dirname(dir_name))
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        return 0
    return get_num_gifs_in_dir(dir_name)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = DataGatherApp()
    a.show()
    sys.exit(app.exec())
