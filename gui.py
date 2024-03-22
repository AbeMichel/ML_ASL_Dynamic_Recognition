import os.path
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageQt
# import ml_classification
import hand_recognition
import pose_recognition
import utils
from utils import ACTION_DIRECTORY, check_action_directory, create_gif
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton
from PyQt6.QtWidgets import QProgressBar, QDialog
from PyQt6.QtCore import QTimer, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QPixelFormat

SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self._last_img = None

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                self._last_img = pil_img
                cv_img = hand_recognition.draw_landmarks(pil_img)
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

    def get_last_img(self):
        return self._last_img

    def save_current(self):
        if self._last_img is None:
            return
        utils.create_gif([self._last_img], "testingGUI.gif")


class GIFPreviewWidget(QDialog):
    def __init__(self, gif_images, frame_duration):
        super().__init__()
        self.gif_images = gif_images
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
        self.use_cached = False
        self.cached_frames = []

    def keyPressEvent(self, a0) -> None:
        key = a0.key()
        if key == Qt.Key.Key_Escape:
            self.done(QDialog.DialogCode.Accepted)

    def update_frame(self):
        if self.use_cached is True:
            pixmap = self.cached_frames[self.current_frame]
        else:
            frame_image = self.gif_images[self.current_frame]
            frame_image = hand_recognition.draw_landmarks(frame_image)  # draws the landmarks for the hand recognition
            tensor_image = utils.convert_pil_qt_tensor(frame_image)
            frame_image = pose_recognition.analyze_image_from_image(tensor_image)
            frame_image = Image.fromarray(frame_image.astype('uint8'))#.convert('BGR')
            pixmap = utils.convert_pil_qt(frame_image)
            self.cached_frames.append(pixmap)
        self.label.setPixmap(pixmap)

        self.current_frame += 1
        if self.current_frame >= len(self.gif_images):
            self.current_frame = 0
            self.use_cached = True
            # self.done(QDialog.DialogCode.Accepted)


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


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Create Training Data")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.images = []
        self.curr_img_index = 0
        self.curr_action_label = ''
        self.max_frames = 20
        self.frame_time_milli = 100

        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(640, 480)

        # create a text label
        self.textLabel = QPushButton('Save')
        self.textLabel.clicked.connect(self.save_img)
        self.textLabel.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # create a line edit to set the training label
        self.trainingLabelInput = ActionLabelInput()
        self.trainingLabelInput.label_updated_signal.connect(self.update_current_label)
        self.trainingLabelInput.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, self.max_frames)
        self.progressBar.setValue(0)

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.progressBar)
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        vbox.addWidget(self.trainingLabelInput)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create a timer for capturing actions
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.capture_frame_recording)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

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
        self.progressBar.setValue(self.progressBar.maximum())
        while self.progressBar.value() > 0:
            time.sleep(self.frame_time_milli / 2000)
            self.progressBar.setValue(self.progressBar.value() - 1)

        self.timer.start(self.frame_time_milli)

    def preview_current_gif(self):
        if len(self.images) > 0:
            dialog = GIFPreviewWidget(self.images, self.frame_time_milli)
            dialog.exec()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = utils.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def save_img(self):
        if len(self.images) > 0:
            utils.create_gif(self.images, self.curr_action_label + str(self.curr_img_index) + '.gif')
            self.curr_img_index += 1
            self.images.clear()

    def capture_frame_recording(self):
        if len(self.images) < self.max_frames:
            self.capture_frame()
            self.progressBar.setValue(len(self.images))
        else:
            self.timer.stop()

    def capture_frame(self):
        img = self.thread.get_last_img()
        if img is not None:
            self.images.append(img)

    def clear_current(self):
        self.images.clear()
        self.progressBar.setValue(0)

    @pyqtSlot(str)
    def update_current_label(self, new_label: str):
        self.curr_action_label, self.curr_img_index = check_action_directory(new_label)
        self.clear_current()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec())
