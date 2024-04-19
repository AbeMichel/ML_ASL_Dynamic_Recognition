import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480


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
            cv_img = cv2.resize(cv_img, (SCREEN_WIDTH, SCREEN_HEIGHT))
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            if ret:
                self._last_img = cv_img
                # cv_img = hand_recognition.draw_landmarks(pil_img)
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

    def get_last_img(self):
        return self._last_img
