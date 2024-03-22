import os.path
import numpy as np
from PIL import Image, ImageDraw, ImageTk, ImageSequence
from PyQt6.QtGui import QImage, QPixmap
import cv2
import PySimpleGUI as sg
import tensorflow as tf

ACTION_DIRECTORY = "Actions"


def check_action_directory(path: str) -> tuple[str, int]:
    dir_name = ACTION_DIRECTORY + '/' + path.strip(".gif") + '/'
    if not os.path.exists(dir_name):
        if not os.path.exists(ACTION_DIRECTORY):
            os.mkdir(ACTION_DIRECTORY)
        os.mkdir(dir_name)
        return dir_name, 0
    i = 0
    for file in os.listdir(dir_name):
        print(file)
        if os.path.isfile(dir_name + file):
            i += 1
    return dir_name, i


def get_actions() -> dict[str, list[str]]:  # returns label and paths
    if not os.path.exists(ACTION_DIRECTORY):
        return {}
    out = {}
    for action_dir in os.listdir(ACTION_DIRECTORY):
        rel_path = f'{ACTION_DIRECTORY}/{action_dir}'
        if os.path.isdir(rel_path):
            for action in os.listdir(rel_path):
                action_rel_path = f'{rel_path}/{action}'
                if os.path.isfile(action_rel_path):
                    if action_dir not in out.keys():
                        out[action_dir] = []
                    out[action_dir].append(action_rel_path)
    return out


def create_gif(images: list[Image], save_name: str) -> str:
    if images is None or len(images) == 0:
        return ''
    images[0].save(save_name,
                   save_all=True, append_images=images[1:],
                   optimize=False, duration=10,
                   loop=0)
    return save_name


def display_gif(gif_file_path: str):
    layout = [[sg.Image(key='-IMAGE-')]]
    window = sg.Window(gif_file_path, layout, element_justification='c', margins=(0, 0), element_padding=(0, 0),
                       finalize=True, location=(0, 0))
    # window.close_destroys_window = True
    interframe_duration = Image.open(gif_file_path).info['duration'] * 2
    done = False
    while True:
        for frame in ImageSequence.Iterator(Image.open(gif_file_path)):
            event, values = window.read(timeout=interframe_duration)
            if event == sg.WIN_CLOSED:
                done = True
                break
            window['-IMAGE-'].update(data=ImageTk.PhotoImage(frame))
        if done:
            break


def convert_cv_qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv_img  # cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    p = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(p)


def convert_pil_qt(pil_img):
    img_array = np.array(pil_img)
    h, w, ch = img_array.shape
    q_image = QImage(img_array.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q_image)


def convert_pil_qt_tensor(img):
    # Convert OpenCV frame to PIL image if needed
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    # Convert PIL image to TensorFlow tensor
    image_tensor = tf.convert_to_tensor(np.array(img))
    # print(type(image_tensor))
    # Decode JPEG image
    # decoded_image = tf.io.decode_jpeg(image_tensor, channels=3)

    return image_tensor

