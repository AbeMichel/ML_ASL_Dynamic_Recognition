import json
import os
import numpy as np
from PIL import Image, ImageTk
from PyQt6.QtGui import QImage, QPixmap
import tkinter as tk

from Scripts import hand_recognition
from Scripts.hand_recognition import get_landmarks

DEFAULT_POINT_VALUE: int = 9999999  # The value we will put in place of "missing" data
FRAMES_PER_GIF: int = 30
POINTS_PER_FRAME: int = 63


class GIF:
    def __init__(self) -> None:
        self._class = ''

    def set_class(self, class_name: str) -> None:
        self._class = class_name

    def get_class(self) -> str:
        return self._class


class GIFSimple(GIF):
    def __init__(self):
        super().__init__()
        self._data: list[int] = []

    def get_data(self) -> list[int]:
        return self._data

    def set_data(self, data: list[int]) -> None:
        self._data = data


class GIFJSON(GIF):
    def __init__(self) -> None:
        super().__init__()
        self._frames: list[list[list[int]]] = []

    def to_simple(self) -> GIFSimple:
        simple_frames: list[int] = []

        # Ensure there are the correct amount of frames inside the gif
        while len(self._frames) < FRAMES_PER_GIF:
            self._frames.append([])

        for landmarks in self._frames:
            while len(landmarks) < POINTS_PER_FRAME:
                landmarks.append([DEFAULT_POINT_VALUE, DEFAULT_POINT_VALUE])
            if (len(landmarks)) > POINTS_PER_FRAME:
                print(f"LANDMARK: {len(landmarks)}")

            for point in landmarks:
                x = point[0]
                y = point[1]
                simple_frames.append(x)
                simple_frames.append(y)
            # print(f"Number of hands: {len(simple_frames) / 21}")
            # print(f"Number of points: {len(simple_frames)}")
            # print(f"Number of default points: {simple_frames.count(DEFAULT_POINT_VALUE)}")
            # print(f"% are default: {simple_frames.count(DEFAULT_POINT_VALUE) / len(simple_frames)}")
        simple_gif = GIFSimple()
        simple_gif.set_class(self.get_class())
        simple_gif.set_data(simple_frames)
        return simple_gif

    def get_frames(self) -> list:
        return self._frames

    def set_frames(self, frames: list) -> None:
        self._frames = frames


class GIFQT(GIF):
    def __init__(self):
        super().__init__()
        self._frames: list[QPixmap] = []

    def get_frames(self) -> list[QPixmap]:
        return self._frames

    def get_frame(self, index: int) -> QPixmap | None:
        if index >= self.frame_count():
            return None
        return self._frames[index]

    def set_frames(self, frames: list[QPixmap]) -> None:
        self._frames = frames

    def frame_count(self) -> int:
        return len(self._frames)


class GIFCV(GIF):
    def __init__(self) -> None:
        super().__init__()
        self._frames: list[np.ndarray] = []
        self._cached_pixmap: list[QPixmap] = []

    def get_frames(self) -> list[np.ndarray]:
        return self._frames

    def get_cached_pixmap(self) -> list[QPixmap]:
        if len(self._cached_pixmap) == 0:
            pil_gif = self.to_pil()
            self._cached_pixmap = pil_gif.get_cached_pixmap()
        return self._cached_pixmap

    def get_frame(self, index: int) -> np.ndarray | None:
        if index >= self.frame_count():
            return None
        return self._frames[index]

    def set_frames(self, frames: list[np.ndarray]) -> None:
        self._frames = frames

    def add_frame(self, frame: np.ndarray) -> None:
        self._frames.append(frame)

    def frame_count(self) -> int:
        return len(self._frames)

    def to_pil(self):# -> GIFPIL:
        new_frames = []
        for frame in self._frames:
            new_frame = Image.fromarray(frame)
            new_frames.append(new_frame)
        pil_gif = GIFPIL()
        pil_gif.set_class(self.get_class())
        pil_gif.set_frames(new_frames)
        return pil_gif

    def to_json(self) -> GIFJSON:
        pil_gif: GIFPIL = self.to_pil()
        return pil_gif.to_json()

    def to_simple(self) -> GIFSimple:
        return self.to_json().to_simple()

    def to_qt(self) -> GIFQT:
        pil_gif: GIFPIL = self.to_pil()
        return pil_gif.to_qt()

    def save_gif(self, path: str) -> None:
        pil_gif: GIFPIL = self.to_pil()
        pil_gif.save_gif(path)

    def flip(self):  # -> GIFCV:
        return self.to_pil().flip().to_cv()


class GIFPIL(GIF):
    def __init__(self):
        super().__init__()
        self._frames: list[Image] = []
        self._cached_pixmap: list[QPixmap] = []

    def get_frames(self) -> list[Image]:
        return self._frames

    def get_cached_pixmap(self) -> list[QPixmap]:
        if len(self._cached_pixmap) == 0:
            for frame in self._frames:
                img_arr = hand_recognition.draw_landmarks(frame)
                pil_img = Image.fromarray(img_arr.astype('uint8'))
                pixmap = convert_pil_frame_to_qt(pil_img)
                self._cached_pixmap.append(pixmap)
        return self._cached_pixmap

    def get_frame(self, index: int) -> Image:
        if index >= self.frame_count():
            return None
        return self._frames[index]

    def set_frames(self, frames: list[Image]) -> None:
        self._frames = frames

    def add_frame(self, frame: Image) -> None:
        self._frames.append(frame)

    def frame_count(self) -> int:
        return len(self._frames)

    def to_cv(self) -> GIFCV:
        cv_frames = []
        for pil_frame in self._frames:
            img_array = np.array(pil_frame)
            cv_frame = img_array  # [:, :, ::-1].copy()
            cv_frames.append(cv_frame)
        cv_gif = GIFCV()
        cv_gif.set_class(self.get_class())
        cv_gif.set_frames(cv_frames)
        return cv_gif

    def to_json(self) -> GIFJSON:
        json_frames = []
        for pil_frame in self._frames:
            np_frame = np.array(pil_frame)
            if len(np_frame.shape) == 3:
                landmarks = get_landmarks(np_frame)
            else:
                landmarks = []
            json_frames.append(landmarks)
        json_gif = GIFJSON()
        json_gif.set_class(self.get_class())
        json_gif.set_frames(json_frames)
        return json_gif

    def to_qt(self) -> GIFQT:
        qt_frames = []
        for frame in self._frames:
            img_arr = np.array(frame)
            h, w, ch = img_arr.shape
            q_img = QImage(img_arr.data, w, h, ch * w, QImage.Format.Format_RGB888)
            qt_frames.append(QPixmap.fromImage(q_img))
        qt_gif = GIFQT()
        qt_gif.set_class(self.get_class())
        qt_gif.set_frames(qt_frames)
        return qt_gif

    def save_gif(self, path: str) -> None:
        if self._frames is None or self.frame_count() == 0:
            return
        print(f"Number of frames saved: {self.frame_count()}")
        self._frames[0].save(
                            path,
                            save_all=True,
                            append_images=self._frames[1:],
                            optimize=False,
                            duration=10,
                            loop=0
        )

    def flip(self): # -> GIFPIL:
        flipped_frames: list[Image] = []
        for frame in self._frames:
            flipped = frame.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_frames.append(flipped)
        pil_gif = GIFPIL()
        pil_gif.set_class(self.get_class())
        pil_gif.set_frames(flipped_frames)
        return pil_gif


def get_action_gifs(directory_path: str) -> dict[str, list[str]]:  # returns label and paths
    if not os.path.exists(directory_path):
        return {}
    out = {}
    for action_dir in os.listdir(directory_path):
        rel_path = f'{directory_path}/{action_dir}'
        if os.path.isdir(rel_path):
            for action in os.listdir(rel_path):
                action_rel_path = f'{rel_path}/{action}'
                if os.path.isfile(action_rel_path):
                    if action_dir not in out.keys():
                        out[action_dir] = []
                    out[action_dir].append(os.path.abspath(action_rel_path))
    return out


def load_from_json(path: str) -> dict[str, list[GIFSimple]]:
    out: dict[str, list[GIFSimple]] = {}
    json_data: dict = {}
    with open(path, 'r') as f:
        json_data = json.load(f)
    for gif_class, gifs in json_data.items():
        if gif_class not in out.keys():
            out[gif_class] = []
        for gif_data in gifs:
            simple_gif = GIFSimple()
            simple_gif.set_class(gif_class)
            simple_gif.set_data(gif_data)
            out[gif_class].append(simple_gif)
    return out


def save_to_json_from_jsons(gifs: list[GIFJSON], path: str) -> None:
    simple_gifs: list[GIFSimple] = []
    for gif in gifs:
        simple_gifs.append(gif.to_simple())
    save_to_json_from_simples(simple_gifs, path)


def save_to_json_from_simples(gifs: list[GIFSimple], path: str) -> None:
    if not path.endswith('.json'):
        path += ".json"
    json_dict: dict[str, list[list[int]]] = {}
    for gif in gifs:
        if gif.get_class() not in json_dict.keys():
            json_dict[gif.get_class()] = []
        json_dict[gif.get_class()].append(gif.get_data())
    with open(path, 'w') as f:
        json.dump(json_dict, f, indent=4)


def convert_pil_frame_to_qt(pil_img):
    img_array = np.array(pil_img)
    h, w, ch = img_array.shape
    q_image = QImage(img_array.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q_image)


def convert_cv_frame_to_qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv_img  # cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    p = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(p)


def convert_pil_or_cv_to_tensor(img):
    # Convert OpenCV frame to PIL image if needed
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    # Convert PIL image to TensorFlow tensor
    image_tensor = tf.convert_to_tensor(np.array(img))
    # print(type(image_tensor))
    # Decode JPEG image
    # decoded_image = tf.io.decode_jpeg(image_tensor, channels=3)

    return image_tensor


def open_gif_as_pil(gif_file_path: str, debug: bool) -> GIFPIL:
    gif = Image.open(gif_file_path)

    # Extract frames from the GIF
    pil_gif = GIFPIL()
    while True:
        try:
            # Copy the current frame and append it to the frames list
            pil_gif.add_frame(gif.copy())
            gif.seek(gif.tell() + 1)
        except EOFError:
            break
    if debug:
        print(f"Successfully loaded GIF:\n\tPath: {gif_file_path}\n\tFrames: {pil_gif.frame_count()}")
    # Close the GIF file
    gif.close()
    return pil_gif


def display_gif_from_path(gif_file_path: str):
    gif = open_gif_as_pil(gif_file_path, True)
    display_gif(gif.get_frames())


def display_gif(frames: list[Image]):
    tk_window = tk.Tk()
    tk_label = tk.Label(tk_window)
    tk_label.pack()

    def update_tk_label(idx):
        frame = frames[idx]
        photo = ImageTk.PhotoImage(frame)
        tk_label.config(image=photo)
        tk_label.image = photo
        tk_window.after(100, update_tk_label, (idx + 1) % len(frames))

    update_tk_label(0)
    tk_window.mainloop()


def convert_all_gifs_to_simple_json(gif_dir: str, save_path: str, add_flipped: bool = False):
    action_dict = get_action_gifs(gif_dir)
    simple_gifs: list[GIFSimple] = []
    for label, gifs in action_dict.items():
        print(label)
        for gif_path in gifs:
            gif = open_gif_as_pil(gif_path, False)
            gif.set_class(label)
            json_gif = gif.to_json()
            simple_gifs.append(json_gif.to_simple())
            if add_flipped:
                flipped = gif.flip()
                flipped_json = flipped.to_json()
                simple_gifs.append(flipped_json.to_simple())

    save_to_json_from_simples(simple_gifs, save_path)


if __name__ == "__main__":
    tmp_path = os.path.abspath("./") + "/" + "Actions_New"
    print(tmp_path)
    convert_all_gifs_to_simple_json(tmp_path, "./Simple_Gifs.json")

