import os
from PIL import Image, ImageSequence
import numpy as np
import hand_recognition as hr
import json


def save_to_json(filename, data):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {filename} successfully.")


def load_from_json(filename) -> dict[str, list[list[list[int]]]]:
    with open(filename, 'r') as file:
        data = json.load(file)
        return data


def get_action_gifs(directory) -> dict[str, list[list[list[int]]]]:
    gif_classes = os.listdir(directory)
    gif_files = {}
    for gif_class in gif_classes:
        if gif_class.strip() == "newTest":
            continue
        gif_dir = os.path.join(directory, gif_class)
        gif_files[gif_class] = os.listdir(gif_dir)

    gifs_processed = {}
    for _class, _files in gif_files.items():
        gifs_processed[_class] = []  # the class that has all of the gifs
        gif_class_path = os.path.join(directory, _class)
        for _file in _files:
            gif_path = os.path.join(gif_class_path, _file)
            gifs_processed[_class].append([])  # the gif that has frames
            with Image.open(gif_path) as gif:
                gif_sequence = ImageSequence.Iterator(gif)
                for frame in gif_sequence:
                    numpy_frame = np.array(frame)
                    if len(numpy_frame.shape) == 3:
                        landmarks = hr.get_landmarks(frame)
                    else:
                        landmarks = []
                    #  if landmarks has a length of:
                    #  0  - then there are no hands present
                    #  21 - then there is one hand present
                    #  42 - then there are two hands present
                    #  etc...
                    gifs_processed[_class][-1].append(landmarks)

    return gifs_processed  # dict[str, list[list[list[int]]]]


if __name__ == "__main__":
    MAIN_DIR = "Actions"
    actions = get_action_gifs(MAIN_DIR)
    save_to_json("Actions_processed.json", actions)
    actions = load_from_json("Actions_processed.json")
    print(len(actions.keys()))
    print(len(actions["thank_you"][0]))
