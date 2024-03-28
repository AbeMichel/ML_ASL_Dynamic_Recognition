import os
import json
from PIL import Image, ImageSequence
import numpy as np
import hand_recognition as hr

DEFAULT_POINT_VALUE = 9999999  # The value we will put in place of "missing" data


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
            gifs_processed[_class].append(process_gif(gif_path))  # the gif that has frames

    return gifs_processed  # dict[str, list[list[list[int]]]]


def process_gif(gif_path):
    gif_frames = []
    with Image.open(gif_path) as gif:
        gif_sequence = ImageSequence.Iterator(gif)
        for frame in gif_sequence:
            numpy_frame = np.array(frame)
            if len(numpy_frame.shape) == 3:
                landmarks = hr.get_landmarks(frame)
            else:
                landmarks = []
            gif_frames.append(landmarks)
    return gif_frames


def simplify_gif(gif):
    gif_simple = []

    if len(gif) < 15:
        while len(gif) < 15:
            gif.append([])
    # if len(gif) > max_landmarks:
    #     max_landmarks = len(gif)
    # if len(gif) < min_landmarks:
    #     min_landmarks = len(gif)
    for landmarks in gif:
        if len(landmarks) < 42:
            while len(landmarks) < 42:
                landmarks.append([DEFAULT_POINT_VALUE, DEFAULT_POINT_VALUE])
        elif len(landmarks) > 42:
            while len(landmarks) > 42:
                landmarks.pop(-1)
        # if len(landmarks) > max_points:
        #     max_points = len(landmarks)
        # if len(landmarks) < min_points:
        #     min_points = len(landmarks)
        for point in landmarks:
            x = point[0]
            y = point[1]
            gif_simple.append(x)
            gif_simple.append(y)
    return gif_simple


def convert_to_simple(class_dict) -> dict[str, list[list[int]]]:
    # Max landmarks:  15
    # Min landmarks:  14
    # Max points:  63
    # Min points:  0
    out: dict[str, list[list[int]]] = {}
    for cls, gifs in class_dict.items():
        out[cls] = []
        for gif in gifs:
            out[cls].append(simplify_gif(gif))
    return out


def reaccess_data():
    main_dir = "Actions"
    actions = get_action_gifs(main_dir)
    save_to_json("Actions_processed.json", actions)
    actions = load_from_json("Actions_processed.json")
    actions_simple = convert_to_simple(actions)
    save_to_json("Actions_processed_simple.json", actions_simple)


if __name__ == "__main__":
    reaccess_data()
