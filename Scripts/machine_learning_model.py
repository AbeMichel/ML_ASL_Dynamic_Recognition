import json
import os.path
import random
from datetime import datetime
import numpy as np
import pickle

import matplotlib
import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import Scripts.gif as gif_utils

matplotlib.use('TkAgg')


def shuffle_list(list_to_shuffle: list):
    if len(list_to_shuffle) == 0:
        return list_to_shuffle
    random.shuffle(list_to_shuffle)
    return list(list_to_shuffle)


def shuffle_data_and_labels(data: list, labels: list):
    if len(data) == 0:
        return data, labels
    data1 = shuffle_list(data)
    labels1 = shuffle_list(labels)
    combined = list(zip(data1, labels1))
    random.shuffle(combined)
    shuffled_data, shuffled_labels = zip(*combined)
    return list(shuffled_data), list(shuffled_labels)


def create_model_from_json_path(json_file_path: str, batch_size: int = 15, num_epochs: int = 20, val_split: float = 0.4):
    with open(json_file_path, 'r') as json_file:
        gifs_json_data = json.load(json_file)
    return create_model_from_json_data(gifs_json_data, batch_size, num_epochs, val_split)


def create_model_from_json_data(gifs_json_data: dict[str, list[list[int]]], batch_size: int = 15, num_epochs: int = 20, val_split: float = 0.4):
    class_names = list(gifs_json_data.keys())
    data: list[list[int]]
    labels: list[str]
    val_data: list[list[int]] = []
    val_labels: list[str] = []
    train_data: list[list[int]] = []
    train_labels: list[str] = []

    # take the validation split from each class
    # and create a train and validation list
    for class_name, class_data in gifs_json_data.items():
        total_instances = len(class_data)
        num_for_validation = int(total_instances * val_split)
        shuffled_class_data = shuffle_list(class_data.copy())
        i = 0
        for gif in shuffled_class_data:
            if i < num_for_validation:
                val_data.append(gif)
                val_labels.append(class_name)
            else:
                train_data.append(gif)
                train_labels.append(class_name)
            i += 1
    # shuffle them
    val_data, val_labels = shuffle_data_and_labels(val_data, val_labels)
    train_data, train_labels = shuffle_data_and_labels(train_data, train_labels)
    # recombine them
    data = train_data + val_data
    labels = train_labels + val_labels

    np_data = np.array(data)
    np_labels = np.array(labels)

    input_shape = len(np_data[0])

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(np_labels)

    model = tf.keras.Sequential()
    model.add(layers.Reshape((input_shape, 1), input_shape=(input_shape, 1)))

    model.add(layers.Conv1D(32, 3, padding='same', activation='relu'))
    model.add(layers.Conv1D(64, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D())
    model.add(layers.Dropout(rate=0.1))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(len(class_names), activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    model_metrics = model.fit(np_data,
                              labels_encoded,
                              epochs=num_epochs,
                              batch_size=batch_size,
                              validation_split=val_split)

    plot_model_metrics(model_metrics.history)
    return model, label_encoder, model_metrics.history


def plot_model_metrics(metrics: dict):
    # Plot training loss
    plt.plot(metrics['loss'], label='Training Loss')

    # Check if validation loss is available
    if 'val_loss' in metrics:
        # Plot validation loss
        plt.plot(metrics['val_loss'], label='Validation Loss')

    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training accuracy
    plt.plot(metrics['accuracy'], label='Training Accuracy')

    # Check if validation accuracy is available
    if 'val_accuracy' in metrics:
        # Plot validation accuracy
        plt.plot(metrics['val_accuracy'], label='Validation Accuracy')

    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def predict_gif(gif: gif_utils.GIFSimple, model, label_encoder=None):
    np_simple_gif = np.array(gif.get_data())
    np_simple_gif = np_simple_gif.reshape(1, -1, 1)  # TODO: FIGURE THIS OUT
    predictions = model.predict(np_simple_gif)
    print(predictions)
    predicted_classes = np.argmax(predictions, axis=1)

    if label_encoder:
        predicted_classes = label_encoder.inverse_transform(predicted_classes)

    return predicted_classes[0]


def save_tf_model(model, save_path: str) -> None:
    tf.keras.models.save_model(model, save_path)


def save_label_encoder(label_encoder, save_path: str) -> None:
    with open(save_path, 'wb') as file:
        pickle.dump((label_encoder, label_encoder.classes_), file)


def save_model_metrics(metrics: dict, save_path: str) -> None:
    with open(save_path, 'w') as file:
        json.dump(metrics, file)


def save_model_labels_and_metrics(model, label_encoder: LabelEncoder, metrics: dict, save_path: str) -> str:
    def ensure_path_exists(path: str):
        if os.path.exists(path):
            return
        base_path = os.path.dirname(path)
        ensure_path_exists(base_path)
        os.mkdir(path)

    date = datetime.now()
    date_formatted = date.strftime("%m-%d-%Y")
    time_formatted = date.strftime("%H%M")
    save_path += f"\\model_{date_formatted}_{time_formatted}"
    ensure_path_exists(save_path)
    model_path = save_path + "\\model.keras"
    encoder_path = save_path + "\\encoder.pkl"
    metrics_path = save_path + "\\metrics.json"
    save_tf_model(model, model_path)
    save_label_encoder(label_encoder, encoder_path)
    save_model_metrics(metrics, metrics_path)
    return save_path


def load_tf_model(model_path: str):
    if not os.path.exists(model_path) or not os.path.isfile(model_path):
        return None
    return tf.keras.models.load_model(model_path)


def load_label_encoder(encoder_path: str):
    if not os.path.exists(encoder_path) or not os.path.isfile(encoder_path):
        return None
    with open(encoder_path, 'rb') as file:
        encoder, classes = pickle.load(file)
        encoder.classes_ = classes
    return encoder


def load_model_metrics(metrics_path: str):
    if not os.path.exists(metrics_path) or not os.path.isfile(metrics_path):
        return None
    with open(metrics_path, 'r') as file:
        return json.load(file)


def load_model_labels_and_metrics(folder_path):
    model_path = folder_path + "\\model.keras"
    encoder_path = folder_path + "\\encoder.pkl"
    metrics_path = folder_path + "\\metrics.json"
    model = load_tf_model(model_path)
    encoder = load_label_encoder(encoder_path)
    metrics = load_model_metrics(metrics_path)
    return model, encoder, metrics


if __name__ == "__main__":
    json_path = os.path.abspath('../test_data.json')
    # mdl, encdr = create_model_from_json_path(json_path)
    # save_model_and_labels(mdl, encdr, "../")
    mdl, encdr, mets = load_model_labels_and_metrics('../model_04-18-2024_2204')
    print(encdr.classes_)
