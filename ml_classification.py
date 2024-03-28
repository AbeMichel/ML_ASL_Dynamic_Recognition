import json
import random

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import PIL
import pathlib
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from assess_gif_data import process_gif, simplify_gif
import utils


matplotlib.use('TkAgg')

batch_size = 15
epochs = 20
val_split = 0.4

'''
The data we want to train on will either be a directory of gifs or a json file with all the
images preprocessed resulting in a dictionary of the form: dict[str, list[list[list[int]]]]
'''


def create_model_from_json(json_file: str):
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    class_names = list(json_data.keys())
    data = []
    labels = []

    for class_name, class_data in json_data.items():
        for gif in class_data:
            data.append(gif)
            labels.append(class_name)

    data = np.array(data)
    labels = np.array(labels)

    input_shape = len(data[0])

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=val_split, random_state=random.randint(0,100))

    model = tf.keras.Sequential([
        # Input layer (reshape input if necessary)
        layers.Reshape((input_shape, 1), input_shape=(input_shape, 1)),

        # Convolutional layers
        layers.Conv1D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling1D(),
        layers.Conv1D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling1D(),
        layers.Conv1D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling1D(),

        # Flatten layer
        layers.Flatten(),

        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model_metrics = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    plot_model_metrics(model_metrics)
    return model, label_encoder


def plot_model_metrics(metrics):
    # Plot training loss
    plt.plot(metrics.history['loss'], label='Training Loss')

    # Check if validation loss is available
    if 'val_loss' in metrics.history:
        # Plot validation loss
        plt.plot(metrics.history['val_loss'], label='Validation Loss')

    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training accuracy
    plt.plot(metrics.history['accuracy'], label='Training Accuracy')

    # Check if validation accuracy is available
    if 'val_accuracy' in metrics.history:
        # Plot validation accuracy
        plt.plot(metrics.history['val_accuracy'], label='Validation Accuracy')

    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def predict(input_gif_path: str, model, label_encoder=None):
    input_gif = process_gif(input_gif_path)
    simple_gif = simplify_gif(input_gif)
    simple_gif = np.array(simple_gif)
    simple_gif = simple_gif.reshape(1, -1, 1)
    # print(simple_gif)
    predictions = model.predict(simple_gif)
    print(predictions)
    predicted_classes = np.argmax(predictions, axis=1)

    if label_encoder:
        predicted_classes = label_encoder.inverse_transform(predicted_classes)

    print(f"Path: {input_gif_path}\nPredicted Class: {predicted_classes[0]}")


if __name__ == "__main__":
    encoder = None
    # utils.display_gif_with_hr("Input_GIFS/goodbye.gif")
    # utils.display_gif_with_hr("Input_GIFS/hello.gif")
    # utils.display_gif_with_hr("Input_GIFS/thank_you.gif")
    # utils.display_gif_with_hr("Input_GIFS/nice_to_meet_you.gif")
    # utils.display_gif_with_hr("Input_GIFS/how_are_you.gif")
    # new_model, encoder = create_model_from_json("Actions_processed_simple.json")
    # ['goodbye' 'hello' 'how_are_you' 'nice_to_meet_you' 'thank_you']
    # tf.keras.models.save_model(new_model, "jsonBasedModel.keras")
    #
    saved_model = tf.keras.models.load_model("jsonBasedModel.keras")
    predict("Input_GIFS/goodbye.gif", saved_model, encoder)
    predict("Input_GIFS/hello.gif", saved_model, encoder)
    predict("Input_GIFS/how_are_you.gif", saved_model, encoder)
    predict("Input_GIFS/nice_to_meet_you.gif", saved_model, encoder)
    predict("Input_GIFS/thank_you.gif", saved_model, encoder)
    predict("Actions/how_are_you/0.gif", saved_model, encoder)
    predict("Actions/thank_you/0.gif", saved_model, encoder)
    predict("Actions/hello/0.gif", saved_model, encoder)
    predict("Actions/nice_to_meet_you/0.gif", saved_model, encoder)
