# Dynamic Action Sign Language Recognition

### Table of Contents
- [Goals](#goals)
- [Setup](#setup)
- [Python Libraries Utilized](#libraries-utilized)
- [Training Data](#training-data)
- [The Model](#the-model)
- [How to](#how-to)
  - [Step 1: Create Data](#step-1-create-data)
  - [Step 2: Build Model](#step-2-build-model)
  - [Step 3: Predict](#step-3-predict)
- [Additional Resources and References](#additional-resources)

### Goals
- Support 5 dynamic American Sign Language (ASL) signs
  - Thank you
  - Nice to meet you
  - How are you
  - Hello
  - Goodbye
- Provide an accessible program for people to learn about the machine learning process and ASL
- Add more ASL signs to the model
- Create better error display
- Expand on the UI to make it more intuitive and the model customizable

### Setup
***Starting***:
1. (Optional) Download my models that support 5 signs [here (Google Drive)](https://drive.google.com/file/d/1_da7gQPMACUVHiLSe8W0MWTbSPwh0YP7/view?usp=sharing)
1. Install Python
2. Run *DoSetup.bat* from *./MLGIFProject/*
3. Run *RunMain.bat* from *./MLGIFProject/*

***Python Version***: 3.11.4

***Bat files***:
- DoSetup.bat: Installs needed libraries and does some python package setup
```
pip install -r "requirements.txt"
python setup.py install
```

- RunMain.bat: Starts the application
```
python "Scripts/main.py"
```

### Libraries Utilized
This project is coded in Python and as such involves a variety of libraries. The libraries and their versions are listed
below.

| Library      | Version     |
|--------------|-------------|
| OpenCV       | 4.9.0.80    |
| pillow       | 10.2.0      |
| PyQt6        | 6.6.1       |
| tensorflow   | 2.16.1      |
| mediapipe    | 0.10.11     |
| matplotlib   | 3.8.3       |
| numpy        | 1.26.4      |
| pathlib      | 1.0.1       |
| scikit-learn | 1.4.1.post1 |
| setuptools   | 60.2.0      |
| jsonschema   | 4.21.1      |

### Training Data
Data for this project is stored in two formats: JSON and GIF. 

GIFs were utilized as a storage friendly alternative to common video formats while still allowing for a lot of control over the data within. Each GIF is recorded as 30 frames over a period of 1.5 seconds.

The JSON data is created from GIFs by passing each frame of the GIFs through a hand landmark detection model. The points returned from this model (up to 63 per frame) are then added to corresponding class in the JSON file. 

An example of the 5 signs are below:

Hello
- <img alt="hello" src=".\README Resources\Action Examples\hello.gif" width="200px">

Goodbye
- <img alt="goodbye" src=".\README Resources\Action Examples\goodbye.gif" width="200px">

Thank you
- <img alt="thank you" src=".\README Resources\Action Examples\thank_you.gif" width="200px">

How are you
- <img alt="how are you" src=".\README Resources\Action Examples\how_are_you.gif" width="200px">

Nice to meet you
- <img alt="nice to meet you" src=".\README Resources\Action Examples\nice_to_meet_you.gif" width="200px">


### The Model
The model used in this project is built using the Tensorflow Keras framework.

While certain parameters are adjustable while using the GUI the layers have been hardcoded in the _machine\_learning\_model.py_ script. If one were to want to customize their model further it would be done by editing that script.
The hardcoded parameters and options I have as of writing this are:
- ***Model***: Sequential
- ***Layers***:
  - Reshape - Used to convert the data into a Tensorflow friendly format.
  - Convolutional - Applies a number of filters (32 here) to the data and subsequently uses an activation function (relu) on the data.
  - Convolutional - Applies a number of filters (64 here) to the data and subsequently uses an activation function (relu) on the data.
  - Max Pooling - Extracts subregions of the data, decreasing the processing time.
  - Dropout - Essentially throws away a percentage (10%) of neurons to attempt to avoid overfitting the model.
  - Flatten - Ensures the data is in a 1 dimensional format
  - Dense - A fully connected layer of neurons (128) where every neuron from the previous layer connects to every neuron of this layer and perform classification. This layer also utilizes an activation function (relu).
  - Dense - A fully connected layer of neurons (number of classes) where every neuron from the previous layer connects to every neuron of this layer and perform classification. This layer also utilizes an activation function (softmax).
- ***Optimizer***: Adam
- ***Training Metric***: Accuracy

The code for creating the default model is below,
      
```python
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
```
      


### How to

#### Step 1: Create Data
Begin by creating your data in the _Create Data_ tab and select a save directory.

<img alt="select save directory button" src=".\README Resources\create_data_select_save_dir.jpg" width="500px">

<img alt="selecting save directory" src=".\README Resources\create_data_save_dir.jpg" width="500px">

Set the label of the action you want to create data for.

<img alt="setting action label" src=".\README Resources\create_data_set_label.jpg" width="500px">

To record a new gif:
1. Starts a new recording. Also clears the last recorded GIF from the buffer.
2. The progress bar will empty as a countdown then it will fill up as an indicator of how much of the GIF has been recorded.
3. After recording a GIF you can preview it with the detected hand landmarks.
4. Saves the last recorded GIF in the selected save directory.
5. Clears the last recorded GIF from the buffer.

<img alt="recording ui" src=".\README Resources\create_data_recording_ui.jpg" width="500px">

Finally when you are happy with the collected data use the _Create JSON Data from Actions_ button to convert the GIFs in the selected save directory to a JSON file to use for building the model.

<img alt="convert and save data to json" src=".\README Resources\create_data_to_json.jpg" width="500px">

#### Step 2: Build Model
Select a JSON file created in the previous step.

<img alt="" src=".\README Resources\build_model_selecting_json.jpg" width="500px">

Adjust the parameters for the machine learning model
- **Batch Size**: The number of training samples in a single full pass.
  - _If there are 100 data samples and a batch size of 20 there will be 5 full passes in a single epoch._
- **Number of Epochs**: How many forward and backward passes through all the data to do during training.
- **Val Split**: The percentage of the data that will be used as validation data.

<img alt="" src=".\README Resources\build_model_set_params.jpg" width="500px">

Once parameters are set to your liking hit the _Build Model_ button. This may take a couple minutes and the program may become "unresponsive" depending on the parameters. If you've started the program from a terminal you can follow the progress through epochs there.

<img alt="" src=".\README Resources\build_model_btn.jpg" width="500px">

Once training is finished there will be two graphs that pop up.
- **Model Loss**: Quantifies the error margin between the predicted values and actual values. Lower is better most of the time.
- **Model Accuracy**: Percentage of correct classifications.

<img alt="" src=".\README Resources\build_model_loss_graph.jpg" width="500px">
<img alt="" src=".\README Resources\build_model_accuracy_graph.jpg" width="500px">

The numerical results of the training will be displayed above the _Build Model_ button.

<img alt="" src=".\README Resources\build_model_results.jpg" width="500px">

If you're happy with the results select a save directory for the model and hit the save button.

<img alt="" src=".\README Resources\build_model_select_save_dir.jpg" width="500px">
<img alt="" src=".\README Resources\build_model_selecting_save_dir.jpg" width="500px">

<img alt="" src=".\README Resources\build_model_save_model.jpg" width="500px">

This saved model will be a folder containing the encoded labels as a pickle file and the keras model.
1. The save directory selected. 
2. The name of the model. The naming format is "model_month-day-year_time" based on when the model was saved.
3. The label encoder. This is a pickle file used to decode the output of the model to legible class names.
4. The keras model file.

<img alt="" src=".\README Resources\build_model_saved_model.jpg" width="500px">


#### Step 3: Predict
Once you've built a model you're ready to predict on new data so load your model by selecting the model folder.

<img alt="" src=".\README Resources\predict_load_model.jpg" width="500px">

<img alt="" src=".\README Resources\predict_loading_model.jpg" width="500px">

The UI for creating new data to predict on is very similar to creating data for the model.
1. Model loading status and the classes the model has been trained with.
2. Starts a new GIF recording.
3. Progress bar used to count you in and visualize the recording time left.
4. Sends the recorded GIF through the model.
5. Displays the predicted class.

<img alt="" src=".\README Resources\predict_prediction_ui.jpg" width="500px">

### Additional Resources
Below are some of the resources and sites I utilized while working on this project. While not an exhaustive list it does help one in their journey to understanding ASL and machine learning.
- [MoveNet](https://www.kaggle.com/models/google/movenet)
- [MoveNet Pose Detection Model](https://www.tensorflow.org/hub/tutorials/movenet)
- [Real-time Hand Gesture Recognition](https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/)
- [Top 10 (and 25) Sign Language Signs for Beginners](https://www.startasl.com/top-10-25-american-sign-language-signs-for-beginners-the-most-know-top-10-25-asl-signs-to-learn-first/)
- [A Guide to TF Layers: Building a Convolutional Neural Network](https://docs.w3cub.com/tensorflow~guide/tutorials/layers.html)