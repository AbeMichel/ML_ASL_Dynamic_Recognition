# Dynamic Action Sign Language Recognition
### Goals
The goal for this project is to create a program that will take in a gif of 1 of 5 ASL signs and return the 
action shown. The actions supported as of now are:
- Thank you
- Nice to meet you
- How are you
- Hello
- Goodbye

### Training Data
The training data used is stored in the _Actions_ folder organized into sub-folders representing their classes.
An alternative preprocessed storage of this data is in the _Actions_processed.json_ file. Both hold the same data just 
in different formats.

### Libraries Utilized
This project is coded in Python and as such involves a variety of libraries. The libraries and their versions are listed
below.

| Library    | Version  |
|------------|----------|
| numpy      | 1.26.4   |
| tensorflow | 2.16.2   |
| pillow     | 10.2.0   |
| PyQt6      | 6.6.1    |
| OpenCV     | 4.9.0.80 |
| matplotlib | 3.8.3    |
| mediapipe  | 0.10.11  |
| jsonschema | 4.21.1   |

### Additional Resources
- [MoveNet](https://www.kaggle.com/models/google/movenet)
- [MoveNet Pose Detection Model](https://www.tensorflow.org/hub/tutorials/movenet)
- [Real-time Hand Gesture Recognition](https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/)
- [Top 10 (and 25) Sign Language Signs for Beginners](https://www.startasl.com/top-10-25-american-sign-language-signs-for-beginners-the-most-know-top-10-25-asl-signs-to-learn-first/)