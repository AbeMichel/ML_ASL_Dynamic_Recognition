import cv2
import numpy as np
import mediapipe as mp

# initialize mediapipe
mpHands = mp.solutions.hands  # module that performs the hand recognition algorithm
hands = mpHands.Hands(max_num_hands=2,
                      min_detection_confidence=0.5)  # configured model
mpDraw = mp.solutions.drawing_utils  # draws the key points


def draw_landmarks(pil_frame):
    frame = np.array(pil_frame)
    h, w, ch = frame.shape
    frame_rgb = frame  # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * h)
                lmy = int(lm.y * w)
                landmarks.append([lmx, lmy])

                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
    return frame



