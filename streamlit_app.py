import pickle
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import streamlit as st

st.set_page_config(page_title="GestureSpeak — Morse Decoder", layout="wide")
st.title("GestureSpeak Morse Code Recognition (Upload Video)")

# ---------------------------
# Load TFLite model and label encoder
# ---------------------------
interpreter = tf.lite.Interpreter(model_path='morse_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# ---------------------------
# Mediapipe Hands
# ---------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------
# Gesture mappings & Morse
# ---------------------------
labels_dict = {'0': 'Dot', '1': 'Dash', '2': 'BlankSpace', '3': 'BackSpace', '4': 'Next'}
display_map = {'Dot': '.', 'Dash': '-', 'BlankSpace': ' '}
MORSE_CODE_DICT = {
    '.-':'A', '-...':'B', '-.-.':'C', '-..':'D', '.':'E',
    '..-.':'F', '--.':'G', '....':'H', '..':'I', '.---':'J',
    '-.-':'K', '.-..':'L', '--':'M', '-.':'N', '---':'O',
    '.--.':'P', '--.-':'Q', '.-.':'R', '...':'S', '-':'T',
    '..-':'U', '...-':'V', '.--':'W', '-..-':'X', '-.--':'Y',
    '--..':'Z', '-----':'0', '.----':'1', '..---':'2', '...--':'3',
    '....-':'4', '.....':'5', '-....':'6', '--...':'7', '---..':'8', '----.':'9'
}

# ---------------------------
# State variables
# ---------------------------
displayed_text = ""
current_morse = ""
last_gesture = ""
gesture_stable_count = 0
min_stable_frames = 5
next_detected = False
symbol_added = False
buffer = deque(maxlen=5)

# ---------------------------
# Predict function
# ---------------------------
def predict_frame(frame):
    global displayed_text, current_morse, last_gesture
    global gesture_stable_count, next_detected, symbol_added

    data_aux, x_, y_ = [], [], []
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        gesture_stable_count = 0
        last_gesture = ""
        next_detected = False
        symbol_added = False
        return None

    for hand_landmarks in results.multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

    if len(data_aux) == 42:
        input_data = np.array([data_aux], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_idx = np.argmax(output_data[0])
        confidence = np.max(output_data[0])

        if confidence > 0.85:
            predicted_class = label_encoder.classes_[predicted_class_idx]
            predicted_character = labels_dict[predicted_class]

            if predicted_character == last_gesture:
                gesture_stable_count += 1
            else:
                gesture_stable_count = 0
                last_gesture = predicted_character
                symbol_added = False

            if gesture_stable_count >= min_stable_frames:
                if predicted_character in ["Dot", "Dash"] and not symbol_added:
                    current_morse += display_map[predicted_character]
                    symbol_added = True
                    next_detected = False
                elif predicted_character == "Next" and current_morse and not symbol_added:
                    letter = MORSE_CODE_DICT.get(current_morse, "")
                    displayed_text += letter
                    current_morse = ""
                    symbol_added = True
                    next_detected = True
                elif predicted_character == "BlankSpace" and not symbol_added:
                    if current_morse:
                        letter = MORSE_CODE_DICT.get(current_morse, "")
                        displayed_text += letter
                        current_morse = ""
                    displayed_text += " "
                    symbol_added = True
                    next_detected = False
                elif predicted_character == "BackSpace" and not symbol_added:
                    if current_morse:
                        current_morse = current_morse[:-1]
                    elif displayed_text:
                        displayed_text = displayed_text[:-1]
                    symbol_added = True

            buffer.append(predicted_character)
            smoothed_prediction = max(set(buffer), key=buffer.count)
            return smoothed_prediction
    return None

# ---------------------------
# Streamlit UI
# ---------------------------
uploaded_file = st.file_uploader("Upload a video file (mp4, mov)", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = "temp_video.mp4"
    with open(tfile, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        predict_frame(frame)

        # Display overlay
        overlay = frame.copy()
        cv2.putText(overlay, f"Text: {displayed_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(overlay, f"Morse: {current_morse}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        stframe.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    st.success(f"Decoded Text: {displayed_text}")
