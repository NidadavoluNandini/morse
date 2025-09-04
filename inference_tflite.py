import pickle
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# ---------------------------
# Load model and label encoder
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
# Gesture mappings
# ---------------------------
labels_dict = {'0': 'Dot', '1': 'Dash', '2': 'BlankSpace', '3': 'BackSpace', '4': 'Next'}
display_map = {'Dot': '.', 'Dash': '-', 'BlankSpace': ' '}

# Morse code dictionary
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
current_morse = ""   # Accumulates dots and dashes for a single letter
last_gesture = ""
gesture_stable_count = 0
min_stable_frames = 10
next_detected = False
symbol_added = False   # <-- NEW: to prevent repeated symbols

# Buffer for smoothing
buffer = deque(maxlen=5)

# ---------------------------
# Predict function
# ---------------------------
def predict(frame: np.ndarray) -> dict:
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
        return {"gesture": None, "text": displayed_text, "morse": current_morse}

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

            # Stability check
            if predicted_character == last_gesture:
                gesture_stable_count += 1
            else:
                gesture_stable_count = 0
                last_gesture = predicted_character
                symbol_added = False  # reset when gesture changes

            if gesture_stable_count >= min_stable_frames:
                # --- Add Dot/Dash only once ---
                if predicted_character in ["Dot", "Dash"] and not symbol_added:
                    current_morse += display_map[predicted_character]
                    symbol_added = True
                    next_detected = False

                # --- Decode letter on Next ---
                elif predicted_character == "Next" and current_morse and not symbol_added:
                    letter = MORSE_CODE_DICT.get(current_morse, "")
                    displayed_text += letter
                    current_morse = ""
                    symbol_added = True
                    next_detected = True

                # --- Decode word on BlankSpace ---
                elif predicted_character == "BlankSpace" and not symbol_added:
                    if current_morse:
                        letter = MORSE_CODE_DICT.get(current_morse, "")
                        displayed_text += letter
                        current_morse = ""
                    displayed_text += " "  # add space for word
                    symbol_added = True
                    next_detected = False

                # --- Handle BackSpace ---
                elif predicted_character == "BackSpace" and not symbol_added:
                    if current_morse:
                        current_morse = current_morse[:-1]
                    elif displayed_text:
                        displayed_text = displayed_text[:-1]
                    symbol_added = True

            # --- Smoothing buffer ---
            buffer.append(predicted_character)
            smoothed_prediction = max(set(buffer), key=buffer.count)

            return {"gesture": smoothed_prediction, "text": displayed_text, "morse": current_morse}

    return {"gesture": None, "text": displayed_text, "morse": current_morse}
