import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Set up input and output directories
DATA_DIR = './augmented_data'

# Separate data and labels for single and double hand signs
single_hand_data = []
single_hand_labels = []
double_hand_data = []
double_hand_labels = []

# Process each folder (each alphabet)
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # If two hands are detected
            if len(results.multi_hand_landmarks) == 2:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_x = []
                    hand_y = []

                    # Extract landmarks for each hand
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        hand_x.append(x)
                        hand_y.append(y)

                    # Normalize and append landmarks of each hand
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(hand_x))
                        data_aux.append(y - min(hand_y))
                
                # Append the data and labels to the double-hand lists
                double_hand_data.append(data_aux)
                double_hand_labels.append(dir_)
            
            # If only one hand is detected, add zero padding for the second hand
            elif len(results.multi_hand_landmarks) == 1:
                hand_landmarks = results.multi_hand_landmarks[0]
                hand_x = []
                hand_y = []

                # Extract landmarks for the detected hand
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    hand_x.append(x)
                    hand_y.append(y)

                # Normalize and append landmarks of the detected hand
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(hand_x))
                    data_aux.append(y - min(hand_y))

                # Append the data and labels to the single-hand lists
                single_hand_data.append(data_aux)
                single_hand_labels.append(dir_)

# Check lengths and pad sequences for both single and double-hand data
single_hand_lengths = [len(seq) for seq in single_hand_data]
double_hand_lengths = [len(seq) for seq in double_hand_data]
print(f"Unique lengths of single-hand sequences: {set(single_hand_lengths)}")
print(f"Unique lengths of double-hand sequences: {set(double_hand_lengths)}")

# Pad sequences to make them the same length
max_len_single = 42  # 21 landmarks * 2 (x, y) for one hand
max_len_double = 84  # 42 landmarks * 2 (x, y) for two hands

single_hand_data = pad_sequences(single_hand_data, maxlen=max_len_single, padding='post', dtype='float32')
double_hand_data = pad_sequences(double_hand_data, maxlen=max_len_double, padding='post', dtype='float32')

# Convert labels to NumPy arrays
single_hand_labels = np.asarray(single_hand_labels)
double_hand_labels = np.asarray(double_hand_labels)

# Save single-hand data to a pickle file
with open('single_hand_data.pickle', 'wb') as f:
    pickle.dump({'data': single_hand_data, 'labels': single_hand_labels}, f)

print("Single-hand data has been processed and saved to 'single_hand_data.pickle'.")

# Save double-hand data to a pickle file
with open('double_hand_data.pickle', 'wb') as f:
    pickle.dump({'data': double_hand_data, 'labels': double_hand_labels}, f)

print("Double-hand data has been processed and saved to 'double_hand_data.pickle'.")
