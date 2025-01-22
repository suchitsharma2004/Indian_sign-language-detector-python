import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load both trained models
try:
    model_single_hand_dict = pickle.load(open('./model_single_hand.p', 'rb'))
    model_single_hand = model_single_hand_dict['model']
except FileNotFoundError:
    model_single_hand = None
    print("Single-hand model not found.")

try:
    model_double_hand_dict = pickle.load(open('./model_double_hand.p', 'rb'))
    model_double_hand = model_double_hand_dict['model']
except FileNotFoundError:
    model_double_hand = None
    print("Double-hand model not found.")

# Initialize video capture and mediapipe
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# Create labels dictionary
labels_dict = {i: chr(ord('a') + i) for i in range(26)}  # Letters 'a' to 'z'
labels_dict.update({26 + i: str(i) for i in range(10)})  # Numbers '0' to '9'

print(labels_dict)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret:
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        num_hands_detected = len(results.multi_hand_landmarks)
        data_aux = []  # Clear data for each frame

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Extract features from each detected hand
            x_.clear()
            y_.clear()

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Prepare the features for prediction based on the number of hands detected
        if num_hands_detected == 1 and model_single_hand:
            # Single-hand prediction (42 features)
            if len(data_aux) == 42:
                features = np.asarray(data_aux).reshape(1, -1)
                prediction = model_single_hand.predict(features)
                predicted_character = labels_dict[int(prediction[0])]
                display_text = f"Prediction (1 hand): {predicted_character}"
            else:
                display_text = "Invalid features for single-hand model."

        elif num_hands_detected == 2 and model_double_hand:
            # Two-hands prediction (84 features)
            if len(data_aux) == 84:
                features = np.asarray(data_aux).reshape(1, -1)
                prediction = model_double_hand.predict(features)
                predicted_character = labels_dict[int(prediction[0])]
                display_text = f"Prediction (2 hands): {predicted_character}"
            else:
                display_text = "Invalid features for double-hand model."
        else:
            display_text = "Model not trained or wrong input feature length."

        # Display prediction
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
