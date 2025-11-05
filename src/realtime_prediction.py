import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model

# ✅ Correct action list
actions = ['hello', 'thank you', 'yes', 'no']

# ✅ Load your trained model
model = load_model('model/sign_language_model.h5')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ✅ Always return consistent shape (63 values)
def extract_keypoints(results):
    try:
        if results.multi_hand_landmarks:
            keypoints = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
            return np.array(keypoints)  # Shape (63,)
        else:
            return np.zeros(63)  # No hand → zero array
    except:
        return np.zeros(63)

def run_realtime_prediction():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    sequence = []  # Stores last 30 frames
    threshold = 0.8

    print("✅ Real-time Sign Detection Started")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera not detected!")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # ✅ Extract and append fixed-size keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        # ✅ Keep only last 30 frames
        sequence = sequence[-30:]

        # ✅ Only predict when full 30 frames are ready
        if len(sequence) == 30:
            try:
                input_data = np.expand_dims(sequence, axis=0)  # (1, 30, 63)
                yhat = model.predict(input_data, verbose=0)[0]

                if np.max(yhat) > threshold:
                    predicted_action = actions[np.argmax(yhat)]
                else:
                    predicted_action = "..."

                cv2.putText(frame, f'Prediction: {predicted_action}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                print("Prediction error:", e)

        # ✅ Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Sign Language Detection", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("❌ Real-time detection stopped")

if __name__ == '__main__':
    run_realtime_prediction()
