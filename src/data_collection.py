import cv2
import os
import numpy as np
import time
import mediapipe as mp

# Path where data will be stored
DATA_PATH = os.path.join('sample_data')

# Function to extract keypoints from MediaPipe hand landmarks
def extract_keypoints(results):
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints = np.zeros(21 * 3)  # If no hand detected, save zeros
    return keypoints

def collect_sign_data(actions, num_sequences=30, sequence_length=30):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    # Create directory for each action and each sequence
    for action in actions:
        for seq in range(num_sequences):
            os.makedirs(os.path.join(DATA_PATH, action, str(seq)), exist_ok=True)

    print("--- Starting Data Collection ---")
    time.sleep(2)

    for action in actions:
        print(f"\nGet ready to perform: {action}")
        time.sleep(2)

        for seq in range(num_sequences):
            print(f"Collecting {action} - Sequence {seq + 1}/{num_sequences}")

            # Display countdown before recording
            for countdown in range(3, 0, -1):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Starting in {countdown}", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Collecting Data", frame)
                cv2.waitKey(500)

            for frame_idx in range(sequence_length):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Draw landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get keypoints for each frame
                keypoints = extract_keypoints(results)

                # Save keypoints to file
                npy_file = os.path.join(DATA_PATH, action, str(seq), str(frame_idx))
                np.save(npy_file, keypoints)

                # Show frame
                cv2.putText(frame, f"Action: {action} | Seq: {seq}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow("Collecting Data", frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("--- Data Collection Completed ---")
