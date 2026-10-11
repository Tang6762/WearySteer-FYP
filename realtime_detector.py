import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque, Counter

# MediaPipe Solutions
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# Facial Landmark Indices matching feature_extractor.py
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 81, 13, 312, 317, 14, 87, 178]

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    pts = [(int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h)) for idx in eye_indices]
    d_v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    d_v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    d_h = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if d_h == 0: 
        return 0.0
    return (d_v1 + d_v2) / (2.0 * d_h)

def calculate_mar(landmarks, mouth_indices, img_w, img_h):
    pts = [(int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h)) for idx in mouth_indices]
    d_v = np.linalg.norm(np.array(pts[2]) - np.array(pts[5]))
    d_h = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if d_h == 0: 
        return 0.0
    return d_v / d_h

def main():
    try:
        model = joblib.load("wearysteer_rf.pkl")
        scaler = joblib.load("wearysteer_scaler.pkl")
        print("[+] Model and scaler loaded successfully.")
    except Exception as e:
        print(f"[-] Error loading model files: {e}")
        return

    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    pose = mp_pose.Pose(min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[-] Could not open webcam.")
        return

    label_map = {0: "Alert", 1: "Low Vigilance", 2: "Drowsy"}
    color_map = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255)}

    # Buffer for temporal smoothing (stores predictions of the last 15 frames)
    prediction_buffer = deque(maxlen=15)

    print("[*] Starting real-time detection with temporal smoothing. Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = face_mesh.process(rgb_frame)
        pose_results = pose.process(rgb_frame)

        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            
            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(landmarks, MOUTH, w, h)

            shoulder_slump = 0.0
            if pose_results.pose_landmarks:
                p_landmarks = pose_results.pose_landmarks.landmark
                nose_y = p_landmarks[mp_pose.PoseLandmark.NOSE].y
                l_shoulder_y = p_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                r_shoulder_y = p_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
                shoulder_slump = ((l_shoulder_y + r_shoulder_y) / 2.0) - nose_y

            # Construct feature vector and predict
            feature_vector = np.array([[avg_ear, mar, shoulder_slump]])
            scaled_features = scaler.transform(feature_vector)
            raw_prediction = model.predict(scaled_features)[0]
            
            # Push prediction to smoothing buffer
            prediction_buffer.append(raw_prediction)

            # Majority voting across buffered frames
            smoothed_prediction = Counter(prediction_buffer).most_common(1)[0][0]

            status_text = label_map.get(smoothed_prediction, "Unknown")
            status_color = color_map.get(smoothed_prediction, (255, 255, 255))

            # Display real-time telemetry and smoothed classification
            cv2.putText(frame, f"State: {status_text}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
            cv2.putText(frame, f"EAR: {avg_ear:.2f} | MAR: {mar:.2f} | Slump: {shoulder_slump:.2f}", 
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if smoothed_prediction == 2:
                cv2.putText(frame, "DROWSINESS ALERT!", (30, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow("FocusSphere Real-Time Drowsiness Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    pose.close()

if __name__ == "__main__":
    main()