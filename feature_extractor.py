import cv2
import numpy as np
import pandas as pd
import time
import os
import mediapipe as mp
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize MediaPipe Holistic (combines Face Mesh and Pose tracking)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two 2D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    """
    Calculates Eye Aspect Ratio (EAR) given landmark indices.
    indices format: [p1, p2, p3, p4, p5, p6]
    p1, p4: horizontal corners | p2, p6 & p3, p5: vertical pairs
    """
    pts = [np.array([landmarks[idx].x * img_w, landmarks[idx].y * img_h]) for idx in eye_indices]
    
    # Vertical distances
    v1 = calculate_distance(pts[1], pts[5])
    v2 = calculate_distance(pts[2], pts[4])
    # Horizontal distance
    h = calculate_distance(pts[0], pts[3])
    
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)

def calculate_mar(landmarks, mouth_indices, img_w, img_h):
    """
    Calculates Mouth Aspect Ratio (MAR).
    indices format: [p1, p2, p3, p4, p5, p6, p7, p8]
    p1, p5: horizontal corners | p2, p8 & p3, p7 & p4, p6: vertical pairs
    """
    pts = [np.array([landmarks[idx].x * img_w, landmarks[idx].y * img_h]) for idx in mouth_indices]
    
    v1 = calculate_distance(pts[1], pts[7])
    v2 = calculate_distance(pts[2], pts[6])
    v3 = calculate_distance(pts[3], pts[5])
    h = calculate_distance(pts[0], pts[4])
    
    if h == 0:
        return 0.0
    return (v1 + v2 + v3) / (2.0 * h)

def estimate_head_pose(landmarks, img_w, img_h):
    """
    Estimates Pitch, Yaw, and Roll angles using perspective-n-point (PnP).
    """
    # Key face points for PnP
    face_3d = []
    face_2d = []
    
    key_indices = [1, 33, 263, 61, 291, 199] # Nose tip, Left eye outer, Right eye outer, Mouth left, Mouth right, Chin
    
    for idx in key_indices:
        lm = landmarks[idx]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])
        
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    # Camera intrinsics matrix estimation
    focal_length = img_w
    cam_matrix = np.array([[focal_length, 0, img_w / 2],
                           [0, focal_length, img_h / 2],
                           [0, 0, 1]], dtype=np.float64)
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch = angles[0] * 360
    yaw = angles[1] * 360
    roll = angles[2] * 360

    return pitch, yaw, roll

def calculate_shoulder_angle(pose_landmarks, img_w, img_h):
    """
    Calculates the tilt angle of the shoulders relative to horizontal.
    Pose landmarks: 11 = Left Shoulder, 12 = Right Shoulder
    """
    l_shoulder = pose_landmarks[11]
    r_shoulder = pose_landmarks[12]
    
    lx, ly = l_shoulder.x * img_w, l_shoulder.y * img_h
    rx, ry = r_shoulder.x * img_w, r_shoulder.y * img_h
    
    delta_y = ry - ly
    delta_x = rx - lx
    
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    return angle

def main():
    # Landmark indices for EAR and MAR
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    MOUTH = [61, 185, 40, 39, 291, 269, 270, 267] # Outer mouth contours

    csv_file = "fatigue_features.csv"
    
    # Initialize CSV header if file doesn't exist
    if not os.path.exists(csv_file):
        df = pd.DataFrame(columns=["timestamp", "ear", "mar", "pitch", "yaw", "roll", "shoulder_angle", "label"])
        df.to_csv(csv_file, index=False)

    cap = cv2.VideoCapture(0)
    
    # Optional: Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = 0
    current_label = 0  # 0: Alert, 1: Drowsy (Toggleable using 'd' and 'a' keys for ground truth collection)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        print("=== Feature Extractor Running ===")
        print("Press 'a' to set ground truth label to ALERT (0)")
        print("Press 'd' to set ground truth label to DROWSY (1)")
        print("Press 'q' to QUIT")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror frame for natural UI interaction
            frame = cv2.flip(frame, 1)
            img_h, img_w, _ = frame.shape

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = holistic.process(rgb_frame)
            rgb_frame.flags.writeable = True

            ear, mar = 0.0, 0.0
            pitch, yaw, roll = 0.0, 0.0, 0.0
            shoulder_angle = 0.0
            valid_detection = False

            # 1. Process Face Mesh Landmarks
            if results.face_landmarks:
                face_lms = results.face_landmarks.landmark
                
                # Calculate EAR and MAR
                left_ear = calculate_ear(face_lms, LEFT_EYE, img_w, img_h)
                right_ear = calculate_ear(face_lms, RIGHT_EYE, img_w, img_h)
                ear = (left_ear + right_ear) / 2.0
                mar = calculate_mar(face_lms, MOUTH, img_w, img_h)
                
                # Head Pose Estimation
                pitch, yaw, roll = estimate_head_pose(face_lms, img_w, img_h)
                
                # Draw Face Mesh contours
                mp_drawing.draw_landmarks(
                    frame,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                valid_detection = True

            # 2. Process Pose Landmarks (Shoulders/Posture)
            if results.pose_landmarks:
                pose_lms = results.pose_landmarks.landmark
                shoulder_angle = calculate_shoulder_angle(pose_lms, img_w, img_h)
                
                # Draw Pose upper body skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # Record Data Frame to CSV
            timestamp = time.time()
            if valid_detection:
                new_row = pd.DataFrame([{
                    "timestamp": timestamp,
                    "ear": round(ear, 4),
                    "mar": round(mar, 4),
                    "pitch": round(pitch, 2),
                    "yaw": round(yaw, 2),
                    "roll": round(roll, 2),
                    "shoulder_angle": round(shoulder_angle, 2),
                    "label": current_label
                }])
                new_row.to_csv(csv_file, mode='a', header=False, index=False)

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # On-screen visual telemetry overlay
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f} Yaw: {yaw:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Shoulder Tilt: {shoulder_angle:.1f}deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            label_text = "DROWSY" if current_label == 1 else "ALERT"
            label_color = (0, 0, 255) if current_label == 1 else (0, 255, 0)
            cv2.putText(frame, f"Record Label: {label_text}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

            cv2.imshow("WearySteer - Milestone 1 Feature Extractor", frame)

            # Key Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                current_label = 0
            elif key == ord('d'):
                current_label = 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()