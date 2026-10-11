import os
import sys
import json
import msvcrt  # Built-in Windows keyboard listener
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 81, 13, 312, 317, 14, 87, 178]

PROGRESS_FILE = "processed_videos.json"

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    pts = [(int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h)) for idx in eye_indices]
    d_v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    d_v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    d_h = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if d_h == 0: return 0.0
    return (d_v1 + d_v2) / (2.0 * d_h)

def calculate_mar(landmarks, mouth_indices, img_w, img_h):
    pts = [(int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h)) for idx in mouth_indices]
    d_v = np.linalg.norm(np.array(pts[2]) - np.array(pts[5]))
    d_h = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if d_h == 0: return 0.0
    return d_v / d_h

def load_processed_videos():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_processed_videos(processed_set):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(processed_set), f, indent=4)

def check_pause_key():
    """Non-blocking keyboard check for Windows ('z' to stop safely)"""
    if msvcrt.kbhit():
        key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
        if key == 'z':
            return True
    return False

def extract_dataset(base_path, output_csv="wearysteer_features.csv"):
    processed_videos = load_processed_videos()
    video_files = []
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                video_files.append(os.path.join(root, file))
    
    # Filter out already processed videos
    unprocessed_videos = [v for v in video_files if v not in processed_videos]

    print(f"Total videos found: {len(video_files)}")
    print(f"Already completed: {len(processed_videos)}")
    print(f"Remaining to process: {len(unprocessed_videos)}")
    print("Press 'z' at any time in terminal to gracefully PAUSE/SAVE progress.\n")

    if not unprocessed_videos:
        print("All videos are already processed!")
        return

    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    pose = mp_pose.Pose(min_detection_confidence=0.5)

    stop_requested = False

    for vid_idx, vid_path in enumerate(unprocessed_videos, 1):
        filename = os.path.splitext(os.path.basename(vid_path))[0]
        
        if filename == '0': label = 0
        elif filename == '5': label = 1
        elif filename == '10': label = 2
        else: continue

        subject_id = os.path.basename(os.path.dirname(vid_path))
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        video_data = []
        frame_count = 0

        # Progress bar per video
        desc = f"[{vid_idx}/{len(unprocessed_videos)}] Subj {subject_id} ({filename})"
        pbar = tqdm(total=total_frames, desc=desc, unit="frame", leave=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            pbar.update(1)

            # Check for 'z' pause keypress
            if check_pause_key():
                print("\n\n'z' pressed! Stopping after current video finishes...")
                stop_requested = True

            if frame_count % 5 != 0:
                continue

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

                video_data.append({
                    "subject_id": subject_id,
                    "ear": avg_ear,
                    "mar": mar,
                    "shoulder_slump": shoulder_slump,
                    "label": label
                })

        cap.release()
        pbar.close()

        # Append data to CSV immediately after video finishes
        if video_data:
            df_batch = pd.DataFrame(video_data)
            header_needed = not os.path.exists(output_csv)
            df_batch.to_csv(output_csv, mode='a', index=False, header=header_needed)
            
            # Record completed video in progress tracking
            processed_videos.add(vid_path)
            save_processed_videos(processed_videos)

        if stop_requested:
            print(f"\nProgress safely saved to {output_csv}. You can resume anytime by re-running the script!")
            break

    face_mesh.close()
    pose.close()
    print("\nBatch operation complete.")

if __name__ == "__main__":
    DATASET_ROOT = r"E:\FYP-Dataset\UTA-RLDD"
    extract_dataset(DATASET_ROOT)