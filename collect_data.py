"""
Sign Language Data Collector
Interactively captures hand landmark data from webcam, normalizes to 63 features,
and saves samples to data/data.pickle for model training.
"""

import os
import sys
import time
import pickle
import argparse
import cv2
import numpy as np

# Import unified hand detector and feature extractor
from hand_utils import HandDetector, extract_features, get_pixel_landmarks_and_bbox, HAND_CONNECTIONS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_FILE = os.path.join(DATA_DIR, "data.pickle")


def load_existing_dataset():
    """Loads existing dataset if present, or initializes an empty one."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "rb") as f:
                data_dict = pickle.load(f)
                return data_dict.get("data", []), data_dict.get("labels", [])
        except Exception as e:
            print(f"[Warning] Error reading {DATA_FILE}: {e}. Creating new dataset.")
    return [], []


def save_dataset(data, labels):
    """Saves data and labels to pickle file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATA_FILE, "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)
    print(f"[Saved] Total dataset size: {len(data)} samples across {len(set(labels))} classes.")


def draw_hud(frame, label, recorded_count, target_count, state, countdown=0):
    """Renders professional cyberpunk/clean HUD overlay on the OpenCV frame."""
    h, w, _ = frame.shape
    overlay = frame.copy()

    # Top banner background
    cv2.rectangle(overlay, (0, 0), (w, 80), (18, 18, 28), -1)
    # Bottom banner background
    cv2.rectangle(overlay, (0, h - 60), (w, h), (18, 18, 28), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title & State
    title_text = f"SIGN DATA COLLECTOR | Target Label: [{label}]"
    cv2.putText(frame, title_text, (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 240, 255), 2)

    # Status indicator
    if state == "READY":
        status_text = "READY - Press 'S' to Start 3s Countdown"
        status_color = (0, 255, 128)
    elif state == "COUNTDOWN":
        status_text = f"GET READY: {countdown:.1f}s"
        status_color = (0, 215, 255)
    elif state == "RECORDING":
        status_text = f"RECORDING SAMPLES... ({recorded_count}/{target_count})"
        status_color = (50, 50, 255)
        # Pulsing recording dot
        cv2.circle(frame, (w - 40, 35), 10, (0, 0, 255), -1)
    elif state == "COMPLETE":
        status_text = f"FINISHED! Captured {target_count} samples for '{label}'"
        status_color = (0, 255, 0)
    else:
        status_text = state
        status_color = (200, 200, 200)

    cv2.putText(frame, status_text, (20, 65), cv2.FONT_HERSHEY_DUPLEX, 0.6, status_color, 1)

    # Progress Bar at bottom if recording
    if target_count > 0:
        progress = min(1.0, recorded_count / target_count)
        bar_w = int((w - 40) * progress)
        cv2.rectangle(frame, (20, h - 35), (w - 20, h - 20), (50, 50, 70), -1)
        cv2.rectangle(frame, (20, h - 35), (20 + bar_w, h - 20), (0, 240, 255), -1)

    controls_text = "Controls: [S] Start/Record | [R] New Label | [Q] Quit & Save"
    cv2.putText(frame, controls_text, (20, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def draw_skeleton(frame, pixel_points, bbox=None):
    """Draws glowing landmarks and bone connections."""
    if not pixel_points or len(pixel_points) < 21:
        return

    # Draw Connections
    for p1_idx, p2_idx in HAND_CONNECTIONS:
        pt1 = tuple(pixel_points[p1_idx])
        pt2 = tuple(pixel_points[p2_idx])
        cv2.line(frame, pt1, pt2, (255, 180, 0), 2, cv2.LINE_AA)

    # Draw Landmark nodes
    for i, pt in enumerate(pixel_points):
        # Finger tips in vibrant cyan, joints in yellow/white
        color = (0, 255, 255) if i in [4, 8, 12, 16, 20] else (255, 255, 255)
        radius = 5 if i in [4, 8, 12, 16, 20] else 3
        cv2.circle(frame, tuple(pt), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(pt), radius + 1, (0, 128, 255), 1, cv2.LINE_AA)

    # Draw Bounding Box
    if bbox:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 128), 2)
        cv2.putText(frame, "Hand Detected", (x_min, max(20, y_min - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)


def run_collector(label=None, target_samples=80, camera_index=0):
    dataset_data, dataset_labels = load_existing_dataset()
    print("=" * 60)
    print("             SIGN LANGUAGE DATA COLLECTOR")
    print("=" * 60)
    print(f"Existing dataset: {len(dataset_data)} samples | Classes: {set(dataset_labels)}")

    if not label:
        try:
            label = input("\nEnter Sign/Letter Label to record (e.g. A, B, HELLO): ").strip().upper()
        except EOFError:
            label = "A"
    if not label:
        label = "SAMPLE"

    print(f"\n[Collector] Initializing Camera #{camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[Error] Could not open camera {camera_index}. Please check connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[Collector] Initializing Hand Detector...")
    detector = HandDetector()

    state = "READY"
    countdown_start = 0.0
    recorded_for_label = 0
    new_features = []
    new_labels = []

    window_name = "Sign Language Dataset Collector (Press Q to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 640)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to grab camera frame.")
            break

        # Mirror frame horizontally for natural intuitive interaction
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Detect Hand
        landmarks, handedness = detector.detect(frame)
        pixel_pts = []
        bbox = None

        if landmarks:
            pixel_pts, bbox = get_pixel_landmarks_and_bbox(landmarks, w, h)
            draw_skeleton(frame, pixel_pts, bbox)

        # State Machine Logic
        now = time.time()
        countdown_rem = 0.0

        if state == "COUNTDOWN":
            elapsed = now - countdown_start
            countdown_rem = max(0.0, 3.0 - elapsed)
            if elapsed >= 3.0:
                state = "RECORDING"
                recorded_for_label = 0

        elif state == "RECORDING":
            if landmarks:
                features = extract_features(landmarks)
                if features is not None and len(features) == 63:
                    new_features.append(features.tolist())
                    new_labels.append(label)
                    recorded_for_label += 1

                    if recorded_for_label >= target_samples:
                        state = "COMPLETE"
                        # Append and save immediately
                        dataset_data.extend(new_features)
                        dataset_labels.extend(new_labels)
                        save_dataset(dataset_data, dataset_labels)
                        new_features = []
                        new_labels = []

        # Render HUD
        draw_hud(frame, label, recorded_for_label, target_samples, state, countdown_rem)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("\n[Collector] Exiting...")
            break
        elif key == ord('s') or key == 32:  # 's' or Space
            if state in ["READY", "COMPLETE"]:
                state = "COUNTDOWN"
                countdown_start = time.time()
                print(f"\n[Collector] Countdown started for sign '{label}'...")
        elif key == ord('r'):  # Change label
            cv2.destroyAllWindows()
            try:
                new_l = input("\nEnter New Sign/Letter Label: ").strip().upper()
                if new_l:
                    label = new_l
            except EOFError:
                pass
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1024, 640)
            state = "READY"
            recorded_for_label = 0

    cap.release()
    detector.close()
    cv2.destroyAllWindows()

    if new_features:
        dataset_data.extend(new_features)
        dataset_labels.extend(new_labels)
        save_dataset(dataset_data, dataset_labels)

    print("\n[Collector] Data collection finished.")
    print("To train the model now, run: python train_model.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Data Collector")
    parser.add_argument("--label", type=str, default="", help="Label to record (e.g. A, B, HELLO)")
    parser.add_argument("--samples", type=int, default=80, help="Number of frames to record")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    args = parser.parse_args()

    run_collector(label=args.label, target_samples=args.samples, camera_index=args.camera)
