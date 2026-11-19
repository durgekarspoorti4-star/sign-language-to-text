"""
Hand Landmark Extraction and Feature Normalization Utilities
Supports both MediaPipe Solutions (legacy) and MediaPipe Tasks API (modern)
"""

import os
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe Hand Connections (Pairs of landmark indices)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm base cross-connections
    (5, 9), (9, 13), (13, 17)
]


class HandDetector:
    """Unified Hand Detector supporting both MediaPipe legacy & Tasks APIs."""

    def __init__(self, model_asset_path="hand_landmarker.task", min_detection_confidence=0.5):
        self.min_detection_confidence = min_detection_confidence
        self.mode = None
        self.detector = None

        # Determine path to model asset
        if not os.path.isabs(model_asset_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.join(current_dir, model_asset_path)
        else:
            self.model_path = model_asset_path

        # Check if legacy mp.solutions is available
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            try:
                self.mp_hands = mp.solutions.hands
                self.detector = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=0.5
                )
                self.mode = "solutions"
                print("[HandDetector] Initialized with MediaPipe Solutions API")
                return
            except Exception as e:
                print(f"[HandDetector] Failed to init Solutions API: {e}")

        # Fallback / Modern MediaPipe Tasks API
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            if not os.path.exists(self.model_path):
                # Try finding in scratch or parent directory
                alt_path = os.path.join(os.path.dirname(__file__), "..", "..", "brain", "scratch", "hand_landmarker.task")
                if os.path.exists(alt_path):
                    self.model_path = alt_path

            if os.path.exists(self.model_path):
                base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=1,
                    min_hand_detection_confidence=self.min_detection_confidence
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
                self.mode = "tasks"
                print(f"[HandDetector] Initialized with MediaPipe Tasks API ({self.model_path})")
            else:
                print(f"[HandDetector] Warning: Model asset not found at {self.model_path}")
                self.mode = "none"
        except Exception as e:
            print(f"[HandDetector] Error initializing Tasks API: {e}")
            self.mode = "none"

    def detect(self, bgr_image):
        """
        Detects 21 hand landmarks in a BGR OpenCV image.
        Returns:
            landmarks_list: List of 21 (x, y, z) objects with normalized coords [0, 1]
            handedness: 'Right' or 'Left' or 'Unknown'
        """
        h, w, _ = bgr_image.shape
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        if self.mode == "solutions":
            results = self.detector.process(rgb_image)
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                handedness = "Right"
                if results.multi_handedness:
                    handedness = results.multi_handedness[0].classification[0].label
                return landmarks, handedness

        elif self.mode == "tasks":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            results = self.detector.detect(mp_image)
            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                landmarks = results.hand_landmarks[0]
                handedness = "Right"
                if results.handedness and len(results.handedness) > 0:
                    handedness = results.handedness[0][0].category_name
                return landmarks, handedness

        return None, None

    def close(self):
        if self.mode == "solutions" and self.detector:
            self.detector.close()


def extract_features(landmarks):
    """
    Extracts 63 normalized coordinates from 21 hand landmarks.
    Normalization:
      1. Translates all landmarks relative to Landmark 0 (wrist at (0, 0, 0)).
      2. Scales by maximum coordinate spread across x, y, z for scale invariance.
    Returns:
        features: 1D numpy array of length 63.
    """
    if landmarks is None or len(landmarks) < 21:
        return None

    # Landmark 0 is the wrist
    wrist_x = landmarks[0].x
    wrist_y = landmarks[0].y
    wrist_z = getattr(landmarks[0], 'z', 0.0)

    rel_x = []
    rel_y = []
    rel_z = []

    for lm in landmarks:
        rel_x.append(lm.x - wrist_x)
        rel_y.append(lm.y - wrist_y)
        rel_z.append(getattr(lm, 'z', 0.0) - wrist_z)

    # Find maximum absolute coordinate offset for scale invariance
    max_x = max(abs(x) for x in rel_x)
    max_y = max(abs(y) for y in rel_y)
    max_z = max(abs(z) for z in rel_z)
    max_dist = max(max_x, max_y, max_z, 1e-6)

    # 63 normalized values: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
    features = []
    for i in range(21):
        features.append(rel_x[i] / max_dist)
        features.append(rel_y[i] / max_dist)
        features.append(rel_z[i] / max_dist)

    return np.array(features, dtype=np.float32)


def get_pixel_landmarks_and_bbox(landmarks, width, height):
    """
    Converts normalized landmark coordinates into pixel coordinates and computes bounding box.
    Returns:
        pixel_points: list of [x_px, y_px]
        bbox: [x_min, y_min, x_max, y_max]
    """
    if not landmarks:
        return [], []

    pixel_points = []
    x_coords = []
    y_coords = []

    for lm in landmarks:
        px = int(lm.x * width)
        py = int(lm.y * height)
        pixel_points.append([px, py])
        x_coords.append(px)
        y_coords.append(py)

    pad = 20
    x_min = max(0, min(x_coords) - pad)
    y_min = max(0, min(y_coords) - pad)
    x_max = min(width, max(x_coords) + pad)
    y_max = min(height, max(y_coords) + pad)

    return pixel_points, [x_min, y_min, x_max, y_max]
