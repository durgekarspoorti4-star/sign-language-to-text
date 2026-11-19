"""
Sign Language Model Trainer
Trains a Random Forest classifier on 63 normalized hand landmark coordinates.
Saves the trained model and label map to model.p.
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "data.pickle")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.p")


def create_seed_dataset():
    """
    Generates a realistic anatomical benchmark dataset for standard ASL signs
    so the web app is immediately functional without initial manual recording.
    Includes A, B, C, D, E, I, L, O, U, V, W, Y, HELLO, YES, NO, SPACE, I LOVE YOU.
    """
    print("[Seed Generator] Synthesizing anatomical landmark templates for core signs...")

    # Canonical base hand template (Wrist at 0,0,0)
    # 21 points:
    # 0: Wrist
    # 1-4: Thumb (CMC, MCP, IP, Tip)
    # 5-8: Index (MCP, PIP, DIP, Tip)
    # 9-12: Middle (MCP, PIP, DIP, Tip)
    # 13-16: Ring (MCP, PIP, DIP, Tip)
    # 17-20: Pinky (MCP, PIP, DIP, Tip)

    def build_hand(thumb_ext=0.5, index_ext=1.0, middle_ext=1.0, ring_ext=1.0, pinky_ext=1.0,
                   thumb_spread=0.4, finger_spread=0.1, curl_inward=False):
        """Constructs 21 3D points based on finger extension parameters [0.0 = curled, 1.0 = extended]."""
        pts = np.zeros((21, 3), dtype=np.float32)
        pts[0] = [0.0, 0.0, 0.0]  # Wrist

        # Thumb
        t_dir = np.array([-thumb_spread, -0.3 * thumb_ext, 0.2 * (1 - thumb_ext)])
        pts[1] = pts[0] + np.array([-0.15, -0.15, 0.0])
        pts[2] = pts[1] + t_dir * 0.3
        pts[3] = pts[2] + t_dir * 0.35
        pts[4] = pts[3] + t_dir * 0.35

        # Fingers: Index, Middle, Ring, Pinky
        fingers = [
            (5, index_ext, -0.15 - finger_spread * 1.5),
            (9, middle_ext, -0.05 - finger_spread * 0.5),
            (13, ring_ext, 0.05 + finger_spread * 0.5),
            (17, pinky_ext, 0.15 + finger_spread * 1.5)
        ]

        for base_idx, ext, spread_x in fingers:
            mcp = np.array([spread_x, -0.4, 0.0])
            pts[base_idx] = mcp

            if ext > 0.6:  # Extended
                up_dir = np.array([spread_x * 0.2, -0.22, 0.0])
                pts[base_idx + 1] = pts[base_idx] + up_dir
                pts[base_idx + 2] = pts[base_idx + 1] + up_dir
                pts[base_idx + 3] = pts[base_idx + 2] + up_dir
            elif ext < 0.3:  # Curled fist
                pts[base_idx + 1] = pts[base_idx] + np.array([spread_x * 0.1, -0.12, 0.1])
                pts[base_idx + 2] = pts[base_idx + 1] + np.array([0.0, 0.08, 0.12])
                pts[base_idx + 3] = pts[base_idx + 2] + np.array([0.0, 0.06, 0.02])
            else:  # Half curved (e.g., C, O)
                pts[base_idx + 1] = pts[base_idx] + np.array([spread_x * 0.1, -0.16, 0.08])
                pts[base_idx + 2] = pts[base_idx + 1] + np.array([-0.05 if curl_inward else 0.0, -0.10, 0.15])
                pts[base_idx + 3] = pts[base_idx + 2] + np.array([-0.1 if curl_inward else 0.0, 0.02, 0.15])

        # Normalize relative to wrist & scale
        max_dist = np.max(np.abs(pts))
        if max_dist > 0:
            pts = pts / max_dist
        return pts.flatten()

    # Define gesture signature parameters
    gestures = {
        'A': {'thumb_ext': 0.8, 'index_ext': 0.1, 'middle_ext': 0.1, 'ring_ext': 0.1, 'pinky_ext': 0.1, 'thumb_spread': 0.1},
        'B': {'thumb_ext': 0.1, 'index_ext': 1.0, 'middle_ext': 1.0, 'ring_ext': 1.0, 'pinky_ext': 1.0, 'finger_spread': 0.0},
        'C': {'thumb_ext': 0.5, 'index_ext': 0.5, 'middle_ext': 0.5, 'ring_ext': 0.5, 'pinky_ext': 0.5, 'curl_inward': True},
        'D': {'thumb_ext': 0.4, 'index_ext': 1.0, 'middle_ext': 0.2, 'ring_ext': 0.2, 'pinky_ext': 0.2},
        'E': {'thumb_ext': 0.1, 'index_ext': 0.15, 'middle_ext': 0.15, 'ring_ext': 0.15, 'pinky_ext': 0.15},
        'I': {'thumb_ext': 0.2, 'index_ext': 0.1, 'middle_ext': 0.1, 'ring_ext': 0.1, 'pinky_ext': 1.0},
        'L': {'thumb_ext': 1.0, 'index_ext': 1.0, 'middle_ext': 0.1, 'ring_ext': 0.1, 'pinky_ext': 0.1, 'thumb_spread': 0.7},
        'O': {'thumb_ext': 0.4, 'index_ext': 0.4, 'middle_ext': 0.4, 'ring_ext': 0.4, 'pinky_ext': 0.4, 'curl_inward': True},
        'U': {'thumb_ext': 0.1, 'index_ext': 1.0, 'middle_ext': 1.0, 'ring_ext': 0.1, 'pinky_ext': 0.1, 'finger_spread': 0.0},
        'V': {'thumb_ext': 0.1, 'index_ext': 1.0, 'middle_ext': 1.0, 'ring_ext': 0.1, 'pinky_ext': 0.1, 'finger_spread': 0.3},
        'W': {'thumb_ext': 0.2, 'index_ext': 1.0, 'middle_ext': 1.0, 'ring_ext': 1.0, 'pinky_ext': 0.1, 'finger_spread': 0.25},
        'Y': {'thumb_ext': 1.0, 'index_ext': 0.1, 'middle_ext': 0.1, 'ring_ext': 0.1, 'pinky_ext': 1.0, 'thumb_spread': 0.8},
        'HELLO': {'thumb_ext': 0.9, 'index_ext': 1.0, 'middle_ext': 1.0, 'ring_ext': 1.0, 'pinky_ext': 1.0, 'finger_spread': 0.35},
        'YES': {'thumb_ext': 0.3, 'index_ext': 0.1, 'middle_ext': 0.1, 'ring_ext': 0.1, 'pinky_ext': 0.1},
        'NO': {'thumb_ext': 0.4, 'index_ext': 0.7, 'middle_ext': 0.7, 'ring_ext': 0.1, 'pinky_ext': 0.1, 'finger_spread': 0.05},
        'I LOVE YOU': {'thumb_ext': 1.0, 'index_ext': 1.0, 'middle_ext': 0.1, 'ring_ext': 0.1, 'pinky_ext': 1.0, 'thumb_spread': 0.6},
        'SPACE': {'thumb_ext': 0.9, 'index_ext': 0.3, 'middle_ext': 0.3, 'ring_ext': 0.3, 'pinky_ext': 0.3, 'thumb_spread': 0.9}
    }

    data = []
    labels = []
    samples_per_sign = 160

    np.random.seed(42)
    for label, params in gestures.items():
        base_features = build_hand(**params)
        for _ in range(samples_per_sign):
            # Realistic hand jitter & sensor noise
            noise = np.random.normal(0.0, 0.035, size=base_features.shape)
            scale = np.random.uniform(0.92, 1.08)
            sample = (base_features + noise) * scale
            # Re-normalize
            max_val = np.max(np.abs(sample))
            if max_val > 0:
                sample = sample / max_val
            data.append(sample.tolist())
            labels.append(label)

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)

    print(f"[Seed Generator] Saved {len(data)} training samples across {len(gestures)} signs to {DATA_PATH}")
    return data, labels


def train(data_path=DATA_PATH, model_path=MODEL_PATH):
    """Loads dataset, trains Random Forest model, and saves to model.p."""
    print("=" * 60)
    print("      AI SIGN LANGUAGE RECOGNITION - MODEL TRAINING")
    print("=" * 60)

    if not os.path.exists(data_path):
        print(f"[Train] Dataset not found at {data_path}. Creating initial seed dataset...")
        data, labels = create_seed_dataset()
    else:
        print(f"[Train] Loading existing dataset from {data_path}...")
        with open(data_path, "rb") as f:
            dataset = pickle.load(f)
            data = dataset["data"]
            labels = dataset["labels"]

    X = np.asarray(data, dtype=np.float32)
    y = np.asarray(labels)

    print(f"[Train] Total samples: {X.shape[0]} | Feature dimension: {X.shape[1]}")
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"[Train] Classes ({len(unique_labels)}):")
    for lbl, count in zip(unique_labels, counts):
        print(f"   - {lbl:<12}: {count} samples")

    # Split dataset: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )

    print("\n[Train] Training Random Forest Classifier (100 estimators)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[Train] >>> Test Set Accuracy: {accuracy * 100:.2f}% <<<")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save to model.p
    model_payload = {
        "model": clf,
        "labels": sorted(list(unique_labels)),
        "accuracy": float(accuracy),
        "feature_count": 63
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_payload, f)

    print(f"[Train] Model successfully saved to: {model_path}")
    print("=" * 60)
    return clf, accuracy


if __name__ == "__main__":
    train()
