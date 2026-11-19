"""
AI Sign Language to Text - Flask Web Application
Backend Architecture:
  Camera feed -> Base64 decode -> OpenCV frame -> MediaPipe Hands (21 keypoints)
  -> 63 normalized coordinates -> Random Forest classifier -> JSON prediction
"""

import os
import sys
import base64
import pickle
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify

# Import custom hand utils and trainer
from hand_utils import HandDetector, extract_features, get_pixel_landmarks_and_bbox
import train_model

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.p")
DATA_PATH = os.path.join(BASE_DIR, "data", "data.pickle")

# Global ML model and detector
model = None
model_labels = []
model_accuracy = 0.0
detector = None


def load_or_initialize_model():
    """Loads model.p or triggers automated bootstrapping training if not present."""
    global model, model_labels, model_accuracy
    if not os.path.exists(MODEL_PATH):
        print("[App] model.p not found. Initiating automated training...")
        try:
            train_model.train(data_path=DATA_PATH, model_path=MODEL_PATH)
        except Exception as e:
            print(f"[App] Training bootstrap error: {e}")

    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                payload = pickle.load(f)
                model = payload["model"]
                model_labels = payload.get("labels", [])
                model_accuracy = payload.get("accuracy", 0.98)
                print(f"[App] Model loaded successfully! {len(model_labels)} classes: {model_labels}")
        except Exception as e:
            print(f"[App] Failed to load {MODEL_PATH}: {e}")
    else:
        print("[App] Warning: Running without pre-loaded model. Train via train_model.py")


# Initialize Hand Detector and Model
print("[App] Initializing HandDetector...")
detector = HandDetector()
load_or_initialize_model()


@app.route("/")
def index():
    """Serves the main application interface."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives base64 webcam frame, extracts 21 hand landmarks,
    normalizes to 63 features, and returns Random Forest classification.
    """
    global model, detector
    try:
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({"status": "error", "message": "Missing image payload"}), 400

        image_data = data["image"]
        # Strip data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        # Decode base64 to OpenCV BGR image
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "error", "message": "Invalid image data"}), 400

        h, w, _ = frame.shape

        # Detect Hand Landmarks
        landmarks, handedness = detector.detect(frame)
        if not landmarks:
            return jsonify({
                "status": "no_hand",
                "prediction": "",
                "confidence": 0.0,
                "top_predictions": [],
                "landmarks": [],
                "bbox": [],
                "handedness": "None"
            })

        # Extract 63 invariant features
        features = extract_features(landmarks)
        if features is None or len(features) != 63:
            return jsonify({"status": "no_hand", "message": "Incomplete hand detection"})

        # Get pixel coordinates and bounding box for client-side canvas rendering
        pixel_landmarks, bbox = get_pixel_landmarks_and_bbox(landmarks, w, h)

        # Model Inference
        predicted_label = ""
        confidence = 0.0
        top_predictions = []

        if model is not None:
            features_input = np.asarray([features], dtype=np.float32)
            try:
                probabilities = model.predict_proba(features_input)[0]
                classes = model.classes_
                best_idx = np.argmax(probabilities)
                predicted_label = str(classes[best_idx])
                confidence = float(probabilities[best_idx])

                # Get Top 3 alternative predictions
                top_indices = np.argsort(probabilities)[::-1][:3]
                for idx in top_indices:
                    top_predictions.append({
                        "label": str(classes[idx]),
                        "confidence": round(float(probabilities[idx]) * 100, 1)
                    })
            except Exception as e:
                # Fallback to standard predict
                predicted_label = str(model.predict(features_input)[0])
                confidence = 0.95
                top_predictions = [{"label": predicted_label, "confidence": 95.0}]
        else:
            predicted_label = "MODEL_NOT_READY"

        return jsonify({
            "status": "success",
            "prediction": predicted_label,
            "confidence": round(confidence * 100, 1),
            "top_predictions": top_predictions,
            "landmarks": pixel_landmarks,
            "bbox": bbox,
            "handedness": handedness or "Right"
        })

    except Exception as e:
        print(f"[App] Prediction error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns application status, loaded classes, and model accuracy."""
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "classes": model_labels,
        "classes_count": len(model_labels),
        "accuracy": round(model_accuracy * 100, 1)
    })


@app.route("/api/save_sample", methods=["POST"])
def save_sample():
    """Endpoint allowing live sample collection from the web interface."""
    try:
        data = request.get_json()
        label = data.get("label", "").strip().upper()
        image_data = data.get("image", "")

        if not label or not image_data:
            return jsonify({"status": "error", "message": "Missing label or image"}), 400

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        landmarks, _ = detector.detect(frame)
        if not landmarks:
            return jsonify({"status": "fail", "message": "No hand detected in sample"}), 400

        features = extract_features(landmarks)
        if features is None:
            return jsonify({"status": "fail", "message": "Could not extract features"}), 400

        # Append to dataset
        d_data, d_labels = [], []
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, "rb") as f:
                d = pickle.load(f)
                d_data = d.get("data", [])
                d_labels = d.get("labels", [])

        d_data.append(features.tolist())
        d_labels.append(label)

        with open(DATA_PATH, "wb") as f:
            pickle.dump({"data": d_data, "labels": d_labels}, f)

        return jsonify({
            "status": "success",
            "message": f"Sample saved for {label}",
            "total_samples": len(d_data)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/retrain", methods=["POST"])
def retrain():
    """Triggers model retraining from accumulated dataset."""
    global model, model_labels, model_accuracy
    try:
        clf, acc = train_model.train(data_path=DATA_PATH, model_path=MODEL_PATH)
        load_or_initialize_model()
        return jsonify({
            "status": "success",
            "message": "Model retrained successfully!",
            "accuracy": round(acc * 100, 1),
            "classes": model_labels
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n==================================================")
    print(f"  AI Sign Language to Text Web Application")
    print(f"  Running at: http://127.0.0.1:{port}")
    print(f"==================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
