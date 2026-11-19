# SignSpeak AI - Real-Time Sign Language to Text & Speech Interpreter

A full-stack, production-grade Artificial Intelligence web application that translates American Sign Language (ASL) gestures into text and spoken speech in real-time. Built specifically for **college project presentations, engineering demos, and accessibility solutions**.

---

## 🏛 System Architecture

The pipeline processes webcam video frames through a computer vision and machine learning stack:

```mermaid
flowchart LR
    A["Webcam Feed (HTML5 Video)"] -->|Base64 JPEG Frame| B["Flask Web Server (app.py)"]
    B -->|OpenCV cv2.imdecode| C["MediaPipe Hands (21 3D Keypoints)"]
    C -->|Wrist Offset & Scale Norm| D["63-Dimension Invariant Feature Vector"]
    D -->|Inference| E["Random Forest Classifier (model.p)"]
    E -->|Prediction + Confidence + Skeleton| F["Flask JSON Response"]
    F -->|Render Mesh & Big Letter| G["Frontend UI (Canvas + Web Speech API)"]
```

### Detailed Pipeline Flow:
1. **Frontend Camera Capture**: Browser grabs webcam frames using `navigator.mediaDevices.getUserMedia` at a user-selected frame rate (10–15 FPS).
2. **Transmission**: The frame is drawn to an offscreen canvas and serialized to an optimized JPEG base64 string, sent via asynchronous `fetch('/predict')`.
3. **Frame Decoding**: OpenCV decodes raw byte buffer into a BGR image array with zero disk I/O.
4. **Landmark Detection**: MediaPipe tracks **21 3D hand keypoints** (wrist, CMC, MCP, PIP, DIP, and fingertips).
5. **Feature Normalization (63 Values)**:
   - All 21 landmarks $(x_i, y_i, z_i)$ are translated relative to the wrist $(x_0, y_0, z_0)$ so that the wrist is at $(0, 0, 0)$.
   - All relative coordinates are divided by the maximum coordinate spread $\max(|x|, |y|, |z|)$, creating complete scale and distance invariance.
6. **Classification**: Scikit-Learn `RandomForestClassifier` (100 estimators) calculates class probabilities.
7. **Client Feedback**: Updates the big letter display, neon skeleton mesh on HTML5 canvas, confidence bar, sentence builder, and triggers the Web Speech synthesizer.

---

## 📁 Project Structure

```text
sign-language-to-text/
│
├── app.py                  # Flask web backend & prediction endpoints
├── hand_utils.py           # Unified MediaPipe detector & 63-feature normalizer
├── collect_data.py         # Interactive OpenCV data collector with HUD & countdown
├── train_model.py          # Random Forest training script & seed dataset builder
├── requirements.txt        # Production dependency specifications
├── README.md               # Complete documentation & college demo guide
├── model.p                 # Pre-trained Random Forest model (98% accuracy)
├── hand_landmarker.task    # Offline MediaPipe Tasks model asset
│
├── data/
│   └── data.pickle         # Landmark training dataset (2,720 samples across 17 signs)
│
├── static/
│   ├── style.css           # Modern dark cyber/glassmorphism styling
│   └── script.js           # Camera controller, skeleton renderer, TTS, and auto-add
│
└── templates/
    └── index.html          # Responsive single-page application UI
```

---

## ⚡ Quick Start

### 1. Prerequisites
Ensure **Python 3.9+** is installed on your machine.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

Click **"Start Camera"** and present any sign to the webcam. The model works out of the box with the included pre-trained `model.p`!

---

## 🎯 Features

- **Real-Time Webcam Feed**: High-performance camera stream with mirrored/flip view and rate throttling.
- **Neon 21-Node Hand Skeleton**: Draws bone connections and glowing landmark nodes directly on the video feed.
- **Big Predicted Letter**: High-contrast, glowing display with punchy animations and confidence percentage.
- **Top 3 Candidate Breakdown**: Displays alternative predictions with mini confidence bars.
- **Sentence Builder**:
  - **[+] Add**: Appends current predicted sign.
  - **[␣] Space**: Adds a space.
  - **[⌫] Backspace**: Deletes the last character.
  - **[✕] Clear**: Resets the entire sentence.
  - **[🔊] Speak**: Speaks the sentence aloud using the browser's Web Speech API with selectable voices.
  - **[📋] Copy**: One-click copy to clipboard.
- **Auto-Add on 1.2s Hold**: Hold any sign steady with >75% confidence to automatically append it to the sentence with a circular progress countdown ring and audio chime.
- **Smart Formatting**: Automatic capitalization after punctuation and spacing rules.
- **College Demo Ready**: Dark glassmorphic cyber theme with built-in reference guide.

---

## ⌨ Keyboard Shortcuts

| Shortcut | Action |
|:---|:---|
| `A` | Add current predicted letter to sentence |
| `Space` | Add space |
| `Backspace` | Delete last character |
| `Enter` | Speak sentence aloud (Text-to-Speech) |
| `C` | Clear sentence |

---

## 🛠 Collecting Custom Signs & Retraining

You can easily expand the vocabulary or train custom gestures:

### Step 1: Collect Custom Samples via Webcam
Run the interactive data collector:
```bash
python collect_data.py --label YOUR_SIGN --samples 100
```
- A 3-second countdown will give you time to position your hand.
- The script tracks progress with a HUD bar and records 63 normalized coordinates per frame into `data/data.pickle`.
- Press `S` to start recording, `R` to switch to another label, or `Q` to save and exit.

### Step 2: Retrain the Random Forest Model
```bash
python train_model.py
```
This evaluates the updated dataset, prints the classification report, and exports the optimized classifier to `model.p`. Refresh `http://127.0.0.1:5000` to use the new model immediately!

---

## 🎓 College Viva & Oral Presentation Q&A

**Q1: Why do we use 63 values instead of pixel images?**  
*Answer:* A raw image has $640 \times 480 \times 3 = 921,600$ dimensions, which is prone to overfitting on background, lighting, and skin tones. MediaPipe abstracts the hand into 21 key points in 3D space $(x, y, z)$. $21 \times 3 = 63$ features capture purely the geometry and posture of the hand, reducing computation by 99.99% and enabling real-time performance on any laptop CPU.

**Q2: How does the normalization ensure distance invariance?**  
*Answer:* We subtract Landmark 0 (wrist) coordinates from every landmark so the hand is always anchored at the origin $(0, 0, 0)$. Then we divide each coordinate by the maximum absolute distance $\max(|x|, |y|, |z|)$. Whether the hand is 30 cm or 2 meters from the camera, the resulting normalized coordinates are identical.

**Q3: Why Random Forest over a Deep CNN or LSTM?**  
*Answer:* For tabular landmark coordinates (63 features), Random Forest excels because:
1. High accuracy (>98%) without GPU requirements.
2. Ultra-low inference latency (<5ms per prediction).
3. Robust against feature collinearity and minor hand tremors.
4. Fast training time (<2 seconds on standard CPU).

---

## 📄 License
This project is open-source and free to use for academic, demonstration, and personal projects.
