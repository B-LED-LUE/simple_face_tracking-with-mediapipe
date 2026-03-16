# Face Tracking Vtuber

A real-time face tracking Vtuber prototype built with MediaPipe and OpenCV.  
Each facial feature is tracked frame-by-frame and mapped to a 2D sprite overlay.

---

## Features

- **3-axis head rotation** — Roll, Yaw, Pitch tracked simultaneously
- **Blink & wink detection** — per-eye blink via MediaPipe blendshapes
- **Mouth tracking** — open/close detection using jawOpen blendshape
- **Smile detection** — left/right smile via mouthSmile blendshapes
- **Gaze tracking** — iris position relative to eye center
- **Real-time calibration** — sets a neutral baseline on startup
- **Moving average smoothing** — reduces jitter on all tracked values

---

## Installation

```bash
pip install -r requirements.txt
```

Download the MediaPipe face landmark model:  
https://developers.google.com/mediapipe/solutions/vision/face_landmarker

---

## Usage

1. Edit `config.py` and set your file paths:
   ```python
   MODEL_PATH = r"C:\your\path\face_landmarker.task"
   EYE_OPEN_PATH = r"C:\your\path\eye_open.png"
   # ... (set all sprite paths)
   ```
2. Adjust thresholds in `config.py` to fit your face
3. Run:
   ```bash
   python main.py
   ```
4. Look straight ahead and wait for calibration to complete
5. Press `q` to quit

---

## Project Structure

```
├── main.py          # Main loop — webcam capture, calibration, rendering
├── face_utils.py    # Face tracking functions & sprite overlay
├── config.py        # Paths, landmark indices, hyperparameters
├── requirements.txt
└── README.md
```

---

## How It Works

MediaPipe detects 478 facial landmarks and 52 blendshape scores per frame.  
The landmark coordinates are used to compute head rotation (Roll/Yaw/Pitch) and gaze offset.  
Blendshape scores drive blink, mouth, and smile detection.  
All values are smoothed with a moving average buffer before being applied to sprites.

---

## Notes

- Sprite images are excluded from this repo via `.gitignore` — bring your own assets
- Thresholds in `config.py` may need tuning depending on your face and lighting
- Currently uses `VisionRunningMode.IMAGE` — switching to `LIVE_STREAM` can improve FPS
