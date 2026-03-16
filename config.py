# ── Model path ─────────────────────────────────────────────
# Download from: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
MODEL_PATH = r"path/to/face_landmarker.task"

# ── Sprite paths ───────────────────────────────────────────
# All sprites should be PNG or WEBP with transparency (alpha channel)
EYE_OPEN_PATH    = r"path/to/eye_open.png"
L_EYE_OPEN_PATH  = r"path/to/left_eye_open.png"
R_EYE_OPEN_PATH  = r"path/to/right_eye_open.png"
EYE_CLOSED_PATH  = r"path/to/eye_closed.png"
MOUTH_OPEN_PATH  = r"path/to/mouth_open.png"
MOUTH_CLOSED_PATH= r"path/to/mouth_closed.png"
L_BROW_PATH      = r"path/to/left_brow.png"
R_BROW_PATH      = r"path/to/right_brow.png"

# ── Landmark indices ───────────────────────────────────────
LEFT_EYE   = [33, 160, 158, 133, 153, 144]
RIGHT_EYE  = [362, 385, 387, 263, 373, 380]
MOUTH      = [61, 39, 0, 269, 291, 405, 17, 181]
LEFT_BROW  = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]
NOSE_TIP   = 1

# ── Thresholds — tune these to fit your face ──────────────
EAR_THRESHOLD   = 0.17   # below this -> eye closed
MAR_THRESHOLD   = 0.3    # above this -> mouth open
SMILE_THRESHOLD = 0.035  # above this -> smile detected
GAZE_SCALE      = 8.0    # iris movement sensitivity
CALIB_FRAMES    = 30     # frames collected during calibration
SMOOTH          = 10     # smoothing buffer size (larger = smoother but slower)
