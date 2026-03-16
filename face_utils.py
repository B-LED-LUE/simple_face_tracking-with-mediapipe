import cv2
import mediapipe as mp
import numpy as np
from config import NOSE_TIP, MOUTH

# ── Detector ───────────────────────────────────────────────

def create_detector(model_path):
    """Create and return a MediaPipe FaceLandmarker detector"""
    BaseOptions           = mp.tasks.BaseOptions
    FaceLandmarker        = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        output_face_blendshapes=True
    )
    return FaceLandmarker.create_from_options(options)

# ── Coordinate helpers ─────────────────────────────────────

def get_pts(landmarks, indices, w, h):
    """Convert landmark indices to pixel coordinates"""
    if isinstance(indices, int):
        indices = [indices]
    return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]

def get_center(landmarks, indices, w, h):
    """Mean pixel position of a landmark group"""
    pts = get_pts(landmarks, indices, w, h)
    return int(np.mean([p[0] for p in pts])), int(np.mean([p[1] for p in pts]))

def get_size(landmarks, indices, w, h):
    """Horizontal span of a landmark group — used as sprite size"""
    pts = get_pts(landmarks, indices, w, h)
    return int(max(p[0] for p in pts) - min(p[0] for p in pts)) * 2

# ── Head rotation ──────────────────────────────────────────

def get_roll(landmarks, w, h):
    """Head tilt angle (Z-axis rotation)"""
    l = (int(landmarks[33].x  * w), int(landmarks[33].y  * h))
    r = (int(landmarks[263].x * w), int(landmarks[263].y * h))
    return np.degrees(np.arctan2(r[1] - l[1], r[0] - l[0]))

def get_yaw(landmarks, w, h):
    """Head left/right rotation via eye-corner to nose distance ratio"""
    l    = (int(landmarks[33].x       * w), int(landmarks[33].y       * h))
    r    = (int(landmarks[263].x      * w), int(landmarks[263].y      * h))
    nose = (int(landmarks[NOSE_TIP].x * w), int(landmarks[NOSE_TIP].y * h))
    ld = np.linalg.norm(np.array(l) - np.array(nose))
    rd = np.linalg.norm(np.array(r) - np.array(nose))
    if rd == 0:
        return 1.0
    return np.clip(ld / rd, 0.5, 2.0)

def get_pitch(landmarks, w, h):
    """Head up/down rotation via eye-midpoint to nose vs mouth to nose ratio"""
    l    = (int(landmarks[33].x       * w), int(landmarks[33].y       * h))
    r    = (int(landmarks[263].x      * w), int(landmarks[263].y      * h))
    nose = (int(landmarks[NOSE_TIP].x * w), int(landmarks[NOSE_TIP].y * h))
    eye_mid    = ((l[0]+r[0])//2, (l[1]+r[1])//2)
    mouth_c    = get_center(landmarks, MOUTH, w, h)
    eye_dist   = np.linalg.norm(np.array(eye_mid) - np.array(nose))
    mouth_dist = np.linalg.norm(np.array(mouth_c) - np.array(nose))
    if mouth_dist == 0:
        return 1.0
    return np.clip(eye_dist / mouth_dist, 0.5, 1.5)

# ── Facial feature tracking ────────────────────────────────

def get_ear(landmarks, eye_indices, w, h):
    """Eye Aspect Ratio — low value means eye is closed"""
    pts = get_pts(landmarks, eye_indices, w, h)
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

def get_gaze_offset(landmarks, eye_indices, iris_idx, w, h):
    """Iris center minus eye center = gaze direction offset"""
    eye_cx, eye_cy = get_center(landmarks, eye_indices, w, h)
    iris_x = int(landmarks[iris_idx].x * w)
    iris_y = int(landmarks[iris_idx].y * h)
    return iris_x - eye_cx, iris_y - eye_cy

def get_mouth_open(face_blendshapes):
    """Mouth openness: jawOpen - mouthClose"""
    jaw_open = mouth_close = 0.0
    for b in face_blendshapes:
        if b.category_name == 'jawOpen':    jaw_open    = b.score
        if b.category_name == 'mouthClose': mouth_close = b.score
    return jaw_open - mouth_close

def get_smile(face_blendshapes):
    """Returns (left_smile, right_smile) blendshape scores"""
    l = r = 0.0
    for b in face_blendshapes:
        if b.category_name == 'mouthSmileLeft':  l = b.score
        if b.category_name == 'mouthSmileRight': r = b.score
    return l, r

# ── Rendering ──────────────────────────────────────────────

def overlay_sprite_rotated(frame, sprite, cx, cy, size, angle, yaw=1.0, pitch=1.0):
    """Paste a sprite onto the frame with rotation, yaw skew, and pitch scale"""
    if sprite is None or size <= 0:
        return
    yaw   = np.clip(yaw,   0.1, 3.0)
    pitch = np.clip(pitch, 0.1, 3.0)
    w_size = int(size * yaw)
    h_size = int(size * pitch)
    s = cv2.resize(sprite, (w_size, h_size))

    # Expand canvas to avoid corner clipping during rotation
    diag         = int(np.sqrt(2) * max(w_size, h_size)) + 2
    canvas_bgr   = np.zeros((diag, diag, 3), dtype=np.uint8)
    canvas_alpha = np.zeros((diag, diag),    dtype=np.uint8)

    # Place sprite at canvas center
    ox = (diag - w_size) // 2
    oy = (diag - h_size) // 2
    if s.shape[2] == 4:
        canvas_bgr  [oy:oy+h_size, ox:ox+w_size] = s[:, :, :3]
        canvas_alpha[oy:oy+h_size, ox:ox+w_size] = s[:, :,  3]
    else:
        canvas_bgr  [oy:oy+h_size, ox:ox+w_size] = s
        canvas_alpha[oy:oy+h_size, ox:ox+w_size] = 255

    # Rotate canvas
    M     = cv2.getRotationMatrix2D((diag//2, diag//2), -angle, 1.0)
    bgr   = cv2.warpAffine(canvas_bgr,   M, (diag, diag))
    alpha = cv2.warpAffine(canvas_alpha, M, (diag, diag))

    # Compute paste region on frame
    x1, y1 = cx - diag//2, cy - diag//2
    x2, y2 = x1 + diag,    y1 + diag

    # Clip to frame bounds
    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    sx1, sy1 = fx1 - x1, fy1 - y1
    sx2, sy2 = sx1 + (fx2 - fx1), sy1 + (fy2 - fy1)
    if fx1 >= fx2 or fy1 >= fy2:
        return

    # Alpha blending
    a = alpha[sy1:sy2, sx1:sx2] / 255.0
    for c in range(3):
        frame[fy1:fy2, fx1:fx2, c] = (
            a * bgr[sy1:sy2, sx1:sx2, c] + (1-a) * frame[fy1:fy2, fx1:fx2, c]
        )
