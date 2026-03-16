import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from config     import *
from face_utils import *

# ── Load detector & sprites ────────────────────────────────
detector     = create_detector(MODEL_PATH)
eye_open     = cv2.imread(EYE_OPEN_PATH,     cv2.IMREAD_UNCHANGED)
l_eye_open   = cv2.imread(L_EYE_OPEN_PATH,  cv2.IMREAD_UNCHANGED)
r_eye_open   = cv2.imread(R_EYE_OPEN_PATH,  cv2.IMREAD_UNCHANGED)
eye_closed   = cv2.imread(EYE_CLOSED_PATH,  cv2.IMREAD_UNCHANGED)
mouth_open   = cv2.imread(MOUTH_OPEN_PATH,  cv2.IMREAD_UNCHANGED)
mouth_closed = cv2.imread(MOUTH_CLOSED_PATH,cv2.IMREAD_UNCHANGED)
l_brow       = cv2.imread(L_BROW_PATH,      cv2.IMREAD_UNCHANGED)
r_brow       = cv2.imread(R_BROW_PATH,      cv2.IMREAD_UNCHANGED)

# ── Calibration state ──────────────────────────────────────
calib_l, calib_r          = [], []
l_base_ox = l_base_oy     = 0
r_base_ox = r_base_oy     = 0
pitch_base = yaw_base     = 1.0
nose_base_x = nose_base_y = 0
calibrated                = False

# ── Smoothing buffers ──────────────────────────────────────
smooth_lx   = deque(maxlen=SMOOTH)
smooth_ly   = deque(maxlen=SMOOTH)
smooth_rx   = deque(maxlen=SMOOTH)
smooth_ry   = deque(maxlen=SMOOTH)
smooth_mx   = deque(maxlen=SMOOTH)
smooth_my   = deque(maxlen=SMOOTH)
smooth_roll = deque(maxlen=SMOOTH)

# ── Main loop ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results  = detector.detect(mp_image)

    if not calibrated:
        cv2.putText(frame, f"Look straight ahead... ({len(calib_l)}/{CALIB_FRAMES})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            h, w, _ = frame.shape

            l_ox, l_oy = get_gaze_offset(face_landmarks, LEFT_EYE,  468, w, h)
            r_ox, r_oy = get_gaze_offset(face_landmarks, RIGHT_EYE, 473, w, h)
            roll  = get_roll(face_landmarks,  w, h)
            yaw   = get_yaw(face_landmarks,   w, h) / yaw_base
            pitch = get_pitch(face_landmarks, w, h) / pitch_base

            nose_x = int(face_landmarks[NOSE_TIP].x * w)
            nose_y = int(face_landmarks[NOSE_TIP].y * h)

            # Calibration phase
            if not calibrated:
                calib_l.append((l_ox, l_oy))
                calib_r.append((r_ox, r_oy))
                if len(calib_l) >= CALIB_FRAMES:
                    l_base_ox   = int(np.mean([x[0] for x in calib_l]))
                    l_base_oy   = int(np.mean([x[1] for x in calib_l]))
                    r_base_ox   = int(np.mean([x[0] for x in calib_r]))
                    r_base_oy   = int(np.mean([x[1] for x in calib_r]))
                    nose_base_x = nose_x
                    nose_base_y = nose_y
                    pitch_base  = get_pitch(face_landmarks, w, h)
                    yaw_base    = get_yaw(face_landmarks,   w, h)
                    calibrated  = True
                continue

            anchor_dx = nose_x - nose_base_x
            anchor_dy = nose_y - nose_base_y

            # Blendshape values
            mouth_val        = get_mouth_open(results.face_blendshapes[0])
            l_smile, r_smile = get_smile(results.face_blendshapes[0])

            # EAR for blink / wink
            l_ear = get_ear(face_landmarks, LEFT_EYE,  w, h)
            r_ear = get_ear(face_landmarks, RIGHT_EYE, w, h)

            # Iris positions
            l_cx = int(face_landmarks[468].x * w)
            l_cy = int(face_landmarks[468].y * h)
            r_cx = int(face_landmarks[473].x * w)
            r_cy = int(face_landmarks[473].y * h)
            m_cx, m_cy = get_center(face_landmarks, MOUTH, w, h)

            # Correct gaze with calibration baseline
            l_ox -= l_base_ox;  l_oy -= l_base_oy
            r_ox -= r_base_ox;  r_oy -= r_base_oy

            # Push to smoothing buffers
            smooth_lx.append(l_cx - int(l_ox * GAZE_SCALE) + anchor_dx)
            smooth_ly.append(l_cy + int(l_oy * GAZE_SCALE) + anchor_dy)
            smooth_rx.append(r_cx - int(r_ox * GAZE_SCALE) + anchor_dx)
            smooth_ry.append(r_cy + int(r_oy * GAZE_SCALE) + anchor_dy)
            smooth_mx.append(m_cx + anchor_dx)
            smooth_my.append(m_cy + anchor_dy)
            smooth_roll.append(roll)

            # Smoothed values
            sl_cx = int(np.mean(smooth_lx))
            sl_cy = int(np.mean(smooth_ly))
            sr_cx = int(np.mean(smooth_rx))
            sr_cy = int(np.mean(smooth_ry))
            sm_cx = int(np.mean(smooth_mx))
            sm_cy = int(np.mean(smooth_my))
            sroll = float(np.mean(smooth_roll))

            # Part sizes and centers
            l_size       = get_size(face_landmarks, LEFT_EYE,   w, h)
            r_size       = get_size(face_landmarks, RIGHT_EYE,  w, h)
            m_size       = get_size(face_landmarks, MOUTH,      w, h)
            l_brow_size  = get_size(face_landmarks, LEFT_BROW,  w, h)
            r_brow_size  = get_size(face_landmarks, RIGHT_BROW, w, h)
            l_brow_cx, l_brow_cy = get_center(face_landmarks, LEFT_BROW,  w, h)
            r_brow_cx, r_brow_cy = get_center(face_landmarks, RIGHT_BROW, w, h)

            # Eye sprite selection (blink / wink / open)
            if l_ear < EAR_THRESHOLD and r_ear < EAR_THRESHOLD:
                l_sprite, r_sprite = eye_closed, eye_closed
            elif l_ear < EAR_THRESHOLD:   # left wink
                l_sprite, r_sprite = l_brow, r_eye_open
            elif r_ear < EAR_THRESHOLD:   # right wink
                l_sprite, r_sprite = l_eye_open, r_brow
            else:
                l_sprite, r_sprite = l_eye_open, r_eye_open

            # Mouth sprite selection
            if l_smile > SMILE_THRESHOLD and r_smile <= SMILE_THRESHOLD:
                m_sprite = cv2.flip(r_brow, 0)
            elif r_smile > SMILE_THRESHOLD and l_smile <= SMILE_THRESHOLD:
                m_sprite = cv2.flip(l_brow, 0)
            elif l_smile > SMILE_THRESHOLD and r_smile > SMILE_THRESHOLD:
                m_sprite = mouth_open
            else:
                m_sprite = mouth_open if mouth_val > MAR_THRESHOLD else mouth_closed

            # Render sprites
            overlay_sprite_rotated(frame, l_sprite, sl_cx,      sl_cy,      l_size,        sroll, yaw,   pitch)
            overlay_sprite_rotated(frame, r_sprite, sr_cx,      sr_cy,      r_size,        sroll, 1/yaw, pitch)
            overlay_sprite_rotated(frame, m_sprite, sm_cx,      sm_cy,      m_size,        sroll)
            overlay_sprite_rotated(frame, l_brow,   l_brow_cx + anchor_dx,  l_brow_cy + anchor_dy, l_brow_size//2, sroll)
            overlay_sprite_rotated(frame, r_brow,   r_brow_cx + anchor_dx,  r_brow_cy + anchor_dy, r_brow_size//2, sroll)

            # Debug overlay
            cv2.putText(frame, f"EAR: {l_ear:.2f}  MOUTH: {mouth_val:.2f}  Roll: {sroll:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"SMILE L: {l_smile:.2f}  R: {r_smile:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Vtuber', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
