import time
from collections import deque, Counter
import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from utils import normalize_landmarks, draw_hand, draw_cube

MIN_SIZE, MAX_SIZE = 30, 250
PINCH_RANGE = 0.35
ROT_SPEED = 0.01
SMOOTH = 0.3
CONF_MIN = 0.6
START = dict(rx=0.5, ry=0.5, size=100.0)
ACTIONS = {"open_palm": "MOVE", "pinch": "SCALE",
           "point": "ROTATE", "fist": "FROZEN"}

model = joblib.load("models/gesture_model.pkl")
options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)
detector = vision.HandLandmarker.create_from_options(options)

rx, ry, size = START["rx"], START["ry"], START["size"]
center, prev_tip = None, None
history = deque(maxlen=5)
cap = cv2.VideoCapture(0)
timestamp, prev_time = 0, time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    if center is None:
        center = np.array([w / 2, h / 2])

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp += 33
    result = detector.detect_for_video(mp_image, timestamp)

    label, conf, active = "No hand", 0.0, None
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        draw_hand(frame, hand)

        features = normalize_landmarks(hand).reshape(1, -1)
        probs = model.predict_proba(features)[0]
        best = probs.argmax()
        history.append(model.classes_[best])
        label = Counter(history).most_common(1)[0][0]
        conf = probs[best]
        active = label if conf >= CONF_MIN else None

        def pt(i):
            return np.array([hand[i].x * w, hand[i].y * h])

        if active == "open_palm":
            center = center + SMOOTH * (pt(9) - center)
        elif active == "pinch":
            ratio = np.linalg.norm(pt(4) - pt(8)) / np.linalg.norm(pt(0) - pt(9))
            t = np.clip(ratio / PINCH_RANGE, 0, 1)
            target = MIN_SIZE + t * (MAX_SIZE - MIN_SIZE)
            size = size + SMOOTH * (target - size)
        elif active == "point":
            tip = pt(8)
            if prev_tip is not None:
                dx, dy = tip - prev_tip
                ry -= dx * ROT_SPEED
                rx += dy * ROT_SPEED
            prev_tip = tip
        if active != "point":
            prev_tip = None
    else:
        history.clear()
        prev_tip = None

    color = (0, 140, 255) if active == "fist" else (0, 255, 255)
    draw_cube(frame, center, size, rx, ry, color)

    now = time.time()
    fps = 1 / (now - prev_time) if now > prev_time else 0
    prev_time = now

    cv2.putText(frame, f"Gesture: {label} ({conf * 100:.0f}%)", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Action: {ACTIONS.get(active, '-')}", (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}  Size: {size:.0f}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, "R reset | Q quit", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imshow("GestureAR", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        rx, ry, size = START["rx"], START["ry"], START["size"]
        center = np.array([w / 2, h / 2])

cap.release()
cv2.destroyAllWindows()