import time
from collections import deque, Counter
import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from utils import normalize_landmarks, draw_hand

model = joblib.load("models/gesture_model.pkl")

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)
detector = vision.HandLandmarker.create_from_options(options)

history = deque(maxlen=5)
cap = cv2.VideoCapture(0)
timestamp, prev_time = 0, time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp += 33
    result = detector.detect_for_video(mp_image, timestamp)

    label, conf = "No hand", 0.0
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        draw_hand(frame, hand)
        features = normalize_landmarks(hand).reshape(1, -1)
        probs = model.predict_proba(features)[0]
        best = probs.argmax()
        history.append(model.classes_[best])
        label = Counter(history).most_common(1)[0][0]
        conf = probs[best]
    else:
        history.clear()

    now = time.time()
    fps = 1 / (now - prev_time) if now > prev_time else 0
    prev_time = now

    cv2.putText(frame, f"Gesture: {label} ({conf * 100:.0f}%)", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Live recognition (press q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()