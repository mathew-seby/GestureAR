import csv
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

GESTURES = {ord("1"): "open_palm", ord("2"): "pinch",
            ord("3"): "point", ord("4"): "fist"}
CSV_PATH = os.path.join("data", "gestures.csv")
os.makedirs("data", exist_ok=True)


def normalize_landmarks(hand):
    pts = np.array([[p.x, p.y, p.z] for p in hand])
    pts = pts - pts[0]
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    return (pts / (scale if scale > 0 else 1)).flatten()


options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)
detector = vision.HandLandmarker.create_from_options(options)

counts = {name: 0 for name in GESTURES.values()}
new_file = not os.path.exists(CSV_PATH)
if not new_file:
    with open(CSV_PATH) as f:
        for row in csv.reader(f):
            if row and row[0] in counts:
                counts[row[0]] += 1

current, recording, timestamp = None, False, 0
cap = cv2.VideoCapture(0)

with open(CSV_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    if new_file:
        header = ["label"] + [f"{a}{i}" for i in range(21) for a in "xyz"]
        writer.writerow(header)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp += 33
        result = detector.detect_for_video(mp_image, timestamp)

        h, w = frame.shape[:2]
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            for p in hand:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 0, 255), -1)
            if recording and current:
                writer.writerow([current] + normalize_landmarks(hand).tolist())
                counts[current] += 1

        status = "RECORDING" if recording else "paused"
        cv2.putText(frame, f"Gesture: {current}  [{status}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        for i, (name, c) in enumerate(counts.items()):
            cv2.putText(frame, f"{name}: {c}", (10, 65 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Collect data (1-4 select, SPACE record, q quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in GESTURES:
            current, recording = GESTURES[key], False
        if key == 32 and current:
            recording = not recording

cap.release()
cv2.destroyAllWindows()