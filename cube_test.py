import cv2
from utils import draw_cube

rx, ry, size = 0.5, 0.5, 100
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    draw_cube(frame, (w // 2, h // 2), size, rx, ry)

    cv2.putText(frame, "W/S/A/D rotate | E bigger | C smaller | R reset | Q quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.imshow("Cube test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("a"):
        ry -= 0.1
    elif key == ord("d"):
        ry += 0.1
    elif key == ord("w"):
        rx -= 0.1
    elif key == ord("s"):
        rx += 0.1
    elif key == ord("e"):
        size = min(size + 10, 300)
    elif key == ord("c"):
        size = max(size - 10, 20)
    elif key == ord("r"):
        rx, ry, size = 0.5, 0.5, 100

cap.release()
cv2.destroyAllWindows()