import cv2
import numpy as np

CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
               (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
               (15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


def normalize_landmarks(hand):
    pts = np.array([[p.x, p.y, p.z] for p in hand])
    pts = pts - pts[0]
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    return (pts / (scale if scale > 0 else 1)).flatten()


def draw_hand(frame, hand):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in hand]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


CUBE_VERTS = np.array([[x, y, z] for x in (-1, 1)
                       for y in (-1, 1) for z in (-1, 1)], dtype=float)
CUBE_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8)
              if np.sum(CUBE_VERTS[i] != CUBE_VERTS[j]) == 1]


def rotation_matrix(rx, ry):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Ry @ Rx


def draw_cube(frame, center, size, rx, ry, color=(0, 255, 255)):
    pts3d = CUBE_VERTS @ rotation_matrix(rx, ry).T
    cam_dist = 4.0
    pts2d = []
    for x, y, z in pts3d:
        f = cam_dist / (cam_dist + z)
        pts2d.append((int(center[0] + x * size * f),
                      int(center[1] + y * size * f)))
    for i, j in CUBE_EDGES:
        cv2.line(frame, pts2d[i], pts2d[j], color, 2)
    for p in pts2d:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)