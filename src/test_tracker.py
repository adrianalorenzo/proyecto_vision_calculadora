import time 
import cv2 
import numpy as np
import tracker

#Live test of tracker.py: landmarks, bounding box, gesture, fingertip with and without Kalman, lost frames and FPS.
#Write with the index finger: red = raw fingertip, green = Kalman
#c: clear strokes | s: save screenshot | q: quit
#Run from src. At the end it prints the measures for the report.

output_dir = "../output"
camera_index = 0

def add_point(stroke: list, point):
    """Adds a point to a stroke; None separates strokes (only one None in a row)"""
    if point is not None or (stroke and stroke[-1] is not None):
        stroke.append(point)


def draw_stroke(img: np.ndarray, stroke: list, color: tuple):
    """Draws a stroke as lines between consecutive points (skipping the None separators)."""
    for p, q in zip(stroke[:-1], stroke[1:]):
        if p is not None and q is not None:
            cv2.line(img, p, q, color, 3)


def jitter(stroke: list) -> float:
    """
    Tremor of a stroke: mean size of the second difference |p[i+1] - 2 p[i] + p[i-1]| in pixels.
    The smoother the stroke, the smaller it is.
    """
    values = []
    for a, b, c in zip(stroke[:-2], stroke[1:-1], stroke[2:]):
        if a is not None and b is not None and c is not None:
            values.append(np.hypot(c[0] - 2 * b[0] + a[0], c[1] - 2 * b[1] + a[1]))
    return np.mean(values) if values else 0.0


cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
hands = tracker.create_hands()
kalman = tracker.create_kalman()

raw_stroke, kalman_stroke = [], []
n_frames, n_lost = 0, 0
start_time = prev_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received")
        break
    display = cv2.flip(frame, 1)
    n_frames += 1

    landmarks = tracker.get_landmarks(display, hands)
    gesture = None
    if landmarks is None:
        n_lost += 1
    else:
        pen = tracker.get_pen(landmarks, kalman)
        gesture = tracker.get_gesture(landmarks)
        x1, y1, x2, y2 = tracker.get_bbox(landmarks)
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 200, 0), 2)
        for p in landmarks:
            cv2.circle(display, p, 3, (0, 255, 255), -1)
        cv2.circle(display, landmarks[8], 6, (0, 0, 255), -1)    # raw fingertip
        cv2.circle(display, pen, 6, (0, 255, 0), -1)             # Kalman fingertip

    if gesture == "index":
        add_point(raw_stroke, landmarks[8])
        add_point(kalman_stroke, pen)
    else:
        add_point(raw_stroke, None)
        add_point(kalman_stroke, None)

    draw_stroke(display, raw_stroke, (0, 0, 255))
    draw_stroke(display, kalman_stroke, (0, 255, 0))

    now = time.time()
    fps = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
    prev_time = now
    cv2.putText(display, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display, f"FPS: {fps:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display, f"Lost: {100 * n_lost / n_frames:.1f} %", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Test tracker", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        raw_stroke, kalman_stroke = [], []
    elif key == ord("s"):
        path = f"{output_dir}/tracker_{int(now)}.png"
        cv2.imwrite(path, display)
        print("Saved:", path)

cap.release()
cv2.destroyAllWindows()

print(f"Frames: {n_frames} | Lost: {100 * n_lost / max(n_frames, 1):.1f} %")
print(f"Mean FPS: {n_frames / (time.time() - start_time):.1f}")
print(f"Jitter raw: {jitter(raw_stroke):.2f} px | Jitter Kalman: {jitter(kalman_stroke):.2f} px")