import time
import cv2
import numpy as np

#Classic tracker to compare with MediaPipe (as in Lab 1):
#a fluorescent piece of paper on the fingertip, found by HSV thresholding.
#   h: take the colour of the centre of the image (put the paper on the white cross)
#   c: clear stroke | s: save screenshot | q: quit
#Run from src. At the end it prints the same measures as test_tracker.py.

output_dir = "../output"
camera_index = 0
lower_hsv = np.array([140, 80, 80])     # fluorescent pink (OpenCV hue goes from 0 to 179)
upper_hsv = np.array([175, 255, 255])
hue_margin = 10                          # used by the key h
min_area = 150                           # [px] smaller blobs are noise
kernel = np.ones((5, 5), np.uint8)


def find_marker(img_hsv: np.ndarray):
    """
    Finds the coloured marker by HSV thresholding.

    Args:
        img_hsv: HSV image.

    Returns:
        (center, mask): center (x, y) of the biggest blob, or None if there is none; and the binary mask.
    """
    mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)     # remove small noise
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < min_area:
        return None, mask
    m = cv2.moments(biggest)
    center = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
    return center, mask


def add_point(stroke: list, point):
    """Adds a point to a stroke; None separates strokes (only one None in a row)."""
    if point is not None or (stroke and stroke[-1] is not None):
        stroke.append(point)


def draw_stroke(img: np.ndarray, stroke: list, color: tuple):
    """Draws a stroke as lines between consecutive points (skipping the None separators)."""
    for p, q in zip(stroke[:-1], stroke[1:]):
        if p is not None and q is not None:
            cv2.line(img, p, q, color, 3)


def jitter(stroke: list) -> float:
    """Tremor of a stroke: mean size of the second difference |p[i+1] - 2 p[i] + p[i-1]| in pixels."""
    values = []
    for a, b, c in zip(stroke[:-2], stroke[1:-1], stroke[2:]):
        if a is not None and b is not None and c is not None:
            values.append(np.hypot(c[0] - 2 * b[0] + a[0], c[1] - 2 * b[1] + a[1]))
    return np.mean(values) if values else 0.0


cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

stroke = []
n_frames, n_lost = 0, 0
start_time = prev_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received")
        break
    display = cv2.flip(frame, 1)
    img_hsv = cv2.cvtColor(display, cv2.COLOR_BGR2HSV)
    n_frames += 1

    center, mask = find_marker(img_hsv)
    if center is None:
        n_lost += 1
    else:
        cv2.circle(display, center, 8, (0, 255, 0), -1)
    add_point(stroke, center)
    draw_stroke(display, stroke, (0, 255, 0))

    #white cross in the centre, for the key h
    cx, cy = display.shape[1] // 2, display.shape[0] // 2
    cv2.drawMarker(display, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

    now = time.time()
    fps = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
    prev_time = now
    cv2.putText(display, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display, f"Lost: {100 * n_lost / n_frames:.1f} %", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    #image and mask side by side
    cv2.imshow("Color tracker", cv2.hconcat([display, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        stroke = []
    elif key == ord("s"):
        path = f"{output_dir}/color_tracker_{int(now)}.png"
        cv2.imwrite(path, display)
        print("Saved:", path)
    elif key == ord("h"):
        h, s, v = img_hsv[cy, cx]
        lower_hsv = np.array([max(int(h) - hue_margin, 0), 80, 80])
        upper_hsv = np.array([min(int(h) + hue_margin, 179), 255, 255])
        print(f"HSV in the centre: ({h}, {s}, {v}) -> range {lower_hsv} - {upper_hsv}")

cap.release()
cv2.destroyAllWindows()

print(f"Frames: {n_frames} | Lost: {100 * n_lost / max(n_frames, 1):.1f} %")
print(f"Mean FPS: {n_frames / (time.time() - start_time):.1f}")
print(f"Jitter: {jitter(stroke):.2f} px")