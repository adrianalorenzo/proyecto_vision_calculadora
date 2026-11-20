import time
import cv2

camera_index = 0                  # built-in webcam (change to 1 if it does not open)
width, height = 640, 480


def draw_fps(img, fps: float):
    cv2.putText(img, f"FPS {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
if not cap.isOpened():
    print(f"Could not open the camera (index {camera_index})")
    exit()

prev_time = time.time()
fps = 0.0
print("q or ESC: quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received")
        break

    frame = cv2.flip(frame, 1)          # mirror image, more natural to write in the air

    # FPS with a simple moving average to avoid flickering numbers
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
    prev_time = now

    draw_fps(frame, fps)
    cv2.imshow("AirMath - camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
