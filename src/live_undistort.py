import time
import cv2
import numpy as np

# Shows the live webcam image: original (left) | undistorted (right), with FPS
camera_index = 0
params_path = "../data/calibration/params.npz"

params = np.load(params_path)
intrinsics = params["intrinsics"]
dist_coeffs = params["dist_coeffs"]
width, height = [int(v) for v in params["image_size"]]
print("Intrinsics:\n", intrinsics)
print("Distortion coefficients:\n", dist_coeffs)
print("RMS:", float(params["rms"]))

# The maps are computed only once, then cv2.remap() is cheap for every frame
map1, map2 = cv2.initUndistortRectifyMap(intrinsics, dist_coeffs, None, intrinsics, (width, height), cv2.CV_16SC2)

cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

prev_time = time.time()
fps = 0.0
print("s: save a comparison image | q: quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received")
        break

    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    comparison = np.hstack((frame, undistorted))     # original (left) vs undistorted (right)

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
    prev_time = now
    cv2.putText(comparison, f"FPS {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(comparison, "original", (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, "undistorted", (width + 10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Original | Undistorted", comparison)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        cv2.imwrite(f"../output/live_comparison_{int(now)}.png", comparison)
        print("Saved")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
