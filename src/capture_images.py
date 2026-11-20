import os
import glob
import cv2

camera_index = 0                  # built-in webcam (change to 1 if it does not open)
width, height = 640, 480          # same resolution used later in the project
chessboard_shape = (9, 6)         # inner corners of the printed pattern (10x7 squares)
output_folder = "../data/calibration/images"


def write_image(output_folder: str, img_name: str, img):
    os.makedirs(output_folder, exist_ok=True)
    img_path = os.path.join(output_folder, img_name)
    cv2.imwrite(img_path, img)


cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

n_saved = len(glob.glob(os.path.join(output_folder, "*.png")))   # continue numbering, never overwrite
print("SPACE: save image | q: quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received")
        break

    # Fast check only to know if the pattern is visible (the real detection is done in the notebook)
    found, corners = cv2.findChessboardCorners(frame, chessboard_shape, flags=cv2.CALIB_CB_FAST_CHECK)
    preview = cv2.drawChessboardCorners(frame.copy(), chessboard_shape, corners, found)
    cv2.putText(preview, f"Saved: {n_saved}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Calibration capture", preview)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" ") and found:
        write_image(output_folder, f"calib_{n_saved:03d}.png", frame)   # save the clean frame
        n_saved += 1
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
