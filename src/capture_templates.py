import os
import cv2
import numpy as np
from security import find_card_corners, warp_card, card_interior, templates_folder

# Saves the template of every card with OUR camera and lighting.
# Show one card, wait until the contour is green and press its key:
#   1: pi   2: sqrt   3: inf   4: sigma   5: ac   |   q: quit
camera_index = 0
width, height = 640, 480
keys = {ord("1"): "pi", ord("2"): "sqrt", ord("3"): "inf", ord("4"): "sigma", ord("5"): "ac"}

os.makedirs(templates_folder, exist_ok=True)
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
print("1 pi | 2 sqrt | 3 inf | 4 sigma | 5 ac | q quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received")
        break

    # No mirror here: the cards are always read on the non-mirrored frame (also in main.py)
    cards = find_card_corners(frame)
    preview = frame.copy()
    interior = None
    if cards:
        corners = cards[0]                        # biggest card
        cv2.polylines(preview, [corners.astype(np.int32)], True, (0, 255, 0), 3)
        interior = card_interior(warp_card(frame, corners))
        cv2.imshow("Card interior (template)", cv2.resize(interior, None, fx=2, fy=2))
    cv2.imshow("Capture templates", preview)

    key = cv2.waitKey(1) & 0xFF
    if key in keys and interior is not None:
        path = os.path.join(templates_folder, f"{keys[key]}.png")
        cv2.imwrite(path, interior)
        print("Saved", path)
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
