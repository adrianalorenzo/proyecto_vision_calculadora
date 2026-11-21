import os
import time
import cv2
import numpy as np
import ui
from digits import predict_number, predict_symbol

# Test of digits.py without the tracker: the mouse is the pen.
# Draws the same boxes as the calculator of main.py, reads them and saves every crop as an image,
# so the failures can be analysed later and the thresholds of digits.py adjusted.
#   Hold the left button: draw | SPACE: read and save | c: clear | q: quit

stroke_thickness = 12                   # the same as main.py, so the strokes are like the real ones
save_folder = "../output/digits_tests"
window = "Test digits"

# Characters that cannot be used in a file name on Windows
file_name_text = {"/": "div", "?": "unknown"}

os.makedirs(save_folder, exist_ok=True)
canvas = np.zeros((ui.height, ui.width), dtype=np.uint8)    # strokes: white (255) on black (0), as in main.py
results = {}                                                 # what was read in every box: {"num1": "12", ...}
mouse = {"pos": None, "down": False}


def on_mouse(event, x, y, flags, param):
    """Saves the position of the mouse and whether the left button is held (called by OpenCV)."""
    mouse["pos"] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse["down"] = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse["down"] = False


def crop_box(rect: tuple) -> np.ndarray:
    """
    Crops one box of the canvas, the same as main.py.

    Args:
        rect: (x1, y1, x2, y2) of the box.

    Returns:
        The part of the canvas inside the box.
    """
    x1, y1, x2, y2 = rect
    return canvas[y1:y2, x1:x2]


def read_and_save():
    """Reads every box with digits.py, shows the result in the terminal and saves every crop as a PNG."""
    stamp = time.strftime("%H%M%S")
    for name, rect in ui.calculator_boxes.items():
        crop = crop_box(rect)
        results[name] = predict_symbol(crop) if name == "op" else predict_number(crop)

        # Empty boxes are not saved
        if cv2.countNonZero(crop) > 0:
            text = file_name_text.get(results[name], results[name])
            cv2.imwrite(f"{save_folder}/{stamp}_{name}_read-{text}.png", crop)
    print(f"Read: {results['num1']} {results['op']} {results['num2']}")


cv2.namedWindow(window)
cv2.setMouseCallback(window, on_mouse)
prev_pos = None
print("Hold the left button: draw | SPACE: read and save | c: clear | q: quit")

while True:
    # 1. Draw: a line from the previous position to the current one while the button is held.
    #    As in main.py, only if both positions are inside the same box (strokes never cross into another box)
    if mouse["down"] and mouse["pos"] is not None:
        box = ui.get_box(mouse["pos"], ui.calculator_boxes)
        if prev_pos is not None and box is not None and box == ui.get_box(prev_pos, ui.calculator_boxes):
            cv2.line(canvas, prev_pos, mouse["pos"], 255, stroke_thickness)
        prev_pos = mouse["pos"]
    else:
        prev_pos = None     # the button was released: the next stroke starts a new line

    # 2. Show the canvas in colour, with the boxes and what was read above each one
    display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for name, (x1, y1, x2, y2) in ui.calculator_boxes.items():
        cv2.rectangle(display, (x1, y1), (x2, y2), ui.accent, 2)
        if name in results:
            cv2.putText(display, results[name], (x1 + 5, y1 - 10), ui.font, 1, ui.ink, 2)
    cv2.putText(display, "SPACE: read | c: clear | q: quit", (20, ui.height - 20), ui.font, 0.6, ui.white, 1)
    cv2.imshow(window, display)

    # 3. Keys
    key = cv2.waitKey(10) & 0xFF
    if key == ord(" "):
        read_and_save()
    elif key == ord("c"):
        canvas[:] = 0
        results.clear()
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
