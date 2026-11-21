import os
import cv2
import numpy as np
import ui

# ---------------------------------------------------------------
# Record own digit strokes to retrain the SVM (MNIST + our strokes)
#   Write the digit shown on screen inside the box, then:
#   SPACE: save and go to the next digit | c: clear | q: quit
# Run from src. The crops are saved in ../data/strokes/<digit>/<name>_<n>.png
# ---------------------------------------------------------------

output_dir = "../data/strokes"
samples_per_digit = 7           # per person (about 20 per digit between the three)
stroke_thickness = 12           # same thickness as main.py
min_pixels = 250                # an almost empty box is not saved

# Same canvas and number box as the Calculator, taken from ui.py so they always match the app
canvas_width, canvas_height = ui.width, ui.height
box = ui.calculator_boxes["num1"]       # x1, y1, x2, y2 (210 x 190 px)

mouse = {"pos": None, "down": False}


def on_mouse(event, x, y, flags, param):
    """Saves the position of the mouse and whether the left button is held (called by OpenCV)."""
    mouse["pos"] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse["down"] = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse["down"] = False


def inside_box(point: tuple) -> bool:
    """
    Checks if a point is inside the writing box.

    Args:
        point: (x, y) position, or None.

    Returns:
        True if the point is inside the box, False otherwise (also if it is None).
    """
    if point is None:
        return False
    x1, y1, x2, y2 = box
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2


def make_sequence(per_digit: int) -> list:
    """
    Order in which the digits are asked: 0 1 2 ... 9 repeated in rounds,
    so the same digit is never written twice in a row.

    Args:
        per_digit: how many times every digit is asked.

    Returns:
        List of digits, e.g. [0, 1, ..., 9, 0, 1, ..., 9] for per_digit = 2.
    """
    return [digit for _ in range(per_digit) for digit in range(10)]


def next_path(digit: int, name: str) -> str:
    """
    First free file name for a new sample, so nothing is ever overwritten.

    Args:
        digit: the digit written (0-9); it is also the name of its folder.
        name: name of the person who writes it.

    Returns:
        Path of the new file, e.g. "../data/strokes/7/sofia_003.png". The folder is created if needed.
    """
    folder = os.path.join(output_dir, str(digit))
    os.makedirs(folder, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(folder, f"{name}_{n:03d}.png")):
        n += 1
    return os.path.join(folder, f"{name}_{n:03d}.png")


def draw_screen(canvas: np.ndarray, digit: int, done: int, total: int, message: str) -> np.ndarray:
    """
    Draws the screen: the strokes, the box, the digit to write, the progress and the keys.

    Args:
        canvas: the strokes (white on black, uint8).
        digit: the digit the user has to write now.
        done: how many samples have been saved.
        total: how many samples have to be saved in total.
        message: short text shown under the progress (e.g. "Saved!").

    Returns:
        Colour image (BGR) ready for cv2.imshow.
    """
    screen = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    x1, y1, x2, y2 = box
    cv2.rectangle(screen, (x1, y1), (x2, y2), (180, 180, 0), 2)
    cv2.putText(screen, f"Write: {digit}", (280, 160), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 200, 255), 4)
    cv2.putText(screen, f"{done} / {total}", (280, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(screen, message, (280, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(screen, "SPACE: save | c: clear | q: quit", (30, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return screen


if __name__ == "__main__":
    name = input("Your name: ").strip().lower().replace(" ", "_") or "anon"
    sequence = make_sequence(samples_per_digit)

    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)   # strokes: white (255) on black (0)
    prev_pos = None
    index, message = 0, ""

    window = "Capture strokes"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)   # with the tracker, the pen position and gesture go here instead

    while index < len(sequence):
        digit = sequence[index]

        # Writing: only while the button is down and both points are inside the box
        pos = mouse["pos"] if mouse["down"] else None
        if pos is not None and prev_pos is not None and inside_box(pos) and inside_box(prev_pos):
            cv2.line(canvas, prev_pos, pos, 255, stroke_thickness)
        prev_pos = pos

        cv2.imshow(window, draw_screen(canvas, digit, index, len(sequence), message))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas[:] = 0
            message = ""
        elif key == 32:     # SPACE
            x1, y1, x2, y2 = box
            crop = canvas[y1:y2, x1:x2]
            if cv2.countNonZero(crop) < min_pixels:
                message = "Write something first"
                continue
            path = next_path(digit, name)
            cv2.imwrite(path, crop)
            print("Saved:", path)
            canvas[:] = 0
            index += 1
            message = "Saved!"

    cv2.destroyAllWindows()
    print(f"Done: {index} strokes saved in {output_dir}")