"""
Reads what the user has written in the boxes of the canvas.

main.py calls two functions with a crop of one box (white strokes = 255 on a black background = 0):
    predict_number(crop) -> the number written, as text, e.g. "12"
    predict_symbol(crop) -> the operator written: "+", "-", "x" or "/"

Digits are recognised with the SVM trained in digits.ipynb (MNIST).
Operators are NOT in MNIST, so they are recognised with simple rules about the shape of the strokes.
If something cannot be read, the functions return "?".
"""
import os
import cv2
import numpy as np


# ---------------- Parameters ----------------
model_path = "../data/models/svm_digits.xml"   

# Starting values, chosen from the stroke thickness of main.py (12 px). Confirm them with the camera.

# Pieces with fewer white pixels are noise (bits of strokes from the next box that get into the crop).
# It must be well below the size of the dots of the ÷ (a circle of 12 px diameter ≈ 113 pixels).
min_pixels = 30

# Pieces with fewer white pixels are not digits. A tap of the finger is about 110-150 pixels,
# while a real digit (even a short "1") has more than 400.
min_digit_pixels = 250

# Two pieces are the same digit if they overlap horizontally at least this fraction of the narrower one.
# Example: a 5 whose top bar does not touch the rest.
join_overlap = 0.5
# Two digits that touch are read as one. Splitting them by the column with the least ink was tried, but
# it also split a 7 with a long bar ("87"), so the user is asked to leave a gap between digits instead.

# In an operator, pieces smaller than this fraction of the biggest one are specks and are ignored.
speck_ratio = 0.2

# A single stroke this many times longer than thick is a line ("-" or "/").
# A line of 30 px is (30 + 12) / 12 = 3.5. A "+" or an "x" is about 1 (at most 1.8 in the air tests).
# A small hook at one end makes the line look thicker: a "-" with a hook gave 2.97, so 3 was too high.
line_ratio = 2.5

# A line tilted up to this angle (in degrees) is a minus. More tilted, it is a slash "/".
minus_max_angle = 40


# ---------------- Model ----------------
# It is loaded only once, when main.py imports this file
if os.path.exists(model_path):
    svm = cv2.ml.SVM_load(model_path)
else:
    svm = None
    print(f"digits.py: model not found ({model_path}), every digit will be read as '?'")


# ---------------- Steps ----------------
def find_pieces(crop: np.ndarray, min_size: int) -> tuple:
    """
    Finds the separate pieces of stroke in a box.

    A "piece" is a group of white pixels that touch each other (a connected component).
    Example: in a box with "12" there are 2 pieces: the "1" and the "2".

    Args:
        crop: image of one box of the canvas (uint8), white strokes (255) on a black background (0).
        min_size: pieces with fewer white pixels than this are ignored.

    Returns:
        labels: image of the same size as crop where every pixel has the number of its piece (0 = background).
        stats: one row per piece number: (x, y, width, height, number of pixels).
        pieces: the numbers of the pieces that are big enough, sorted from left to right.
    """
    binary = (crop > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    # Piece 0 is the background, so we start at 1
    pieces = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_size]
    pieces.sort(key=lambda i: stats[i, cv2.CC_STAT_LEFT])
    return labels, stats, pieces


def join_digits(stats: np.ndarray, pieces: list) -> list:
    """
    Groups the pieces into digits: pieces that overlap horizontally are the same digit.

    When writing in the air the finger is often lifted in the middle of a digit (4, 5, 7 with a bar),
    so one digit can be made of several pieces, one above the other.

    Args:
        stats: the stats returned by find_pieces.
        pieces: the piece numbers returned by find_pieces, sorted from left to right.

    Returns:
        List of digits, from left to right. Each digit is a list of piece numbers. Example: [[1], [2, 3]].
    """
    digits = []
    for i in pieces:
        x1 = stats[i, cv2.CC_STAT_LEFT]
        x2 = x1 + stats[i, cv2.CC_STAT_WIDTH]
        if digits:
            # Horizontal overlap with the previous digit (negative if there is a gap between them)
            prev_x1, prev_x2 = digits[-1]["x1"], digits[-1]["x2"]
            overlap = min(x2, prev_x2) - max(x1, prev_x1)
            if overlap >= join_overlap * min(x2 - x1, prev_x2 - prev_x1):
                digits[-1]["pieces"].append(i)
                digits[-1]["x2"] = max(x2, prev_x2)
                continue
        digits.append({"x1": x1, "x2": x2, "pieces": [i]})
    return [d["pieces"] for d in digits]


def to_mnist(digit: np.ndarray) -> np.ndarray:
    """
    Converts the image of ONE digit to the format the SVM was trained with (MNIST).

    MNIST format: 28x28 pixels, the digit fitted in a 20x20 square and centred by its centre of mass,
    values from 0 to 1. If the image is not like the MNIST images, the SVM fails even if the digit is clear.

    Args:
        digit: image (uint8) with only one digit in white (255) on black (0), of any size.
               It must have at least one white pixel.

    Returns:
        Array of shape (1, 784) and type float32 with values from 0 to 1: one row ready for svm.predict.
    """
    # 1. Crop the digit: keep only the rectangle that contains the white pixels
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(digit))
    digit = digit[y:y+h, x:x+w]

    # 2. Scale it so its largest side is 20 pixels, keeping the proportions
    #    (a "1" must stay thin: if we stretched it to 20x20 it would become a black square)
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 3. Paste it in the middle of a black 28x28 image
    img = np.zeros((28, 28), dtype=np.uint8)
    x0, y0 = (28 - new_w) // 2, (28 - new_h) // 2
    img[y0:y0 + new_h, x0:x0 + new_w] = digit

    # 4. Move it so its centre of mass (the "average position" of the white pixels) is at the centre (14, 14).
    #    MNIST was prepared this way, so we do the same.
    m = cv2.moments(img)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]          # centre of mass
    shift = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])     # translation matrix: move (14 - cx, 14 - cy)
    img = cv2.warpAffine(img, shift, (28, 28))

    # 5. The SVM expects one row of 784 float values between 0 and 1 (as in digits.ipynb)
    return img.reshape(1, -1).astype(np.float32) / 255.0


def is_division(stats: np.ndarray, pieces: list) -> bool:
    """
    Checks if the pieces form a ÷: a wide line with a dot above it and another dot below it.

    Args:
        stats: the stats returned by find_pieces.
        pieces: the piece numbers returned by find_pieces.

    Returns:
        True if there is a dot above and a dot below the biggest piece (the line), False otherwise.
    """
    line = max(pieces, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    lx, ly, lw, lh = stats[line, :4]
    if lw < 2 * lh:     # the line of a ÷ must be wider than tall
        return False

    dot_above = dot_below = False
    for i in pieces:
        x, y, w, h = stats[i, :4]
        is_dot = i != line and w < lw / 2 and h < lw / 2      # small compared with the line
        is_over_line = lx <= x + w / 2 <= lx + lw              # its centre is within the width of the line
        if is_dot and is_over_line:
            dot_above = dot_above or y + h / 2 < ly            # its centre is above the line
            dot_below = dot_below or y + h / 2 > ly + lh       # its centre is below the line
    return dot_above and dot_below


# ---------------- Functions used by main.py ----------------
def predict_number(crop: np.ndarray) -> str:
    """
    Reads the number written in a box.

    Args:
        crop: image of one box of the canvas (uint8), white strokes (255) on a black background (0).

    Returns:
        The number as text, e.g. "12" for a box with a 1 and a 2.
        "?" if the box is empty or the model could not be loaded.
    """
    # Taps of the finger are removed BEFORE joining, so they do not get stuck to a digit
    labels, stats, pieces = find_pieces(crop, min_digit_pixels)
    if svm is None or not pieces:
        return "?"

    # One image per digit (with all its pieces), converted to MNIST, and all of them predicted at once
    digit_images = [np.isin(labels, d).astype(np.uint8) * 255 for d in join_digits(stats, pieces)]
    samples = np.vstack([to_mnist(img) for img in digit_images])     # one row per digit
    _, predictions = svm.predict(samples)                            # one predicted digit per row

    # Join the digits as text, from left to right: [1, 2] -> "12"
    return "".join(str(int(d)) for d in predictions.ravel())


def predict_symbol(crop: np.ndarray) -> str:
    """
    Reads the operator written in a box.

    The SVM cannot be used because MNIST has no operators, so simple rules are checked in this order:
        /  -> a wide line with a dot above and a dot below (÷)
        -  -> a single long and thin stroke, tilted 40° at most (a more tilted one is a slash: /)
        +  -> the edges of the strokes are mostly horizontal or vertical
        x  -> the edges of the strokes are mostly diagonal

    Args:
        crop: image of the operator box of the canvas (uint8), white strokes (255) on a black background (0).

    Returns:
        "+", "-", "x" (times) or "/" (divided by). "?" if the box is empty.
        "x" and "/" are used instead of "×" and "÷" because cv2.putText cannot draw those symbols.
    """
    labels, stats, pieces = find_pieces(crop, min_pixels)
    if not pieces:
        return "?"

    # 1. Division. It is checked BEFORE removing the specks, because its dots are small pieces
    if is_division(stats, pieces):
        return "/"

    # 2. Remove the specks: a small speck in a corner would change the rectangle of the strokes
    biggest = max(stats[i, cv2.CC_STAT_AREA] for i in pieces)
    pieces = [i for i in pieces if stats[i, cv2.CC_STAT_AREA] >= speck_ratio * biggest]
    strokes = np.isin(labels, pieces).astype(np.uint8) * 255
    points = cv2.findNonZero(strokes)

    # 3. Minus (or slash): a single stroke much longer than thick.
    #    minAreaRect is the smallest rectangle around the stroke, rotated with it, so its sides are
    #    the length and the thickness of the stroke even if it is tilted.
    if len(pieces) == 1:
        length, thickness = sorted(cv2.minAreaRect(points)[1], reverse=True)
        if length >= line_ratio * thickness:
            # fitLine gives the direction (vx, vy) of the stroke: its angle with the horizontal
            vx, vy = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).ravel()[:2]
            angle = np.degrees(np.arctan2(abs(vy), abs(vx)))
            return "-" if angle <= minus_max_angle else "/"

    # 4. Plus or times: look at the DIRECTION of the edges of the strokes.
    #    The edges of a "+" are horizontal or vertical; the edges of an "x" are diagonal.
    #    This works even if the crossing is not centred or a stroke is cut by the edge of the box.

    # 4a. Stretch the strokes to a square, so a tall and narrow "x" also has its strokes at about 45°
    x, y, w, h = cv2.boundingRect(points)
    square = cv2.resize(strokes[y:y+h, x:x+w], (60, 60))

    # 4b. Sobel gives the gradient at every pixel: how strong the edge is and its direction (0-360°)
    gx = cv2.Sobel(square, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(square, cv2.CV_32F, 0, 1, ksize=5)
    strength, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # 4c. "Fold" the directions into 0-90° (angle % 90): horizontal and vertical edges end up near 0° or 90°,
    #     diagonal edges near 45°. An edge is diagonal if it is less than 22.5° away from 45°.
    diagonal = np.abs(angle % 90 - 45) < 22.5

    # 4d. Compare how much edge strength is straight and how much is diagonal
    return "+" if strength[~diagonal].sum() > strength[diagonal].sum() else "x"
