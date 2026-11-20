import os
import glob
import time
import cv2
import numpy as np

# ---------------------------------------------------------------
# Security block: card detector (threshold + contours + homography
# + template matching) and decoder of the 4-card password
# ---------------------------------------------------------------

password = ["sigma", "pi", "inf", "sqrt"]     # order of the cards to unlock (change it if you want)
symbols = ["pi", "sqrt", "inf", "sigma", "ac"]
templates_folder = "../data/templates"

# Detector parameters
card_threshold = 80           # the black frame is darker than this gray level
min_card_area = 4000          # [px^2] smaller quadrilaterals are ignored
card_size = (140, 200)        # (width, height) of the straightened card, same ratio as the printed frame
frame_margin = 25             # [px] removed on each side of the straightened card to keep only the symbol
match_threshold = 0.45        # below this template matching score the card is unknown (None)

# Decoder parameters
stable_frames = 10            # a symbol must be seen this number of frames in a row to count
forget_time = 10.0            # [s] without showing any card -> the sequence is forgotten
max_fails = 3                 # wrong sequences before locking
lock_time = 30.0              # [s] locked after max_fails


# ---------------- Detector ----------------

def order_corners(corners: np.array) -> np.array:
    # Order the 4 corners as top-left, top-right, bottom-right, bottom-left
    corners = corners.reshape(4, 2).astype(np.float32)
    s = corners.sum(axis=1)
    d = np.diff(corners, axis=1).ravel()
    ordered = np.array([corners[np.argmin(s)], corners[np.argmin(d)],
                        corners[np.argmax(s)], corners[np.argmax(d)]], dtype=np.float32)
    # The card is vertical: if the top side is longer than the left side, rotate the order
    top = np.linalg.norm(ordered[1] - ordered[0])
    left = np.linalg.norm(ordered[3] - ordered[0])
    if top > left:
        ordered = np.roll(ordered, -1, axis=0)
    return ordered


def find_card_corners(frame: np.array) -> list:
    # Gray -> blur -> binarize (black frame becomes white) -> external contours -> quadrilaterals
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, card_threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_card_area:
            continue
        approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            cards.append((area, order_corners(approx)))

    cards.sort(key=lambda c: c[0], reverse=True)     # biggest card first
    return [corners for _, corners in cards]


def warp_card(frame: np.array, corners: np.array) -> np.array:
    # Homography from the 4 corners to a straight card of fixed size
    w, h = card_size
    target = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners, target)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.warpPerspective(gray, H, (w, h))


def card_interior(warped: np.array) -> np.array:
    # Remove the black frame and binarize the symbol (Otsu): white background, black symbol
    interior = warped[frame_margin:-frame_margin, frame_margin:-frame_margin]
    _, interior = cv2.threshold(interior, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return interior


def load_templates(folder: str = templates_folder) -> dict:
    templates = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.png"))):
        name = os.path.splitext(os.path.basename(path))[0]
        templates[name] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return templates


def match_card(interior: np.array, templates: dict):
    # Template matching against every template (also rotated 180 degrees, in case the card is upside down)
    best_name, best_score = None, -1.0
    for name, template in templates.items():
        for t in [template, cv2.rotate(template, cv2.ROTATE_180)]:
            score = cv2.matchTemplate(interior, t, cv2.TM_CCOEFF_NORMED)[0][0]
            if score > best_score:
                best_name, best_score = name, score
    if best_score < match_threshold:
        best_name = None
    return best_name, best_score


def detect_cards(frame: np.array, templates: dict) -> list:
    # Returns a list of (symbol, corners), biggest card first. symbol is None if unknown
    results = []
    for corners in find_card_corners(frame):
        interior = card_interior(warp_card(frame, corners))
        symbol, _ = match_card(interior, templates)
        results.append((symbol, corners))
    return results


# ---------------- Decoder ----------------

def new_decoder_state() -> dict:
    return {"entered": [],            # symbols accepted so far (0 to 4)
            "fails": 0,               # wrong sequences in a row
            "locked_until": 0.0,      # time.time() until the system is locked
            "unlocked": False,
            "candidate": None,        # symbol being seen now
            "count": 0,               # frames in a row seeing the candidate
            "last_seen": time.time(), # last time a card was seen
            "message": "Show your cards"}


def update_decoder(state: dict, symbol) -> dict:
    now = time.time()
    if state["unlocked"] or now < state["locked_until"]:
        return state

    # No card: after forget_time the sequence is forgotten
    if symbol is None:
        state["candidate"], state["count"] = None, 0
        if state["entered"] and now - state["last_seen"] > forget_time:
            state["entered"] = []
            state["message"] = "Sequence forgotten"
        return state

    state["last_seen"] = now
    if symbol == state["candidate"]:
        state["count"] += 1
    else:
        state["candidate"], state["count"] = symbol, 1

    # The symbol only counts once, when it has been stable for stable_frames
    if state["count"] != stable_frames:
        return state

    if symbol == "ac":
        state["entered"] = []
        state["message"] = "Sequence reset"
    elif state["entered"] and symbol == state["entered"][-1]:
        pass                                            # same card still in front of the camera
    else:
        state["entered"].append(symbol)
        if state["entered"] != password[:len(state["entered"])]:
            state["entered"] = []
            state["fails"] += 1
            state["message"] = f"Wrong card ({state['fails']}/{max_fails})"
            if state["fails"] >= max_fails:
                state["fails"] = 0
                state["locked_until"] = now + lock_time
                state["message"] = "Too many fails"
        elif len(state["entered"]) == len(password):
            state["unlocked"] = True
            state["message"] = "Unlocked"
        else:
            state["message"] = "Correct, next card"
    return state
