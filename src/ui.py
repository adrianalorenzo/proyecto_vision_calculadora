import cv2
import numpy as np

# ---------------------------------------------------------------
# Visual interface: everything is drawn with OpenCV on top of the
# camera frame. The modes only return data; this file draws them.
# ---------------------------------------------------------------

width, height = 640, 480
font = cv2.FONT_HERSHEY_SIMPLEX
font_bold = cv2.FONT_HERSHEY_DUPLEX

# Color palette (BGR)
dark = (45, 30, 22)          # navy panels
white = (245, 245, 245)
muted = (200, 190, 180)      # secondary text
accent = (166, 184, 20)      # teal: buttons, borders, player 1
ink = (36, 191, 251)         # amber: strokes, player 2
green = (94, 197, 34)
red = (68, 68, 239)

# Text shown for every card (cv2.putText cannot draw symbols like the Greek letters)
symbol_text = {"pi": "pi", "sqrt": "sqrt", "inf": "inf", "sigma": "sigma", "ac": "AC"}

# ---------------- Layouts (x1, y1, x2, y2) ----------------
# Writing boxes: only the strokes inside these boxes are used
calculator_boxes = {"num1": (30, 105, 240, 295),
                    "op": (255, 105, 385, 295),
                    "num2": (400, 105, 610, 295)}
mental_math_boxes = {"answer": (200, 140, 440, 330)}
versus_boxes = {"answer1": (20, 140, 300, 330),
                "answer2": (340, 140, 620, 330)}


def make_button(name: str, text: str, rect: tuple) -> dict:
    return {"name": name, "text": text, "rect": rect}


menu_button = make_button("menu", "Menu", (540, 12, 628, 46))     # top-right corner in every mode

menu_buttons = [make_button("calculator", "Calculator", (190, 125, 450, 172)),
                make_button("mm_easy", "Mental Math  Easy", (190, 182, 450, 229)),
                make_button("mm_hard", "Mental Math  Hard", (190, 239, 450, 286)),
                make_button("versus", "Versus", (190, 296, 450, 343)),
                make_button("lock", "Lock", (190, 365, 450, 405))]

calculator_buttons = [make_button("equals", "=", (170, 400, 310, 448)),
                      make_button("clear", "Clear", (330, 400, 470, 448)),
                      menu_button]

mental_math_buttons = [make_button("submit", "Submit", (110, 400, 300, 448)),
                       make_button("not_what_i_wrote", "Not what I wrote", (320, 400, 530, 448)),
                       menu_button]

versus_buttons = [make_button("submit1", "Submit P1", (60, 400, 260, 448)),
                  make_button("submit2", "Submit P2", (380, 400, 580, 448)),
                  menu_button]

results_buttons = [make_button("back", "Back to menu", (220, 400, 420, 448))]


# ---------------- Basic drawing functions ----------------

def is_inside(point, rect: tuple) -> bool:
    if point is None:
        return False
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def get_box(point, boxes: dict):
    # Name of the writing box that contains the point, or None
    for name, rect in boxes.items():
        if is_inside(point, rect):
            return name
    return None


def get_button(point, buttons: list):
    # Name of the button under the point, or None
    for button in buttons:
        if is_inside(point, button["rect"]):
            return button["name"]
    return None


def rounded_rect(img: np.array, rect: tuple, color: tuple, radius: int = 12, thickness: int = -1):
    # Rectangle with rounded corners (filled if thickness = -1)
    x1, y1, x2, y2 = rect
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def draw_panel(img: np.array, rect: tuple, color: tuple = dark, alpha: float = 0.65, radius: int = 12):
    # Semi-transparent rounded panel on top of the camera image
    overlay = img.copy()
    rounded_rect(overlay, rect, color, radius)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def draw_text(img: np.array, text: str, pos: tuple, scale: float = 0.7, color: tuple = white,
              thickness: int = 1, bold: bool = False):
    cv2.putText(img, text, (int(pos[0]), int(pos[1])), font_bold if bold else font, scale, color, thickness, cv2.LINE_AA)


def draw_text_centered(img: np.array, text: str, center: tuple, scale: float = 0.7, color: tuple = white,
                       thickness: int = 1, bold: bool = False):
    (w, h), _ = cv2.getTextSize(text, font_bold if bold else font, scale, thickness)
    draw_text(img, text, (center[0] - w / 2, center[1] + h / 2), scale, color, thickness, bold)


def draw_header(img: np.array, title: str, subtitle: str = ""):
    # Pill in the top-left corner: app name + current screen
    (w1, _), _ = cv2.getTextSize("AirMath", font_bold, 0.7, 1)
    (w2, _), _ = cv2.getTextSize(subtitle, font, 0.6, 1)
    x2 = 24 + w1 + (w2 + 22 if subtitle else 0) + 14
    draw_panel(img, (12, 12, x2, 46), dark, 0.75, 17)
    draw_text(img, "AirMath", (26, 37), 0.7, accent, 1, True)
    if subtitle:
        cv2.circle(img, (int(32 + w1), 29), 3, muted, -1, cv2.LINE_AA)
        draw_text(img, subtitle, (42 + w1, 36), 0.6, white, 1)


def draw_button(img: np.array, button: dict, pen=None, primary: bool = False):
    # Hover (pen over the button): filled with the accent color
    x1, y1, x2, y2 = button["rect"]
    radius = (y2 - y1) // 2
    hover = is_inside(pen, button["rect"])
    if hover or primary:
        draw_panel(img, button["rect"], accent, 0.9 if hover else 0.75, radius)
    else:
        draw_panel(img, button["rect"], dark, 0.75, radius)
        rounded_rect(img, button["rect"], accent, radius, 1)
    draw_text_centered(img, button["text"], ((x1 + x2) / 2, (y1 + y2) / 2), 0.6, white, 1, True)


def draw_buttons(img: np.array, buttons: list, pen=None):
    for button in buttons:
        draw_button(img, button, pen)


def draw_boxes(img: np.array, boxes: dict, labels: dict = None, colors: dict = None):
    # Writing boxes: dark translucent area with a thin colored border and a small label inside
    for name, rect in boxes.items():
        color = colors.get(name, accent) if colors else accent
        draw_panel(img, rect, dark, 0.45, 14)
        rounded_rect(img, rect, color, 14, 2)
        if labels is not None and name in labels:
            draw_text(img, labels[name], (rect[0] + 12, rect[1] + 22), 0.45, muted, 1)


def draw_strokes(img: np.array, canvas: np.array, color: tuple = ink):
    # Paint on the image the strokes stored in the canvas (white on black, one channel)
    img[canvas > 0] = color


def draw_fps(img: np.array, fps: float):
    draw_panel(img, (12, height - 34, 92, height - 10), dark, 0.7, 12)
    cv2.circle(img, (26, height - 22), 4, green, -1, cv2.LINE_AA)
    draw_text(img, f"{fps:.0f} FPS", (36, height - 16), 0.45, white, 1)


def draw_cursor(img: np.array, pen, gesture):
    # Fingertip: amber when writing, teal ring when clicking (pinch), white otherwise
    if pen is None:
        return
    p = (int(pen[0]), int(pen[1]))
    if gesture == "pinch":
        cv2.circle(img, p, 14, accent, 2, cv2.LINE_AA)
        cv2.circle(img, p, 5, accent, -1, cv2.LINE_AA)
    else:
        color = ink if gesture == "index" else white
        cv2.circle(img, p, 9, dark, -1, cv2.LINE_AA)
        cv2.circle(img, p, 7, color, -1, cv2.LINE_AA)


def draw_corners(img: np.array, rect: tuple, color: tuple, length: int = 18, thickness: int = 3):
    # Only the four corners of a rectangle (viewfinder style)
    x1, y1, x2, y2 = [int(v) for v in rect]
    for (x, y, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (x, y), (x + dx * length, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + dy * length), color, thickness, cv2.LINE_AA)


def draw_hand_box(img: np.array, bbox):
    # Bounding box of the hand given by the tracker
    if bbox is None:
        return
    draw_corners(img, bbox, accent, 16, 2)


def draw_recording(img: np.array):
    draw_panel(img, (width - 82, height - 34, width - 12, height - 10), dark, 0.7, 12)
    cv2.circle(img, (width - 66, height - 22), 5, red, -1, cv2.LINE_AA)
    draw_text(img, "REC", (width - 54, height - 16), 0.45, white, 1, True)


def draw_lock_icon(img: np.array, center: tuple, color: tuple, opened: bool = False):
    # Simple padlock: body + shackle
    cx, cy = int(center[0]), int(center[1])
    rounded_rect(img, (cx - 11, cy - 2, cx + 11, cy + 15), color, 3)
    shift = 6 if opened else 0
    cv2.ellipse(img, (cx + shift, cy - 3), (7, 9), 0, 180, 360, color, 3, cv2.LINE_AA)
    if not opened:
        cv2.line(img, (cx - 7, cy - 3), (cx - 7, cy), color, 3)
    cv2.line(img, (cx + 7 + shift, cy - 3), (cx + 7 + shift, cy), color, 3)
    cv2.circle(img, (cx, cy + 6), 2, dark, -1)


# ---------------- Screens ----------------

def draw_card(img: np.array, corners: np.array, symbol):
    # Detected card: polygon + label above it
    pts = corners.reshape(-1, 1, 2).astype(np.int32)
    color = green if symbol is not None else red
    cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
    text = symbol_text.get(symbol, "?")
    x, y = int(pts[:, 0, 0].mean()), int(pts[:, 0, 1].min())
    (w, _), _ = cv2.getTextSize(text, font_bold, 0.6, 1)
    draw_panel(img, (x - w // 2 - 12, y - 36, x + w // 2 + 12, y - 8), color, 0.85, 14)
    draw_text_centered(img, text, (x, y - 22), 0.6, white, 1, True)


def draw_progress(img: np.array, n_entered: int, n_total: int, center: tuple):
    # Password progress: filled dots = cards accepted
    x0 = int(center[0] - (n_total - 1) * 17)
    for i in range(n_total):
        c = (x0 + 34 * i, int(center[1]))
        if i < n_entered:
            cv2.circle(img, c, 10, accent, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, c, 10, muted, 2, cv2.LINE_AA)


def draw_locked_screen(img: np.array, decoder_state: dict, cards: list, n_total: int, now: float):
    draw_header(img, "AirMath", "Locked")
    for symbol, corners in cards:
        draw_card(img, corners, symbol)

    locked = now < decoder_state["locked_until"]
    draw_panel(img, (170, height - 128, width - 170, height - 50), dark, 0.75, 18)
    draw_lock_icon(img, (205, height - 96), red if locked else white)
    if locked:
        draw_text(img, "Too many fails", (235, height - 96), 0.6, red, 1, True)
        draw_text(img, f"Try again in {decoder_state['locked_until'] - now:.0f} s", (235, height - 70), 0.5, muted, 1)
    else:
        draw_progress(img, len(decoder_state["entered"]), n_total, (width / 2 + 25, height - 101))
        draw_text_centered(img, decoder_state["message"], (width / 2 + 25, height - 70), 0.5, muted, 1)


def draw_menu(img: np.array, pen=None):
    draw_header(img, "AirMath", "Menu")
    draw_panel(img, (170, 70, 470, 420), dark, 0.6, 20)
    draw_text_centered(img, "Choose a mode", (width / 2, 97), 0.65, white, 1, True)
    draw_buttons(img, menu_buttons, pen)
    cv2.line(img, (210, 354), (430, 354), muted, 1, cv2.LINE_AA)


def draw_calculator(img: np.array, canvas: np.array, read_text: str, result_text: str, pen=None):
    draw_header(img, "AirMath", "Calculator")
    draw_boxes(img, calculator_boxes, {"num1": "number", "op": "op", "num2": "number"})
    draw_strokes(img, canvas)
    if read_text:
        text = f"{read_text} = {result_text}"
        (w, _), _ = cv2.getTextSize(text, font_bold, 1.1, 2)
        draw_panel(img, (320 - w // 2 - 28, 312, 320 + w // 2 + 28, 380), dark, 0.8, 34)
        draw_text_centered(img, text, (width / 2, 346), 1.1, white, 2, True)
    draw_buttons(img, calculator_buttons, pen)


def draw_problem(img: np.array, problem_text: str):
    text = f"{problem_text} = ?"
    (w, _), _ = cv2.getTextSize(text, font_bold, 1.2, 2)
    draw_panel(img, (320 - w // 2 - 30, 64, 320 + w // 2 + 30, 124), dark, 0.8, 30)
    draw_text_centered(img, text, (width / 2, 94), 1.2, white, 2, True)


def draw_mental_math(img: np.array, canvas: np.array, problem_text: str, time_left: float,
                     hits: int, misses: int, feedback: str, pen=None):
    draw_header(img, "AirMath", "Mental Math")
    # Timer as a pill with a progress bar (2 minutes)
    draw_panel(img, (250, 12, 400, 46), dark, 0.75, 17)
    draw_text(img, f"{int(time_left) // 60}:{int(time_left) % 60:02d}", (264, 36), 0.6, white, 1, True)
    frac = max(0.0, min(1.0, time_left / 120))
    cv2.line(img, (325, 29), (385, 29), muted, 4, cv2.LINE_AA)
    cv2.line(img, (325, 29), (325 + int(60 * frac), 29), red if time_left < 15 else accent, 4, cv2.LINE_AA)
    # Score chips
    draw_panel(img, (410, 12, 525, 46), dark, 0.75, 17)
    cv2.circle(img, (428, 29), 5, green, -1, cv2.LINE_AA)
    draw_text(img, str(hits), (438, 35), 0.55, white, 1, True)
    cv2.circle(img, (475, 29), 5, red, -1, cv2.LINE_AA)
    draw_text(img, str(misses), (485, 35), 0.55, white, 1, True)

    draw_problem(img, problem_text)
    draw_boxes(img, mental_math_boxes, {"answer": "your answer"})
    draw_strokes(img, canvas)
    if feedback:
        color = green if feedback.startswith("Correct") else red
        (w, _), _ = cv2.getTextSize(feedback, font_bold, 0.6, 1)
        draw_panel(img, (320 - w // 2 - 18, 343, 320 + w // 2 + 18, 377), color, 0.85, 17)
        draw_text_centered(img, feedback, (width / 2, 360), 0.6, white, 1, True)
    draw_buttons(img, mental_math_buttons, pen)


def draw_versus(img: np.array, canvas: np.array, problem_text: str, points: list, target: int,
                feedback: str, pen=None):
    draw_header(img, "AirMath", "Versus")
    # Scoreboard
    draw_panel(img, (232, 12, 408, 46), dark, 0.75, 17)
    draw_text(img, f"P1 {points[0]}", (248, 36), 0.6, accent, 1, True)
    draw_text_centered(img, f"to {target}", (320, 29), 0.4, muted, 1)
    draw_text(img, f"{points[1]} P2", (352, 36), 0.6, ink, 1, True)

    draw_problem(img, problem_text)
    draw_boxes(img, versus_boxes, {"answer1": "Player 1", "answer2": "Player 2"},
               {"answer1": accent, "answer2": ink})
    draw_strokes(img, canvas, white)
    if feedback:
        (w, _), _ = cv2.getTextSize(feedback, font_bold, 0.6, 1)
        draw_panel(img, (320 - w // 2 - 18, 343, 320 + w // 2 + 18, 377), dark, 0.85, 17)
        draw_text_centered(img, feedback, (width / 2, 360), 0.6, white, 1, True)
    draw_buttons(img, versus_buttons, pen)


def draw_results(img: np.array, title: str, lines: list, pen=None):
    draw_header(img, "AirMath", "Results")
    draw_panel(img, (110, 66, 530, 385), dark, 0.8, 20)
    draw_text_centered(img, title, (width / 2, 96), 0.65, accent, 1, True)
    cv2.line(img, (140, 116), (500, 116), muted, 1, cv2.LINE_AA)
    for i, line in enumerate(lines[:8]):
        draw_text(img, line, (145, 148 + 30 * i), 0.55, white, 1)
    draw_buttons(img, results_buttons, pen)
