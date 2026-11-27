import os
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------
# Visual interface (neon style). Everything is drawn on top of the
# camera frame. The modes only return data; this file draws them.
#
# All the positions are in the 640x480 image of the camera. The
# screen is drawn S times bigger so the text looks sharp in full screen.
#
# Every frame:  screen = ui.new_frame(display)
#               ui.draw_xxx(screen, ...)
#               ui.finish_frame(screen)
# ---------------------------------------------------------------

width, height = 640, 480
S = 2                                    # drawing scale (screen = 1280x960)
fonts_folder = "../assets/fonts"

# Neon palette (BGR)
cyan = (255, 230, 0)
magenta = (230, 40, 255)
lime = (110, 255, 90)
red = (90, 60, 255)
yellow = (60, 220, 255)
white = (255, 255, 255)
muted = (205, 180, 195)
panel = (40, 12, 22)                     # very dark purple
black = (0, 0, 0)
green = lime                             # old names, still used in some scripts
accent = cyan

# Text shown for every card
symbol_text = {"pi": "π", "sqrt": "√", "inf": "∞", "sigma": "Σ", "ac": "AC"}

# ---------------- Layouts (x1, y1, x2, y2), in 640x480 ----------------
# Writing boxes: only the strokes inside these boxes are used
calculator_boxes = {"num1": (30, 105, 240, 295),
                    "op": (255, 105, 385, 295),
                    "num2": (400, 105, 610, 295)}
mental_math_boxes = {"answer": (200, 140, 440, 330)}
versus_boxes = {"answer1": (20, 140, 300, 330),
                "answer2": (340, 140, 620, 330)}


def make_button(name: str, text: str, rect: tuple, color: tuple = cyan) -> dict:
    return {"name": name, "text": text, "rect": rect, "color": color}


menu_button = make_button("menu", "Menu", (540, 12, 628, 46))     # top-right corner in every mode

menu_buttons = [make_button("calculator", "Calculator", (190, 150, 450, 194)),
                make_button("mm_easy", "Mental Math  ·  Easy", (190, 204, 450, 248)),
                make_button("mm_hard", "Mental Math  ·  Hard", (190, 258, 450, 302)),
                make_button("versus", "Versus", (190, 312, 450, 356)),
                make_button("lock", "Lock", (250, 378, 390, 414), magenta)]

calculator_buttons = [make_button("equals", "=", (170, 400, 310, 448)),
                      make_button("clear", "Clear", (330, 400, 470, 448), magenta),
                      menu_button]

mental_math_buttons = [make_button("submit", "Submit", (110, 400, 300, 448)),
                       make_button("not_what_i_wrote", "Not what I wrote", (320, 400, 530, 448), magenta),
                       menu_button]

versus_buttons = [make_button("submit1", "Submit P1", (100, 400, 280, 448)),
                  make_button("submit2", "Submit P2", (360, 400, 540, 448), magenta),
                  menu_button]

results_buttons = [make_button("back", "Back to menu", (220, 400, 420, 448))]


# ---------------- Geometry helpers ----------------

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


def sc(rect: tuple) -> tuple:
    # From 640x480 coordinates to screen coordinates
    return tuple(int(v * S) for v in rect)


# ---------------- Frame: background, glow layer, flash ----------------
glow = np.zeros((height, width, 3), dtype=np.uint8)   # neon lines are also drawn here and blurred
flash_state = {"color": None, "start": 0.0}
trail = []                                              # last positions of the fingertip
vignette = None
tint = None


def pretty(text: str) -> str:
    # Nicer operators on the screen: x -> ×, / -> ÷, - -> −
    return text.replace(" x ", " × ").replace(" / ", " ÷ ").replace(" - ", " − ")


def new_frame(display: np.array, dim: float = 0.55) -> np.array:
    # Darker copy of the camera image with a purple tint and vignette, then S times bigger
    global vignette, tint
    glow[:] = 0
    if vignette is None:
        yy, xx = np.mgrid[0:height, 0:width]
        d = np.sqrt(((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2)
        v = np.clip((d - 0.55) * 90, 0, 90).astype(np.uint8)
        vignette = cv2.merge([v, v, v])
        tint = np.full((height, width, 3), (18, 4, 10), dtype=np.uint8)
    small = cv2.convertScaleAbs(display, alpha=dim, beta=0)
    cv2.subtract(small, vignette, dst=small)
    cv2.add(small, tint, dst=small)
    return cv2.resize(small, (width * S, height * S), interpolation=cv2.INTER_LINEAR)


def flash(color: tuple):
    # Short colored flash of the screen border (correct / wrong / unlocked...)
    flash_state["color"] = color
    flash_state["start"] = time.time()


def finish_frame(screen: np.array):
    # Add the blurred neon layer (glow) and the flash, if any
    t = time.time() - flash_state["start"]
    if flash_state["color"] is not None and t < 0.5:
        k = 1 - t / 0.5
        c = tuple(int(v * k) for v in flash_state["color"])
        cv2.rectangle(glow, (0, 0), (width - 1, height - 1), c, 14)
        cv2.rectangle(screen, (0, 0), (width * S - 1, height * S - 1), c, 6)
    # The glow is blurred at quarter size (much faster) and added to the screen
    halo = cv2.resize(glow, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    halo = cv2.GaussianBlur(halo, (0, 0), 2.5)
    halo = cv2.convertScaleAbs(halo, alpha=1.5)
    halo = cv2.resize(halo, (width * S, height * S), interpolation=cv2.INTER_LINEAR)
    cv2.add(screen, halo, dst=screen)


def fit_to_window(img: np.array, window: str):
    # Black bars on the sides so the 4:3 image is not stretched when the window is wider (full screen)
    _, _, w, h = cv2.getWindowImageRect(window)
    if w <= 0 or h <= 0:
        return img, 0
    pad = max((int(round(img.shape[0] * w / h)) - img.shape[1]) // 2, 0)
    return cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=black), pad


def screen_to_image(point, pad: int = 0):
    # Mouse position on the screen -> position in the 640x480 image
    return ((point[0] - pad) / S, point[1] / S)


# ---------------- Text (TrueType fonts with Pillow) ----------------
font_files = {"title": "Tektur-Medium.ttf", "body": "Outfit-Regular.ttf", "bold": "Outfit-Bold.ttf"}
loaded_fonts = {}
text_masks = {}


def get_font(kind: str, size: int):
    key = (kind, size)
    if key not in loaded_fonts:
        loaded_fonts[key] = ImageFont.truetype(os.path.join(fonts_folder, font_files[kind]), size * S)
    return loaded_fonts[key]


def text_mask(text: str, size: int, kind: str) -> np.array:
    # Gray mask (0-1) of the text, computed once and kept in memory
    key = (text, size, kind)
    if key not in text_masks:
        if len(text_masks) > 600:
            text_masks.clear()
        font = get_font(kind, size)
        ascent, descent = font.getmetrics()
        x0, _, x1, _ = font.getbbox(text)
        img = Image.new("L", (max(x1 - x0, 1) + 4, ascent + descent + 4), 0)
        ImageDraw.Draw(img).text((2 - x0, 2), text, font=font, fill=255)
        text_masks[key] = np.array(img, dtype=np.float32) / 255.0
    return text_masks[key]


def text_width(text: str, size: int = 16, kind: str = "body") -> float:
    return text_mask(text, size, kind).shape[1] / S


def draw_text(screen: np.array, text: str, pos: tuple, size: int = 16, color: tuple = white,
              kind: str = "body", center: bool = False, neon: bool = False):
    # pos = (x, y) in 640x480; y is the vertical center of the text; x is the left side (or the center)
    mask = text_mask(text, size, kind)
    h, w = mask.shape
    x = int(pos[0] * S - (w / 2 if center else 0))
    y = int(pos[1] * S - h / 2)
    x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, screen.shape[1]), min(y + h, screen.shape[0])
    if x1 >= x2 or y1 >= y2:
        return
    m = mask[y1 - y:y2 - y, x1 - x:x2 - x, None]
    roi = screen[y1:y2, x1:x2].astype(np.float32)
    screen[y1:y2, x1:x2] = (roi * (1 - m) + np.array(color, dtype=np.float32) * m).astype(np.uint8)
    if neon:
        # Same text, smaller, on the glow layer
        small = cv2.resize(mask, (max(w // S, 1), max(h // S, 1)), interpolation=cv2.INTER_AREA)
        gx, gy = x // S, y // S
        gx1, gy1 = max(gx, 0), max(gy, 0)
        gx2, gy2 = min(gx + small.shape[1], width), min(gy + small.shape[0], height)
        if gx1 < gx2 and gy1 < gy2:
            sm = small[gy1 - gy:gy2 - gy, gx1 - gx:gx2 - gx, None]
            groi = glow[gy1:gy2, gx1:gx2].astype(np.float32)
            glow[gy1:gy2, gx1:gx2] = np.maximum(groi, np.array(color, dtype=np.float32) * sm * 0.8).astype(np.uint8)


def draw_text_centered(screen: np.array, text: str, center: tuple, size: int = 16, color: tuple = white,
                       kind: str = "body", neon: bool = False):
    draw_text(screen, text, center, size, color, kind, True, neon)


# ---------------- Shapes ----------------

def rounded_rect(img: np.array, rect: tuple, color: tuple, radius: int, thickness: int = -1):
    # Rectangle with rounded corners (filled if thickness = -1). rect and radius in pixels of img
    x1, y1, x2, y2 = rect
    r = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
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


def draw_panel(screen: np.array, rect: tuple, color: tuple = panel, alpha: float = 0.7, radius: int = 14):
    # Semi-transparent rounded panel (only the region of the panel is blended, to keep the FPS high)
    x1, y1, x2, y2 = sc(rect)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, screen.shape[1] - 1), min(y2, screen.shape[0] - 1)
    roi = screen[y1:y2 + 1, x1:x2 + 1]
    overlay = roi.copy()
    rounded_rect(overlay, (0, 0, x2 - x1, y2 - y1), color, radius * S)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def neon_rect(screen: np.array, rect: tuple, color: tuple, radius: int = 14, thickness: int = 1):
    # Thin bright border + glow around it
    rounded_rect(screen, sc(rect), color, radius * S, thickness * S)
    rounded_rect(glow, tuple(int(v) for v in rect), color, radius, thickness + 2)


def neon_line(screen: np.array, p1: tuple, p2: tuple, color: tuple, thickness: int = 1):
    cv2.line(screen, (int(p1[0] * S), int(p1[1] * S)), (int(p2[0] * S), int(p2[1] * S)), color, thickness * S, cv2.LINE_AA)
    cv2.line(glow, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness + 2, cv2.LINE_AA)


def neon_circle(screen: np.array, center: tuple, radius: int, color: tuple, thickness: int = 1):
    c = (int(center[0] * S), int(center[1] * S))
    cv2.circle(screen, c, radius * S, color, -1 if thickness < 0 else thickness * S, cv2.LINE_AA)
    cv2.circle(glow, (int(center[0]), int(center[1])), radius, color, -1 if thickness < 0 else thickness + 2, cv2.LINE_AA)


# ---------------- Widgets ----------------

def draw_header(screen: np.array, subtitle: str = ""):
    # Logo + current screen, top-left corner
    w = text_width(subtitle, 14) if subtitle else 0
    x2 = 118 + (w + 22 if subtitle else 0)
    draw_panel(screen, (12, 12, x2, 46), panel, 0.75, 17)
    neon_rect(screen, (12, 12, x2, 46), cyan, 17)
    draw_text(screen, "AirMath", (26, 29), 17, cyan, "title", neon=True)
    if subtitle:
        neon_circle(screen, (122, 29), 2, magenta, -1)
        draw_text(screen, subtitle, (132, 29), 14, white)


def draw_button(screen: np.array, button: dict, pen=None, progress: float = 0.0):
    # Neon pill. Hover: brighter. progress (0-1): fills while the pinch is held
    x1, y1, x2, y2 = button["rect"]
    color = button["color"]
    radius = (y2 - y1) // 2
    hover = is_inside(pen, button["rect"])
    draw_panel(screen, button["rect"], panel, 0.8, radius)
    if progress > 0:
        fill = int((x2 - x1) * min(progress, 1.0))
        if fill > 4:
            draw_panel(screen, (x1, y1, x1 + max(fill, 2 * radius), y2), color, 0.55, radius)
    neon_rect(screen, button["rect"], color, radius, 2 if hover else 1)
    draw_text_centered(screen, button["text"], ((x1 + x2) / 2, (y1 + y2) / 2), 15,
                       white if hover or progress > 0 else color, "bold", neon=hover)


def draw_buttons(screen: np.array, buttons: list, pen=None, hold=None):
    # hold = (button name, progress) of the button being pressed
    for button in buttons:
        progress = hold[1] if hold is not None and hold[0] == button["name"] else 0.0
        draw_button(screen, button, pen, progress)


def draw_boxes(screen: np.array, boxes: dict, labels: dict = None, colors: dict = None):
    # Writing boxes: dark translucent area, neon border and a small label inside
    for name, rect in boxes.items():
        color = colors.get(name, cyan) if colors else cyan
        draw_panel(screen, rect, panel, 0.45, 16)
        neon_rect(screen, rect, color, 16, 1)
        if labels is not None and name in labels:
            draw_text(screen, labels[name], (rect[0] + 12, rect[1] + 15), 11, muted)


def draw_strokes(screen: np.array, canvas: np.array, color: tuple = cyan, region: tuple = None):
    # Strokes stored in the canvas (white on black, 640x480), drawn as glowing neon ink
    if region is not None:
        x1, y1, x2, y2 = region
        part = np.zeros_like(canvas)
        part[y1:y2, x1:x2] = canvas[y1:y2, x1:x2]
        canvas = part
    big = cv2.resize(canvas, (width * S, height * S), interpolation=cv2.INTER_LINEAR)
    screen[big > 110] = color
    glow[canvas > 0] = color


def draw_fps(screen: np.array, fps: float):
    draw_panel(screen, (12, height - 34, 92, height - 10), panel, 0.75, 12)
    neon_circle(screen, (25, height - 22), 3, lime, -1)
    draw_text(screen, f"{fps:.0f} FPS", (34, height - 22), 12, white)


def draw_cursor(screen: np.array, pen, gesture):
    # Fingertip with a short fading trail. Cyan = writing, magenta = pinch (click), white = moving
    now = time.time()
    if pen is not None:
        trail.append((pen[0], pen[1], now))
    while trail and now - trail[0][2] > 0.35:
        trail.pop(0)
    if pen is None:
        return
    color = {"index": cyan, "pinch": magenta}.get(gesture, white)
    for i, (x, y, t) in enumerate(trail[:-1]):
        k = 1 - (now - t) / 0.35
        cv2.circle(glow, (int(x), int(y)), max(1, int(6 * k)), tuple(int(c * k) for c in color), -1, cv2.LINE_AA)
    neon_circle(screen, pen, 9, color, 1)
    neon_circle(screen, pen, 3, color, -1)


def draw_corners(screen: np.array, rect: tuple, color: tuple, length: int = 16):
    # Only the four corners of a rectangle (viewfinder style)
    x1, y1, x2, y2 = [int(v) for v in rect]
    for (x, y, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        neon_line(screen, (x, y), (x + dx * length, y), color, 2)
        neon_line(screen, (x, y), (x, y + dy * length), color, 2)


def draw_hand_box(screen: np.array, bbox):
    # Bounding box of the hand given by the tracker
    if bbox is not None:
        draw_corners(screen, bbox, cyan)


def draw_recording(screen: np.array):
    draw_panel(screen, (width - 82, height - 34, width - 12, height - 10), panel, 0.75, 12)
    if int(time.time() * 2) % 2 == 0:
        neon_circle(screen, (width - 66, height - 22), 4, red, -1)
    draw_text(screen, "REC", (width - 55, height - 22), 12, red, "bold")


def draw_chip(screen: np.array, text: str, center: tuple, color: tuple, size: int = 15, kind: str = "bold"):
    # Small rounded label with neon border (feedback messages)
    w = text_width(text, size, kind)
    cx, cy = center
    rect = (int(cx - w / 2 - 16), int(cy - 16), int(cx + w / 2 + 16), int(cy + 16))
    draw_panel(screen, rect, panel, 0.85, 16)
    neon_rect(screen, rect, color, 16, 1)
    draw_text_centered(screen, text, center, size, color, kind, neon=True)


def draw_lock_icon(screen: np.array, center: tuple, color: tuple):
    cx, cy = center
    neon_rect(screen, (cx - 12, cy - 2, cx + 12, cy + 16), color, 4, 2)
    for img, s, t in [(screen, S, 2 * S), (glow, 1, 4)]:
        cv2.ellipse(img, (int(cx * s), int((cy - 2) * s)), (int(8 * s), int(10 * s)), 0, 180, 360, color, t, cv2.LINE_AA)
    neon_circle(screen, (cx, cy + 7), 2, color, -1)


# ---------------- Screens ----------------

def draw_card(screen: np.array, corners: np.array, symbol):
    # Detected card: neon outline + label with the symbol read
    color = lime if symbol is not None else red
    pts = corners.reshape(-1, 2)
    for i in range(4):
        neon_line(screen, pts[i], pts[(i + 1) % 4], color, 2)
    cx, top = pts[:, 0].mean(), pts[:, 1].min()
    draw_chip(screen, symbol_text.get(symbol, "?"), (cx, top - 22), color, 18, "title")


def draw_progress(screen: np.array, n_entered: int, n_total: int, center: tuple):
    # Password progress: glowing dots = cards accepted
    x0 = center[0] - (n_total - 1) * 17
    for i in range(n_total):
        c = (x0 + 34 * i, center[1])
        if i < n_entered:
            neon_circle(screen, c, 9, cyan, -1)
        else:
            neon_circle(screen, c, 9, muted, 1)


def draw_locked_screen(screen: np.array, decoder_state: dict, cards: list, n_total: int, now: float):
    draw_header(screen, "Locked")
    for symbol, corners in cards:
        draw_card(screen, corners, symbol)

    locked = now < decoder_state["locked_until"]
    rect = (160, height - 135, width - 160, height - 50)
    draw_panel(screen, rect, panel, 0.8, 20)
    neon_rect(screen, rect, red if locked else cyan, 20, 1)
    draw_lock_icon(screen, (200, height - 95), red if locked else cyan)
    if locked:
        draw_text(screen, "TOO MANY FAILS", (232, height - 105), 18, red, "title", neon=True)
        draw_text(screen, f"Try again in {decoder_state['locked_until'] - now:.0f} s", (232, height - 78), 14, muted)
    else:
        draw_text(screen, "SHOW YOUR CARDS", (232, height - 112), 16, white, "title")
        draw_progress(screen, len(decoder_state["entered"]), n_total, (300, height - 86))
        draw_text(screen, decoder_state["message"], (232, height - 64), 12, muted)


def draw_menu(screen: np.array, pen=None, hold=None):
    draw_header(screen, "Menu")
    draw_panel(screen, (165, 62, 475, 430), panel, 0.7, 22)
    neon_rect(screen, (165, 62, 475, 430), cyan, 22, 1)
    draw_text_centered(screen, "AirMath", (width / 2, 100), 34, cyan, "title", neon=True)
    draw_text_centered(screen, "write maths in the air", (width / 2, 130), 12, muted)
    draw_buttons(screen, menu_buttons, pen, hold)


def draw_calculator(screen: np.array, canvas: np.array, read_text: str, result_text: str, pen=None, hold=None):
    draw_header(screen, "Calculator")
    draw_boxes(screen, calculator_boxes, {"num1": "NUMBER", "op": "OP", "num2": "NUMBER"},
               {"num1": cyan, "op": magenta, "num2": cyan})
    draw_strokes(screen, canvas, cyan)
    if read_text:
        text = pretty(f"{read_text} = {result_text}")
        w = text_width(text, 30, "title")
        rect = (int(320 - w / 2 - 30), 312, int(320 + w / 2 + 30), 382)
        draw_panel(screen, rect, panel, 0.85, 35)
        neon_rect(screen, rect, yellow, 35, 1)
        draw_text_centered(screen, text, (width / 2, 347), 30, yellow, "title", neon=True)
    draw_buttons(screen, calculator_buttons, pen, hold)


def draw_problem(screen: np.array, problem_text: str):
    text = pretty(f"{problem_text} = ?")
    w = text_width(text, 30, "title")
    rect = (int(320 - w / 2 - 30), 62, int(320 + w / 2 + 30), 124)
    draw_panel(screen, rect, panel, 0.85, 31)
    neon_rect(screen, rect, magenta, 31, 1)
    draw_text_centered(screen, text, (width / 2, 93), 30, white, "title", neon=False)


def draw_mental_math(screen: np.array, canvas: np.array, problem_text: str, time_left: float,
                     hits: int, misses: int, feedback: str, pen=None, hold=None):
    draw_header(screen, "Mental Math")
    # Timer with a bar that empties (red in the last 15 s)
    color = red if time_left < 15 else cyan
    draw_panel(screen, (250, 12, 400, 46), panel, 0.8, 17)
    neon_rect(screen, (250, 12, 400, 46), color, 17, 1)
    draw_text(screen, f"{int(time_left) // 60}:{int(time_left) % 60:02d}", (264, 29), 16, color, "title")
    frac = max(0.0, min(1.0, time_left / 120))
    cv2.line(screen, (325 * S, 29 * S), (385 * S, 29 * S), (90, 60, 80), 4 * S, cv2.LINE_AA)
    neon_line(screen, (325, 29), (325 + 60 * frac, 29), color, 4)
    # Score
    draw_panel(screen, (410, 12, 525, 46), panel, 0.8, 17)
    neon_rect(screen, (410, 12, 525, 46), cyan, 17, 1)
    neon_circle(screen, (428, 29), 5, lime, -1)
    draw_text(screen, str(hits), (439, 29), 16, white, "title")
    neon_circle(screen, (476, 29), 5, red, -1)
    draw_text(screen, str(misses), (487, 29), 16, white, "title")

    draw_problem(screen, problem_text)
    draw_boxes(screen, mental_math_boxes, {"answer": "YOUR ANSWER"})
    draw_strokes(screen, canvas, cyan)
    if feedback:
        draw_chip(screen, feedback, (width / 2, 362), lime if feedback.startswith("Correct") else red)
    draw_buttons(screen, mental_math_buttons, pen, hold)


def draw_versus(screen: np.array, canvas: np.array, problem_text: str, points: list, target: int,
                feedback: str, pen=None, hold=None):
    draw_header(screen, "Versus")
    # Scoreboard: player 1 cyan, player 2 magenta
    draw_panel(screen, (222, 12, 418, 46), panel, 0.8, 17)
    neon_rect(screen, (222, 12, 418, 46), white, 17, 1)
    draw_text(screen, f"P1  {points[0]}", (240, 29), 17, cyan, "title", neon=True)
    draw_text_centered(screen, f"to {target}", (320, 29), 10, muted)
    draw_text(screen, f"{points[1]}  P2", (350, 29), 17, magenta, "title", neon=True)

    draw_problem(screen, problem_text)
    draw_boxes(screen, versus_boxes, {"answer1": "PLAYER 1", "answer2": "PLAYER 2"},
               {"answer1": cyan, "answer2": magenta})
    draw_strokes(screen, canvas, cyan, versus_boxes["answer1"])
    draw_strokes(screen, canvas, magenta, versus_boxes["answer2"])
    if feedback:
        draw_chip(screen, feedback, (width / 2, 362), yellow)
    draw_buttons(screen, versus_buttons, pen, hold)


def draw_results(screen: np.array, title: str, lines: list, pen=None, hold=None):
    draw_header(screen, "Results")
    rect = (110, 62, 530, 385)
    draw_panel(screen, rect, panel, 0.85, 22)
    neon_rect(screen, rect, magenta, 22, 1)
    draw_text_centered(screen, title, (width / 2, 95), 20, magenta, "title", neon=True)
    neon_line(screen, (150, 120), (490, 120), magenta, 1)
    for i, line in enumerate(lines[:9]):         # compute_stats gives up to 8 lines + "New record!"
        color = yellow if "record" in line.lower() or "wins" in line.lower() else white
        draw_text(screen, line, (150, 148 + 29 * i), 15, color, "bold" if color == yellow else "body",
                  neon=color == yellow)
    draw_buttons(screen, results_buttons, pen, hold)
