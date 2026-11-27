import os
import time
import random
import cv2
import numpy as np
import security
import sounds
import ui

# ---------------------------------------------------------------
# AirMath main program: camera -> undistort -> security -> menu -> modes
#   q: quit | f: full screen on/off | m: music on/off | r: start / stop recording the video
# ---------------------------------------------------------------

camera_index = 0
params_path = "../data/calibration/params.npz"
mental_math_time = 120          # [s] duration of a Mental Math game
versus_target = 5               # points to win a Versus game
clear_after = 4.0               # [s] the Calculator result is erased after this time
stroke_thickness = 12           # strokes are drawn white with this thickness on the canvas
hold_time = 0.5                 # [s] the pinch must be held this time over a button to press it
press_cooldown = 1.0            # [s] after a press, no other press (the pinch can flicker for a frame)
stroke_delay = 4                # segments kept on hold: if the hand closes, they are thrown away (no tails)
test_folder = "../output/air_tests"   # every box read is saved here (crop + log.csv) to tune digits.py

# ---------------- Modules of Adriana and Sofia ----------------
# While a module does not exist yet, a simple stand-in is used so main.py always runs.
try:
    import tracker
    hands = tracker.create_hands()
    kalman = tracker.create_kalman()
    use_mouse = False
except ImportError:
    print("tracker.py not found: the mouse is used as the pen (hold the left button)")
    use_mouse = True

try:
    from digits import predict_number, predict_symbol
except ImportError:
    print("digits.py not found: every box is read as '?'")
    def predict_number(crop): return "?"
    def predict_symbol(crop): return "?"

try:
    from modes import evaluate, new_problem, compute_stats, save_record
except ImportError:
    print("modes.py not found: simple stand-ins for the modes")
    def evaluate(expression): return "?"
    def new_problem(level):
        a, b = random.randint(1, 9), random.randint(1, 9)
        return f"{a} + {b}", a + b
    def compute_stats(history, level):
        hits = sum(h["correct"] for h in history)
        return [f"Hits: {hits}", f"Misses: {len(history) - hits}"]
    def save_record(history, level): return False


# ---------------- Mouse as pen (only when there is no tracker) ----------------
mouse = {"pos": None, "down": False, "pad": 0}


def on_mouse(event, x, y, flags, param):
    mouse["pos"] = ui.screen_to_image((x, y), mouse["pad"])     # screen pixels -> 640x480
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse["down"] = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse["down"] = False


def mouse_pen(buttons: list):
    # Left button over a button = pinch (click); left button elsewhere = index (write)
    pen = mouse["pos"]
    if not mouse["down"]:
        return pen, None
    if ui.get_button(pen, buttons) is not None:
        return pen, "pinch"
    return pen, "index"


# ---------------- Helpers ----------------

def load_undistort_maps(path: str):
    if not os.path.exists(path):
        print("No calibration found: the frames are not undistorted")
        return None, None
    params = np.load(path)
    K, dist = params["intrinsics"], params["dist_coeffs"]
    size = tuple(int(v) for v in params["image_size"])
    return cv2.initUndistortRectifyMap(K, dist, None, K, size, cv2.CV_16SC2)


def mirror_corners(cards: list) -> list:
    # The cards are read on the non-mirrored frame; their corners are mirrored to draw them
    return [(symbol, np.column_stack((ui.width - 1 - c[:, 0], c[:, 1]))) for symbol, c in cards]


def crop_box(canvas: np.array, rect: tuple) -> np.array:
    x1, y1, x2, y2 = rect
    return canvas[y1:y2, x1:x2]


def clear_box(canvas: np.array, rect: tuple):
    x1, y1, x2, y2 = rect
    canvas[y1:y2, x1:x2] = 0


def save_test(mode: str, crops: dict, read: str, expected=""):
    # Saves the crops of the boxes that were read and one line in log.csv: time, mode, expected, read
    os.makedirs(test_folder, exist_ok=True)
    stamp = time.strftime("%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    for name, crop in crops.items():
        cv2.imwrite(os.path.join(test_folder, f"{stamp}_{mode}_{name}.png"), crop)
    with open(os.path.join(test_folder, "log.csv"), "a") as f:
        f.write(f"{stamp},{mode},{expected},{read}\n")


def screen_elements(state: str):
    # Buttons and writing boxes of every screen
    if state == "MENU":
        return ui.menu_buttons, {}
    if state == "CALCULATOR":
        return ui.calculator_buttons, ui.calculator_boxes
    if state == "MENTAL_MATH":
        return ui.mental_math_buttons, ui.mental_math_boxes
    if state == "VERSUS":
        return ui.versus_buttons, ui.versus_boxes
    if state == "RESULTS":
        return ui.results_buttons, {}
    return [], {}


# ---------------- Initialization ----------------
map1, map2 = load_undistort_maps(params_path)
templates = security.load_templates()

cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, ui.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ui.height)
if not cap.isOpened():
    print(f"Could not open the camera (index {camera_index})")
    exit()

window = "AirMath"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)             # resizable window, scaled image
cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
if use_mouse:
    cv2.setMouseCallback(window, on_mouse)

state = "LOCKED"
decoder = security.new_decoder_state()
canvas = np.zeros((ui.height, ui.width), dtype=np.uint8)   # strokes: white (255) on black (0)
prev_pen, prev_gesture = None, None
hold_name, hold_start, hold_done = None, 0.0, False    # button being pressed with the pinch
last_press = 0.0
pending = []                                            # last segments of the stroke, not yet in the canvas
prev_time, fps = time.time(), 0.0
writer = None

# Data of the modes
read_text, result_text, result_time = "", "", 0.0         # Calculator
level, problem_text, solution = "easy", "", 0             # Mental Math / Versus
history, end_time, hits, misses, feedback = [], 0.0, 0, 0, ""
points = [0, 0]
results_title, results_lines = "", []

print("q: quit | f: full screen | m: music | r: record | u: unlock without cards")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received")
        break
    now = time.time()

    # 1. Undistort with the calibration (before mirroring)
    if map1 is not None:
        frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    display = cv2.flip(frame, 1)          # mirror image: what the user sees

    # 2. LOCKED: security block on the non-mirrored frame
    if state == "LOCKED":
        screen = ui.new_frame(display, 0.8)
        n_before, fails_before, locked_before = len(decoder["entered"]), decoder["fails"], decoder["locked_until"]
        cards = security.detect_cards(frame, templates)
        decoder = security.update_decoder(decoder, cards[0][0] if cards else None)
        ui.draw_locked_screen(screen, decoder, mirror_corners(cards), len(security.password), now)

        # Sounds and flashes of the security block
        if decoder["unlocked"]:
            state = "MENU"
            sounds.play("unlock")
            ui.flash(ui.lime)
            sounds.start_music()
        elif decoder["locked_until"] != locked_before:
            sounds.play("locked")
            ui.flash(ui.red)
        elif decoder["fails"] != fails_before:
            sounds.play("error")
            ui.flash(ui.red)
        elif len(decoder["entered"]) > n_before:
            sounds.play("card_ok")
            ui.flash(ui.cyan)
        elif len(decoder["entered"]) < n_before:
            sounds.play("reset")

    else:
        buttons, boxes = screen_elements(state)

        # 3. Tracker: pen position, gesture and bounding box of the hand
        bbox = None
        if use_mouse:
            pen, gesture = mouse_pen(buttons)
        else:
            landmarks = tracker.get_landmarks(display, hands)
            if landmarks is None:
                pen, gesture = None, None
            else:
                pen = tracker.get_pen(landmarks, kalman)
                gesture = tracker.get_gesture(landmarks)
                bbox = tracker.get_bbox(landmarks)

        # 4. Writing: only with the index gesture and inside a box.
        # The gesture changes some frames after the hand closes, so the last segments are kept
        # on hold (pending) and thrown away if the index stops: this removes the tails of the strokes
        box = ui.get_box(pen, boxes)
        if gesture == "index" and prev_gesture == "index" and box is not None and box == ui.get_box(prev_pen, boxes):
            pending.append(((int(prev_pen[0]), int(prev_pen[1])), (int(pen[0]), int(pen[1]))))
            if len(pending) > (0 if use_mouse else stroke_delay):     # the mouse has no tails
                p1, p2 = pending.pop(0)
                cv2.line(canvas, p1, p2, 255, stroke_thickness)
        elif gesture != "index":
            pending.clear()
        shown_canvas = canvas.copy()                     # what is drawn on the screen (with the pending segments)
        for p1, p2 in pending:
            cv2.line(shown_canvas, p1, p2, 255, stroke_thickness)

        # 5. Click: the pinch is held over a button for hold_time (the button fills up)
        pressed, hold = None, None
        over = ui.get_button(pen, buttons) if gesture == "pinch" else None
        if over != hold_name:
            hold_name, hold_start, hold_done = over, now, False
        if hold_name is not None and not hold_done:
            hold = (hold_name, (now - hold_start) / hold_time)
            if hold[1] >= 1 and now - last_press > press_cooldown:
                pressed, hold_done, last_press = hold_name, True, now
                sounds.play("click")

        # 6. State machine
        if pressed == "menu" or pressed == "back":
            state = "MENU"
            canvas[:] = 0

        elif state == "MENU":
            if pressed == "calculator":
                state = "CALCULATOR"
                read_text, result_text = "", ""
            elif pressed in ("mm_easy", "mm_hard"):
                state = "MENTAL_MATH"
                level = "easy" if pressed == "mm_easy" else "hard"
                problem_text, solution = new_problem(level)
                history, hits, misses, feedback = [], 0, 0, ""
                end_time = now + mental_math_time
            elif pressed == "versus":
                state = "VERSUS"
                problem_text, solution = new_problem("easy")
                points, feedback = [0, 0], ""
            elif pressed == "lock":
                state = "LOCKED"
                decoder = security.new_decoder_state()

        elif state == "CALCULATOR":
            if pressed == "equals":
                num1 = predict_number(crop_box(canvas, ui.calculator_boxes["num1"]))
                op = predict_symbol(crop_box(canvas, ui.calculator_boxes["op"]))
                num2 = predict_number(crop_box(canvas, ui.calculator_boxes["num2"]))
                read_text = f"{num1} {op} {num2}"
                save_test("calc", {name: crop_box(canvas, rect) for name, rect in ui.calculator_boxes.items()}, read_text)
                result_text = str(evaluate(read_text))
                result_time = now
                if "?" in read_text or result_text in ("?", "Error"):
                    sounds.play("error")
                    ui.flash(ui.red)
                else:
                    sounds.play("correct")
                    ui.flash(ui.yellow)
            elif pressed == "clear":
                canvas[:] = 0
                read_text, result_text = "", ""
            if read_text and now - result_time > clear_after:
                canvas[:] = 0
                read_text, result_text = "", ""

        elif state == "MENTAL_MATH":
            rect = ui.mental_math_boxes["answer"]
            if pressed == "submit":
                answer = predict_number(crop_box(canvas, rect))
                save_test("mental_math", {"answer": crop_box(canvas, rect)}, answer, solution)
                correct = answer == str(solution)
                history.append({"problem": problem_text, "solution": solution, "answer": answer,
                                "correct": correct, "time": now})
                hits, misses = hits + correct, misses + (not correct)
                feedback = "Correct!" if correct else f"Wrong: it was {solution} (read {answer})"
                sounds.play("correct" if correct else "error")
                ui.flash(ui.lime if correct else ui.red)
                problem_text, solution = new_problem(level)
                clear_box(canvas, rect)
            elif pressed == "clear":
                clear_box(canvas, rect)
            elif pressed == "not_what_i_wrote" and history:
                # The last answer was misread: undo it and repeat the same problem
                last = history.pop()
                hits, misses = hits - last["correct"], misses - (not last["correct"])
                problem_text, solution = last["problem"], last["solution"]
                feedback = "Write it again"
                clear_box(canvas, rect)
            if now > end_time:
                state = "RESULTS"
                results_title = f"Mental Math ({level}) - results"
                results_lines = compute_stats(history, level)
                if save_record(history, level):
                    results_lines.append("New record!")
                    sounds.play("record")
                else:
                    sounds.play("game_over")
                canvas[:] = 0

        elif state == "VERSUS":
            if pressed == "clear1":
                clear_box(canvas, ui.versus_boxes["answer1"])
            elif pressed == "clear2":
                clear_box(canvas, ui.versus_boxes["answer2"])
            for player, button, box_name in [(0, "submit1", "answer1"), (1, "submit2", "answer2")]:
                if pressed != button:
                    continue
                rect = ui.versus_boxes[box_name]
                answer = predict_number(crop_box(canvas, rect))
                save_test("versus", {box_name: crop_box(canvas, rect)}, answer, solution)
                if answer == str(solution):
                    points[player] += 1
                    feedback = f"Point for Player {player + 1}!"
                    problem_text, solution = new_problem("easy")
                    canvas[:] = 0
                    sounds.play("point")
                    ui.flash(ui.player1 if player == 0 else ui.player2)
                else:
                    feedback = f"Player {player + 1}: wrong ({answer})"
                    clear_box(canvas, rect)
                    sounds.play("error")
                    ui.flash(ui.red)
            if max(points) >= versus_target:
                state = "RESULTS"
                winner = 1 if points[0] > points[1] else 2
                results_title = "Versus - results"
                results_lines = [f"Player {winner} wins!", f"P1 {points[0]} - {points[1]} P2"]
                canvas[:] = 0
                sounds.play("record")

        # 7. Draw the current screen
        screen = ui.new_frame(display)
        if state == "MENU":
            ui.draw_menu(screen, pen, hold)
        elif state == "CALCULATOR":
            ui.draw_calculator(screen, shown_canvas, read_text, result_text, pen, hold)
        elif state == "MENTAL_MATH":
            ui.draw_mental_math(screen, shown_canvas, problem_text, max(end_time - now, 0), hits, misses, feedback, pen, hold)
        elif state == "VERSUS":
            ui.draw_versus(screen, shown_canvas, problem_text, points, versus_target, feedback, pen, hold)
        elif state == "RESULTS":
            ui.draw_results(screen, results_title, results_lines, pen, hold)
        ui.draw_hand_box(screen, bbox)
        ui.draw_cursor(screen, pen, gesture)
        prev_pen, prev_gesture = pen, gesture

    # 8. FPS, glow, recording and window
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
    prev_time = now
    ui.draw_fps(screen, fps)
    ui.finish_frame(screen)
    if writer is not None:
        writer.write(screen)
        ui.draw_recording(screen)
    shown, mouse["pad"] = ui.fit_to_window(screen, window)
    cv2.imshow(window, shown)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("f"):
        # Toggle full screen (the image is scaled to the screen)
        full = cv2.getWindowProperty(window, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL if full else cv2.WINDOW_FULLSCREEN)
    elif key == ord("m"):
        sounds.toggle_music()
    elif key == ord("u") and state == "LOCKED":
        decoder["unlocked"] = True              # unlock without cards (only for testing)
    elif key == ord("r"):
        if writer is None:
            os.makedirs("../output", exist_ok=True)
            path = f"../output/airmath_{int(now)}.avi"
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), 20, (screen.shape[1], screen.shape[0]))
            print("Recording:", path)
        else:
            writer.release()
            writer = None
            print("Recording stopped")

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
