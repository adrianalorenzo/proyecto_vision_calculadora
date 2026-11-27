import time
import cv2
import numpy as np
import sounds
import ui

# Shows every screen of the interface with fake data, to work on the design without the other modules
#   1 locked | 2 menu | 3 calculator | 4 mental math | 5 versus | 6 results
#   f full screen | m music | s sounds demo | q quit
# The mouse is the pen: hold the left button on a button to fill it (and hear the click)
camera_index = 0
hold_time = 0.5

cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, ui.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ui.height)

mouse = {"pos": None, "down": False, "pad": 0}


def on_mouse(event, x, y, flags, param):
    mouse["pos"] = ui.screen_to_image((x, y), mouse["pad"])
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse["down"] = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse["down"] = False


window = "Test UI"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)            # resizable window, scaled image
cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(window, on_mouse)

# Fake data
canvas = np.zeros((ui.height, ui.width), dtype=np.uint8)
cv2.putText(canvas, "12", (60, 250), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 12)
cv2.putText(canvas, "x", (285, 250), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 12)
cv2.putText(canvas, "7", (460, 250), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 12)
fake_decoder = {"entered": ["sigma", "pi"], "locked_until": 0.0, "message": "Correct, next card"}
fake_card = [("inf", np.array([[250, 150], [380, 160], [370, 330], [240, 320]], dtype=np.float32))]
demo = ["click", "card_ok", "unlock", "correct", "error", "point", "record"]
screen_id = ord("1")
hold_name, hold_start = None, 0.0

while True:
    ret, frame = cap.read()
    display = cv2.flip(frame, 1) if ret else np.full((ui.height, ui.width, 3), 120, np.uint8)
    pen = mouse["pos"]
    now = time.time()
    screen = ui.new_frame(display, 0.8 if screen_id == ord("1") else 0.55)

    buttons = {ord("2"): ui.menu_buttons, ord("3"): ui.calculator_buttons, ord("4"): ui.mental_math_buttons,
               ord("5"): ui.versus_buttons, ord("6"): ui.results_buttons}.get(screen_id, [])

    # Holding the left button over a button fills it, like the pinch in main.py
    over = ui.get_button(pen, buttons) if mouse["down"] else None
    if over != hold_name:
        hold_name, hold_start = over, now
    hold = (hold_name, (now - hold_start) / hold_time) if hold_name else None
    if hold and hold[1] >= 1:
        print("Pressed:", hold_name)
        sounds.play("click")
        ui.flash(ui.cyan)
        hold_name, hold_start = None, now + 999

    if screen_id == ord("1"):
        ui.draw_locked_screen(screen, fake_decoder, fake_card, 4, now)
    elif screen_id == ord("2"):
        ui.draw_menu(screen, pen, hold)
    elif screen_id == ord("3"):
        ui.draw_calculator(screen, canvas, "12 x 7", "84", pen, hold)
    elif screen_id == ord("4"):
        ui.draw_mental_math(screen, np.zeros_like(canvas), "7 x 8", 95, 5, 1, "Correct!", pen, hold)
    elif screen_id == ord("5"):
        ui.draw_versus(screen, np.zeros_like(canvas), "15 - 6", [2, 3], 5, "Point for Player 2!", pen, hold)
    else:
        lines = ["Hits: 14", "Misses: 3", "Operations per minute: 8.5", "Worst operation: /", "New record!"]
        ui.draw_results(screen, "Mental Math (easy)", lines, pen, hold)

    ui.draw_cursor(screen, pen, "pinch" if mouse["down"] else None)
    ui.draw_fps(screen, 30)
    ui.finish_frame(screen)
    shown, mouse["pad"] = ui.fit_to_window(screen, window)
    cv2.imshow(window, shown)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("f"):
        full = cv2.getWindowProperty(window, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL if full else cv2.WINDOW_FULLSCREEN)
    elif key == ord("m"):
        sounds.toggle_music()
    elif key == ord("s"):
        # Plays the next sound effect of the list
        sounds.play(demo[0])
        print("Sound:", demo[0])
        demo.append(demo.pop(0))
    elif ord("1") <= key <= ord("6"):
        screen_id = key

cap.release()
cv2.destroyAllWindows()
