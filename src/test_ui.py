import time
import cv2
import numpy as np
import ui

# Shows every screen of the interface with fake data, to work on the design without the other modules
#   1 locked | 2 menu | 3 calculator | 4 mental math | 5 versus | 6 results | q quit
# The mouse is the pen: move it to see the hover and click to print the button name
camera_index = 0

cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, ui.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ui.height)

mouse = {"pos": None, "click": False}


def on_mouse(event, x, y, flags, param):
    mouse["pos"] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse["click"] = True


cv2.namedWindow("Test UI")
cv2.setMouseCallback("Test UI", on_mouse)

# Fake data
canvas = np.zeros((ui.height, ui.width), dtype=np.uint8)
cv2.putText(canvas, "12", (60, 250), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 12)
cv2.putText(canvas, "x", (285, 250), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 12)
cv2.putText(canvas, "7", (460, 250), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 12)
fake_decoder = {"entered": ["sigma", "pi"], "locked_until": 0.0, "message": "Correct, next card"}
fake_card = [("inf", np.array([[250, 150], [380, 160], [370, 330], [240, 320]], dtype=np.float32))]
screen = ord("1")

while True:
    ret, frame = cap.read()
    display = cv2.flip(frame, 1) if ret else np.full((ui.height, ui.width, 3), 120, np.uint8)
    pen = mouse["pos"]
    now = time.time()

    if screen == ord("1"):
        ui.draw_locked_screen(display, fake_decoder, fake_card, 4, now)
        buttons = []
    elif screen == ord("2"):
        ui.draw_menu(display, pen)
        buttons = ui.menu_buttons
    elif screen == ord("3"):
        ui.draw_calculator(display, canvas, "12 x 7", "84", pen)
        buttons = ui.calculator_buttons
    elif screen == ord("4"):
        ui.draw_mental_math(display, np.zeros_like(canvas), "7 x 8", 95, 5, 1, "Correct!", pen)
        buttons = ui.mental_math_buttons
    elif screen == ord("5"):
        ui.draw_versus(display, np.zeros_like(canvas), "15 - 6", [2, 3], 5, "Point for Player 2!", pen)
        buttons = ui.versus_buttons
    else:
        lines = ["Hits: 14", "Misses: 3", "Operations per minute: 8.5", "Worst operation: /"]
        ui.draw_results(display, "Mental Math (easy) - results", lines, pen)
        buttons = ui.results_buttons

    if mouse["click"]:
        print("Pressed:", ui.get_button(pen, buttons))
        mouse["click"] = False

    ui.draw_cursor(display, pen, None)
    ui.draw_fps(display, 30)
    cv2.imshow("Test UI", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif ord("1") <= key <= ord("6"):
        screen = key

cap.release()
cv2.destroyAllWindows()
