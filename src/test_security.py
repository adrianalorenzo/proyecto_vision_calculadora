import time
import cv2
import security
import ui

# Live test of the security block: card detection + decoder, without the rest of the project
#   r: reset the decoder | q: quit
camera_index = 0

templates = security.load_templates()
print("Templates loaded:", list(templates.keys()))
print("Password:", security.password)

cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, ui.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ui.height)

state = security.new_decoder_state()
prev_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received")
        break

    cards = security.detect_cards(frame, templates)
    symbol = cards[0][0] if cards else None          # only the biggest card counts
    state = security.update_decoder(state, symbol)

    # Show the symbol score of the biggest card (useful to tune match_threshold)
    if cards:
        interior = security.card_interior(security.warp_card(frame, cards[0][1]))
        name, score = security.match_card(interior, templates)
        cv2.imshow("Interior", interior)
        print(f"{str(name):6s} score {score:.2f}", end="\r")

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
    prev_time = now

    screen = ui.new_frame(frame, 0.8)            # here without mirror, as the cards are read
    ui.draw_locked_screen(screen, state, cards, len(security.password), now)
    if state["unlocked"]:
        ui.draw_text_centered(screen, "UNLOCKED!", (ui.width / 2, ui.height / 2), 48, ui.lime, "bold")
    ui.draw_fps(screen, fps)
    ui.finish_frame(screen)
    cv2.imshow("Test security", screen)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        state = security.new_decoder_state()
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
