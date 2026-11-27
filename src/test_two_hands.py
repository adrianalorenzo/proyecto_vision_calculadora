import cv2
import tracker

# Quick test: two hands, one per half of the screen. q: quit
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
hands = tracker.create_hands()
kalmans = [tracker.create_kalman(), tracker.create_kalman()]
colors = [(255, 0, 255), (0, 255, 0)]      # Player 1 pink, Player 2 green

while True:
    ret, frame = cap.read()
    if not ret:
        break
    display = cv2.flip(frame, 1)
    cv2.line(display, (320, 0), (320, 480), (255, 255, 255), 1)

    for landmarks in tracker.get_all_landmarks(display, hands):
        player = 0 if landmarks[0][0] < 320 else 1      # by half of the screen
        pen = tracker.get_pen(landmarks, kalmans[player], player)
        gesture = tracker.get_gesture(landmarks, player)
        cv2.circle(display, pen, 8, colors[player], -1)
        cv2.putText(display, f"P{player + 1}: {gesture}", (10 + 330 * player, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[player], 2)

    cv2.imshow("Two hands", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()