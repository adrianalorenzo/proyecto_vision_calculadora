import cv2
import numpy as np
import mediapipe as mp

#parameters
gesture_frames = 4 #a gesture must rpeat this many frames to count
pinch_ratio = 0.25 #pinch if thumb-index distance < pitch_ratio * hand size
bbox_margin = 20 #[px] margin of the bounding box
reset_distance = 100 #[px] if the fingertip jumps more than this, the Kalman restarts
use_kalman = True #False to compare without filter
width, height =640, 480
max_lost_frames = 5 #[frames] without hand, the pen is predicted by the Kalman at most this many frames

#memory bewtween frames (instead of a class), one per player (0 = Player 1, 1 = Player 2)
last_gestures = [[],[]]
stable_gesture = [None,None]
kalman_started = [False, False]
lost_frames = [0, 0]

def create_hands():
    """
    Creates the MediaPipe Hands detector (up to two hands, for Versus).

    Returns:
        The mp.solutions.hands.Hands object used by get_all_landmarks and get_landmarks.
    """
    hands = mp.solutions.hands.Hands(max_num_hands=2,min_detection_confidence=0.7,min_tracking_confidence=0.5)
    return hands

def create_kalman():
    """
    Kalman filter for the index fingertip (same as Lab 5).

    State (x, y, vx, vy) with a constant velocity model; measurement (x, y).

    Returns:
        The configured cv2.KalmanFilter.
    """
    kf = cv2.KalmanFilter(4,2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
    return kf

def get_all_landmarks(frame, hands):
    """
    Detects up to two hands and returns their landmarks in pixels.

    Args:
        frame: mirrored BGR image (640x480).
        hands: object returned by create_hands().

    Returns:
        List with one list of 21 (x, y) points per hand (0, 1 or 2 hands),
        sorted from left to right by the x of the wrist.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     #MediaPipe works in RGB
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return []
    all_hands = [[(int(p.x * w), int(p.y * h)) for p in hand.landmark]
                 for hand in results.multi_hand_landmarks]
    all_hands.sort(key=lambda lm: lm[0][0])          #left to right
    return all_hands


def get_landmarks(frame, hands):
    """
    Landmarks of ONE hand (the modes with one player). If there are two hands,
    the biggest one is used: the one closest to the camera.

    Returns:
        List of 21 (x, y) points, or None if there is no hand.
    """
    all_hands = get_all_landmarks(frame, hands)
    if not all_hands:
        return None
    return max(all_hands, key=lambda lm: distance(lm[0], lm[9]))   #hand size: wrist - middle knuckle
def distance(p, q):
    """Euclidian distance between two(x,y) points"""
    return np.hypot(p[0]-q[0], p[1]-q[1])


def get_bbox(landmarks):
    """
    Bounding box of the hand, with a margin.

    Returns:
        (x1, y1, x2, y2) inside the image.
    """
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    x1 = max(min(xs) - bbox_margin, 0)
    y1 = max(min(ys) - bbox_margin, 0)
    x2 = min(max(xs) + bbox_margin, width - 1)
    y2 = min(max(ys) + bbox_margin, height - 1)
    return (x1, y1, x2, y2)


def finger_extended(landmarks, tip, pip):
    """A finger is extended if its tip is farther from the wrist than its middle joint."""
    wrist = landmarks[0]
    return distance(wrist, landmarks[tip]) > distance(wrist, landmarks[pip])


def raw_gesture(landmarks):
    """
    Gesture of one frame by geometric rules (without stabilising).

    Returns:
        "pinch", "index", "fist" or None.
    """
    index = finger_extended(landmarks, 8, 6)
    middle = finger_extended(landmarks, 12, 10)
    ring = finger_extended(landmarks, 16, 14)
    pinky = finger_extended(landmarks, 20, 18)

    size = distance(landmarks[0], landmarks[9])               # wrist - middle knuckle
    pinch_distance = distance(landmarks[4], landmarks[8])     # thumb tip - index tip
    index_open = distance(landmarks[0], landmarks[8]) > distance(landmarks[0], landmarks[5])

    if pinch_distance < pinch_ratio * size and index_open:
        return "pinch"
    if index and not middle and not ring and not pinky:
        return "index"
    if not index and not middle and not ring and not pinky:
        return "fist"
    return None


def get_gesture(landmarks, player=0):
    """
    Stable gesture: it only changes if the same gesture is seen during gesture_frames frames.
    Each player has her own memory
    Args:
    -landmarks: the 21 points of the hand
    -player: 0 or 1 (0 by default, fot the modes with one player)
    Returns:
        "index", "fist", "pinch" or None
    """
    last = last_gestures[player]
    last.append(raw_gesture(landmarks))
    if len(last)>gesture_frames:
        last.pop(0)
    if len(last) == gesture_frames and all(g == last[0] for g in last):
        stable_gesture[player] = last[0]
    return stable_gesture[player]

def get_pen(landmarks, kalman, player=0):
    """
    Position of the index fingertip (landmark 8), smoothed with the Kalman filter

    Returns:
        (x,y) in pixels
    """
    
    x,y = landmarks[8]
    if not use_kalman:
        return (x,y)

    measurement = np.array([[x], [y]], np.float32)
    jump = distance((x,y), (kalman.statePost[0,0], kalman.statePost[1,0]))
    #first time, or the hand reappeared far away: start the filter at the fingertip
    if not kalman_started[player] or jump>reset_distance:
        kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
        kalman.errorCovPost = np.eye(4, dtype=np.float32)
        kalman_started[player] = True

    lost_frames[player] = 0
    kalman.predict()
    kalman.correct(measurement)
    return (int(kalman.statePost[0,0]), int(kalman.statePost[1,0]))

def predict_pen(kalman, player=0):
    """
    Position of the fingertip when MediaPipe loses the hand for a few frames.

    Only the prediction step of the Kalman filter is done (there is no measurement):
    the pen keeps moving with the last estimated velocity.

    Args:
        kalman: the Kalman filter of this player.
        player: 0 or 1 (0 by default, for the modes with one player).

    Returns:
        (x, y) in pixels, or None if the filter has not started or the hand
        has been lost for more than max_lost_frames frames.
    """
    if not use_kalman or not kalman_started[player] or lost_frames[player] >= max_lost_frames:
        kalman_started[player] = False      #the next detection restarts the filter at the fingertip
        return None
    lost_frames[player] += 1
    kalman.predict()                        #OpenCV also copies the prediction to statePost
    return (int(kalman.statePost[0, 0]), int(kalman.statePost[1, 0]))