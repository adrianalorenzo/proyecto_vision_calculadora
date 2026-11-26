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

#memory bewtween frames (instead of a class)
last_gestures = []
stable_gesture = None
kalman_started = False 

def create_hands():
    hands = mp.solutions.hands.Hands(max_num_hands=1,min_detection_confidence=0.7,min_tracking_confidence=0.5)
    return hands

def create_kalman():
    #Same as Lab 5: state (x,y,vx, vy), measurement (x,y), constante velocity
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


def get_landmarks(frame, hands):
    """
    Detects the hand and returns its 21 landmarks in pixels.

    Args:
    -frame: mirrored BGR image (640x480)
    -hands: object returned by create_hands()

    Returns:
    -List of 21 (x,y) poinys, or None if there is no hand
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #MediaPipe works in RGB
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    #MediaPipe gives voordinates in [0,1]: convert to pixels
    return [(int(p.x * w), int(p.y * h)) for p in hand.landmark]


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


def get_gesture(landmarks):
    """
    Stable gesture: it only changes if the same gesture is seen during gesture_frames frames.

    Returns:
        "index", "fist", "pinch" or None
    """
    global stable_gesture
    last_gestures.append(raw_gesture(landmarks))
    if len(last_gestures)>gesture_frames:
        last_gestures.pop(0)
    if len(last_gestures) == gesture_frames and all(g == last_gestures[0] for g in last_gestures):
        stable_gesture = last_gestures[0]
    return stable_gesture

def get_pen(landmarks, kalman):
    """
    Position of the index fingertip (landmark 8), smoothed with the Kalman filter

    Returns:
        (x,y) in pixels
    """
    global kalman_started
    x,y = landmarks[8]
    if not use_kalman:
        return (x,y)

    measurement = np.array([[x], [y]], np.float32)
    jump = distance((x,y), (kalman.statePost[0,0], kalman.statePost[1,0]))
    #first time, or the hand reappeared far away: start the filter at the fingertip
    if not kalman_started or jump>reset_distance:
        kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
        kalman.errorCovPost = np.eye(4, dtype=np.float32)
        kalman_started = True

    kalman.predict()
    kalman.correct(measurement)
    return (int(kalman.statePost[0,0]), int(kalman.statePost[1,0]))