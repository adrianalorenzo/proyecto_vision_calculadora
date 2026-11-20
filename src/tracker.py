import cv2
import numpy as np
import mediapipe as mp

#parameters
gesture_frames = 4 #a gesture must rpeat this many frames to count
pinch_ratio = 0.25 #pinch if thumb-index distance < pitch_ratio * hand size
bbox_margin = 20 #[px] margin of the bounding box
reset_distnace = 100 #[px] if the fingertip jumps more than this, the Kalman restarts
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