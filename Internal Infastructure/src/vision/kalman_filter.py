import cv2
import numpy as np

class KalmanBallTracker:
    def __init__(self):
        # Initialize Kalman Filter from OpenCV
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 states: x, y, dx, dy; 2 measurements: x, y

        # Transition matrix (A)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + dx
            [0, 1, 0, 1],  # y = y + dy
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix (H)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

        self.initialized = False

    def update(self, x, y):
        """Update tracker with a new measurement."""
        measurement = np.array([[np.float32(x)], [np.float32(y)]])

        if not self.initialized:
            # Initialize state vector [x, y, dx, dy]
            self.kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.initialized = True

        prediction = self.kalman.predict()
        estimate = self.kalman.correct(measurement)

        smoothed_x, smoothed_y = estimate[0][0], estimate[1][0]
        return smoothed_x, smoothed_y

    def predict_only(self):
        """Use this if no measurement is available (e.g., occlusion)."""
        prediction = self.kalman.predict()
        return prediction[0][0], prediction[1][0]
