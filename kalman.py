# kalman.py
# Implements a 2D Kalman filter for smoothing noisy position data.
#
# The Kalman filter works by maintaining two pieces of information:
#   - A state estimate (where we think the point is and how fast it's moving)
#   - A covariance matrix (how uncertain we are about that estimate)
#
# Each time step it does two things:
#   1. PREDICT — project the state forward using a motion model
#   2. UPDATE  — blend the prediction with the new (noisy) measurement,
#                weighted by how much we trust each source

import numpy as np


class KalmanFilter2D:
    """
    Tracks a 2D point (x, y) and its velocity using a Kalman filter.

    State vector: [x, vx, y, vy]
      x, y  — position
      vx, vy — velocity (not directly measured, estimated by the filter)
    """

    def __init__(self, process_noise=1e-3, measurement_noise=0.1):
        # process_noise:    how much we trust our motion model
        #                   lower → smoother output but slower to respond to real motion
        # measurement_noise: how noisy we believe the sensor is
        #                   higher → trust the sensor less, rely more on prediction

        self.dt = 1.0  # normalized time step between measurements

        # State transition matrix F: constant-velocity motion model
        # New state = F @ old state:
        #   x_new  = x + vx*dt
        #   vx_new = vx  (velocity assumed constant)
        #   (same for y)
        self.F = np.array([
            [1, self.dt, 0,       0],
            [0, 1,       0,       0],
            [0, 0,       1, self.dt],
            [0, 0,       0,       1],
        ])

        # Measurement matrix H: maps state → what we can observe
        # We only measure position (x, y), not velocity
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ])

        # Process noise covariance Q: uncertainty introduced by the motion model
        self.Q = np.eye(4) * process_noise

        # Measurement noise covariance R: uncertainty in sensor readings
        self.R = np.eye(2) * measurement_noise

        # State estimate and covariance — initialized on first measurement
        self.x = None
        self.P = np.eye(4) * 1.0

    def _initialize(self, measurement):
        # Seed the state with the first measurement; assume zero initial velocity
        self.x = np.array([measurement[0], 0.0, measurement[1], 0.0])

    def update(self, measurement):
        """
        Process one noisy (x, y) measurement and return the filtered position.

        On the first call, the filter initializes from the measurement directly.
        On subsequent calls it runs the full predict → update cycle.
        """
        measurement = np.array(measurement)

        if self.x is None:
            self._initialize(measurement)
            return measurement.copy()

        # ── Predict step ─────────────────────────────────────────────────
        # Project the previous state forward one time step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # ── Update step ──────────────────────────────────────────────────
        # Innovation: how far the measurement is from what we predicted
        y = measurement - self.H @ x_pred

        # Innovation covariance: combined uncertainty of prediction + sensor
        S = self.H @ P_pred @ self.H.T + self.R

        # Kalman gain: the blending weight between prediction and measurement
        # High gain → trust measurement more; low gain → trust prediction more
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Fuse prediction and measurement
        self.x = x_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred

        # Return only the position components of the state
        return np.array([self.x[0], self.x[2]])


def filter_path(noisy_path, process_noise=1e-3, measurement_noise=0.1):
    """Run a Kalman filter over an entire path and return the smoothed version."""
    kf = KalmanFilter2D(process_noise=process_noise, measurement_noise=measurement_noise)
    return np.array([kf.update(pt) for pt in noisy_path])
