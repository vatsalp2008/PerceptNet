"""3D Kalman filter with a constant-velocity motion model (the AB3DMOT core).

State (10-dim), kept in the *canonical box order* so it can be compared with
detections and fed to :func:`perceptnet.geometry.iou.iou_3d` with no reordering:

    [x, y, z, dx, dy, dz, heading, vx, vy, vz]

(The project spec writes the state as ``[x, y, z, theta, l, w, h, vx, vy, vz]``;
this is the same information with the angle moved next to the dimensions. We keep
``heading`` at index 6 so ``state[:7]`` *is* a canonical box.)

Measurement (7-dim) is the canonical box ``[x, y, z, dx, dy, dz, heading]``.
"""

from __future__ import annotations

import numpy as np

from perceptnet.geometry.heading import normalize_angle

ANGLE_IDX = 6


class KalmanFilter3D:
    """Constant-velocity 3D Kalman filter over a single object's box."""

    def __init__(self, measurement, dt: float = 1.0):
        measurement = np.asarray(measurement, dtype=np.float64).reshape(7)

        self.dim_x = 10
        self.dim_z = 7

        # State transition: position += velocity * dt; everything else constant.
        self.F = np.eye(self.dim_x)
        self.F[0, 7] = self.F[1, 8] = self.F[2, 9] = dt

        # Measurement: observe the 7 box parameters directly.
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[:7, :7] = np.eye(7)

        # Covariance: confident in the initial box, very unsure of velocity.
        self.P = np.eye(self.dim_x) * 10.0
        self.P[7:, 7:] *= 1000.0

        # Process / measurement noise.
        self.Q = np.eye(self.dim_x)
        self.Q[7:, 7:] *= 0.01          # velocity drifts slowly
        self.R = np.eye(self.dim_z)

        self.x = np.zeros(self.dim_x)
        self.x[:7] = measurement

    def predict(self) -> np.ndarray:
        """Advance the state one step; returns the predicted box ``(7,)``."""
        self.x = self.F @ self.x
        self.x[ANGLE_IDX] = normalize_angle(self.x[ANGLE_IDX])
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:7].copy()

    def update(self, measurement) -> np.ndarray:
        """Correct the state with a new box measurement ``(7,)``."""
        z = np.asarray(measurement, dtype=np.float64).reshape(7).copy()

        # --- orientation correction (AB3DMOT) -------------------------------
        # Headings wrap, and a detector may report a box flipped by pi. Bring the
        # measured heading into agreement with the predicted one before forming
        # the residual so the filter never "snaps" 180 deg.
        self.x[ANGLE_IDX] = normalize_angle(self.x[ANGLE_IDX])
        z[ANGLE_IDX] = normalize_angle(z[ANGLE_IDX])
        diff = abs(z[ANGLE_IDX] - self.x[ANGLE_IDX])
        if np.pi / 2.0 < diff < 3.0 * np.pi / 2.0:
            self.x[ANGLE_IDX] = normalize_angle(self.x[ANGLE_IDX] + np.pi)
        # near-2pi wrap (opposite ends of the [-pi, pi) range)
        if abs(z[ANGLE_IDX] - self.x[ANGLE_IDX]) >= 3.0 * np.pi / 2.0:
            self.x[ANGLE_IDX] += 2.0 * np.pi if z[ANGLE_IDX] > 0 else -2.0 * np.pi

        y = z - self.H @ self.x
        y[ANGLE_IDX] = normalize_angle(y[ANGLE_IDX])

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[ANGLE_IDX] = normalize_angle(self.x[ANGLE_IDX])
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P
        return self.x[:7].copy()

    @property
    def box(self) -> np.ndarray:
        return self.x[:7].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Estimated ``(vx, vy, vz)`` in box units per frame."""
        return self.x[7:10].copy()
