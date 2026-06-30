"""Heading-angle conventions and conversions.

Two classic 3D-detection bugs live here, so every conversion is centralized and
round-trip tested:

1. **Frame mismatch.** KITTI labels store ``rotation_y`` (yaw about the camera's
   down-pointing +Y axis, in the rectified camera frame). LiDAR-frame yaw is about
   the up-pointing +z axis. They differ by a sign *and* a 90 deg offset:

       lidar_yaw = -(rotation_y + pi/2)

   This map is an involution (its own inverse), so the same formula converts both
   directions.

2. **sin(2*theta) ambiguity.** A ``sin(2*theta)`` heading loss is invariant to a
   pi flip, so it only resolves orientation up to 180 deg -- cars can come out facing
   backward. PerceptNet therefore regresses the heading as a (sin, cos) pair (full
   2*pi), optionally paired with a direction classifier. See ADR-002 / ADR for the
   decision. The encode/decode helpers below are the (sin, cos) path.
"""

from __future__ import annotations

import numpy as np

HALF_PI = np.pi / 2.0


def normalize_angle(theta):
    """Wrap angle(s) to [-pi, pi)."""
    theta = np.asarray(theta, dtype=np.float64)
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def rotation_y_to_lidar_yaw(rotation_y):
    """KITTI camera ``rotation_y`` -> LiDAR-frame yaw."""
    return normalize_angle(-(np.asarray(rotation_y, dtype=np.float64) + HALF_PI))


def lidar_yaw_to_rotation_y(yaw):
    """LiDAR-frame yaw -> KITTI camera ``rotation_y`` (same involution as above)."""
    return normalize_angle(-(np.asarray(yaw, dtype=np.float64) + HALF_PI))


def encode_heading_sincos(yaw):
    """Yaw -> (..., 2) array of (sin, cos). Full-2*pi representation."""
    yaw = np.asarray(yaw, dtype=np.float64)
    return np.stack([np.sin(yaw), np.cos(yaw)], axis=-1)


def decode_heading_sincos(sincos):
    """(..., 2) (sin, cos) -> yaw via atan2. Inverse of :func:`encode_heading_sincos`
    up to angle wrapping."""
    sincos = np.asarray(sincos, dtype=np.float64)
    return np.arctan2(sincos[..., 0], sincos[..., 1])


def heading_bin_residual(yaw, num_bins: int = 12):
    """Split yaw into a coarse bin index + residual (the encoding many anchor-free
    heads use). Returns ``(bin_index, residual)``."""
    yaw = normalize_angle(yaw)
    bin_size = 2.0 * np.pi / num_bins
    shifted = (yaw + np.pi) % (2.0 * np.pi)        # [0, 2pi)
    bin_index = np.floor(shifted / bin_size).astype(np.int64)
    bin_center = bin_index * bin_size + bin_size / 2.0 - np.pi
    residual = normalize_angle(yaw - bin_center)
    return bin_index, residual
