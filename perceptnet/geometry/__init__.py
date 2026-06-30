"""Shared geometry primitives (pure NumPy, no heavy deps).

Used across data/calibration, models/fusion, tracking, and evaluation:
boxes (corner generation, format conversions), iou (2D/BEV/3D), and heading
(KITTI camera ``rotation_y`` <-> LiDAR yaw, sin/cos encode/decode).
"""

from perceptnet.geometry import boxes, heading, iou

__all__ = ["boxes", "heading", "iou"]
