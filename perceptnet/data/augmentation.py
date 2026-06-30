"""LiDAR / box data augmentation (operates in the LiDAR frame).

All transforms act jointly on a point cloud ``(N, >=3)`` and its boxes
``(M, 7)`` in the canonical ``[x, y, z, dx, dy, dz, heading]`` format, keeping the
two consistent. The functional cores are deterministic given their parameters (and
unit-tested: flip-twice and rotate-by-+/-a are identities); the ``*_random``
wrappers sample parameters from a NumPy Generator.

Camera-side augmentation (color jitter, crop) is applied by the camera branch's
own training pipeline (Ultralytics) and is not duplicated here.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def flip_along_x(points: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror the scene across the x-axis (negate y and heading)."""
    points = np.asarray(points, dtype=np.float64).copy()
    boxes = np.asarray(boxes, dtype=np.float64).copy()
    points[:, 1] = -points[:, 1]
    if len(boxes):
        boxes[:, 1] = -boxes[:, 1]
        boxes[:, 6] = -boxes[:, 6]
    return points, boxes


def rotate_about_z(points: np.ndarray, boxes: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate the scene about the vertical axis by ``angle`` radians."""
    points = np.asarray(points, dtype=np.float64).copy()
    boxes = np.asarray(boxes, dtype=np.float64).copy()
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    points[:, :2] = points[:, :2] @ rot.T
    if len(boxes):
        boxes[:, :2] = boxes[:, :2] @ rot.T
        boxes[:, 6] += angle
    return points, boxes


def scale_scene(points: np.ndarray, boxes: np.ndarray, factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """Uniformly scale point positions and box centers + dimensions."""
    points = np.asarray(points, dtype=np.float64).copy()
    boxes = np.asarray(boxes, dtype=np.float64).copy()
    points[:, :3] *= factor
    if len(boxes):
        boxes[:, :6] *= factor          # centers (xyz) and dims (dx,dy,dz)
    return points, boxes


# --------------------------------------------------------------------------- #
# Random wrappers + pipeline
# --------------------------------------------------------------------------- #
def flip_along_x_random(points, boxes, rng, prob=0.5):
    if rng.random() < prob:
        return flip_along_x(points, boxes)
    return points, boxes


def rotate_about_z_random(points, boxes, rng, rot_range=(-np.pi / 4, np.pi / 4)):
    angle = rng.uniform(rot_range[0], rot_range[1])
    return rotate_about_z(points, boxes, angle)


def scale_scene_random(points, boxes, rng, scale_range=(0.95, 1.05)):
    factor = rng.uniform(scale_range[0], scale_range[1])
    return scale_scene(points, boxes, factor)


class AugmentationPipeline:
    """Compose the standard PointPillars LiDAR augmentations.

    Args:
        flip_prob: probability of the left-right flip.
        rot_range: global rotation range in radians (spec: +/- pi/4).
        scale_range: global scaling range.
        seed: optional seed for reproducibility.
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        rot_range: Tuple[float, float] = (-np.pi / 4, np.pi / 4),
        scale_range: Tuple[float, float] = (0.95, 1.05),
        seed: Optional[int] = None,
    ):
        self.flip_prob = flip_prob
        self.rot_range = rot_range
        self.scale_range = scale_range
        self.rng = np.random.default_rng(seed)

    def __call__(self, points: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        points, boxes = flip_along_x_random(points, boxes, self.rng, self.flip_prob)
        points, boxes = rotate_about_z_random(points, boxes, self.rng, self.rot_range)
        points, boxes = scale_scene_random(points, boxes, self.rng, self.scale_range)
        return points, boxes
