"""3D bounding-box corner generation and format helpers.

Canonical box format used throughout PerceptNet (LiDAR frame, OpenPCDet convention):

    [x, y, z, dx, dy, dz, heading]

  - (x, y, z): geometric center of the box
  - dx: extent along the heading axis ("length")
  - dy: extent perpendicular in BEV ("width")
  - dz: vertical extent ("height")
  - heading: yaw about the +z axis, counter-clockwise, 0 == pointing along +x

LiDAR frame is x-forward, y-left, z-up. This is distinct from the KITTI *camera*
label frame (x-right, y-down, z-forward, ``rotation_y``); conversions between the two
live in :mod:`perceptnet.data.calibration` and :mod:`perceptnet.geometry.heading`.
"""

from __future__ import annotations

import numpy as np

# Unit-cube corner template, ordered so the first four corners are the bottom face
# (z = -dz/2) traversed clockwise in BEV and the last four are the top face:
#       4 -------- 5
#      /|         /|
#     7 -------- 6 |
#     | |        | |
#     | 0 -------| 1
#     |/         |/
#     3 -------- 2
_CORNER_TEMPLATE = (
    np.array(
        [
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],   # bottom face
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],        # top face
        ],
        dtype=np.float64,
    )
    / 2.0
)


def _yaw_rotation_matrices(yaw: np.ndarray) -> np.ndarray:
    """(N,) yaw angles -> (N, 3, 3) rotation matrices about +z."""
    cos, sin = np.cos(yaw), np.sin(yaw)
    zeros, ones = np.zeros_like(cos), np.ones_like(cos)
    return np.stack(
        [
            np.stack([cos, -sin, zeros], axis=-1),
            np.stack([sin, cos, zeros], axis=-1),
            np.stack([zeros, zeros, ones], axis=-1),
        ],
        axis=-2,
    )


def boxes_to_corners_3d(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes ``[x,y,z,dx,dy,dz,heading]`` to corner coordinates.

    Args:
        boxes: ``(7,)`` or ``(N, 7)`` array.

    Returns:
        ``(8, 3)`` if a single box was given, else ``(N, 8, 3)``.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    single = boxes.ndim == 1
    if single:
        boxes = boxes[None, :]
    if boxes.shape[-1] != 7:
        raise ValueError(f"boxes must have last dim 7, got shape {boxes.shape}")

    dims = boxes[:, 3:6]                                  # (N, 3)
    corners = _CORNER_TEMPLATE[None] * dims[:, None, :]    # (N, 8, 3)
    rot = _yaw_rotation_matrices(boxes[:, 6])              # (N, 3, 3)
    corners = np.einsum("nij,nkj->nki", rot, corners)      # rotate each corner
    corners = corners + boxes[:, None, :3]                 # translate to center
    return corners[0] if single else corners


def boxes_to_bev_corners(boxes: np.ndarray) -> np.ndarray:
    """Bottom-face (BEV) corners in the xy-plane.

    Returns ``(4, 2)`` for a single box, else ``(N, 4, 2)``.
    """
    corners = boxes_to_corners_3d(boxes)
    if corners.ndim == 2:          # single box -> (8, 3)
        return corners[:4, :2]
    return corners[:, :4, :2]


def box_volume(boxes: np.ndarray) -> np.ndarray:
    """Volume dx*dy*dz. Scalar for a single box, else ``(N,)``."""
    boxes = np.asarray(boxes, dtype=np.float64)
    vol = np.prod(boxes[..., 3:6], axis=-1)
    return vol


def box_bev_area(boxes: np.ndarray) -> np.ndarray:
    """BEV footprint area dx*dy. Scalar for a single box, else ``(N,)``."""
    boxes = np.asarray(boxes, dtype=np.float64)
    return boxes[..., 3] * boxes[..., 4]
