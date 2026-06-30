"""2D image visualization: draw 2D detections, projected 3D boxes, and LiDAR overlays.

Pure OpenCV/NumPy (no Open3D) so it runs anywhere and produces real PNGs for the
README. Accepts images as a torch CHW float tensor in [0, 1] or a NumPy HWC array.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from perceptnet.data.calibration import Calibration
from perceptnet.geometry.boxes import boxes_to_corners_3d

# 12 edges of a box given the corner ordering in geometry.boxes
# (bottom 0-1-2-3, top 4-5-6-7, with top i above bottom i).
_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),       # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),       # top
    (0, 4), (1, 5), (2, 6), (3, 7),       # verticals
]

_CLASS_COLORS = [(0, 255, 0), (0, 165, 255), (255, 0, 0)]   # BGR: Car, Ped, Cyclist


def _to_bgr_uint8(image) -> np.ndarray:
    """Normalize an image to an HWC uint8 BGR array (a copy, safe to draw on)."""
    arr = image
    try:
        import torch

        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):   # CHW -> HWC
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            # incoming tensors are RGB (KITTIDataset); convert to BGR for cv2
            import cv2

            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return np.ascontiguousarray(arr)
    except ImportError:
        pass

    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if arr.ndim == 2:
        import cv2

        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(arr)


def draw_2d_boxes(image, boxes_xyxy, labels=None, scores=None, class_names=None) -> np.ndarray:
    """Draw axis-aligned 2D detection boxes."""
    import cv2

    img = _to_bgr_uint8(image)
    boxes_xyxy = np.asarray(boxes_xyxy).reshape(-1, 4)
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy.astype(int)):
        label = int(labels[i]) if labels is not None else 0
        color = _CLASS_COLORS[label % len(_CLASS_COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if scores is not None:
            name = class_names[label] if class_names else str(label)
            cv2.putText(img, f"{name} {scores[i]:.2f}", (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def draw_projected_boxes_3d(image, boxes_3d, calib: Calibration, labels=None) -> np.ndarray:
    """Project 3D LiDAR boxes onto the image and draw their wireframes."""
    import cv2

    img = _to_bgr_uint8(image)
    boxes_3d = np.atleast_2d(np.asarray(boxes_3d, dtype=np.float64))
    if boxes_3d.size == 0:
        return img

    corners = boxes_to_corners_3d(boxes_3d)               # (N, 8, 3) LiDAR
    for i in range(len(boxes_3d)):
        uv, depth = calib.velo_to_image(corners[i])
        if np.any(depth <= 1e-3):                          # box crosses the image plane
            continue
        color = _CLASS_COLORS[int(labels[i]) % len(_CLASS_COLORS)] if labels is not None else (0, 255, 255)
        pts = uv.astype(int)
        for a, b in _BOX_EDGES:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 1, cv2.LINE_AA)
    return img


def draw_lidar_overlay(image, points: np.ndarray, calib: Calibration, max_depth: float = 70.0) -> np.ndarray:
    """Overlay LiDAR points on the image, colored by depth."""
    import cv2

    img = _to_bgr_uint8(image)
    h, w = img.shape[:2]
    uv, depth, _ = calib.project_lidar_to_image(points, image_shape=(h, w))
    for (u, v), d in zip(uv.astype(int), depth):
        c = int(255 * (1.0 - min(d, max_depth) / max_depth))
        cv2.circle(img, (u, v), 1, (c, 255 - c, 0), -1)
    return img


def save_image(image, path) -> None:
    import cv2

    cv2.imwrite(str(path), _to_bgr_uint8(image))


def visualize_frame(
    image,
    calib: Calibration,
    boxes_3d: Optional[np.ndarray] = None,
    boxes_2d: Optional[np.ndarray] = None,
    labels: Optional[Sequence[int]] = None,
    points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compose a debug frame: optional LiDAR overlay + 3D wireframes + 2D boxes."""
    img = _to_bgr_uint8(image)
    if points is not None:
        img = draw_lidar_overlay(img, points, calib)
    if boxes_3d is not None and len(boxes_3d):
        img = draw_projected_boxes_3d(img, boxes_3d, calib, labels)
    if boxes_2d is not None and len(boxes_2d):
        img = draw_2d_boxes(img, boxes_2d, labels)
    return img
