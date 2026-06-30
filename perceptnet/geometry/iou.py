"""IoU primitives: axis-aligned 2D (image boxes) and rotated BEV / 3D (LiDAR boxes).

Rotated-rectangle overlap is computed exactly with the Sutherland-Hodgman convex
polygon-clipping algorithm (pure NumPy, no Shapely dependency). Rectangles are
convex, so clipping the subject polygon against each edge of the (convex) clip
polygon yields the exact intersection polygon, whose area follows from the shoelace
formula.
"""

from __future__ import annotations

import numpy as np

from perceptnet.geometry.boxes import (
    box_bev_area,
    box_volume,
    boxes_to_bev_corners,
)


# --------------------------------------------------------------------------- #
# Axis-aligned 2D IoU (image-plane boxes: x1, y1, x2, y2)
# --------------------------------------------------------------------------- #
def iou_2d(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of axis-aligned boxes.

    Args:
        boxes_a: ``(N, 4)`` x1,y1,x2,y2.
        boxes_b: ``(M, 4)``.

    Returns:
        ``(N, M)`` IoU matrix.
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])      # (N, M, 2)
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


# --------------------------------------------------------------------------- #
# Convex polygon intersection (Sutherland-Hodgman)
# --------------------------------------------------------------------------- #
def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    signed = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return poly if signed >= 0 else poly[::-1]


def _line_intersect(s, e, a, b):
    """Intersection of segment s->e with the (infinite) line through a->b."""
    d1 = e - s
    d2 = b - a
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return e
    t = ((a[0] - s[0]) * d2[1] - (a[1] - s[1]) * d2[0]) / denom
    return s + t * d1


def convex_intersection_area(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Area of the intersection of two convex polygons given as ``(k, 2)`` corners."""
    clip = _ensure_ccw(np.asarray(poly2, dtype=np.float64))
    output = [pt for pt in np.asarray(poly1, dtype=np.float64)]

    n = len(clip)
    for i in range(n):
        a = clip[i]
        b = clip[(i + 1) % n]
        edge = b - a
        normal = np.array([-edge[1], edge[0]])   # inward normal for CCW clip
        if not output:
            break
        prev = output[-1]
        clipped = []
        prev_inside = np.dot(normal, prev - a) >= 0
        for cur in output:
            cur_inside = np.dot(normal, cur - a) >= 0
            if cur_inside:
                if not prev_inside:
                    clipped.append(_line_intersect(prev, cur, a, b))
                clipped.append(cur)
            elif prev_inside:
                clipped.append(_line_intersect(prev, cur, a, b))
            prev, prev_inside = cur, cur_inside
        output = clipped

    if len(output) < 3:
        return 0.0
    return _polygon_area(np.asarray(output))


# --------------------------------------------------------------------------- #
# Rotated BEV / 3D IoU (LiDAR boxes: x,y,z,dx,dy,dz,heading)
# --------------------------------------------------------------------------- #
def _bev_intersection_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    a_corners = boxes_to_bev_corners(boxes_a).reshape(-1, 4, 2)
    b_corners = boxes_to_bev_corners(boxes_b).reshape(-1, 4, 2)
    n, m = len(a_corners), len(b_corners)
    inter = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            inter[i, j] = convex_intersection_area(a_corners[i], b_corners[j])
    return inter


def iou_bev(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise rotated BEV IoU. Boxes ``(N,7)`` / ``(M,7)`` -> ``(N, M)``."""
    a = np.atleast_2d(np.asarray(boxes_a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(boxes_b, dtype=np.float64))
    inter = _bev_intersection_matrix(a, b)
    area_a = box_bev_area(a)[:, None]
    area_b = box_bev_area(b)[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def iou_3d(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise 3D IoU. Boxes ``(N,7)`` / ``(M,7)`` -> ``(N, M)``.

    Combines the rotated BEV intersection area with the vertical (z) overlap.
    """
    a = np.atleast_2d(np.asarray(boxes_a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(boxes_b, dtype=np.float64))
    inter_area = _bev_intersection_matrix(a, b)

    a_zmin = (a[:, 2] - a[:, 5] / 2.0)[:, None]
    a_zmax = (a[:, 2] + a[:, 5] / 2.0)[:, None]
    b_zmin = (b[:, 2] - b[:, 5] / 2.0)[None, :]
    b_zmax = (b[:, 2] + b[:, 5] / 2.0)[None, :]
    overlap_h = np.clip(np.minimum(a_zmax, b_zmax) - np.maximum(a_zmin, b_zmin), 0, None)

    inter_vol = inter_area * overlap_h
    vol_a = box_volume(a)[:, None]
    vol_b = box_volume(b)[None, :]
    union = vol_a + vol_b - inter_vol
    return np.where(union > 0, inter_vol / union, 0.0)
