"""Detection-to-track association via the Hungarian algorithm.

Thin wrapper over :func:`scipy.optimize.linear_sum_assignment` that turns a
similarity (IoU) matrix into matches plus the leftover detections and tracks,
applying a minimum-IoU gate so low-overlap "matches" are rejected.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def associate(
    iou_matrix: np.ndarray, iou_threshold: float = 0.1
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match detections (rows) to tracks (cols) to maximize total IoU.

    Args:
        iou_matrix: ``(num_det, num_trk)`` IoU/similarity matrix.
        iou_threshold: matches with IoU below this are rejected.

    Returns:
        ``(matches, unmatched_detections, unmatched_tracks)`` where ``matches`` is a
        list of ``(det_idx, trk_idx)`` pairs.
    """
    iou_matrix = np.asarray(iou_matrix, dtype=np.float64)
    num_det, num_trk = iou_matrix.shape

    if num_det == 0 or num_trk == 0:
        return [], list(range(num_det)), list(range(num_trk))

    # linear_sum_assignment minimizes cost -> negate IoU to maximize overlap.
    det_idx, trk_idx = linear_sum_assignment(-iou_matrix)

    matches: List[Tuple[int, int]] = []
    matched_det, matched_trk = set(), set()
    for d, t in zip(det_idx, trk_idx):
        if iou_matrix[d, t] >= iou_threshold:
            matches.append((int(d), int(t)))
            matched_det.add(int(d))
            matched_trk.add(int(t))

    unmatched_det = [d for d in range(num_det) if d not in matched_det]
    unmatched_trk = [t for t in range(num_trk) if t not in matched_trk]
    return matches, unmatched_det, unmatched_trk
