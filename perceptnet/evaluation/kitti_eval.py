"""KITTI-style 3D / BEV detection mAP.

Implements the standard detection-AP machinery (sort detections by score, greedily
match each to an unmatched ground-truth box above an IoU threshold, integrate the
precision-recall curve). Pure NumPy; the IoU primitive is shared with the tracker
and fusion modules (:mod:`perceptnet.geometry.iou`).

KITTI's official thresholds: Car @ IoU 0.7, Pedestrian/Cyclist @ IoU 0.5. The
easy/moderate/hard split is a filter on ground-truth box height / occlusion /
truncation; :func:`kitti_difficulty_mask` builds that filter when the metadata is
available.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from perceptnet.geometry.iou import iou_3d, iou_bev

# Default KITTI IoU thresholds by class name.
DEFAULT_IOU_THRESHOLDS = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}

# KITTI difficulty definitions (min 2D box height in px, max occlusion, max truncation).
KITTI_DIFFICULTY = {
    "easy": dict(min_height=40, max_occlusion=0, max_truncation=0.15),
    "moderate": dict(min_height=25, max_occlusion=1, max_truncation=0.30),
    "hard": dict(min_height=25, max_occlusion=2, max_truncation=0.50),
}


def average_precision(precision: np.ndarray, recall: np.ndarray, method: str = "area") -> float:
    """Average precision from a precision-recall curve.

    Args:
        precision, recall: equal-length arrays, ordered by decreasing score.
        method: ``"area"`` (all-point interpolation, VOC2010+/COCO) or ``"11point"``
            (VOC2007).
    """
    precision = np.asarray(precision, dtype=np.float64)
    recall = np.asarray(recall, dtype=np.float64)
    if precision.size == 0:
        return 0.0

    if method == "11point":
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recall >= t
            ap += (precision[mask].max() if np.any(mask) else 0.0) / 11.0
        return float(ap)

    # All-point interpolation: area under the monotonic upper envelope of the curve.
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _ap_for_class(
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    iou_fn,
    iou_threshold: float,
    ap_method: str = "area",
) -> Dict[str, float]:
    """AP for a single class across frames.

    Each ``*_per_frame`` list is indexed by frame; entry shapes are ``(Ni, 7)`` /
    ``(Ni,)`` / ``(Mi, 7)``.
    """
    # Flatten detections, remembering their frame.
    flat = []  # (score, frame_idx, det_idx_in_frame)
    for f, scores in enumerate(pred_scores):
        for j, s in enumerate(np.asarray(scores).reshape(-1)):
            flat.append((float(s), f, j))
    flat.sort(key=lambda r: r[0], reverse=True)

    total_gt = int(sum(len(np.asarray(g).reshape(-1, 7)) for g in gt_boxes))
    if total_gt == 0:
        return {"ap": 0.0, "num_gt": 0, "num_pred": len(flat)}

    matched = {f: np.zeros(len(np.asarray(gt_boxes[f]).reshape(-1, 7)), dtype=bool) for f in range(len(gt_boxes))}
    tp = np.zeros(len(flat))
    fp = np.zeros(len(flat))

    for rank, (_, f, j) in enumerate(flat):
        gts = np.asarray(gt_boxes[f]).reshape(-1, 7)
        if len(gts) == 0:
            fp[rank] = 1
            continue
        det = np.asarray(pred_boxes[f]).reshape(-1, 7)[j][None, :]
        ious = iou_fn(det, gts)[0]
        best = int(np.argmax(ious))
        if ious[best] >= iou_threshold and not matched[f][best]:
            tp[rank] = 1
            matched[f][best] = True
        else:
            fp[rank] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    ap = average_precision(precision, recall, method=ap_method)
    return {"ap": ap, "num_gt": total_gt, "num_pred": len(flat)}


def eval_kitti_3d(
    predictions: Sequence[dict],
    ground_truths: Sequence[dict],
    classes: Sequence[str],
    iou_thresholds: Optional[Dict[str, float]] = None,
    metric: str = "3d",
    ap_method: str = "area",
) -> Dict[str, float]:
    """Per-class detection AP over a dataset.

    Args:
        predictions: per-frame dicts with keys ``boxes`` ``(N,7)``, ``scores`` ``(N,)``,
            ``labels`` ``(N,)`` (label values must be in ``classes``).
        ground_truths: per-frame dicts with keys ``boxes`` ``(M,7)``, ``labels`` ``(M,)``.
        classes: class names to evaluate, e.g. ``["Car", "Pedestrian", "Cyclist"]``.
        iou_thresholds: per-class IoU threshold (defaults to KITTI's).
        metric: ``"3d"`` or ``"bev"``.

    Returns:
        ``{class_name: AP}`` plus ``"mAP"`` (mean over classes).
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have the same number of frames")
    iou_thresholds = iou_thresholds or DEFAULT_IOU_THRESHOLDS
    iou_fn = iou_3d if metric == "3d" else iou_bev

    results: Dict[str, float] = {}
    for cls in classes:
        thr = iou_thresholds.get(cls, 0.5)
        pred_boxes, pred_scores, gt_boxes = [], [], []
        for pred, gt in zip(predictions, ground_truths):
            pm = np.asarray(pred.get("labels", [])).reshape(-1) == cls
            gm = np.asarray(gt.get("labels", [])).reshape(-1) == cls
            pb = np.asarray(pred.get("boxes", np.zeros((0, 7)))).reshape(-1, 7)
            ps = np.asarray(pred.get("scores", np.zeros(0))).reshape(-1)
            gb = np.asarray(gt.get("boxes", np.zeros((0, 7)))).reshape(-1, 7)
            pred_boxes.append(pb[pm])
            pred_scores.append(ps[pm])
            gt_boxes.append(gb[gm])
        results[cls] = _ap_for_class(pred_boxes, pred_scores, gt_boxes, iou_fn, thr, ap_method)["ap"]

    results["mAP"] = float(np.mean([results[c] for c in classes])) if classes else 0.0
    return results


def kitti_difficulty_mask(
    heights: np.ndarray, occlusions: np.ndarray, truncations: np.ndarray, difficulty: str
) -> np.ndarray:
    """Boolean mask selecting ground-truth objects of a given KITTI difficulty."""
    cfg = KITTI_DIFFICULTY[difficulty]
    heights = np.asarray(heights)
    occlusions = np.asarray(occlusions)
    truncations = np.asarray(truncations)
    return (
        (heights >= cfg["min_height"])
        & (occlusions <= cfg["max_occlusion"])
        & (truncations <= cfg["max_truncation"])
    )
