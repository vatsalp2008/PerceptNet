"""CLEAR-MOT tracking metrics: MOTA, MOTP, ID switches, fragmentations.

    MOTA = 1 - (FN + FP + IDSW) / num_gt
    MOTP = mean IoU of matched (gt, pred) pairs

Matching each frame prefers to *preserve* the previous frame's gt->pred assignment
when it is still valid (IoU >= threshold), then applies the Hungarian algorithm to
the remainder. An ID switch is counted when a ground-truth object that was tracked
before is now matched to a different predicted track id.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from perceptnet.geometry.iou import iou_bev
from perceptnet.tracking.hungarian import associate


def eval_mot(
    frames_gt: Sequence[dict],
    frames_pred: Sequence[dict],
    iou_threshold: float = 0.25,
    metric: str = "bev",
) -> Dict[str, float]:
    """Compute CLEAR-MOT metrics over a sequence.

    Args:
        frames_gt: per-frame dicts with ``ids`` ``(M,)`` and ``boxes`` ``(M,7)``.
        frames_pred: per-frame dicts with ``ids`` ``(N,)`` and ``boxes`` ``(N,7)``.
        iou_threshold: minimum IoU for a valid match.
        metric: currently ``"bev"`` (rotated BEV IoU).

    Returns:
        dict with MOTA, MOTP, num_switches, num_fragmentations, fp, fn, tp, num_gt.
    """
    if len(frames_gt) != len(frames_pred):
        raise ValueError("gt and pred must have the same number of frames")
    iou_fn = iou_bev  # 'metric' reserved for future 3d option

    total_gt = 0
    fp = fn = tp = idsw = frag = 0
    motp_sum = 0.0

    last_match: Dict[int, int] = {}        # gt_id -> pred_id (most recent)
    was_tracked: Dict[int, bool] = {}      # gt_id -> tracked in previous frame

    for gt, pred in zip(frames_gt, frames_pred):
        gt_ids = np.asarray(gt.get("ids", [])).reshape(-1)
        pred_ids = np.asarray(pred.get("ids", [])).reshape(-1)
        gt_boxes = np.asarray(gt.get("boxes", np.zeros((0, 7)))).reshape(-1, 7)
        pred_boxes = np.asarray(pred.get("boxes", np.zeros((0, 7)))).reshape(-1, 7)
        total_gt += len(gt_ids)

        matches = {}  # gt_idx -> pred_idx
        if len(gt_ids) and len(pred_ids):
            iou = iou_fn(gt_boxes, pred_boxes)

            # 1. preserve previous assignments still above threshold
            used_pred = set()
            remaining_gt = []
            for gi, gid in enumerate(gt_ids):
                pid = last_match.get(int(gid))
                if pid is not None and pid in pred_ids:
                    pj = int(np.where(pred_ids == pid)[0][0])
                    if iou[gi, pj] >= iou_threshold and pj not in used_pred:
                        matches[gi] = pj
                        used_pred.add(pj)
                        continue
                remaining_gt.append(gi)

            # 2. Hungarian on the leftovers
            free_pred = [pj for pj in range(len(pred_ids)) if pj not in used_pred]
            if remaining_gt and free_pred:
                sub = iou[np.ix_(remaining_gt, free_pred)]
                m, _, _ = associate(sub, iou_threshold)
                for gi_sub, pj_sub in m:
                    matches[remaining_gt[gi_sub]] = free_pred[pj_sub]

        # tally
        matched_gt = set(matches.keys())
        matched_pred = set(matches.values())
        tp += len(matches)
        fn += len(gt_ids) - len(matched_gt)
        fp += len(pred_ids) - len(matched_pred)

        current_match: Dict[int, int] = {}
        for gi, pj in matches.items():
            gid, pid = int(gt_ids[gi]), int(pred_ids[pj])
            motp_sum += iou[gi, pj]
            if gid in last_match and last_match[gid] != pid:
                idsw += 1
            current_match[gid] = pid

        # fragmentation: a gt that was tracked, then lost, regardless of id
        for gid in gt_ids.astype(int):
            tracked_now = gid in current_match
            if was_tracked.get(gid, False) and not tracked_now:
                frag += 1
            was_tracked[gid] = tracked_now

        last_match.update(current_match)

    mota = 1.0 - (fn + fp + idsw) / total_gt if total_gt else 0.0
    motp = motp_sum / tp if tp else 0.0
    return {
        "MOTA": float(mota),
        "MOTP": float(motp),
        "num_switches": int(idsw),
        "num_fragmentations": int(frag),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "num_gt": int(total_gt),
    }
