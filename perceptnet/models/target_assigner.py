"""Anchor target assignment for PointPillars training.

Matches anchors to ground-truth boxes by BEV IoU (positive above ``pos_iou``,
negative below ``neg_iou``, the band between is ignored) and encodes the matched
boxes into SECOND-style residuals. ``encode_box_residuals`` is the exact inverse of
:func:`perceptnet.models.lidar_branch.decode_box_predictions` (round-trip tested).
"""

from __future__ import annotations

import numpy as np
import torch

from perceptnet.geometry.iou import iou_bev


def encode_box_residuals(boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Encode absolute boxes as residuals relative to anchors (inverse of decode)."""
    xa, ya, za, dxa, dya, dza, ra = anchors.unbind(-1)
    diag = torch.sqrt(dxa ** 2 + dya ** 2)
    xg, yg, zg, dxg, dyg, dzg, rg = boxes.unbind(-1)
    return torch.stack(
        [
            (xg - xa) / diag,
            (yg - ya) / diag,
            (zg - za) / dza,
            torch.log(dxg / dxa),
            torch.log(dyg / dya),
            torch.log(dzg / dza),
            rg - ra,
        ],
        dim=-1,
    )


class AnchorTargetAssigner:
    """Assign GT boxes to anchors and produce classification / box / direction targets."""

    def __init__(self, pos_iou: float = 0.6, neg_iou: float = 0.45):
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou

    def assign(self, anchors: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor) -> dict:
        """Args:
            anchors: ``(A, 7)``.
            gt_boxes: ``(M, 7)`` (empty allowed).
            gt_labels: ``(M,)`` int class ids.

        Returns dict of ``cls_targets (A,)`` (-1 background, >=0 class; -2 ignore),
        ``box_targets (A, 7)``, ``dir_targets (A,)``, ``reg_mask (A,)`` bool.
        """
        A = anchors.shape[0]
        cls_targets = torch.full((A,), -1, dtype=torch.long)       # background
        box_targets = torch.zeros((A, 7), dtype=torch.float32)
        dir_targets = torch.zeros((A,), dtype=torch.long)

        if len(gt_boxes) == 0:
            return {"cls_targets": cls_targets, "box_targets": box_targets,
                    "dir_targets": dir_targets, "reg_mask": cls_targets >= 0}

        iou = torch.as_tensor(iou_bev(anchors.numpy(), gt_boxes.numpy()), dtype=torch.float32)  # (A, M)
        max_iou, gt_idx = iou.max(dim=1)

        positive = max_iou >= self.pos_iou
        ignore = (max_iou >= self.neg_iou) & (~positive)
        cls_targets[ignore] = -2

        # ensure each GT has at least one positive anchor (its best-overlapping anchor)
        best_anchor = iou.argmax(dim=0)
        positive[best_anchor] = True
        gt_idx[best_anchor] = torch.arange(len(gt_boxes))

        pos_idx = positive.nonzero(as_tuple=True)[0]
        matched_gt = gt_idx[pos_idx]
        cls_targets[pos_idx] = gt_labels[matched_gt]
        box_targets[pos_idx] = encode_box_residuals(gt_boxes[matched_gt], anchors[pos_idx])
        # direction class: 1 if heading points to the negative half, else 0
        dir_targets[pos_idx] = (gt_boxes[matched_gt][:, 6] < 0).long()

        return {"cls_targets": cls_targets, "box_targets": box_targets,
                "dir_targets": dir_targets, "reg_mask": positive}
