"""PointPillars losses: focal (classification) + smooth-L1 (box) + direction.

These match the spec's "Focal loss (cls) + SmoothL1 (reg) + heading" recipe and are
numerically unit-tested. They consume *assigned* targets from
:mod:`perceptnet.models.target_assigner`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Sigmoid focal loss (RetinaNet). ``targets`` is a 0/1 float tensor of ``logits`` shape."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0 / 9.0,
                   reduction: str = "mean") -> torch.Tensor:
    """Smooth-L1 (Huber-like) regression loss."""
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class PointPillarsLoss(torch.nn.Module):
    """Weighted sum of classification, box-regression, and direction losses.

    Expects flattened, anchor-aligned predictions and targets:
        cls_preds   (A, num_classes)     box_preds (A, 7)     dir_preds (A, 2)
        cls_targets (A,) int (-1 ignore, 0..nc-1; nc == background omitted via mask)
        box_targets (A, 7)               dir_targets (A,) int
    """

    def __init__(self, num_classes: int = 3, cls_weight: float = 1.0,
                 box_weight: float = 2.0, dir_weight: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.dir_weight = dir_weight

    def forward(self, cls_preds, box_preds, dir_preds, cls_targets, box_targets, dir_targets) -> dict:
        positives = cls_targets >= 0
        valid = cls_targets > -2          # -2 reserved for "ignore"
        num_pos = positives.sum().clamp(min=1).float()

        # classification: one-hot over valid anchors (background = all-zero row)
        cls_onehot = torch.zeros_like(cls_preds)
        pos_idx = positives.nonzero(as_tuple=True)[0]
        cls_onehot[pos_idx, cls_targets[pos_idx]] = 1.0
        cls_loss = sigmoid_focal_loss(cls_preds[valid], cls_onehot[valid], reduction="sum") / num_pos

        # box + direction only on positives
        if positives.any():
            box_loss = smooth_l1_loss(box_preds[positives], box_targets[positives], reduction="sum") / num_pos
            dir_loss = F.cross_entropy(dir_preds[positives], dir_targets[positives])
        else:
            box_loss = box_preds.sum() * 0.0
            dir_loss = dir_preds.sum() * 0.0

        total = self.cls_weight * cls_loss + self.box_weight * box_loss + self.dir_weight * dir_loss
        return {"loss": total, "cls_loss": cls_loss, "box_loss": box_loss, "dir_loss": dir_loss}
