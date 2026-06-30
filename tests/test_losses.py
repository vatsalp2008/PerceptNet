"""Tests for losses, box residual encode/decode roundtrip, and anchor assignment."""

import torch

from perceptnet.models.lidar_branch import decode_box_predictions
from perceptnet.models.losses import PointPillarsLoss, sigmoid_focal_loss, smooth_l1_loss
from perceptnet.models.target_assigner import AnchorTargetAssigner, encode_box_residuals


# --------------------------------------------------------------------------- #
# Focal / smooth-L1
# --------------------------------------------------------------------------- #
def test_focal_loss_low_for_confident_correct():
    logits = torch.tensor([[8.0]])      # very confident positive
    target = torch.tensor([[1.0]])
    confident = sigmoid_focal_loss(logits, target)
    wrong = sigmoid_focal_loss(torch.tensor([[-8.0]]), target)
    assert confident < wrong
    assert confident < 1e-3


def test_smooth_l1_zero_for_identical():
    x = torch.randn(10, 7)
    assert smooth_l1_loss(x, x).item() == 0.0


def test_smooth_l1_linear_region():
    # large error -> ~ |diff| - 0.5*beta
    loss = smooth_l1_loss(torch.tensor([10.0]), torch.tensor([0.0]), beta=1.0, reduction="sum")
    assert torch.isclose(loss, torch.tensor(9.5))


# --------------------------------------------------------------------------- #
# Box residual roundtrip (encode is the inverse of decode)
# --------------------------------------------------------------------------- #
def test_encode_decode_roundtrip():
    torch.manual_seed(0)
    anchors = torch.tensor([[0.0, 0, 0, 3.9, 1.6, 1.56, 0.0]]).repeat(6, 1)
    boxes = anchors.clone()
    boxes[:, :3] += torch.randn(6, 3) * 0.5         # perturb centers
    boxes[:, 3:6] *= torch.empty(6, 3).uniform_(0.8, 1.2)
    boxes[:, 6] += torch.randn(6) * 0.2
    residuals = encode_box_residuals(boxes, anchors)
    recovered = decode_box_predictions(residuals, anchors)
    assert torch.allclose(recovered, boxes, atol=1e-5)


# --------------------------------------------------------------------------- #
# Anchor assignment
# --------------------------------------------------------------------------- #
def test_assigner_positive_and_background():
    anchors = torch.tensor([
        [0.0, 0, 0, 3.9, 1.6, 1.56, 0.0],     # overlaps the GT
        [50.0, 0, 0, 3.9, 1.6, 1.56, 0.0],    # far away
    ])
    gt = torch.tensor([[0.0, 0, 0, 3.9, 1.6, 1.56, 0.0]])
    labels = torch.tensor([0])
    out = AnchorTargetAssigner(pos_iou=0.6, neg_iou=0.45).assign(anchors, gt, labels)
    assert out["cls_targets"][0].item() == 0          # matched to class 0
    assert out["cls_targets"][1].item() == -1         # background
    assert out["reg_mask"][0].item() and not out["reg_mask"][1].item()


def test_assigner_no_gt_is_all_background():
    anchors = torch.zeros((4, 7))
    anchors[:, 3:6] = 1.0
    out = AnchorTargetAssigner().assign(anchors, torch.zeros((0, 7)), torch.zeros((0,), dtype=torch.long))
    assert (out["cls_targets"] == -1).all()


# --------------------------------------------------------------------------- #
# Combined loss
# --------------------------------------------------------------------------- #
def test_pointpillars_loss_runs_and_is_finite():
    A = 20
    # predictions come from the model -> they require grad
    cls_preds = torch.randn(A, 3, requires_grad=True)
    box_preds = torch.randn(A, 7, requires_grad=True)
    dir_preds = torch.randn(A, 2, requires_grad=True)
    cls_targets = torch.full((A,), -1, dtype=torch.long)
    cls_targets[:3] = torch.tensor([0, 1, 2])         # 3 positives
    box_targets = torch.randn(A, 7)
    dir_targets = torch.zeros(A, dtype=torch.long)
    out = PointPillarsLoss(num_classes=3)(cls_preds, box_preds, dir_preds, cls_targets, box_targets, dir_targets)
    assert torch.isfinite(out["loss"])
    assert out["loss"].requires_grad
