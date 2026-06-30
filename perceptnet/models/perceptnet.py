"""Unified PerceptNet model — ties the LiDAR branch, camera branch, and ROI fusion
head into one inference path with selectable modality.

``modality``:
  - ``"fusion"``       : LiDAR proposals refined with camera ROI features (default).
  - ``"lidar_only"``   : skip the camera entirely; fusion head runs zero-filled image.
  - ``"camera_only"``  : LiDAR used only to seed proposals; report camera-refined boxes.

The wrapper is *wired and shape-correct* but untrained — proposal scores are not
calibrated and there is no NMS tuning yet (those land with the trained weights from
the GPU box). See README "Status".
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from perceptnet.data.calibration import Calibration
from perceptnet.models.fusion import ROIFusionHead, boxes_lidar_to_image_rois, roi_align_features
from perceptnet.models.lidar_branch import PointPillars, decode_box_predictions, generate_anchors


def gather_bev_features(bev_features: torch.Tensor, boxes: torch.Tensor, cfg) -> torch.Tensor:
    """Sample the BEV feature vector under each box center.

    Args:
        bev_features: ``(1, C, H, W)``.
        boxes: ``(N, 7)`` canonical boxes (uses x, y).
        cfg: PointPillarsConfig (for the point-cloud range).

    Returns:
        ``(N, C)`` features.
    """
    _, C, H, W = bev_features.shape
    if len(boxes) == 0:
        return torch.zeros((0, C), device=bev_features.device)
    x0, y0, _, x1, y1, _ = cfg.pc_range
    ix = ((boxes[:, 0] - x0) / (x1 - x0) * W).long().clamp(0, W - 1)
    iy = ((boxes[:, 1] - y0) / (y1 - y0) * H).long().clamp(0, H - 1)
    return bev_features[0, :, iy, ix].transpose(0, 1)        # (N, C)


class PerceptNet:
    """Inference wrapper over the three branches."""

    def __init__(
        self,
        lidar_model: PointPillars,
        fusion_head: ROIFusionHead,
        camera_branch=None,
        score_threshold: float = 0.1,
        top_k: int = 100,
    ):
        self.lidar = lidar_model
        self.fusion = fusion_head
        self.camera = camera_branch
        self.cfg = lidar_model.cfg
        self.score_threshold = score_threshold
        self.top_k = top_k

    # ------------------------------------------------------------------ #
    def _decode_proposals(self, lidar_out: dict):
        """Decode LiDAR head outputs to (boxes, scores, labels, per-proposal feats)."""
        cls = lidar_out["cls_preds"]          # (1, A*nc, H, W)
        box = lidar_out["box_preds"]          # (1, A*7,  H, W)
        bev = lidar_out["bev_features"]       # (1, C,    H, W)
        H, W = cls.shape[-2:]
        A = self.cfg.num_anchors_per_loc
        nc = self.cfg.num_classes

        anchors = generate_anchors(H, W, self.cfg).to(cls.device)  # (H*W*A, 7)
        cls = cls.permute(0, 2, 3, 1).reshape(H * W * A, nc)        # anchor-major channels
        box = box.permute(0, 2, 3, 1).reshape(H * W * A, 7)

        scores, labels = torch.sigmoid(cls).max(dim=1)
        boxes = decode_box_predictions(box, anchors)

        keep = scores >= self.score_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        if len(scores) > self.top_k:
            topk = torch.topk(scores, self.top_k).indices
            boxes, scores, labels = boxes[topk], scores[topk], labels[topk]

        feats = gather_bev_features(bev, boxes, self.cfg)
        return boxes, scores, labels, feats

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(
        self,
        points: torch.Tensor,
        image=None,
        calib: Optional[Calibration] = None,
        modality: str = "fusion",
        image_shape=None,
        fpn_stride: int = 8,
    ) -> dict:
        """Run the full pipeline on one frame.

        Returns ``{boxes (N,7), scores (N,), labels (N,), box_refine (N,7), modality}``.
        """
        if modality not in ("fusion", "lidar_only", "camera_only"):
            raise ValueError(f"unknown modality {modality!r}")

        lidar_out = self.lidar([torch.as_tensor(points, dtype=torch.float32)])
        boxes, scores, labels, feats = self._decode_proposals(lidar_out)

        camera_available = (
            modality in ("fusion", "camera_only")
            and self.camera is not None
            and image is not None
            and calib is not None
        )

        roi_feats = None
        if camera_available and len(boxes) > 0:
            _, fpn = self.camera(image)
            fmap = fpn["p3"]                                # finest FPN level
            if image_shape is None:
                # infer from feature map and stride
                image_shape = (fmap.shape[-2] * fpn_stride, fmap.shape[-1] * fpn_stride)
            rois, valid = boxes_lidar_to_image_rois(boxes.cpu().numpy(), calib, image_shape)
            roi_feats = roi_align_features(fmap, torch.as_tensor(rois), output_size=7, spatial_scale=1.0 / fpn_stride)

        fused = self.fusion(feats, roi_feats, camera_available=camera_available)
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "cls_logits": fused["cls_logits"],
            "box_refine": fused["box_refine"],
            "modality": modality,
        }
