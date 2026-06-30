"""ROI-based late fusion (Module 4).

Camera-guided, LiDAR-refined fusion that mirrors the production Mobileye/Waymo
approach (ADR-001):

  1. Project each 3D LiDAR proposal's box corners into the image to get a 2D ROI.
  2. ROIAlign the camera FPN features at that ROI -> a fixed 7x7 grid -> image vector.
  3. Concatenate [LiDAR feature (128) | image feature (64)] and refine with an MLP
     into a class score + 3D box refinement.

Modality dropout: if the camera is unavailable the image features are zero-filled and
a flag is passed, so the same head runs LiDAR-only with no architecture change
(ADR-006). The projection geometry and the dropout path are pure and unit-tested; the
MLP is untrained (shape-tested).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

from perceptnet.data.calibration import Calibration
from perceptnet.geometry.boxes import boxes_to_corners_3d


def boxes_lidar_to_image_rois(
    boxes: np.ndarray, calib: Calibration, image_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D LiDAR boxes to 2D image ROIs.

    Args:
        boxes: ``(N, 7)`` canonical LiDAR boxes.
        calib: frame calibration.
        image_shape: ``(height, width)``.

    Returns:
        ``(rois (N,4) xyxy clipped to image, valid (N,) bool)``. A box is invalid if
        none of its corners are in front of the camera.
    """
    boxes = np.atleast_2d(np.asarray(boxes, dtype=np.float64))
    n = len(boxes)
    h, w = image_shape[:2]
    if n == 0:
        return np.zeros((0, 4)), np.zeros((0,), dtype=bool)

    corners = boxes_to_corners_3d(boxes).reshape(-1, 3)        # (N*8, 3)
    uv, depth = calib.velo_to_image(corners)
    uv = uv.reshape(n, 8, 2)
    depth = depth.reshape(n, 8)

    rois = np.zeros((n, 4))
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        front = depth[i] > 1e-3
        if not front.any():
            continue
        pts = uv[i][front]
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        rois[i] = [
            np.clip(x1, 0, w - 1), np.clip(y1, 0, h - 1),
            np.clip(x2, 0, w - 1), np.clip(y2, 0, h - 1),
        ]
        valid[i] = (rois[i, 2] > rois[i, 0]) and (rois[i, 3] > rois[i, 1])
    return rois, valid


def roi_align_features(
    feature_map: torch.Tensor, rois: torch.Tensor, output_size: int = 7, spatial_scale: float = 1 / 8.0
) -> torch.Tensor:
    """ROIAlign image features for a set of ROIs.

    Args:
        feature_map: ``(1, C, H, W)`` FPN map.
        rois: ``(N, 4)`` xyxy in *image* pixel coords.
        output_size: square output side (spec: 7).
        spatial_scale: image-pixel -> feature-map scale (1/stride of the FPN level).

    Returns:
        ``(N, C, output_size, output_size)``.
    """
    from torchvision.ops import roi_align

    rois = torch.as_tensor(rois, dtype=torch.float32)
    batch_index = torch.zeros((len(rois), 1), dtype=torch.float32)
    rois_b = torch.cat([batch_index, rois], dim=1)            # (N, 5): [batch, x1,y1,x2,y2]
    return roi_align(feature_map, rois_b, output_size=output_size, spatial_scale=spatial_scale, aligned=True)


class ROIFusionHead(nn.Module):
    """Fuse per-proposal LiDAR and image features -> class + box refinement."""

    def __init__(
        self,
        lidar_channels: int = 384,
        image_channels: int = 64,
        roi_size: int = 7,
        lidar_feat_dim: int = 128,
        image_feat_dim: int = 64,
        num_classes: int = 3,
    ):
        super().__init__()
        self.image_feat_dim = image_feat_dim
        self.lidar_proj = nn.Linear(lidar_channels, lidar_feat_dim)
        self.image_proj = nn.Linear(image_channels * roi_size * roi_size, image_feat_dim)
        fused_dim = lidar_feat_dim + image_feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Linear(64, num_classes)
        self.box_head = nn.Linear(64, 7)        # box refinement residual

    def forward(
        self,
        lidar_feats: torch.Tensor,
        roi_image_feats: Optional[torch.Tensor] = None,
        camera_available: bool = True,
    ) -> dict:
        """Args:
            lidar_feats: ``(N, lidar_channels)`` per-proposal LiDAR features.
            roi_image_feats: ``(N, C, roi, roi)`` ROIAligned image features, or ``None``.
            camera_available: when False (or features missing), image branch is
                zero-filled (modality dropout).
        """
        n = lidar_feats.shape[0]
        lidar_vec = self.lidar_proj(lidar_feats)

        if camera_available and roi_image_feats is not None:
            img_vec = self.image_proj(roi_image_feats.reshape(n, -1))
        else:
            img_vec = torch.zeros((n, self.image_feat_dim), device=lidar_feats.device, dtype=lidar_feats.dtype)

        fused = self.fusion(torch.cat([lidar_vec, img_vec], dim=1))
        return {"cls_logits": self.cls_head(fused), "box_refine": self.box_head(fused)}
