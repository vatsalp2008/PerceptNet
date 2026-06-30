"""Fusion tests: ROI projection geometry, ROIAlign, modality dropout, wrapper wiring."""

import numpy as np
import torch

from perceptnet.data.calibration import Calibration
from perceptnet.models.fusion import (
    ROIFusionHead,
    boxes_lidar_to_image_rois,
    roi_align_features,
)
from perceptnet.models.lidar_branch import PointPillars, PointPillarsConfig
from perceptnet.models.perceptnet import PerceptNet, gather_bev_features

F, CX, CY = 700.0, 600.0, 180.0


def make_calib():
    V2C = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0]], dtype=float)
    P2 = np.array([[F, 0, CX, 0], [0, F, CY, 0], [0, 0, 1, 0]], dtype=float)
    return Calibration(P2=P2, R0_rect=np.eye(3), Tr_velo_to_cam=V2C)


def small_cfg():
    return PointPillarsConfig(pc_range=(0.0, -8.0, -3.0, 11.2, 8.0, 1.0))


# --------------------------------------------------------------------------- #
# ROI projection geometry
# --------------------------------------------------------------------------- #
def test_box_projects_to_roi_containing_center():
    calib = make_calib()
    box = np.array([[10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])     # 10 m ahead
    rois, valid = boxes_lidar_to_image_rois(box, calib, image_shape=(375, 1242))
    assert valid[0]
    x1, y1, x2, y2 = rois[0]
    assert x1 <= CX <= x2 and y1 <= CY <= y2                   # center inside ROI
    assert x1 >= 0 and y1 >= 0 and x2 <= 1241 and y2 <= 374    # clipped to image


def test_behind_camera_box_is_invalid():
    calib = make_calib()
    box = np.array([[-5.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])     # behind the camera
    _, valid = boxes_lidar_to_image_rois(box, calib, image_shape=(375, 1242))
    assert not valid[0]


def test_empty_boxes():
    calib = make_calib()
    rois, valid = boxes_lidar_to_image_rois(np.zeros((0, 7)), calib, (375, 1242))
    assert rois.shape == (0, 4) and valid.shape == (0,)


# --------------------------------------------------------------------------- #
# ROIAlign
# --------------------------------------------------------------------------- #
def test_roi_align_output_shape():
    fmap = torch.rand(1, 64, 50, 160)
    rois = torch.tensor([[10.0, 10.0, 100.0, 50.0], [0.0, 0.0, 80.0, 40.0]])
    out = roi_align_features(fmap, rois, output_size=7, spatial_scale=1 / 8.0)
    assert out.shape == (2, 64, 7, 7)


# --------------------------------------------------------------------------- #
# Fusion head + modality dropout
# --------------------------------------------------------------------------- #
def test_fusion_head_output_shapes():
    head = ROIFusionHead(lidar_channels=384, image_channels=64).eval()
    lidar = torch.rand(5, 384)
    img = torch.rand(5, 64, 7, 7)
    out = head(lidar, img, camera_available=True)
    assert out["cls_logits"].shape == (5, 3)
    assert out["box_refine"].shape == (5, 7)


def test_modality_dropout_ignores_image():
    head = ROIFusionHead(lidar_channels=384, image_channels=64).eval()
    lidar = torch.rand(4, 384)
    img_a = torch.rand(4, 64, 7, 7)
    img_b = torch.rand(4, 64, 7, 7)
    with torch.no_grad():
        out_none = head(lidar, None, camera_available=False)
        out_a = head(lidar, img_a, camera_available=False)
        out_b = head(lidar, img_b, camera_available=False)
    # With the camera dropped, the image input must not affect the output.
    assert torch.allclose(out_none["cls_logits"], out_a["cls_logits"])
    assert torch.allclose(out_a["cls_logits"], out_b["cls_logits"])


def test_dropout_changes_output_vs_available():
    head = ROIFusionHead(lidar_channels=384, image_channels=64).eval()
    lidar = torch.rand(4, 384)
    img = torch.rand(4, 64, 7, 7)
    with torch.no_grad():
        with_cam = head(lidar, img, camera_available=True)
        without = head(lidar, img, camera_available=False)
    assert not torch.allclose(with_cam["cls_logits"], without["cls_logits"])


# --------------------------------------------------------------------------- #
# BEV feature gather
# --------------------------------------------------------------------------- #
def test_gather_bev_features():
    cfg = small_cfg()
    bev = torch.rand(1, 8, 100, 70)
    # a box at the range origin maps to grid cell (0, 0)
    boxes = torch.tensor([[cfg.pc_range[0], cfg.pc_range[1], 0, 1, 1, 1, 0]])
    feats = gather_bev_features(bev, boxes, cfg)
    assert feats.shape == (1, 8)
    assert torch.allclose(feats[0], bev[0, :, 0, 0])


def test_gather_bev_empty():
    cfg = small_cfg()
    feats = gather_bev_features(torch.rand(1, 8, 10, 10), torch.zeros((0, 7)), cfg)
    assert feats.shape == (0, 8)


# --------------------------------------------------------------------------- #
# PerceptNet wrapper
# --------------------------------------------------------------------------- #
def test_perceptnet_lidar_only_runs():
    cfg = small_cfg()
    model = PerceptNet(
        PointPillars(cfg).eval(),
        ROIFusionHead(lidar_channels=384).eval(),
        camera_branch=None,
        score_threshold=0.0,
        top_k=15,
    )
    cloud = torch.rand(400, 4) * torch.tensor([11.0, 16.0, 3.0, 1.0]) - torch.tensor([0.0, 8.0, 3.0, 0.0])
    out = model.predict(cloud, modality="lidar_only")
    assert out["modality"] == "lidar_only"
    n = out["boxes"].shape[0]
    assert n <= 15
    assert out["cls_logits"].shape == (n, 3)
    assert out["box_refine"].shape == (n, 7)


def test_perceptnet_fusion_without_camera_falls_back():
    cfg = small_cfg()
    model = PerceptNet(PointPillars(cfg).eval(), ROIFusionHead(lidar_channels=384).eval(),
                       camera_branch=None, score_threshold=0.0, top_k=10)
    cloud = torch.rand(300, 4) * torch.tensor([11.0, 16.0, 3.0, 1.0]) - torch.tensor([0.0, 8.0, 3.0, 0.0])
    # modality="fusion" but no camera -> must not error, behaves LiDAR-only
    out = model.predict(cloud, modality="fusion")
    assert out["cls_logits"].shape[0] == out["boxes"].shape[0]
