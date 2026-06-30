"""LiDAR-branch tests: pillarization indexing (real logic) + forward-pass shapes.

Forward tests use a deliberately small BEV grid so they run fast on CPU; the shape
relationships are identical to the full KITTI grid.
"""

import numpy as np
import torch

from perceptnet.models.lidar_branch import (
    AnchorHead,
    Backbone2D,
    PillarFeatureNet,
    PointPillars,
    PointPillarsConfig,
    decode_box_predictions,
    generate_anchors,
    points_to_pillars,
    scatter_to_bev,
)


def small_cfg(**kw):
    return PointPillarsConfig(pc_range=(0.0, -8.0, -3.0, 11.2, 8.0, 1.0), **kw)


# --------------------------------------------------------------------------- #
# grid arithmetic
# --------------------------------------------------------------------------- #
def test_full_grid_size_matches_spec():
    cfg = PointPillarsConfig()
    assert cfg.grid_size == (440, 500)              # nx, ny
    assert cfg.num_anchors_per_loc == 6             # 3 classes x 2 rotations


# --------------------------------------------------------------------------- #
# pillarization
# --------------------------------------------------------------------------- #
def test_points_in_same_cell_form_one_pillar():
    cfg = small_cfg()
    pts = torch.tensor([[0.05, -7.95, 0.0, 1.0],
                        [0.10, -7.90, 0.0, 1.0],
                        [0.02, -7.99, 0.0, 1.0]])
    voxels, coords, num_points = points_to_pillars(pts, cfg)
    assert voxels.shape == (1, cfg.max_points_per_pillar, 4)
    assert num_points.tolist() == [3]
    assert coords.tolist() == [[0, 0]]              # (iy, ix)


def test_points_in_different_cells_form_separate_pillars():
    cfg = small_cfg()
    pts = torch.tensor([[0.05, -7.95, 0.0, 1.0],     # cell (0,0)
                        [5.0, 0.0, 0.0, 1.0]])        # a far cell
    voxels, coords, num_points = points_to_pillars(pts, cfg)
    assert voxels.shape[0] == 2 and num_points.tolist() == [1, 1]


def test_out_of_range_points_filtered():
    cfg = small_cfg()
    pts = torch.tensor([[0.05, -7.95, 0.0, 1.0],     # in range
                        [100.0, 0.0, 0.0, 1.0],       # x out of range
                        [0.0, 0.0, 50.0, 1.0]])       # z out of range
    voxels, coords, num_points = points_to_pillars(pts, cfg)
    assert num_points.sum().item() == 1


def test_max_points_per_pillar_caps():
    cfg = small_cfg(max_points_per_pillar=2)
    pts = torch.tensor([[0.05, -7.95, 0.0, 1.0]] * 5)   # all same cell
    voxels, coords, num_points = points_to_pillars(pts, cfg)
    assert voxels.shape == (1, 2, 4)
    assert num_points.tolist() == [2]


def test_empty_cloud():
    cfg = small_cfg()
    voxels, coords, num_points = points_to_pillars(torch.zeros((0, 4)), cfg)
    assert voxels.shape[0] == 0 and coords.shape[0] == 0


# --------------------------------------------------------------------------- #
# pillar feature net + scatter
# --------------------------------------------------------------------------- #
def test_pillar_feature_net_shape():
    cfg = small_cfg()
    pfn = PillarFeatureNet(cfg).eval()
    pts = torch.rand(200, 4) * torch.tensor([11.0, 16.0, 3.0, 1.0]) - torch.tensor([0.0, 8.0, 3.0, 0.0])
    voxels, coords, num_points = points_to_pillars(pts, cfg)
    feats = pfn(voxels, coords, num_points)
    assert feats.shape == (voxels.shape[0], cfg.pillar_feat_channels)


def test_scatter_places_features_at_coords():
    feats = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    coords = torch.tensor([[0, 0], [1, 2]])           # (iy, ix)
    batch_idx = torch.tensor([0, 0])
    bev = scatter_to_bev(feats, coords, batch_idx, batch_size=1, nx=5, ny=3)
    assert bev.shape == (1, 2, 3, 5)
    assert torch.allclose(bev[0, :, 0, 0], feats[0])
    assert torch.allclose(bev[0, :, 1, 2], feats[1])
    assert bev[0, :, 2, 4].sum() == 0                 # untouched cell stays zero


# --------------------------------------------------------------------------- #
# backbone + head + anchors
# --------------------------------------------------------------------------- #
def test_backbone_output_channels():
    bb = Backbone2D(64).eval()
    out = bb(torch.rand(1, 64, 100, 70))
    assert out.shape[1] == 384


def test_anchor_head_channels():
    cfg = small_cfg()
    head = AnchorHead(384, cfg)
    out = head(torch.rand(1, 384, 50, 35))
    assert out["cls_preds"].shape[1] == cfg.num_anchors_per_loc * cfg.num_classes  # 18
    assert out["box_preds"].shape[1] == cfg.num_anchors_per_loc * 7                # 42
    assert out["dir_preds"].shape[1] == cfg.num_anchors_per_loc * 2                # 12


def test_generate_anchors_count():
    cfg = small_cfg()
    anchors = generate_anchors(50, 35, cfg)
    assert anchors.shape == (50 * 35 * cfg.num_anchors_per_loc, 7)


def test_decode_zero_residual_returns_anchor():
    cfg = small_cfg()
    anchors = generate_anchors(4, 4, cfg)
    decoded = decode_box_predictions(torch.zeros_like(anchors), anchors)
    assert torch.allclose(decoded, anchors, atol=1e-5)


# --------------------------------------------------------------------------- #
# full forward
# --------------------------------------------------------------------------- #
def test_pointpillars_forward_shapes():
    cfg = small_cfg()
    model = PointPillars(cfg).eval()
    nx, ny = cfg.grid_size
    cloud = torch.rand(500, 4) * torch.tensor([11.0, 16.0, 3.0, 1.0]) - torch.tensor([0.0, 8.0, 3.0, 0.0])
    with torch.no_grad():
        out = model([cloud])
    for key in ("cls_preds", "box_preds", "dir_preds", "bev_features"):
        assert out[key].shape[0] == 1
    h, w = out["cls_preds"].shape[-2:]
    assert out["box_preds"].shape[-2:] == (h, w)
    assert out["bev_features"].shape[1] == 384
    assert out["cls_preds"].shape[1] == cfg.num_anchors_per_loc * cfg.num_classes


def test_pointpillars_batch_of_two():
    cfg = small_cfg()
    model = PointPillars(cfg).eval()
    clouds = [torch.rand(300, 4) * torch.tensor([11.0, 16.0, 3.0, 1.0]) - torch.tensor([0.0, 8.0, 3.0, 0.0])
              for _ in range(2)]
    with torch.no_grad():
        out = model(clouds)
    assert out["cls_preds"].shape[0] == 2
