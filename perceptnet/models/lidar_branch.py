"""PointPillars LiDAR branch — pure PyTorch (no spconv / custom CUDA).

PointPillars operates on a *dense* BEV pseudo-image, so it needs no sparse 3D
convolution library. Keeping it pure PyTorch means the module imports and runs a
forward pass on CPU/MPS (shape-testable here) while still training on a CUDA box.
See ADR-002.

Pipeline (Module 3 of the spec):
  1. Pillar Feature Net : group points into BEV pillars, augment to 9-d features,
     PointNet MLP -> max-pool -> one 64-d vector per pillar.
  2. Scatter           : place pillar vectors back onto the BEV grid -> pseudo-image.
  3. Backbone (2D CNN) : SECOND/SSD-style top-down + upsample, multi-scale concat.
  4. Anchor head       : per-location class scores, 7-d box regression, direction.

Box code is the canonical ``[x, y, z, dx, dy, dz, heading]`` (LiDAR frame).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class PointPillarsConfig:
    num_point_features: int = 4                       # x, y, z, intensity
    voxel_size: Sequence[float] = (0.16, 0.16, 4.0)
    pc_range: Sequence[float] = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0)  # xmin..zmax
    max_points_per_pillar: int = 100
    max_pillars: int = 12000
    pillar_feat_channels: int = 64
    num_classes: int = 3
    num_rotations: int = 2                            # 0 and 90 degrees
    # anchor dimensions [dx, dy, dz] and z-center per class (Car, Ped, Cyclist)
    anchor_sizes: Sequence[Sequence[float]] = field(
        default_factory=lambda: [[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]]
    )
    anchor_z: Sequence[float] = (-1.0, -0.6, -0.6)
    anchor_rotations: Sequence[float] = (0.0, 1.5707963)

    @property
    def grid_size(self):
        """(nx, ny) — pseudo-image width (x) and height (y)."""
        nx = int(round((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0]))
        ny = int(round((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1]))
        return nx, ny

    @property
    def num_anchors_per_loc(self) -> int:
        return self.num_classes * self.num_rotations


# --------------------------------------------------------------------------- #
# 1. Pillarization (pure indexing — unit-tested)
# --------------------------------------------------------------------------- #
def points_to_pillars(points: torch.Tensor, cfg: PointPillarsConfig):
    """Group an ``(N, C)`` point cloud into pillars.

    Returns:
        voxels: ``(P, max_points, C)``
        coords: ``(P, 2)`` integer ``(iy, ix)`` BEV grid indices
        num_points: ``(P,)`` valid points per pillar (clamped to ``max_points``)

    Runs on CPU (index-heavy ``unique``/scatter are unreliable on MPS); the model
    moves the returned pillars to the compute device.
    """
    points = torch.as_tensor(points, dtype=torch.float32).cpu()
    x0, y0, z0, x1, y1, z1 = cfg.pc_range
    vx, vy = cfg.voxel_size[0], cfg.voxel_size[1]
    nx, ny = cfg.grid_size
    C = points.shape[1]

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    in_range = (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & (z >= z0) & (z < z1)
    pts = points[in_range]
    if len(pts) == 0:
        return (
            torch.zeros((0, cfg.max_points_per_pillar, C)),
            torch.zeros((0, 2), dtype=torch.long),
            torch.zeros((0,), dtype=torch.long),
        )

    ix = ((pts[:, 0] - x0) / vx).floor().long().clamp(0, nx - 1)
    iy = ((pts[:, 1] - y0) / vy).floor().long().clamp(0, ny - 1)
    keys = iy * nx + ix

    unique_keys, inverse, counts = torch.unique(keys, return_inverse=True, return_counts=True)
    P = len(unique_keys)

    # within-pillar slot index for each point (stable sort by pillar)
    order = torch.argsort(inverse, stable=True)
    sorted_inv = inverse[order]
    group_start = torch.zeros(P, dtype=torch.long)
    group_start[1:] = torch.cumsum(counts, 0)[:-1]
    within = torch.arange(len(pts)) - group_start[sorted_inv]
    keep = within < cfg.max_points_per_pillar

    voxels = torch.zeros((P, cfg.max_points_per_pillar, C))
    sel = order[keep]
    voxels[sorted_inv[keep], within[keep]] = pts[sel]

    num_points = counts.clamp(max=cfg.max_points_per_pillar)
    coords = torch.stack([unique_keys // nx, unique_keys % nx], dim=1)  # (iy, ix)

    # cap the number of pillars (keep the densest)
    if P > cfg.max_pillars:
        topk = torch.topk(num_points, cfg.max_pillars).indices
        voxels, coords, num_points = voxels[topk], coords[topk], num_points[topk]
    return voxels, coords, num_points


class PillarFeatureNet(nn.Module):
    """Augment per-point features to 9-d, run a PointNet MLP, max-pool per pillar."""

    def __init__(self, cfg: PointPillarsConfig):
        super().__init__()
        self.cfg = cfg
        in_ch = cfg.num_point_features + 5          # + xc,yc,zc (cluster) + xp,yp (center)
        self.linear = nn.Linear(in_ch, cfg.pillar_feat_channels, bias=False)
        self.norm = nn.BatchNorm1d(cfg.pillar_feat_channels)

    def forward(self, voxels: torch.Tensor, coords: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if voxels.shape[0] == 0:
            return torch.zeros((0, cfg.pillar_feat_channels), device=voxels.device)

        P, T, _ = voxels.shape
        xyz = voxels[:, :, :3]
        valid = (torch.arange(T, device=voxels.device)[None, :] < num_points[:, None]).float()  # (P,T)

        # cluster offset: distance from the mean of the pillar's points
        mean = (xyz * valid[..., None]).sum(1) / num_points.clamp(min=1)[:, None]
        f_cluster = xyz - mean[:, None, :]

        # pillar-center offset
        vx, vy = cfg.voxel_size[0], cfg.voxel_size[1]
        x0, y0 = cfg.pc_range[0], cfg.pc_range[1]
        cx = coords[:, 1].float() * vx + vx / 2 + x0
        cy = coords[:, 0].float() * vy + vy / 2 + y0
        f_center = torch.stack(
            [voxels[:, :, 0] - cx[:, None], voxels[:, :, 1] - cy[:, None]], dim=2
        )

        features = torch.cat([voxels, f_cluster, f_center], dim=2)   # (P, T, 9)
        features = features * valid[..., None]                       # zero padded points

        x = self.linear(features)                                    # (P, T, C)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        x = x * valid[..., None]                                     # re-mask after activation
        return x.max(dim=1).values                                   # (P, C)


def scatter_to_bev(pillar_features: torch.Tensor, coords: torch.Tensor, batch_idx: torch.Tensor,
                   batch_size: int, nx: int, ny: int) -> torch.Tensor:
    """Scatter ``(P, C)`` pillar features onto a ``(B, C, ny, nx)`` pseudo-image."""
    C = pillar_features.shape[1]
    canvas = torch.zeros((batch_size, C, ny * nx), device=pillar_features.device, dtype=pillar_features.dtype)
    if pillar_features.shape[0] > 0:
        flat = coords[:, 0] * nx + coords[:, 1]                      # iy*nx + ix
        canvas[batch_idx, :, flat] = pillar_features
    return canvas.view(batch_size, C, ny, nx)


# --------------------------------------------------------------------------- #
# 3. Backbone
# --------------------------------------------------------------------------- #
def _conv_block(in_ch, out_ch, num_conv, stride):
    layers = [nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
              nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    for _ in range(num_conv - 1):
        layers += [nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                   nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def _deconv_block(in_ch, out_ch, stride):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, stride, stride=stride, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class Backbone2D(nn.Module):
    """SECOND/SSD-style top-down backbone with upsampled multi-scale concatenation.

    The upsampled branches are resized to the first branch's resolution before
    concatenation, so the module is robust to any input grid size (avoids the
    classic PointPillars off-by-one between down/up sampling).
    """

    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.block1 = _conv_block(in_channels, 64, num_conv=3, stride=2)
        self.block2 = _conv_block(64, 128, num_conv=5, stride=2)
        self.block3 = _conv_block(128, 256, num_conv=5, stride=2)
        self.up1 = _deconv_block(64, 128, stride=1)
        self.up2 = _deconv_block(128, 128, stride=2)
        self.up3 = _deconv_block(256, 128, stride=4)
        self.out_channels = 384

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        u1 = self.up1(x1)
        size = u1.shape[-2:]
        u2 = F.interpolate(self.up2(x2), size=size, mode="bilinear", align_corners=False)
        u3 = F.interpolate(self.up3(x3), size=size, mode="bilinear", align_corners=False)
        return torch.cat([u1, u2, u3], dim=1)


# --------------------------------------------------------------------------- #
# 4. Anchor head + anchors
# --------------------------------------------------------------------------- #
class AnchorHead(nn.Module):
    """1x1 conv head producing class / box / direction maps."""

    def __init__(self, in_channels: int, cfg: PointPillarsConfig):
        super().__init__()
        self.cfg = cfg
        n_anchor = cfg.num_anchors_per_loc
        self.conv_cls = nn.Conv2d(in_channels, n_anchor * cfg.num_classes, 1)
        self.conv_box = nn.Conv2d(in_channels, n_anchor * 7, 1)
        self.conv_dir = nn.Conv2d(in_channels, n_anchor * 2, 1)

    def forward(self, x: torch.Tensor) -> dict:
        return {
            "cls_preds": self.conv_cls(x),
            "box_preds": self.conv_box(x),
            "dir_preds": self.conv_dir(x),
        }


def generate_anchors(feature_h: int, feature_w: int, cfg: PointPillarsConfig) -> torch.Tensor:
    """Generate anchors over a feature map.

    Returns ``(feature_h * feature_w * num_anchors_per_loc, 7)`` boxes in the LiDAR
    frame (canonical box code).
    """
    x0, y0, _, x1, y1, _ = cfg.pc_range
    xs = torch.linspace(x0, x1, feature_w)
    ys = torch.linspace(y0, y1, feature_h)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")          # (H, W)

    anchors = []
    for size, z in zip(cfg.anchor_sizes, cfg.anchor_z):
        for rot in cfg.anchor_rotations:
            a = torch.stack(
                [
                    grid_x, grid_y, torch.full_like(grid_x, z),
                    torch.full_like(grid_x, size[0]),
                    torch.full_like(grid_x, size[1]),
                    torch.full_like(grid_x, size[2]),
                    torch.full_like(grid_x, rot),
                ],
                dim=-1,
            )
            anchors.append(a.reshape(-1, 7))
    # interleave per location: (H*W, num_anchors, 7) -> (H*W*num_anchors, 7)
    return torch.stack(anchors, dim=1).reshape(-1, 7)


def decode_box_predictions(box_preds: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Decode SECOND-style box residuals to absolute boxes.

    Args:
        box_preds: ``(num_anchors, 7)`` residuals ``(dx, dy, dz, dl, dw, dh, dtheta)``.
        anchors: ``(num_anchors, 7)`` anchor boxes.
    """
    xa, ya, za, dxa, dya, dza, ra = anchors.unbind(-1)
    diag = torch.sqrt(dxa ** 2 + dya ** 2)
    tx, ty, tz, tdx, tdy, tdz, tr = box_preds.unbind(-1)
    x = tx * diag + xa
    y = ty * diag + ya
    z = tz * dza + za
    dx = torch.exp(tdx) * dxa
    dy = torch.exp(tdy) * dya
    dz = torch.exp(tdz) * dza
    r = tr + ra
    return torch.stack([x, y, z, dx, dy, dz, r], dim=-1)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class PointPillars(nn.Module):
    """End-to-end PointPillars LiDAR detector (untrained — see Status in README)."""

    def __init__(self, cfg: PointPillarsConfig | None = None):
        super().__init__()
        self.cfg = cfg or PointPillarsConfig()
        self.pfn = PillarFeatureNet(self.cfg)
        self.backbone = Backbone2D(self.cfg.pillar_feat_channels)
        self.head = AnchorHead(self.backbone.out_channels, self.cfg)

    def forward(self, points_list: List[torch.Tensor]) -> dict:
        """Args: list of ``(N_i, C)`` point clouds (one per batch element)."""
        cfg = self.cfg
        nx, ny = cfg.grid_size
        device = next(self.parameters()).device

        all_feats, all_coords, all_batch = [], [], []
        for b, points in enumerate(points_list):
            voxels, coords, num_points = points_to_pillars(points, cfg)   # CPU
            voxels = voxels.to(device)
            coords = coords.to(device)
            num_points = num_points.to(device)
            feats = self.pfn(voxels, coords, num_points)
            all_feats.append(feats)
            all_coords.append(coords)
            all_batch.append(torch.full((len(coords),), b, dtype=torch.long, device=device))

        pillar_features = torch.cat(all_feats, 0)
        coords = torch.cat(all_coords, 0)
        batch_idx = torch.cat(all_batch, 0)

        bev = scatter_to_bev(pillar_features, coords, batch_idx, len(points_list), nx, ny)
        features = self.backbone(bev)
        out = self.head(features)
        out["bev_features"] = features            # exposed for the fusion head
        return out

    def get_anchors(self, feature_h: int, feature_w: int) -> torch.Tensor:
        return generate_anchors(feature_h, feature_w, self.cfg)
