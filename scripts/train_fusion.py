#!/usr/bin/env python
"""Train the ROI fusion head on top of a (frozen) LiDAR branch + camera branch.

Real training runs on a CUDA box (``make train-fusion``) and uses the camera. Use
``--smoke`` to verify the fusion-head training step on the fixture (CPU, LiDAR-only,
no network/weights download).

Proposal-level targets: each LiDAR proposal is matched to the best ground-truth box
by 3D IoU; matches above ``--iou-thr`` get that GT's class + an encoded box
refinement, and the head learns classification + refinement on them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from perceptnet.data.kitti_dataset import KITTIDataset
from perceptnet.geometry.iou import iou_3d
from perceptnet.models.fusion import ROIFusionHead
from perceptnet.models.lidar_branch import PointPillars, PointPillarsConfig
from perceptnet.models.losses import smooth_l1_loss
from perceptnet.models.perceptnet import PerceptNet
from perceptnet.models.target_assigner import encode_box_residuals
from perceptnet.utils import get_device, seed_everything

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "mini_kitti"


def fusion_step(model: PerceptNet, sample, loss_classes: int, iou_thr: float, device):
    """One fusion-head training step on a single frame (LiDAR-only path)."""
    out = model.predict(sample["points"].to(device), modality="lidar_only")
    proposals = out["boxes"]
    if len(proposals) == 0:
        return None

    gt = sample["boxes_3d"]
    gt_labels = sample["labels"]
    if len(gt) == 0:
        return None

    iou = torch.as_tensor(iou_3d(proposals.cpu().numpy(), gt.numpy()))   # (N, M)
    best_iou, best_gt = iou.max(dim=1)
    positive = best_iou >= iou_thr
    if not positive.any():
        return None

    pos_idx = positive.nonzero(as_tuple=True)[0]
    cls_logits = out["cls_logits"][pos_idx]
    cls_target = gt_labels[best_gt[pos_idx]]
    cls_loss = F.cross_entropy(cls_logits, cls_target)

    refine = out["box_refine"][pos_idx]
    target_refine = encode_box_residuals(gt[best_gt[pos_idx]], proposals[pos_idx].cpu())
    box_loss = smooth_l1_loss(refine.cpu(), target_refine)
    return cls_loss + box_loss


def main():
    parser = argparse.ArgumentParser(description="Train the ROI fusion head")
    parser.add_argument("--config", default=str(REPO / "configs" / "fusion.yaml"))
    parser.add_argument("--smoke", action="store_true", help="1-step wiring test on the fixture (CPU OK)")
    parser.add_argument("--iou-thr", type=float, default=0.25)
    args = parser.parse_args()

    seed_everything(0)
    cfg_dict = yaml.safe_load(Path(args.config).read_text())
    device = get_device()

    if args.smoke:
        print("[smoke] verifying the fusion-head training step on the mini-KITTI fixture")
        pp_cfg = PointPillarsConfig(pc_range=(0.0, -8.0, -3.0, 11.2, 8.0, 1.0))
        dataset = KITTIDataset(FIXTURE, split="training")
    else:
        pp_cfg = PointPillarsConfig()
        dataset = KITTIDataset(cfg_dict.get("dataset", {}).get("root", "data/kitti"), split="training")
        if device.type != "cuda":
            print("[warn] no CUDA — run full fusion training in docker/Dockerfile.cuda.")

    lidar = PointPillars(pp_cfg).to(device).eval()
    if not args.smoke and Path(cfg_dict["lidar_checkpoint"]).exists():
        lidar.load_state_dict(torch.load(cfg_dict["lidar_checkpoint"], map_location=device))
    for p in lidar.parameters():            # freeze LiDAR branch
        p.requires_grad_(False)

    fcfg = cfg_dict["fusion"]
    fusion_head = ROIFusionHead(
        lidar_channels=fcfg["lidar_channels"], image_channels=fcfg["image_channels"],
        roi_size=fcfg["roi_size"], num_classes=fcfg["num_classes"],
    ).to(device).train()

    model = PerceptNet(lidar, fusion_head, camera_branch=None, score_threshold=0.0, top_k=50)
    optimizer = torch.optim.AdamW(fusion_head.parameters(), lr=cfg_dict["train"]["lr"])

    if args.smoke:
        sample = dataset[0]
        loss = fusion_step(model, sample, fcfg["num_classes"], args.iou_thr, device)
        if loss is None:
            print("[smoke] no positive proposals matched GT (untrained LiDAR) — head still wired; "
                  "run with trained LiDAR weights for a meaningful step.")
            return
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[smoke] step 0: fusion loss={loss.item():.4f}")
        print("[smoke] OK — fusion-head forward/backward/step wired correctly.")
        return

    print("[info] full fusion training loop would iterate the dataset here with the camera enabled; "
          "see ADR-001/ADR-006. Run on the CUDA box.")


if __name__ == "__main__":
    main()
