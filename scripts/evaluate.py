#!/usr/bin/env python
"""Run the full KITTI evaluation and print the mAP table.

Loads a (trained) PerceptNet, runs it over the val split, and reports per-class 3D AP
via :func:`perceptnet.evaluation.kitti_eval.eval_kitti_3d`. With no checkpoint the
model is untrained and numbers will be ~0 (the harness is what's being exercised);
real numbers come from the GPU-trained weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from perceptnet.data.kitti_dataset import KITTIDataset
from perceptnet.evaluation.kitti_eval import eval_kitti_3d
from perceptnet.models.fusion import ROIFusionHead
from perceptnet.models.lidar_branch import PointPillars, PointPillarsConfig
from perceptnet.models.perceptnet import PerceptNet
from perceptnet.utils import get_device

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "mini_kitti"


def format_map_table(results: dict, classes) -> str:
    header = "| Class | 3D AP |\n|---|---|"
    rows = [f"| {c} | {results.get(c, 0.0) * 100:.2f}% |" for c in classes]
    rows.append(f"| **mAP** | **{results.get('mAP', 0.0) * 100:.2f}%** |")
    return "\n".join([header, *rows])


def main():
    parser = argparse.ArgumentParser(description="Evaluate PerceptNet on KITTI (mAP)")
    parser.add_argument("--config", default=str(REPO / "configs" / "kitti_pointpillars.yaml"))
    parser.add_argument("--lidar-checkpoint", default=None)
    parser.add_argument("--smoke", action="store_true", help="run on the mini-KITTI fixture")
    parser.add_argument("--modality", default="lidar_only", choices=["fusion", "lidar_only", "camera_only"])
    args = parser.parse_args()

    cfg_dict = yaml.safe_load(Path(args.config).read_text())
    device = get_device()
    classes = cfg_dict["dataset"]["classes"]

    if args.smoke:
        pp_cfg = PointPillarsConfig(pc_range=(0.0, -8.0, -3.0, 11.2, 8.0, 1.0))
        dataset = KITTIDataset(FIXTURE, split="training", classes=classes)
    else:
        pp_cfg = PointPillarsConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                                       for k, v in cfg_dict["model"].items()})
        dataset = KITTIDataset(cfg_dict["dataset"]["root"], split="training", classes=classes)

    lidar = PointPillars(pp_cfg).to(device).eval()
    if args.lidar_checkpoint and Path(args.lidar_checkpoint).exists():
        lidar.load_state_dict(torch.load(args.lidar_checkpoint, map_location=device))
    else:
        print("[warn] no trained checkpoint — reported AP will be ~0 (untrained model).")

    model = PerceptNet(lidar, ROIFusionHead(lidar_channels=384).to(device).eval(),
                       camera_branch=None, score_threshold=0.1, top_k=100)

    predictions, ground_truths = [], []
    idx_to_class = {i: c for i, c in enumerate(classes)}
    for i in range(len(dataset)):
        s = dataset[i]
        out = model.predict(s["points"].to(device), modality=args.modality)
        predictions.append({
            "boxes": out["boxes"].cpu().numpy(),
            "scores": out["scores"].cpu().numpy(),
            "labels": np.array([idx_to_class[int(l)] for l in out["labels"].cpu().numpy()]),
        })
        ground_truths.append({
            "boxes": s["boxes_3d"].numpy(),
            "labels": np.array([idx_to_class[int(l)] for l in s["labels"].numpy()]),
        })

    results = eval_kitti_3d(predictions, ground_truths, classes=classes,
                            iou_thresholds=cfg_dict["eval"]["iou_thresholds"])
    print("\nKITTI 3D detection results ({} frames, modality={}):\n".format(len(dataset), args.modality))
    print(format_map_table(results, classes))


if __name__ == "__main__":
    main()
