#!/usr/bin/env python
"""Run the modality-dropout robustness study and print the mAP-degradation table.

Demonstrates the study structure on the fixture (LiDAR-only path, so the LiDAR
point-drop scenarios are the meaningful ones). Numbers are only meaningful with a
trained checkpoint — untrained, expect ~0 everywhere; the value here is that the
scenario transforms + evaluation harness run end-to-end.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch

from perceptnet.data.kitti_dataset import KITTIDataset
from perceptnet.evaluation.kitti_eval import eval_kitti_3d
from perceptnet.evaluation.robustness_study import default_scenarios, run_robustness_study
from perceptnet.models.fusion import ROIFusionHead
from perceptnet.models.lidar_branch import PointPillars, PointPillarsConfig
from perceptnet.models.perceptnet import PerceptNet
from perceptnet.utils import get_device

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "mini_kitti"


class _Adapter:
    """Expose model.predict(frame_dict) -> {boxes, scores, labels} for the study runner."""

    def __init__(self, model: PerceptNet, device):
        self.model = model
        self.device = device

    def predict(self, frame):
        pts = frame["points"]
        pts = pts if isinstance(pts, torch.Tensor) else torch.as_tensor(np.asarray(pts), dtype=torch.float32)
        out = self.model.predict(pts.to(self.device), modality="lidar_only")
        return {
            "boxes": out["boxes"].cpu().numpy(),
            "scores": out["scores"].cpu().numpy(),
            "labels": out["labels"].cpu().numpy(),     # integer class ids
        }


def main():
    parser = argparse.ArgumentParser(description="Modality-dropout robustness study")
    parser.add_argument("--data-root", default=str(FIXTURE))
    parser.add_argument("--lidar-checkpoint", default=None)
    args = parser.parse_args()

    device = get_device()
    cfg = PointPillarsConfig(pc_range=(0.0, -8.0, -3.0, 11.2, 8.0, 1.0)) if args.data_root == str(FIXTURE) \
        else PointPillarsConfig()
    lidar = PointPillars(cfg).to(device).eval()
    if args.lidar_checkpoint and Path(args.lidar_checkpoint).exists():
        lidar.load_state_dict(torch.load(args.lidar_checkpoint, map_location=device))
    else:
        print("[warn] no trained checkpoint — degradation numbers will be ~0 (untrained model).")

    model = PerceptNet(lidar, ROIFusionHead(lidar_channels=384).to(device).eval(),
                       camera_branch=None, score_threshold=0.1, top_k=100)
    adapter = _Adapter(model, device)

    dataset = [KITTIDataset(args.data_root, split="training")[i]
               for i in range(len(KITTIDataset(args.data_root, split="training")))]
    evaluate_fn = partial(eval_kitti_3d, classes=[0, 1, 2],
                          iou_thresholds={0: 0.7, 1: 0.5, 2: 0.5})

    results = run_robustness_study(adapter, dataset, default_scenarios(seed=0), evaluate_fn)

    print("\nModality robustness (Car/Ped/Cyclist mAP per scenario):\n")
    print("| Scenario | mAP |\n|---|---|")
    for name, metrics in results.items():
        print(f"| {name} | {metrics.get('mAP', 0.0) * 100:.2f}% |")


if __name__ == "__main__":
    main()
