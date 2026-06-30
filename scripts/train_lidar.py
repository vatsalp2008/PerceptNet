#!/usr/bin/env python
"""Train the PointPillars LiDAR branch on KITTI.

Real training runs on a CUDA box (``make train-lidar``). On a CPU-only machine use
``--smoke`` to verify the whole pipeline is wired (data -> pillarize -> forward ->
anchor assignment -> loss -> backward -> optimizer step) on the committed fixture.

Note on the anchor assigner: it uses the exact pure-NumPy BEV-IoU as a correctness
reference. That is fine for the small ``--smoke`` grid but too slow for the full
KITTI grid; for full-scale training swap in a vectorized/CUDA BEV-IoU (see ADR-002).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from perceptnet.data.kitti_dataset import KITTIDataset, kitti_collate_fn
from perceptnet.models.lidar_branch import PointPillars, PointPillarsConfig, generate_anchors
from perceptnet.models.losses import PointPillarsLoss
from perceptnet.models.target_assigner import AnchorTargetAssigner
from perceptnet.utils import get_device, seed_everything

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "mini_kitti"


def build_config(model_cfg: dict) -> PointPillarsConfig:
    tuples = {k: (tuple(v) if isinstance(v, list) else v) for k, v in model_cfg.items()}
    return PointPillarsConfig(**tuples)


def flatten_head(out: dict, num_classes: int):
    """(1, A*c, H, W) head maps -> per-anchor (A_total, c) tensors aligned with anchors."""
    H, W = out["cls_preds"].shape[-2:]

    def flat(t, last):
        return t.permute(0, 2, 3, 1).reshape(-1, last)

    return flat(out["cls_preds"], num_classes), flat(out["box_preds"], 7), flat(out["dir_preds"], 2), H, W


def run_epoch(model, loader, assigner, loss_fn, optimizer, cfg, device, max_steps=None):
    model.train()
    anchors_cache = {}
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        clouds = [p.to(device) for p in batch["points"]]
        out = model(clouds)
        cls_p, box_p, dir_p, H, W = flatten_head(out, cfg.num_classes)
        if (H, W) not in anchors_cache:
            anchors_cache[(H, W)] = generate_anchors(H, W, cfg)
        anchors = anchors_cache[(H, W)]

        # per-sample anchor assignment (here batch is handled by averaging targets)
        targets = assigner.assign(anchors, batch["boxes_3d"][0].cpu(), batch["labels"][0].cpu())
        n_anchor = anchors.shape[0]
        losses = loss_fn(
            cls_p[:n_anchor].cpu(), box_p[:n_anchor].cpu(), dir_p[:n_anchor].cpu(),
            targets["cls_targets"], targets["box_targets"], targets["dir_targets"],
        )
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()
        yield step, losses


def main():
    parser = argparse.ArgumentParser(description="Train PointPillars on KITTI")
    parser.add_argument("--config", default=str(REPO / "configs" / "kitti_pointpillars.yaml"))
    parser.add_argument("--smoke", action="store_true", help="1-step wiring test on the fixture (CPU OK)")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    seed_everything(0)
    cfg_dict = yaml.safe_load(Path(args.config).read_text())
    device = get_device()

    if args.smoke:
        print("[smoke] verifying the LiDAR training pipeline on the mini-KITTI fixture")
        cfg = PointPillarsConfig(pc_range=(0.0, -8.0, -3.0, 11.2, 8.0, 1.0))   # tiny grid -> fast
        dataset = KITTIDataset(FIXTURE, split="training", augment=False)
    else:
        cfg = build_config(cfg_dict["model"])
        if device.type != "cuda":
            print("[warn] no CUDA detected — full PointPillars training is impractical on CPU. "
                  "Use --smoke here, and run real training in docker/Dockerfile.cuda.")
        dataset = KITTIDataset(cfg_dict["dataset"]["root"], split="training",
                               classes=cfg_dict["dataset"]["classes"], augment=cfg_dict["train"]["augment"])

    loader = DataLoader(dataset, batch_size=1, shuffle=not args.smoke, collate_fn=kitti_collate_fn)
    model = PointPillars(cfg).to(device)
    assigner = AnchorTargetAssigner(**cfg_dict.get("assigner", {}))
    loss_fn = PointPillarsLoss(num_classes=cfg.num_classes, **cfg_dict.get("loss", {}))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_dict["train"]["lr"])

    if args.smoke:
        for step, losses in run_epoch(model, loader, assigner, loss_fn, optimizer, cfg, device, max_steps=1):
            print(f"[smoke] step {step}: loss={losses['loss'].item():.4f} "
                  f"(cls={losses['cls_loss'].item():.4f} box={losses['box_loss'].item():.4f} "
                  f"dir={losses['dir_loss'].item():.4f})")
        print("[smoke] OK — forward/backward/step wired correctly.")
        return

    epochs = args.epochs or cfg_dict["train"]["epochs"]
    ckpt_dir = Path(cfg_dict["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    try:
        import mlflow

        mlflow.set_experiment(cfg_dict["train"].get("mlflow_experiment", "perceptnet-lidar"))
        mlflow.start_run()
        mlflow.log_params({"lr": cfg_dict["train"]["lr"], "epochs": epochs})
    except Exception:  # mlflow optional
        mlflow = None

    for epoch in range(epochs):
        for step, losses in run_epoch(model, loader, assigner, loss_fn, optimizer, cfg, device):
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}: loss={losses['loss'].item():.4f}")
                if mlflow:
                    mlflow.log_metric("loss", losses["loss"].item())
        torch.save(model.state_dict(), ckpt_dir / "pointpillars.pth")
    print(f"saved checkpoint to {ckpt_dir / 'pointpillars.pth'}")


if __name__ == "__main__":
    main()
