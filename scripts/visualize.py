#!/usr/bin/env python
"""Render a frame overlay (LiDAR points + projected 3D boxes) to a PNG.

Runs on CPU/Mac. Defaults to the committed mini-KITTI fixture so it works with no
dataset download; pass --data-root / --frame to render real KITTI frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from perceptnet.data.kitti_dataset import KITTIDataset
from perceptnet.visualization.image_viz import save_image, visualize_frame

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "mini_kitti"


def main():
    parser = argparse.ArgumentParser(description="Render a KITTI frame overlay")
    parser.add_argument("--data-root", default=str(FIXTURE))
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", default=str(REPO / "outputs" / "frame_overlay.png"))
    args = parser.parse_args()

    ds = KITTIDataset(args.data_root, split="training")
    s = ds[args.frame]
    frame = visualize_frame(
        s["image"], s["calib"],
        boxes_3d=s["boxes_3d"].numpy(), labels=s["labels"].numpy(),
        points=s["points"].numpy(),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(frame, out)
    print(f"saved overlay ({len(s['boxes_3d'])} GT boxes) -> {out}")


if __name__ == "__main__":
    main()
