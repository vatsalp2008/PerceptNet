"""Tests for the 2D image visualizer (pure OpenCV path; Open3D is not exercised)."""

from pathlib import Path

import numpy as np

from perceptnet.data.kitti_dataset import KITTIDataset
from perceptnet.visualization.image_viz import (
    draw_2d_boxes,
    draw_projected_boxes_3d,
    save_image,
    visualize_frame,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mini_kitti"


def test_draw_2d_boxes_shape():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out = draw_2d_boxes(img, np.array([[10, 10, 50, 60]]), labels=[0], scores=[0.9])
    assert out.shape == (100, 200, 3) and out.dtype == np.uint8


def test_to_bgr_handles_torch_chw_float():
    import torch

    chw = torch.rand(3, 64, 80)             # CHW float in [0,1] (KITTIDataset style)
    out = draw_2d_boxes(chw, np.zeros((0, 4)))
    assert out.shape == (64, 80, 3) and out.dtype == np.uint8


def test_projected_boxes_3d_on_fixture():
    ds = KITTIDataset(FIXTURE_ROOT, split="training")
    s = ds[0]
    out = draw_projected_boxes_3d(s["image"], s["boxes_3d"].numpy(), s["calib"], labels=s["labels"].numpy())
    assert out.shape == (375, 1242, 3)


def test_visualize_frame_and_save(tmp_path):
    ds = KITTIDataset(FIXTURE_ROOT, split="training")
    s = ds[0]
    frame = visualize_frame(
        s["image"], s["calib"], boxes_3d=s["boxes_3d"].numpy(),
        labels=s["labels"].numpy(), points=s["points"].numpy(),
    )
    assert frame.shape == (375, 1242, 3)
    out = tmp_path / "frame.png"
    save_image(frame, out)
    assert out.exists() and out.stat().st_size > 0
