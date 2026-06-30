"""Data-layer tests against the committed mini-KITTI fixture + augmentation."""

from pathlib import Path

import numpy as np
import pytest

from perceptnet.data.augmentation import flip_along_x, rotate_about_z, scale_scene
from perceptnet.data.kitti_dataset import (
    KITTIDataset,
    kitti_collate_fn,
    load_kitti_labels,
    load_velodyne,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mini_kitti"


def make_dataset(**kw):
    return KITTIDataset(FIXTURE_ROOT, split="training", **kw)


# --------------------------------------------------------------------------- #
# Label parsing
# --------------------------------------------------------------------------- #
def test_load_labels_excludes_dontcare():
    labels = load_kitti_labels(FIXTURE_ROOT / "training" / "label_2" / "000000.txt")
    types = [l.type for l in labels]
    assert "DontCare" not in types
    assert types == ["Car", "Car", "Pedestrian"]


def test_label_box_cam_fields():
    labels = load_kitti_labels(FIXTURE_ROOT / "training" / "label_2" / "000000.txt")
    car = labels[0]
    assert car.box_cam.shape == (7,)
    assert np.isclose(car.box_cam[3], 4.00)      # length l
    assert car.height_px == pytest.approx(80.0)  # 230 - 150


def test_load_velodyne_shape():
    pts = load_velodyne(FIXTURE_ROOT / "training" / "velodyne" / "000000.bin")
    assert pts.ndim == 2 and pts.shape[1] == 4


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def test_dataset_length_and_ids():
    ds = make_dataset()
    assert len(ds) == 2
    assert ds.frame_ids == ["000000", "000001"]


def test_sample_shapes_and_keys():
    ds = make_dataset()
    s = ds[0]
    assert set(s) == {"frame_id", "image", "points", "boxes_3d", "labels", "calib"}
    assert s["image"].shape == (3, 375, 1242)
    assert s["points"].ndim == 2 and s["points"].shape[1] == 4
    assert s["boxes_3d"].shape == (3, 7)         # 3 in-class objects
    assert s["labels"].tolist() == [0, 0, 1]     # Car, Car, Pedestrian


def test_second_frame_classes():
    ds = make_dataset()
    s = ds[1]
    assert s["boxes_3d"].shape == (2, 7)
    assert s["labels"].tolist() == [0, 2]        # Car, Cyclist


def test_gt_boxes_project_in_front_of_camera():
    ds = make_dataset()
    s = ds[0]
    calib = s["calib"]
    centers = s["boxes_3d"][:, :3].numpy()
    _, depth, mask = calib.project_lidar_to_image(centers)
    assert mask.all() and np.all(depth > 0)      # all objects ahead of the camera


def test_box_dims_positive():
    ds = make_dataset()
    boxes = ds[0]["boxes_3d"].numpy()
    assert np.all(boxes[:, 3:6] > 0)


def test_collate_fn_batches():
    ds = make_dataset()
    batch = kitti_collate_fn([ds[0], ds[1]])
    assert batch["image"].shape == (2, 3, 375, 1242)
    assert len(batch["points"]) == 2 and len(batch["boxes_3d"]) == 2
    assert len(batch["calib"]) == 2


def test_augment_preserves_counts():
    ds = make_dataset(augment=True)
    s = ds[0]
    # Augmentation must not drop points or boxes.
    assert s["points"].shape[1] == 4 and len(s["points"]) > 0
    assert s["boxes_3d"].shape == (3, 7)


# --------------------------------------------------------------------------- #
# Augmentation cores (deterministic identities)
# --------------------------------------------------------------------------- #
def test_flip_twice_is_identity():
    pts = np.random.default_rng(0).random((50, 4))
    boxes = np.random.default_rng(1).random((4, 7))
    p2, b2 = flip_along_x(*flip_along_x(pts, boxes))
    assert np.allclose(p2, pts) and np.allclose(b2, boxes)


def test_rotate_then_unrotate_is_identity():
    pts = np.random.default_rng(0).random((50, 4))
    boxes = np.random.default_rng(1).random((4, 7))
    p1, b1 = rotate_about_z(pts, boxes, 0.3)
    p2, b2 = rotate_about_z(p1, b1, -0.3)
    assert np.allclose(p2, pts, atol=1e-9)
    assert np.allclose(b2[:, :6], boxes[:, :6], atol=1e-9)


def test_scale_scene():
    pts = np.ones((10, 4))
    boxes = np.ones((2, 7))
    p, b = scale_scene(pts, boxes, 2.0)
    assert np.allclose(p[:, :3], 2.0) and np.allclose(p[:, 3], 1.0)   # intensity kept
    assert np.allclose(b[:, :6], 2.0) and np.allclose(b[:, 6], 1.0)   # heading kept
