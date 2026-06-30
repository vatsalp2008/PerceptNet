"""Unit tests for mAP, MOTA/MOTP, and robustness transforms."""

import numpy as np

from perceptnet.evaluation.kitti_eval import average_precision, eval_kitti_3d
from perceptnet.evaluation.robustness_study import (
    add_lidar_noise,
    drop_lidar_points,
    zero_camera_features,
)
from perceptnet.evaluation.tracking_eval import eval_mot


def box(x, y=0.0, z=0.0, dx=4.0, dy=2.0, dz=1.5, h=0.0):
    return [x, y, z, dx, dy, dz, h]


# --------------------------------------------------------------------------- #
# Average precision
# --------------------------------------------------------------------------- #
def test_ap_perfect_curve():
    # precision 1 across full recall -> AP 1
    assert np.isclose(average_precision([1.0, 1.0], [0.5, 1.0]), 1.0)


def test_ap_11point_matches_area_for_perfect():
    assert np.isclose(average_precision([1.0, 1.0], [0.5, 1.0], method="11point"), 1.0)


# --------------------------------------------------------------------------- #
# Detection mAP
# --------------------------------------------------------------------------- #
def test_eval_kitti_3d_perfect():
    frame_pred = {"boxes": np.array([box(10), box(20)]), "scores": np.array([0.9, 0.8]),
                  "labels": np.array(["Car", "Car"])}
    frame_gt = {"boxes": np.array([box(10), box(20)]), "labels": np.array(["Car", "Car"])}
    res = eval_kitti_3d([frame_pred], [frame_gt], classes=["Car"])
    assert np.isclose(res["Car"], 1.0, atol=1e-6)
    assert np.isclose(res["mAP"], 1.0, atol=1e-6)


def test_eval_kitti_3d_high_score_false_positive_halves_ap():
    # One GT; a non-overlapping FP scored ABOVE the true positive.
    frame_pred = {
        "boxes": np.array([box(100), box(10)]),     # FP (far away), then TP
        "scores": np.array([0.95, 0.5]),
        "labels": np.array(["Car", "Car"]),
    }
    frame_gt = {"boxes": np.array([box(10)]), "labels": np.array(["Car"])}
    res = eval_kitti_3d([frame_pred], [frame_gt], classes=["Car"])
    assert np.isclose(res["Car"], 0.5, atol=1e-6)


def test_eval_kitti_3d_missing_gt_limits_recall():
    # Two GT, only one detected -> recall caps at 0.5 -> AP 0.5.
    frame_pred = {"boxes": np.array([box(10)]), "scores": np.array([0.9]),
                  "labels": np.array(["Car"])}
    frame_gt = {"boxes": np.array([box(10), box(50)]), "labels": np.array(["Car", "Car"])}
    res = eval_kitti_3d([frame_pred], [frame_gt], classes=["Car"])
    assert np.isclose(res["Car"], 0.5, atol=1e-6)


def test_eval_kitti_3d_respects_iou_threshold():
    # Prediction overlaps GT only partially; below Car's 0.7 threshold -> AP 0.
    frame_pred = {"boxes": np.array([box(12.5)]), "scores": np.array([0.9]),
                  "labels": np.array(["Car"])}    # shifted 2.5m on a 4m box
    frame_gt = {"boxes": np.array([box(10)]), "labels": np.array(["Car"])}
    res = eval_kitti_3d([frame_pred], [frame_gt], classes=["Car"])
    assert res["Car"] == 0.0


# --------------------------------------------------------------------------- #
# Tracking MOTA / MOTP / IDSW
# --------------------------------------------------------------------------- #
def test_mot_perfect_sequence():
    gt = [{"ids": [1, 2], "boxes": np.array([box(0), box(30)])},
          {"ids": [1, 2], "boxes": np.array([box(1), box(31)])}]
    pred = [{"ids": [10, 20], "boxes": np.array([box(0), box(30)])},
            {"ids": [10, 20], "boxes": np.array([box(1), box(31)])}]
    m = eval_mot(gt, pred)
    assert np.isclose(m["MOTA"], 1.0) and m["num_switches"] == 0
    assert m["fp"] == 0 and m["fn"] == 0
    assert np.isclose(m["MOTP"], 1.0, atol=1e-9)


def test_mot_id_switch_counted():
    gt = [{"ids": [1], "boxes": np.array([box(0)])} for _ in range(3)]
    # same object, but the predicted track id changes on frame 3
    pred = [
        {"ids": [10], "boxes": np.array([box(0)])},
        {"ids": [10], "boxes": np.array([box(0)])},
        {"ids": [99], "boxes": np.array([box(0)])},
    ]
    m = eval_mot(gt, pred)
    assert m["num_switches"] == 1


def test_mot_fp_and_fn():
    gt = [{"ids": [1], "boxes": np.array([box(0)])},
          {"ids": [], "boxes": np.zeros((0, 7))}]
    pred = [{"ids": [], "boxes": np.zeros((0, 7))},          # missed -> FN
            {"ids": [7], "boxes": np.array([box(0)])}]        # ghost -> FP
    m = eval_mot(gt, pred)
    assert m["fn"] == 1 and m["fp"] == 1
    assert np.isclose(m["MOTA"], 1.0 - (1 + 1) / 1)           # = -1.0


# --------------------------------------------------------------------------- #
# Robustness transforms
# --------------------------------------------------------------------------- #
def test_drop_lidar_points_ratio():
    pts = np.random.default_rng(0).random((10000, 4))
    kept = drop_lidar_points(pts, 0.5, rng=np.random.default_rng(1))
    assert 0.45 * len(pts) < len(kept) < 0.55 * len(pts)
    assert kept.shape[1] == 4


def test_drop_lidar_points_extremes():
    pts = np.ones((100, 4))
    assert len(drop_lidar_points(pts, 0.0)) == 100
    assert len(drop_lidar_points(pts, 1.0)) == 0


def test_zero_camera_features():
    feats = np.random.default_rng(0).random((64, 7, 7))
    z = zero_camera_features(feats)
    assert z.shape == feats.shape and not np.any(z)


def test_add_lidar_noise_preserves_intensity():
    pts = np.ones((50, 4))
    noisy = add_lidar_noise(pts, sigma=0.1, rng=np.random.default_rng(0))
    assert noisy.shape == pts.shape
    assert np.allclose(noisy[:, 3], 1.0)                      # intensity untouched
    assert not np.allclose(noisy[:, :3], 1.0)                 # xyz jittered
