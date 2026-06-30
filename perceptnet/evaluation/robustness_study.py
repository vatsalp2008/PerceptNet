"""Modality robustness study — degrade one sensor, measure detection impact.

The point of this study (and the bullet it supports) is to show *graceful
degradation*: fusion should lose far less accuracy under LiDAR corruption than a
LiDAR-only model. The transforms below are pure NumPy and unit-tested; the runner
orchestrates a (trained) model over a dataset under each scenario and is therefore
only meaningful once weights exist — it is structured here so the experiment is
one call on the GPU box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np


def drop_lidar_points(points: np.ndarray, drop_ratio: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Randomly remove a fraction of LiDAR points (simulates rain/fog/occlusion).

    Args:
        points: ``(N, C)`` point cloud.
        drop_ratio: fraction in ``[0, 1]`` of points to drop.
        rng: optional NumPy Generator for reproducibility.
    """
    if not 0.0 <= drop_ratio <= 1.0:
        raise ValueError("drop_ratio must be in [0, 1]")
    points = np.asarray(points)
    n = len(points)
    if n == 0 or drop_ratio == 0.0:
        return points
    rng = rng or np.random.default_rng()
    keep = rng.random(n) >= drop_ratio
    return points[keep]


def zero_camera_features(features: np.ndarray) -> np.ndarray:
    """Zero-fill image features to simulate a camera dropout (Module 4 contract)."""
    return np.zeros_like(np.asarray(features))


def add_lidar_noise(points: np.ndarray, sigma: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Add Gaussian jitter to point xyz coordinates (intensity untouched)."""
    points = np.asarray(points, dtype=np.float64).copy()
    if points.size == 0 or sigma == 0.0:
        return points
    rng = rng or np.random.default_rng()
    points[:, :3] += rng.normal(0.0, sigma, size=points[:, :3].shape)
    return points


@dataclass
class Scenario:
    """A named perturbation applied to a frame's inputs before inference."""

    name: str
    transform: Callable[[dict], dict]


def default_scenarios(seed: int = 0) -> List[Scenario]:
    """The standard robustness scenarios from the project spec."""
    rng = np.random.default_rng(seed)

    def identity(frame):
        return frame

    def lidar_drop(ratio):
        def fn(frame):
            out = dict(frame)
            out["points"] = drop_lidar_points(frame["points"], ratio, rng)
            return out
        return fn

    def camera_dropout(frame):
        out = dict(frame)
        out["image"] = None          # signals the model to zero-fill image features
        out["camera_available"] = False
        return out

    return [
        Scenario("full_fusion", identity),
        Scenario("camera_dropout", camera_dropout),
        Scenario("lidar_drop_50", lidar_drop(0.5)),
        Scenario("lidar_drop_90", lidar_drop(0.9)),
    ]


def run_robustness_study(
    model,
    dataset,
    scenarios: Optional[List[Scenario]] = None,
    evaluate_fn: Optional[Callable] = None,
) -> Dict[str, dict]:
    """Run ``model`` over ``dataset`` under each scenario and evaluate.

    Requires a trained model and an ``evaluate_fn`` mapping (predictions, gts) -> metrics
    (e.g. :func:`perceptnet.evaluation.kitti_eval.eval_kitti_3d`). Returns
    ``{scenario_name: metrics}``. Intended to run on the GPU box.
    """
    if evaluate_fn is None:
        raise ValueError("provide evaluate_fn, e.g. functools.partial(eval_kitti_3d, classes=[...])")
    scenarios = scenarios or default_scenarios()

    results: Dict[str, dict] = {}
    for scenario in scenarios:
        preds, gts = [], []
        for frame in dataset:
            perturbed = scenario.transform(frame)
            preds.append(model.predict(perturbed))
            gts.append({"boxes": frame["boxes_3d"], "labels": frame["labels"]})
        results[scenario.name] = evaluate_fn(preds, gts)
    return results
