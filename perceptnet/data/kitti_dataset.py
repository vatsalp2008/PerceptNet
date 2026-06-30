"""KITTI 3D Object Detection dataset loader.

Returns, per frame, the synchronized camera image + LiDAR cloud, ground-truth 3D
boxes in the **LiDAR frame** (canonical ``[x, y, z, dx, dy, dz, heading]``), integer
class labels, and the frame's :class:`~perceptnet.data.calibration.Calibration`.

Directory layout (see README "Dataset setup")::

    root/<split>/{image_2, velodyne, label_2, calib}/<frame_id>.{png,bin,txt,txt}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from perceptnet.data.calibration import Calibration, boxes_camera_to_lidar

try:  # torch is required to use the Dataset, but importing this module must not need it
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only in torch-less envs
    _HAS_TORCH = False

    class _TorchDataset:  # minimal shim so the class can still be defined/imported
        pass


DEFAULT_CLASSES = ("Car", "Pedestrian", "Cyclist")


@dataclass
class KittiLabel:
    """One parsed KITTI ``label_2`` row."""

    type: str
    truncation: float
    occlusion: int
    alpha: float
    bbox2d: np.ndarray        # (4,) x1, y1, x2, y2 in the image
    box_cam: np.ndarray       # (7,) [x, y, z, l, w, h, rotation_y] (camera frame)

    @property
    def height_px(self) -> float:
        return float(self.bbox2d[3] - self.bbox2d[1])


def load_kitti_labels(path) -> List[KittiLabel]:
    """Parse a KITTI ``label_2/<id>.txt`` file."""
    labels: List[KittiLabel] = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 15:
            continue
        typ = parts[0]
        if typ == "DontCare":
            continue
        vals = [float(p) for p in parts[1:15]]
        truncation, occlusion, alpha = vals[0], int(vals[1]), vals[2]
        bbox2d = np.array(vals[3:7])
        h, w, l = vals[7], vals[8], vals[9]
        x, y, z = vals[10], vals[11], vals[12]
        rotation_y = vals[13]
        labels.append(
            KittiLabel(
                type=typ,
                truncation=truncation,
                occlusion=occlusion,
                alpha=alpha,
                bbox2d=bbox2d,
                box_cam=np.array([x, y, z, l, w, h, rotation_y]),
            )
        )
    return labels


def load_velodyne(path) -> np.ndarray:
    """Load a KITTI ``.bin`` Velodyne scan as ``(N, 4)`` float32 (x, y, z, intensity)."""
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)


class KITTIDataset(_TorchDataset):
    """KITTI 3D detection dataset (torch ``Dataset``).

    Args:
        root: dataset root containing the ``<split>`` directory.
        split: ``"training"`` or ``"testing"``.
        classes: class names kept (others are dropped); label index is the position
            in this tuple.
        augment: apply LiDAR augmentation (training only).
        frame_ids: explicit list of frame ids; if ``None`` they are discovered from
            the ``velodyne`` directory.
    """

    def __init__(
        self,
        root,
        split: str = "training",
        classes: Sequence[str] = DEFAULT_CLASSES,
        augment: bool = False,
        frame_ids: Optional[Sequence[str]] = None,
    ):
        if not _HAS_TORCH:
            raise ImportError("KITTIDataset requires torch; install requirements.txt")
        self.root = Path(root) / split
        self.split = split
        self.classes = tuple(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.image_dir = self.root / "image_2"
        self.velodyne_dir = self.root / "velodyne"
        self.label_dir = self.root / "label_2"
        self.calib_dir = self.root / "calib"

        if frame_ids is not None:
            self.frame_ids = list(frame_ids)
        else:
            self.frame_ids = sorted(p.stem for p in self.velodyne_dir.glob("*.bin"))

        self.augment = augment
        self._augmenter = None
        if augment:
            from perceptnet.data.augmentation import AugmentationPipeline

            self._augmenter = AugmentationPipeline()

    def __len__(self) -> int:
        return len(self.frame_ids)

    def _load_image(self, frame_id: str):
        import cv2

        path = self.image_dir / f"{frame_id}.png"
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"could not read image {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # CHW float tensor in [0, 1]
        return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    def get_labels_lidar(self, frame_id: str, calib: Calibration):
        """Return ``(boxes_lidar (M,7), labels (M,), raw_labels)`` for a frame."""
        raw = [lbl for lbl in load_kitti_labels(self.label_dir / f"{frame_id}.txt")
               if lbl.type in self.class_to_idx]
        if not raw:
            return np.zeros((0, 7)), np.zeros((0,), dtype=np.int64), []
        boxes_cam = np.stack([lbl.box_cam for lbl in raw])
        boxes_lidar = boxes_camera_to_lidar(boxes_cam, calib)
        labels = np.array([self.class_to_idx[lbl.type] for lbl in raw], dtype=np.int64)
        return boxes_lidar, labels, raw

    def __getitem__(self, idx: int) -> Dict:
        frame_id = self.frame_ids[idx]
        calib = Calibration.from_file(self.calib_dir / f"{frame_id}.txt")
        points = load_velodyne(self.velodyne_dir / f"{frame_id}.bin")
        boxes_lidar, labels, _ = self.get_labels_lidar(frame_id, calib)

        if self.augment and self._augmenter is not None:
            pts_aug, boxes_aug = self._augmenter(points[:, :4].astype(np.float64), boxes_lidar)
            points = pts_aug.astype(np.float32)
            boxes_lidar = boxes_aug

        sample = {
            "frame_id": frame_id,
            "image": self._load_image(frame_id),
            "points": torch.from_numpy(np.ascontiguousarray(points)).float(),
            "boxes_3d": torch.from_numpy(boxes_lidar).float(),
            "labels": torch.from_numpy(labels).long(),
            "calib": calib,
        }
        return sample


def kitti_collate_fn(batch: List[Dict]) -> Dict:
    """Collate variable-length detection samples.

    Images are stacked (assumed equal size); points / boxes / labels / calibs stay
    as lists because their lengths differ per frame.
    """
    return {
        "frame_id": [b["frame_id"] for b in batch],
        "image": torch.stack([b["image"] for b in batch]) if _HAS_TORCH else [b["image"] for b in batch],
        "points": [b["points"] for b in batch],
        "boxes_3d": [b["boxes_3d"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "calib": [b["calib"] for b in batch],
    }
