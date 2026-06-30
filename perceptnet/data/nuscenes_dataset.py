"""nuScenes loader — secondary dataset for the generalization / domain-gap study.

Deliberately a thin stub: the primary benchmark is KITTI, and nuScenes (6 cameras +
LiDAR + RADAR, ~300 GB, its own devkit and coordinate conventions) is a stretch goal
(README stretch goals; ADR notes). The interface mirrors :class:`KITTIDataset` so the
fusion model and evaluation can run on it once the ``nuscenes-devkit`` data is staged.

To implement: load a sample via ``nuscenes-devkit``, transform boxes from the global
frame into the ego/LiDAR frame, and return the same sample dict keys as KITTIDataset.
"""

from __future__ import annotations

from typing import Dict, Sequence

# nuScenes' 10 detection classes.
NUSCENES_CLASSES = (
    "car", "truck", "bus", "trailer", "construction_vehicle",
    "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier",
)


class NuScenesDataset:
    """Placeholder with the KITTIDataset-compatible surface (not yet implemented)."""

    def __init__(self, root, version: str = "v1.0-mini", classes: Sequence[str] = NUSCENES_CLASSES):
        self.root = root
        self.version = version
        self.classes = tuple(classes)
        raise NotImplementedError(
            "NuScenesDataset is a stretch goal. Install nuscenes-devkit, stage the data, "
            "and map samples to the KITTIDataset sample-dict contract "
            "(image, points, boxes_3d in LiDAR frame, labels, calib)."
        )

    def __len__(self) -> int:  # pragma: no cover
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict:  # pragma: no cover
        raise NotImplementedError
