"""Data loading and sensor calibration.

``calibration`` is pure NumPy and safe to import anywhere. The dataset modules
(``kitti_dataset``, ``nuscenes_dataset``) and ``augmentation`` depend on torch and
are imported explicitly by callers, not re-exported here, to keep ``import perceptnet``
light on CPU-only machines.
"""

from perceptnet.data.calibration import Calibration

__all__ = ["Calibration"]
