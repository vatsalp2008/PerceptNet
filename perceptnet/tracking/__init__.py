"""3D multi-object tracking (AB3DMOT: 3D Kalman filter + Hungarian association).

Pure NumPy / SciPy — safe to import on any machine, no torch required.
"""

from perceptnet.tracking.tracker import Track, Tracker

__all__ = ["Tracker", "Track"]
