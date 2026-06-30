"""PerceptNet — multi-modal camera + LiDAR fusion perception pipeline.

The top-level package intentionally imports nothing heavy. Submodules are imported
explicitly by the caller (e.g. ``from perceptnet.tracking.tracker import Tracker``)
so that ``import perceptnet`` succeeds on a CPU-only machine without torch/open3d/etc.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
