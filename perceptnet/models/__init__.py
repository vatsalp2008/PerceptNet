"""Neural network branches (torch-backed): camera (YOLO v8 + FPN), LiDAR
(PointPillars), the ROI fusion head, and the unified PerceptNet wrapper.

These modules require torch and are imported explicitly by callers (not re-exported
here) so that ``import perceptnet`` stays light on CPU-only machines.
"""
