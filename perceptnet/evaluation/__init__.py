"""Evaluation: KITTI 3D mAP, tracking MOTA/MOTP, and the modality robustness study.

Pure NumPy (shares ``perceptnet.geometry.iou``) — safe to import without torch.
"""

from perceptnet.evaluation.kitti_eval import average_precision, eval_kitti_3d
from perceptnet.evaluation.tracking_eval import eval_mot

__all__ = ["eval_kitti_3d", "average_precision", "eval_mot"]
