"""Pure (ROS-free) conversion helpers for the ROS 2 node.

Keeping the numpy -> message-shaping logic here, free of any ``rclpy``/``vision_msgs``
import, means it runs and is unit-tested on macOS; the node module only has to drop
these dicts into real message objects. (Pitfall: vision_msgs is unavailable
off-platform — don't let it leak into testable code.)
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


def quaternion_from_yaw(yaw: float) -> Dict[str, float]:
    """Yaw about +z -> a quaternion dict ``{x, y, z, w}``."""
    return {"x": 0.0, "y": 0.0, "z": math.sin(yaw / 2.0), "w": math.cos(yaw / 2.0)}


def track_to_dict(track) -> Dict:
    """Shape a :class:`~perceptnet.tracking.tracker.TrackState` into a plain dict
    mirroring a ``vision_msgs/Detection3D`` (center, size, orientation, velocity, id)."""
    box = np.asarray(track.box, dtype=float)
    return {
        "id": int(track.id),
        "label": int(track.label),
        "score": float(track.score),
        "center": box[:3].tolist(),
        "size": box[3:6].tolist(),
        "orientation": quaternion_from_yaw(float(box[6])),
        "velocity": np.asarray(track.velocity, dtype=float).tolist(),
    }


def tracks_to_dicts(tracks) -> List[Dict]:
    return [track_to_dict(t) for t in tracks]
