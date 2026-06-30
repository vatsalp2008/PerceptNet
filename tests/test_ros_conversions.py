"""Tests for the ROS-free conversion helpers (no rclpy needed; run on any platform)."""

import numpy as np

from perceptnet.ros2.conversions import quaternion_from_yaw, track_to_dict, tracks_to_dicts
from perceptnet.tracking.tracker import TrackState


def test_quaternion_from_zero_yaw_is_identity():
    q = quaternion_from_yaw(0.0)
    assert q == {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}


def test_quaternion_from_yaw_pi():
    q = quaternion_from_yaw(np.pi)
    assert np.isclose(q["z"], 1.0) and np.isclose(q["w"], 0.0, atol=1e-9)


def test_track_to_dict_fields():
    ts = TrackState(
        id=7,
        box=np.array([1.0, 2.0, 3.0, 4.0, 1.6, 1.5, 0.0]),
        velocity=np.array([0.5, 0.0, 0.0]),
        score=0.9,
        label=1,
        age=5,
        hits=5,
        time_since_update=0,
    )
    d = track_to_dict(ts)
    assert d["id"] == 7 and d["label"] == 1
    assert d["center"] == [1.0, 2.0, 3.0]
    assert d["size"] == [4.0, 1.6, 1.5]
    assert d["velocity"] == [0.5, 0.0, 0.0]
    assert d["orientation"] == {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}


def test_tracks_to_dicts_list():
    ts = TrackState(id=0, box=np.zeros(7), velocity=np.zeros(3), score=1.0, label=0,
                    age=1, hits=1, time_since_update=0)
    out = tracks_to_dicts([ts, ts])
    assert len(out) == 2 and all("center" in d for d in out)
