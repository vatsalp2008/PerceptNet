"""Unit tests for the AB3DMOT tracker: Kalman filter, association, ID lifecycle."""

import numpy as np

from perceptnet.tracking.hungarian import associate
from perceptnet.tracking.kalman_filter_3d import KalmanFilter3D
from perceptnet.tracking.tracker import Tracker, count_id_switches


def make_box(x, y=0.0, z=0.0, dx=4.0, dy=2.0, dz=1.5, heading=0.0):
    return np.array([x, y, z, dx, dy, dz, heading], dtype=float)


# --------------------------------------------------------------------------- #
# Kalman filter
# --------------------------------------------------------------------------- #
def test_kf_recovers_constant_velocity():
    kf = KalmanFilter3D(make_box(0.0))
    vx = 0.5
    for step in range(1, 15):
        kf.predict()
        kf.update(make_box(vx * step))
    assert np.isclose(kf.velocity[0], vx, atol=0.05)
    assert abs(kf.velocity[1]) < 0.05 and abs(kf.velocity[2]) < 0.05


def test_kf_tracks_position():
    kf = KalmanFilter3D(make_box(0.0))
    for step in range(1, 8):
        kf.predict()
        kf.update(make_box(step * 1.0))
    assert np.isclose(kf.box[0], 7.0, atol=0.3)


def test_kf_heading_no_180_snap():
    # Heading near +pi then a measurement near -pi (same orientation, wrapped).
    kf = KalmanFilter3D(make_box(0.0, heading=np.pi - 0.05))
    kf.predict()
    kf.update(make_box(0.0, heading=-np.pi + 0.05))
    # Should stay near pi/-pi (same physical heading), not jump to ~0.
    assert min(abs(kf.box[6] - np.pi), abs(kf.box[6] + np.pi)) < 0.2


# --------------------------------------------------------------------------- #
# Hungarian association
# --------------------------------------------------------------------------- #
def test_associate_basic_match():
    iou = np.array([[0.9, 0.0], [0.0, 0.8]])
    matches, ud, ut = associate(iou, iou_threshold=0.1)
    assert sorted(matches) == [(0, 0), (1, 1)]
    assert ud == [] and ut == []


def test_associate_threshold_rejects_low_iou():
    iou = np.array([[0.05]])
    matches, ud, ut = associate(iou, iou_threshold=0.1)
    assert matches == [] and ud == [0] and ut == [0]


def test_associate_empty():
    matches, ud, ut = associate(np.zeros((0, 3)), 0.1)
    assert matches == [] and ud == [] and ut == [0, 1, 2]


# --------------------------------------------------------------------------- #
# Tracker lifecycle
# --------------------------------------------------------------------------- #
def test_single_object_keeps_stable_id():
    tracker = Tracker(iou_threshold=0.1, max_age=3, min_hits=1)
    ids = []
    for step in range(6):
        out = tracker.update(make_box(step * 0.5)[None, :])
        assert len(out) == 1
        ids.append(out[0].id)
    assert len(set(ids)) == 1            # never switched


def test_two_objects_distinct_ids():
    tracker = Tracker(iou_threshold=0.1, max_age=3, min_hits=1)
    seen = set()
    for step in range(5):
        boxes = np.stack([make_box(step * 0.5), make_box(step * 0.5 + 30.0)])
        out = tracker.update(boxes)
        assert len(out) == 2
        seen.update(o.id for o in out)
    assert len(seen) == 2                 # exactly two identities total


def test_new_object_gets_new_id():
    tracker = Tracker(iou_threshold=0.1, max_age=3, min_hits=1)
    out1 = tracker.update(make_box(0.0)[None, :])
    # Add a second, well-separated object on frame 2.
    out2 = tracker.update(np.stack([make_box(0.5), make_box(40.0)]))
    ids2 = {o.id for o in out2}
    assert out1[0].id in ids2 and len(ids2) == 2


def test_disappeared_track_is_culled():
    tracker = Tracker(iou_threshold=0.1, max_age=3, min_hits=1)
    for step in range(4):
        tracker.update(make_box(step * 0.5)[None, :])
    assert len(tracker.active_ids) == 1
    # Stop detecting; after max_age missed frames the track is removed.
    for _ in range(5):
        tracker.update(np.zeros((0, 7)))
    assert tracker.active_ids == []


def test_velocity_sign_is_correct():
    tracker = Tracker(min_hits=1)
    out = None
    for step in range(10):
        out = tracker.update(make_box(step * 1.0)[None, :])
    assert out[0].velocity[0] > 0.5      # moving in +x


def test_count_id_switches():
    # gt object 0 assigned track 5 then 5 then 7 (one switch); object 1 stable.
    frames = [{0: 5, 1: 9}, {0: 5, 1: 9}, {0: 7, 1: 9}]
    assert count_id_switches(frames) == 1
