"""AB3DMOT-style 3D multi-object tracker.

Per frame:
  1. **Predict** every active track forward with its constant-velocity Kalman filter.
  2. **Associate** detections to predicted tracks with the Hungarian algorithm on a
     3D-IoU cost matrix (gated by ``iou_threshold``).
  3. **Update** matched tracks; spawn new tracks for unmatched detections; age
     unmatched tracks and delete them once ``time_since_update`` exceeds ``max_age``.

Box format everywhere is the canonical ``[x, y, z, dx, dy, dz, heading]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from perceptnet.geometry.iou import iou_3d
from perceptnet.tracking.hungarian import associate
from perceptnet.tracking.kalman_filter_3d import KalmanFilter3D


@dataclass
class TrackState:
    """A snapshot of one tracked object, returned to the caller each frame."""

    id: int
    box: np.ndarray            # (7,) [x, y, z, dx, dy, dz, heading]
    velocity: np.ndarray       # (3,) [vx, vy, vz] per frame
    score: float
    label: int
    age: int                   # frames since birth
    hits: int                  # number of detections associated so far
    time_since_update: int     # frames since the last associated detection


@dataclass
class Track:
    """Internal per-object track wrapping a Kalman filter."""

    kf: KalmanFilter3D
    id: int
    label: int
    score: float
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    _last_prediction: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_detection(cls, box: np.ndarray, score: float, label: int, track_id: int) -> "Track":
        return cls(kf=KalmanFilter3D(box), id=track_id, label=int(label), score=float(score))

    def predict(self) -> np.ndarray:
        self._last_prediction = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self._last_prediction

    def update(self, box: np.ndarray, score: float, label: int) -> None:
        self.kf.update(box)
        self.hits += 1
        self.time_since_update = 0
        self.score = float(score)
        self.label = int(label)

    def to_state(self) -> TrackState:
        return TrackState(
            id=self.id,
            box=self.kf.box,
            velocity=self.kf.velocity,
            score=self.score,
            label=self.label,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
        )


class Tracker:
    """AB3DMOT tracker.

    Args:
        iou_threshold: minimum 3D IoU for a detection-track match.
        max_age: a track is deleted once ``time_since_update`` exceeds this.
        min_hits: a track is only *reported* once it has this many hits (it still
            exists and is matched before then). Set 0 to report immediately.
    """

    def __init__(self, iou_threshold: float = 0.1, max_age: int = 3, min_hits: int = 1):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[Track] = []
        self._next_id = 0
        self.frame_count = 0

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 0
        self.frame_count = 0

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def update(
        self,
        boxes: np.ndarray,
        scores: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> List[TrackState]:
        """Advance the tracker one frame with the current detections.

        Args:
            boxes: ``(N, 7)`` canonical boxes (may be empty).
            scores: ``(N,)`` confidences (defaults to ones).
            labels: ``(N,)`` integer class ids (defaults to zeros).

        Returns:
            Reported tracks for this frame (those with ``hits >= min_hits`` and a
            fresh update, plus recently-seen tracks during the warm-up window).
        """
        self.frame_count += 1
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
        n = len(boxes)
        scores = np.ones(n) if scores is None else np.asarray(scores, dtype=np.float64).reshape(n)
        labels = np.zeros(n, dtype=int) if labels is None else np.asarray(labels).reshape(n).astype(int)

        # 1. predict
        predicted = np.array([t.predict() for t in self.tracks]).reshape(-1, 7)

        # 2. associate
        if n and len(self.tracks):
            iou = iou_3d(boxes, predicted)
        else:
            iou = np.zeros((n, len(self.tracks)))
        matches, unmatched_det, unmatched_trk = associate(iou, self.iou_threshold)

        # 3a. update matched
        for d, t in matches:
            self.tracks[t].update(boxes[d], scores[d], labels[d])

        # 3b. spawn tracks for unmatched detections
        for d in unmatched_det:
            self.tracks.append(
                Track.from_detection(boxes[d], scores[d], labels[d], self._new_id())
            )

        # 3c. cull stale tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 4. report
        reported: List[TrackState] = []
        for t in self.tracks:
            fresh = t.time_since_update == 0
            warming = self.frame_count <= self.min_hits
            if fresh and (t.hits >= self.min_hits or warming):
                reported.append(t.to_state())
        return reported

    @property
    def active_ids(self) -> List[int]:
        return [t.id for t in self.tracks]


def count_id_switches(gt_id_per_frame: List[Dict[int, int]]) -> int:
    """Count ID switches given, per frame, a mapping ``{gt_object: assigned_track_id}``.

    A switch is counted when a ground-truth object's assigned track id changes
    between consecutive frames in which it appears. Useful for the tracking
    evaluation and tests.
    """
    switches = 0
    last_assignment: Dict[int, int] = {}
    for frame in gt_id_per_frame:
        for gt_obj, trk_id in frame.items():
            if gt_obj in last_assignment and last_assignment[gt_obj] != trk_id:
                switches += 1
            last_assignment[gt_obj] = trk_id
    return switches
