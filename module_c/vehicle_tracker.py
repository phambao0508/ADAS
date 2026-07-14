from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

DEFAULT_MIN_HITS      = 3
DEFAULT_MAX_AGE       = 8
DEFAULT_IOU_THRESHOLD = 0.50

_PROCESS_NOISE = np.array([2.0, 2.0, 4.0, 4.0, 12.0, 12.0, 16.0, 16.0])
_MEASUREMENT_NOISE = np.array([25.0, 25.0, 36.0, 36.0])

TENTATIVE = "TENTATIVE"
CONFIRMED = "CONFIRMED"

@dataclass
class Track:

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int
    hits: int = 1
    time_since_update: int = 0
    state: str = TENTATIVE
    zone: str = "UNKNOWN"

    distance_m: Optional[float] = None
    ttc_s:      Optional[float] = None

    kalman_state: Optional[np.ndarray] = field(default=None, repr=False)
    kalman_cov: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) * 0.5

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    def as_box(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

def _iou(a: Tuple[float, float, float, float],
         b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0

class VehicleTracker:

    def __init__(
        self,
        min_hits: int = DEFAULT_MIN_HITS,
        max_age: int = DEFAULT_MAX_AGE,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ):
        self.min_hits = min_hits
        self.max_age = max_age
        self.iou_threshold = iou_threshold

        self._tracks: Dict[int, Track] = {}
        self._next_id = count(1)

        self._seen_in_zone: Dict[str, Set[int]] = {
            "LEFT": set(), "EGO": set(), "RIGHT": set(), "OUT": set(),
        }

    def update(
        self,
        detections: Iterable[Tuple[float, float, float, float, float, int]],
    ) -> List[Track]:

        det_list = [tuple(d) for d in detections]

        for track in self._tracks.values():
            self._predict(track)

        unmatched_dets, matched = self._match(det_list)

        for track_id, det in matched.items():
            x1, y1, x2, y2, conf, cls_id = det
            t = self._tracks[track_id]
            self._correct(t, (x1, y1, x2, y2))
            t.conf = float(conf)
            t.cls_id = int(cls_id)
            t.hits += 1
            t.time_since_update = 0
            if t.state == TENTATIVE and t.hits >= self.min_hits:
                t.state = CONFIRMED

        to_delete = []
        for track_id, t in self._tracks.items():
            if track_id in matched:
                continue
            t.time_since_update += 1
            if t.state == TENTATIVE:

                to_delete.append(track_id)
            elif t.time_since_update > self.max_age:
                to_delete.append(track_id)
        for track_id in to_delete:
            self._tracks.pop(track_id, None)

        for det in unmatched_dets:
            x1, y1, x2, y2, conf, cls_id = det
            track_id = next(self._next_id)
            track = Track(
                track_id=track_id,
                x1=x1, y1=y1, x2=x2, y2=y2,
                conf=float(conf),
                cls_id=int(cls_id),
                hits=1,
                time_since_update=0,
                state=CONFIRMED if self.min_hits <= 1 else TENTATIVE,
            )
            self._initialise_kalman(track)
            self._tracks[track_id] = track

        return self.confirmed_tracks()

    @staticmethod
    def _box_to_measurement(
        box: Tuple[float, float, float, float]
    ) -> np.ndarray:

        x1, y1, x2, y2 = box
        return np.array([
            (x1 + x2) * 0.5,
            (y1 + y2) * 0.5,
            max(1.0, x2 - x1),
            max(1.0, y2 - y1),
        ], dtype=float)

    @staticmethod
    def _write_box_from_state(track: Track) -> None:

        assert track.kalman_state is not None
        cx, cy, w, h = track.kalman_state[:4]
        w, h = max(1.0, w), max(1.0, h)
        track.kalman_state[2:4] = (w, h)
        track.x1, track.y1 = cx - w * 0.5, cy - h * 0.5
        track.x2, track.y2 = cx + w * 0.5, cy + h * 0.5

    def _initialise_kalman(self, track: Track) -> None:
        measurement = self._box_to_measurement(track.as_box())
        track.kalman_state = np.concatenate((measurement, np.zeros(4)))

        track.kalman_cov = np.diag([25.0, 25.0, 36.0, 36.0,
                                    400.0, 400.0, 625.0, 625.0])

    def _predict(self, track: Track) -> None:

        if track.kalman_state is None or track.kalman_cov is None:
            self._initialise_kalman(track)

        transition = np.eye(8)
        transition[0, 4] = transition[1, 5] = 1.0
        transition[2, 6] = transition[3, 7] = 1.0
        process_cov = np.diag(_PROCESS_NOISE)
        track.kalman_state = transition @ track.kalman_state
        track.kalman_cov = transition @ track.kalman_cov @ transition.T + process_cov
        self._write_box_from_state(track)

    def _correct(self, track: Track, box: Tuple[float, float, float, float]) -> None:

        if track.kalman_state is None or track.kalman_cov is None:
            self._initialise_kalman(track)

        measurement = self._box_to_measurement(box)
        observation = np.zeros((4, 8))
        observation[0, 0] = observation[1, 1] = 1.0
        observation[2, 2] = observation[3, 3] = 1.0
        innovation = measurement - observation @ track.kalman_state
        innovation_cov = observation @ track.kalman_cov @ observation.T + np.diag(_MEASUREMENT_NOISE)
        gain = track.kalman_cov @ observation.T @ np.linalg.pinv(innovation_cov)
        track.kalman_state = track.kalman_state + gain @ innovation
        identity = np.eye(8)

        residual = identity - gain @ observation
        measurement_cov = np.diag(_MEASUREMENT_NOISE)
        track.kalman_cov = residual @ track.kalman_cov @ residual.T + gain @ measurement_cov @ gain.T
        self._write_box_from_state(track)

    def _match(
        self,
        detections: List[Tuple[float, float, float, float, float, int]],
    ) -> Tuple[List[Tuple[float, float, float, float, float, int]], Dict[int, tuple]]:

        if not self._tracks or not detections:
            return list(detections), {}

        track_items = list(self._tracks.items())

        candidates: List[Tuple[float, int, int]] = []
        for di, det in enumerate(detections):
            d_box = det[:4]
            for ti, (track_id, track) in enumerate(track_items):
                iou = _iou(d_box, track.as_box())
                if iou >= self.iou_threshold:
                    candidates.append((iou, di, ti))

        candidates.sort(reverse=True)

        used_dets: Set[int] = set()
        used_tracks: Set[int] = set()
        matched: Dict[int, tuple] = {}

        for iou, di, ti in candidates:
            if di in used_dets or ti in used_tracks:
                continue
            track_id = track_items[ti][0]
            matched[track_id] = detections[di]
            used_dets.add(di)
            used_tracks.add(ti)

        unmatched = [d for i, d in enumerate(detections) if i not in used_dets]
        return unmatched, matched

    def confirmed_tracks(self) -> List[Track]:

        return [t for t in self._tracks.values() if t.state == CONFIRMED]

    def all_tracks(self) -> List[Track]:

        return list(self._tracks.values())

    def record_zones(self, zone_assignments: Dict[int, str]) -> None:

        for track_id, zone in zone_assignments.items():
            if track_id not in self._tracks:
                continue
            self._tracks[track_id].zone = zone
            if zone in self._seen_in_zone:
                self._seen_in_zone[zone].add(track_id)

    def cumulative_seen(self, zone: str) -> int:

        return len(self._seen_in_zone.get(zone, ()))

    def total_seen(self) -> int:

        return len(
            self._seen_in_zone["LEFT"]
            | self._seen_in_zone["EGO"]
            | self._seen_in_zone["RIGHT"]
        )

    def reset(self) -> None:

        self._tracks.clear()
        self._next_id = count(1)
        for s in self._seen_in_zone.values():
            s.clear()
