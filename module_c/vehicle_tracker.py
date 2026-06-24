"""
Module C — Vehicle Tracker (IoU-based, dependency-free)
========================================================
Assigns persistent IDs to YOLO vehicle detections across frames so that
counting and HUD display stay stable when YOLO confidence wobbles.

Why this exists
---------------
Without a tracker, ``GuidancePipeline._count_vehicles_per_zone`` simply
counts raw boxes per frame. A car detected at 0.36 conf one frame and 0.31
the next disappears and reappears, the LEFT-lane count flickers 1 → 0 → 1
→ 0, and the HUD telemetry feels jittery and unreliable.

This module:

  * matches new detections to existing tracks via IoU (greedy, O(N·M))
  * confirms a track only after it has been seen ``min_hits`` consecutive
    frames — filters one-frame ghost detections
  * keeps a confirmed track alive for ``max_age`` missed frames before
    deleting it — bridges short YOLO drop-outs without flicker
  * exposes both the **live** confirmed tracks and the **cumulative**
    set of all unique IDs ever seen per zone (for "vehicles seen so far")

Design choices
--------------
  * No Kalman filter. The boxes update by replacing position with the new
    detection, and missed-frame predictions just hold the last known box.
    For a dashcam at 30 fps, IoU matching with a 1-frame search radius is
    plenty.
  * No external dependencies (no scipy, no filterpy). Greedy IoU matching
    is fine for the typical N ≤ 20 vehicles/frame load.

Public API
----------
    tracker = VehicleTracker(min_hits=3, max_age=8, iou_threshold=0.3)
    confirmed = tracker.update(detections, frame_w, frame_h)
        # detections: list[(x1, y1, x2, y2, conf, cls_id)]
        # confirmed:  list[Track]  — only tracks in CONFIRMED state

    # Per-zone cumulative seen counts (unique IDs that ever entered a zone):
    tracker.cumulative_seen("LEFT")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Dict, Iterable, List, Optional, Set, Tuple

# ── Tunables ─────────────────────────────────────────────────────────────────
DEFAULT_MIN_HITS      = 3      # frames seen before a track becomes CONFIRMED
DEFAULT_MAX_AGE       = 8      # frames missed before a CONFIRMED track is removed
DEFAULT_IOU_THRESHOLD = 0.50   # min IoU to associate a detection with a track

# Track states
TENTATIVE = "TENTATIVE"
CONFIRMED = "CONFIRMED"


@dataclass
class Track:
    """A single tracked vehicle."""
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
    zone: str = "UNKNOWN"   # last-seen zone (LEFT/EGO/RIGHT/OUT) — set externally

    # Populated by DistanceTTCEstimator each frame (None if unavailable).
    distance_m: Optional[float] = None
    ttc_s:      Optional[float] = None

    # ── Convenience views ────────────────────────────────────────────────────
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


# ── IoU helper ───────────────────────────────────────────────────────────────
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


# ── Tracker ──────────────────────────────────────────────────────────────────
class VehicleTracker:
    """
    Greedy-IoU multi-object tracker tailored to short-lived dashcam tracks.

    Parameters
    ----------
    min_hits : int
        Frames a track must be matched before it is reported as CONFIRMED.
    max_age : int
        Frames a CONFIRMED track survives without a match before deletion.
    iou_threshold : float
        Minimum IoU between a detection and an existing track for a match.
    """

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

        # Cumulative unique IDs ever observed in each zone (for "seen so far"
        # counters in the HUD). Populated externally via record_zones().
        self._seen_in_zone: Dict[str, Set[int]] = {
            "LEFT": set(), "EGO": set(), "RIGHT": set(), "OUT": set(),
        }

    # ─────────────────────────────────────────────────────────────────────────
    def update(
        self,
        detections: Iterable[Tuple[float, float, float, float, float, int]],
    ) -> List[Track]:
        """
        Advance the tracker by one frame.

        Parameters
        ----------
        detections : iterable of (x1, y1, x2, y2, conf, cls_id)
            Vehicle-class detections from YOLO for the current frame.

        Returns
        -------
        list[Track]
            Only CONFIRMED tracks for this frame (tentative tracks stay
            internal — caller should not show them in the HUD yet).
        """
        det_list = [tuple(d) for d in detections]

        # 1. Match detections to existing tracks via greedy max-IoU.
        unmatched_dets, matched = self._match(det_list)

        # 2. Update matched tracks with the new detection box.
        for track_id, det in matched.items():
            x1, y1, x2, y2, conf, cls_id = det
            t = self._tracks[track_id]
            t.x1, t.y1, t.x2, t.y2 = x1, y1, x2, y2
            t.conf = float(conf)
            t.cls_id = int(cls_id)
            t.hits += 1
            t.time_since_update = 0
            if t.state == TENTATIVE and t.hits >= self.min_hits:
                t.state = CONFIRMED

        # 3. Age unmatched tracks; delete those past max_age.
        to_delete = []
        for track_id, t in self._tracks.items():
            if track_id in matched:
                continue
            t.time_since_update += 1
            if t.state == TENTATIVE:
                # Tentative tracks die immediately on a miss.
                to_delete.append(track_id)
            elif t.time_since_update > self.max_age:
                to_delete.append(track_id)
        for track_id in to_delete:
            self._tracks.pop(track_id, None)

        # 4. Create new tentative tracks for unmatched detections.
        for det in unmatched_dets:
            x1, y1, x2, y2, conf, cls_id = det
            track_id = next(self._next_id)
            self._tracks[track_id] = Track(
                track_id=track_id,
                x1=x1, y1=y1, x2=x2, y2=y2,
                conf=float(conf),
                cls_id=int(cls_id),
                hits=1,
                time_since_update=0,
                state=CONFIRMED if self.min_hits <= 1 else TENTATIVE,
            )

        return self.confirmed_tracks()

    # ─────────────────────────────────────────────────────────────────────────
    def _match(
        self,
        detections: List[Tuple[float, float, float, float, float, int]],
    ) -> Tuple[List[Tuple[float, float, float, float, float, int]], Dict[int, tuple]]:
        """
        Greedy IoU matching. Returns (unmatched_detections, {track_id: detection}).
        """
        if not self._tracks or not detections:
            return list(detections), {}

        track_items = list(self._tracks.items())   # [(id, Track)]

        # Build an IoU matrix: pairs[i][j] = IoU(detection_i, track_j)
        candidates: List[Tuple[float, int, int]] = []
        for di, det in enumerate(detections):
            d_box = det[:4]
            for ti, (track_id, track) in enumerate(track_items):
                iou = _iou(d_box, track.as_box())
                if iou >= self.iou_threshold:
                    candidates.append((iou, di, ti))

        candidates.sort(reverse=True)   # highest IoU first

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

    # ─────────────────────────────────────────────────────────────────────────
    def confirmed_tracks(self) -> List[Track]:
        """All tracks currently in the CONFIRMED state."""
        return [t for t in self._tracks.values() if t.state == CONFIRMED]

    def all_tracks(self) -> List[Track]:
        """All tracks, including tentative ones (debug / inspection only)."""
        return list(self._tracks.values())

    # ─────────────────────────────────────────────────────────────────────────
    def record_zones(self, zone_assignments: Dict[int, str]) -> None:
        """
        Update each track's `zone` field and the cumulative seen-set.

        Called by the guidance pipeline after it has computed which zone
        (LEFT/EGO/RIGHT/OUT) each confirmed track currently sits in.
        """
        for track_id, zone in zone_assignments.items():
            if track_id not in self._tracks:
                continue
            self._tracks[track_id].zone = zone
            if zone in self._seen_in_zone:
                self._seen_in_zone[zone].add(track_id)

    def cumulative_seen(self, zone: str) -> int:
        """Unique track IDs ever observed in the given zone."""
        return len(self._seen_in_zone.get(zone, ()))

    def total_seen(self) -> int:
        """Unique IDs ever observed in LEFT, EGO or RIGHT (excluding OUT)."""
        return len(
            self._seen_in_zone["LEFT"]
            | self._seen_in_zone["EGO"]
            | self._seen_in_zone["RIGHT"]
        )

    # ─────────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Clear all tracks and cumulative state (e.g. on video switch)."""
        self._tracks.clear()
        self._next_id = count(1)
        for s in self._seen_in_zone.values():
            s.clear()
