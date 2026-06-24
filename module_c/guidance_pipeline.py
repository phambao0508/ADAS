"""
Module C  —  Guidance Pipeline (Orchestrator)
=============================================
TASK
----
Single entry point for Module C. Runs Pre-step → C1 → C2 → C3 → C4 → C7
per frame and returns a GuidanceResult.

PIPELINE FLOW
-------------
    Input:
      raw_detections  — list of [x1, y1, x2, y2, conf, cls] from YOLO
      lane_result     — LaneResult from Module A

    Pre-step: Filter and convert YOLO detections
      - Keep only classes 0 (Car), 1 (bus), 2 (truck)
      - Convert corner format [x1, y1, x2, y2] → centre format (cx, cy, w, h)
      - Result: vehicle_boxes — clean list for C1/C2/C3

    C1: compute_zone_dividers(left_poly, right_poly, W, H)
        → zone_left_x, zone_right_x

    C2: detect_front_proximity(vehicle_boxes, zone_left_x, zone_right_x, W, H)
        → front_proximity  (PROX_NONE | PROX_CLOSE | PROX_VERY_CLOSE)

    C3: check_adjacent_occupancy(vehicle_boxes, zone_left_x, zone_right_x, H)
        → left_clear, right_clear  (bool, bool)

    C4: decide_guidance(front_proximity, left_clear, right_clear,
                        left_type, right_type)
        → raw_guidance  (GUIDE_* constant)

    C7: holder.update(raw_guidance)
        → held_guidance  (hysteresis-smoothed, for HUD display)

    Output: GuidanceResult dataclass

YOLO INPUT FORMAT
-----------------
    results = model(frame)[0]
    boxes   = results.boxes.data.cpu().numpy()  # shape (N, 6)
    # Each row: [x1, y1, x2, y2, confidence, class_id]

    Vehicle classes: 0=Car, 1=bus, 2=truck
    Line classes:    3=white line, 4=yellow line  ← FILTERED OUT here

USAGE EXAMPLE
-------------
    from module_a import LanePipeline
    from module_b import DeparturePipeline
    from module_c import GuidancePipeline

    lane_pipe = LanePipeline(frame_width=1920, frame_height=1080)
    dept_pipe = DeparturePipeline(frame_width=1920, frame_height=1080)
    guid_pipe = GuidancePipeline(frame_width=1920, frame_height=1080)

    while cap.isOpened():
        ret, frame = cap.read()
        yolo_result = model(frame)[0]

        lane_result = lane_pipe.process(frame, yolo_result)
        dept_result = dept_pipe.process(lane_result)
        guid_result = guid_pipe.process(yolo_result, lane_result)

        print(guid_result.guidance)         # e.g. "GUIDE_LEFT"
        print(guid_result.front_proximity)  # e.g. "CLOSE"
        print(guid_result.left_clear)       # True / False
        print(guid_result.zone_left_x)      # pixel x-coordinate
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .guidance_states       import (
    GUIDE_NONE, PROX_NONE,
    ACTIVE_GUIDE_STATES, GUIDE_MESSAGES,
)
from .zone_definer          import compute_zone_dividers, assign_zone
from .proximity_detector    import detect_front_proximity
from .occupancy_checker     import check_adjacent_occupancy
from .guidance_decision     import decide_guidance
from .guidance_hold         import GuidanceHoldLogic, GUIDE_HOLD_FRAMES
from .vehicle_tracker       import VehicleTracker, Track
from .camera_calibration    import CameraIntrinsics
from .distance_estimator    import DistanceTTCEstimator


# ── Vehicle class IDs from the YOLO model ─────────────────────────────────
VEHICLE_CLASSES = {0, 1, 2}   # Car, bus, truck

# Only draw vehicle boxes whose bottom edge sits at or below this fraction
# of the frame height — i.e. the lower half. Keeps the HUD focused on
# nearby traffic and hides far-away cars near the horizon.
NEAR_FIELD_Y_FRAC = 0.5

# Tracks whose foot (y2) sits above this row are considered horizon noise
# and excluded from per-zone counts (they still pass through for proximity
# logic via `_extract_vehicle_boxes`).
COUNT_FAR_FIELD_Y_FRAC = 0.50
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class GuidanceResult:
    """
    All Module C outputs for a single frame.
    Passed directly to Module D (HUD renderer).
    """
    # Final held guidance state (after hysteresis) — used by HUD
    guidance:        str   = GUIDE_NONE

    # Raw guidance state before hold logic (useful for debugging)
    raw_guidance:    str   = GUIDE_NONE

    # Front vehicle proximity
    front_proximity: str   = PROX_NONE

    # Adjacent lane status
    left_clear:      bool  = True
    right_clear:     bool  = True

    # Zone divider x-positions (pixels) — for HUD visualisation
    zone_left_x:     float = 0.0
    zone_right_x:    float = 0.0

    # Tracked-vehicle detections in (x1, y1, x2, y2, conf, cls_id, track_id)
    # format. Only CONFIRMED tracks are emitted — flicker is suppressed.
    vehicle_detections: List[Tuple[float, float, float, float, float, int, int]] = field(
        default_factory=list
    )

    # Live confirmed-track counts per zone (for Lane Status HUD)
    left_count:      int   = 0
    right_count:     int   = 0
    ego_count:       int   = 0

    # Cumulative unique-IDs ever observed per zone (for "vehicles seen"
    # totals). Stable, monotonically increasing values.
    left_seen_total:  int   = 0
    right_seen_total: int   = 0
    total_seen:       int   = 0

    # Front vehicle (closest tracked vehicle in the EGO zone)
    front_track_id:   Optional[int]   = None
    front_distance_m: Optional[float] = None
    front_ttc_s:      Optional[float] = None

    # HUD banner text for the held guidance state
    @property
    def message(self) -> str:
        return GUIDE_MESSAGES.get(self.guidance, "")


class GuidancePipeline:
    """
    Module C orchestrator — runs Pre-step + C1 → C2 → C3 → C4 → C7 per frame.

    Create ONE instance per video (hold logic state is preserved across frames).
    Call process() on every frame.

    Parameters
    ----------
    frame_width  : int
    frame_height : int
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        intrinsics: Optional[CameraIntrinsics] = None,
        fps: float = 30.0,
    ):
        self.w = frame_width
        self.h = frame_height

        # C7: Guidance hold logic (stateful — persists across frames)
        self._holder = GuidanceHoldLogic()

        # Vehicle tracker (stateful — persistent track IDs across frames).
        # Tuned for ~30 fps dashcam: 3-frame confirm, 8-frame coast-through.
        self._tracker = VehicleTracker(
            min_hits=3,
            max_age=8,
            iou_threshold=0.30,
        )

        # Distance + TTC estimator (writes into Track.distance_m / .ttc_s).
        self._intrinsics = intrinsics or CameraIntrinsics.default_for(
            frame_width, frame_height
        )
        self._distance_ttc = DistanceTTCEstimator(self._intrinsics, fps=fps)

    # ─────────────────────────────────────────────────────────────────────
    def process(self, yolo_result, lane_result) -> GuidanceResult:
        """
        Run the full Module C pipeline on one video frame.

        Parameters
        ----------
        yolo_result : ultralytics Results object  (model(frame)[0])
            Must contain .boxes.data with shape (N, 6):
            [x1, y1, x2, y2, conf, cls_id] per detection.
        lane_result : LaneResult  (from module_a.LanePipeline.process)
            Must provide:
              .left_poly   : np.ndarray [a,b,c] or None
              .right_poly  : np.ndarray [a,b,c] or None
              .left_type   : 'solid' or 'dashed'
              .right_type  : 'solid' or 'dashed'

        Returns
        -------
        GuidanceResult
        """
        result = GuidanceResult()

        # ── Pre-step: filter raw YOLO detections to vehicle classes ───────
        raw_vehicle_dets = self._extract_raw_vehicle_dets(yolo_result)

        # ── Tracker update — assigns persistent IDs, suppresses 1-frame
        #    ghosts (TENTATIVE) and bridges short YOLO drop-outs.
        confirmed_tracks: List[Track] = self._tracker.update(raw_vehicle_dets)

        # ── Distance + TTC: writes into Track.distance_m / .ttc_s ─────────
        self._distance_ttc.update(confirmed_tracks)

        # Build downstream views from CONFIRMED tracks only.
        vehicle_boxes = [
            (t.cx, t.cy, t.w, t.h) for t in confirmed_tracks
        ]
        result.vehicle_detections = self._tracks_to_hud_detections(confirmed_tracks)

        # ── C1: Compute zones from boundary polynomials ───────────────────
        zone_left_x, zone_right_x = compute_zone_dividers(
            lane_result.left_poly,
            lane_result.right_poly,
            self.w,
            self.h,
        )
        result.zone_left_x  = zone_left_x
        result.zone_right_x = zone_right_x

        # ── C2: Front vehicle proximity ───────────────────────────────────
        front_proximity = detect_front_proximity(
            vehicle_boxes,
            zone_left_x, zone_right_x,
            self.w, self.h,
        )
        result.front_proximity = front_proximity

        # ── C3: Adjacent lane occupancy ───────────────────────────────────
        left_clear, right_clear = check_adjacent_occupancy(
            vehicle_boxes,
            zone_left_x, zone_right_x,
            self.h,
        )
        result.left_clear  = left_clear
        result.right_clear = right_clear

        # ── C3b: Per-zone counts from CONFIRMED tracks (with far-field cut)
        zone_assignments, counts = self._assign_track_zones(
            confirmed_tracks, zone_left_x, zone_right_x,
        )
        self._tracker.record_zones(zone_assignments)
        result.left_count   = counts["LEFT"]
        result.right_count  = counts["RIGHT"]
        result.ego_count    = counts["EGO"]

        result.left_seen_total  = self._tracker.cumulative_seen("LEFT")
        result.right_seen_total = self._tracker.cumulative_seen("RIGHT")
        result.total_seen       = self._tracker.total_seen()

        # ── Identify the front vehicle (closest tracked EGO-zone car) ─────
        front = self._select_front_track(confirmed_tracks, zone_assignments)
        if front is not None:
            result.front_track_id   = front.track_id
            result.front_distance_m = front.distance_m
            result.front_ttc_s      = front.ttc_s

        # ── C4: Guidance decision ─────────────────────────────────────────
        raw_guidance = decide_guidance(
            front_proximity,
            left_clear,
            right_clear,
            lane_result.left_type,
            lane_result.right_type,
        )
        result.raw_guidance = raw_guidance

        # ── C7: Hysteresis hold ───────────────────────────────────────────
        result.guidance = self._holder.update(raw_guidance)

        return result

    # ─────────────────────────────────────────────────────────────────────
    def _extract_raw_vehicle_dets(
        self, yolo_result
    ) -> List[Tuple[float, float, float, float, float, int]]:
        """
        Filter YOLO detections to vehicle classes and emit corner-format
        rows for the tracker. Non-vehicle classes (lines) are dropped.
        """
        if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
            return []

        boxes = yolo_result.boxes.data.cpu().numpy()   # shape (N, 6)
        out: List[Tuple[float, float, float, float, float, int]] = []
        for row in boxes:
            x1, y1, x2, y2, conf, cls_id = row[:6]
            cls_id = int(cls_id)
            if cls_id not in VEHICLE_CLASSES:
                continue
            out.append((float(x1), float(y1), float(x2), float(y2),
                        float(conf), cls_id))
        return out

    def _tracks_to_hud_detections(
        self, tracks: List[Track]
    ) -> List[tuple]:
        """
        Convert confirmed tracks into the HUD-detection format the renderer
        expects. Far-field tracks (foot above ``NEAR_FIELD_Y_FRAC``) are
        hidden so the HUD stays focused on nearby traffic.

        Each row is
            (x1, y1, x2, y2, conf, cls_id, track_id, distance_m, ttc_s)
        ``distance_m`` and ``ttc_s`` may be None.
        """
        y_min_bottom = NEAR_FIELD_Y_FRAC * self.h
        out: List[tuple] = []
        for t in tracks:
            if t.y2 < y_min_bottom:
                continue
            out.append((
                t.x1, t.y1, t.x2, t.y2,
                t.conf, t.cls_id, t.track_id,
                t.distance_m, t.ttc_s,
            ))
        return out

    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _select_front_track(
        tracks: List[Track],
        zone_assignments: Dict[int, str],
    ) -> Optional[Track]:
        """
        Return the closest-distance track currently in the EGO zone.

        Falls back to the highest-cy track if no track has a distance yet
        (e.g. on the very first frames before TTC history exists).
        """
        ego_tracks = [
            t for t in tracks
            if zone_assignments.get(t.track_id) == "EGO"
        ]
        if not ego_tracks:
            return None

        with_distance = [t for t in ego_tracks if t.distance_m is not None]
        if with_distance:
            return min(with_distance, key=lambda t: t.distance_m)
        # No distance yet — pick the visually-lowest box (closest by row).
        return max(ego_tracks, key=lambda t: t.cy)

    # ─────────────────────────────────────────────────────────────────────
    def _assign_track_zones(
        self,
        tracks: List[Track],
        zone_left_x: float,
        zone_right_x: float,
    ) -> Tuple[Dict[int, str], Dict[str, int]]:
        """
        Decide which zone each confirmed track sits in and return:
          - ``zone_assignments`` : {track_id: zone_label} for *all* tracks
          - ``counts``           : {zone: live_count} with the far-field cut
                                    applied (horizon specks excluded).

        Far-field cut: a track whose foot (y2) sits above
        ``COUNT_FAR_FIELD_Y_FRAC`` of the frame height is treated as horizon
        noise and not counted. It is still assigned to a zone for the
        cumulative seen-set, so the "vehicles seen" total still grows when
        the same vehicle later approaches.
        """
        far_y = COUNT_FAR_FIELD_Y_FRAC * self.h

        zone_assignments: Dict[int, str] = {}
        counts = {"LEFT": 0, "EGO": 0, "RIGHT": 0, "OUT": 0}

        for t in tracks:
            zone = assign_zone(t.cx, zone_left_x, zone_right_x)
            zone_assignments[t.track_id] = zone
            # Apply the far-field cutoff to the live count only — the
            # cumulative seen-set still records the ID.
            if t.y2 < far_y:
                continue
            if zone in counts:
                counts[zone] += 1
        return zone_assignments, counts

    # ─────────────────────────────────────────────────────────────────────
    @property
    def tracker(self) -> VehicleTracker:
        """Expose the tracker (read-only) for diagnostics or external HUD."""
        return self._tracker

    def reset(self):
        """
        Full reset (e.g. when switching video clips).
        Clears guidance hold state, vehicle tracker, distance/TTC history.
        """
        self._holder.reset()
        self._tracker.reset()
        self._distance_ttc.reset()
