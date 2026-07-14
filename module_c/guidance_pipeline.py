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

VEHICLE_CLASSES = {0, 1, 2}

NEAR_FIELD_Y_FRAC = 0.5

COUNT_FAR_FIELD_Y_FRAC = 0.50

@dataclass
class GuidanceResult:

    guidance:        str   = GUIDE_NONE

    raw_guidance:    str   = GUIDE_NONE

    front_proximity: str   = PROX_NONE

    left_clear:      bool  = True
    right_clear:     bool  = True

    zone_left_x:     float = 0.0
    zone_right_x:    float = 0.0

    vehicle_detections: List[Tuple[float, float, float, float, float, int, int]] = field(
        default_factory=list
    )

    left_count:      int   = 0
    right_count:     int   = 0
    ego_count:       int   = 0

    left_seen_total:  int   = 0
    right_seen_total: int   = 0
    total_seen:       int   = 0

    front_track_id:   Optional[int]   = None
    front_distance_m: Optional[float] = None
    front_ttc_s:      Optional[float] = None

    @property
    def message(self) -> str:
        return GUIDE_MESSAGES.get(self.guidance, "")

class GuidancePipeline:

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        intrinsics: Optional[CameraIntrinsics] = None,
        fps: float = 30.0,
    ):
        self.w = frame_width
        self.h = frame_height

        self._holder = GuidanceHoldLogic()

        self._tracker = VehicleTracker(
            min_hits=3,
            max_age=8,
            iou_threshold=0.30,
        )

        self._intrinsics = intrinsics or CameraIntrinsics.default_for(
            frame_width, frame_height
        )
        self._distance_ttc = DistanceTTCEstimator(self._intrinsics, fps=fps)

    def process(self, yolo_result, lane_result) -> GuidanceResult:

        result = GuidanceResult()

        raw_vehicle_dets = self._extract_raw_vehicle_dets(yolo_result)

        confirmed_tracks: List[Track] = self._tracker.update(raw_vehicle_dets)

        self._distance_ttc.update(confirmed_tracks)

        vehicle_boxes = [
            (t.cx, t.cy, t.w, t.h) for t in confirmed_tracks
        ]
        result.vehicle_detections = self._tracks_to_hud_detections(confirmed_tracks)

        zone_left_x, zone_right_x = compute_zone_dividers(
            lane_result.left_poly,
            lane_result.right_poly,
            self.w,
            self.h,
        )
        result.zone_left_x  = zone_left_x
        result.zone_right_x = zone_right_x

        front_proximity = detect_front_proximity(
            vehicle_boxes,
            zone_left_x, zone_right_x,
            self.w, self.h,
        )
        result.front_proximity = front_proximity

        left_clear, right_clear = check_adjacent_occupancy(
            vehicle_boxes,
            zone_left_x, zone_right_x,
            self.h,
        )
        result.left_clear  = left_clear
        result.right_clear = right_clear

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

        front = self._select_front_track(confirmed_tracks, zone_assignments)
        if front is not None:
            result.front_track_id   = front.track_id
            result.front_distance_m = front.distance_m
            result.front_ttc_s      = front.ttc_s

        raw_guidance = decide_guidance(
            front_proximity,
            left_clear,
            right_clear,
            lane_result.left_type,
            lane_result.right_type,
        )
        result.raw_guidance = raw_guidance

        result.guidance = self._holder.update(raw_guidance)

        return result

    def _extract_raw_vehicle_dets(
        self, yolo_result
    ) -> List[Tuple[float, float, float, float, float, int]]:

        if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
            return []

        boxes = yolo_result.boxes.data.cpu().numpy()
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

    @staticmethod
    def _select_front_track(
        tracks: List[Track],
        zone_assignments: Dict[int, str],
    ) -> Optional[Track]:

        ego_tracks = [
            t for t in tracks
            if zone_assignments.get(t.track_id) == "EGO"
        ]
        if not ego_tracks:
            return None

        with_distance = [t for t in ego_tracks if t.distance_m is not None]
        if with_distance:
            return min(with_distance, key=lambda t: t.distance_m)

        return max(ego_tracks, key=lambda t: t.cy)

    def _assign_track_zones(
        self,
        tracks: List[Track],
        zone_left_x: float,
        zone_right_x: float,
    ) -> Tuple[Dict[int, str], Dict[str, int]]:

        far_y = COUNT_FAR_FIELD_Y_FRAC * self.h

        zone_assignments: Dict[int, str] = {}
        counts = {"LEFT": 0, "EGO": 0, "RIGHT": 0, "OUT": 0}

        for t in tracks:
            zone = assign_zone(t.cx, zone_left_x, zone_right_x)
            zone_assignments[t.track_id] = zone

            if t.y2 < far_y:
                continue
            if zone in counts:
                counts[zone] += 1
        return zone_assignments, counts

    @property
    def tracker(self) -> VehicleTracker:

        return self._tracker

    def reset(self):

        self._holder.reset()
        self._tracker.reset()
        self._distance_ttc.reset()
