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

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .guidance_states       import (
    GUIDE_NONE, PROX_NONE,
    ACTIVE_GUIDE_STATES, GUIDE_MESSAGES,
)
from .zone_definer          import compute_zone_dividers
from .proximity_detector    import detect_front_proximity
from .occupancy_checker     import check_adjacent_occupancy
from .guidance_decision     import decide_guidance
from .guidance_hold         import GuidanceHoldLogic, GUIDE_HOLD_FRAMES


# ── Vehicle class IDs from the YOLO model ─────────────────────────────────
VEHICLE_CLASSES = {0, 1, 2}   # Car, bus, truck
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

    def __init__(self, frame_width: int, frame_height: int):
        self.w = frame_width
        self.h = frame_height

        # C7: Guidance hold logic (stateful — persists across frames)
        self._holder = GuidanceHoldLogic()

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

        # ── Pre-step: filter + convert YOLO detections ────────────────────
        vehicle_boxes = self._extract_vehicle_boxes(yolo_result)

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
    def _extract_vehicle_boxes(
        self, yolo_result
    ) -> List[Tuple[float, float, float, float]]:
        """
        Filter YOLO detections to vehicle classes and convert to centre format.

        Parameters
        ----------
        yolo_result : ultralytics Results object

        Returns
        -------
        List of (cx, cy, w, h) tuples — one per vehicle detection.
        Line detections (class 3, 4) are excluded.
        """
        if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
            return []

        boxes = yolo_result.boxes.data.cpu().numpy()   # shape (N, 6)
        vehicle_boxes = []

        for row in boxes:
            x1, y1, x2, y2, conf, cls_id = row[:6]
            if int(cls_id) not in VEHICLE_CLASSES:
                continue   # skip white line / yellow line detections

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w  = x2 - x1
            h  = y2 - y1
            vehicle_boxes.append((cx, cy, w, h))

        return vehicle_boxes

    # ─────────────────────────────────────────────────────────────────────
    def reset(self):
        """
        Full reset (e.g. when switching video clips).
        Clears guidance hold state.
        """
        self._holder.reset()
