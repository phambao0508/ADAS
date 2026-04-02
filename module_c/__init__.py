"""
Module C — Directional Guidance (Overtaking Assist)
=====================================================
Analyses surrounding traffic and advises the driver on whether and
where to change lanes when a vehicle is detected ahead.

Pipeline:
    Pre-step (guidance_pipeline)   → filter+convert YOLO detections
    C1 (zone_definer)              → ego/left/right zone dividers
    C2 (proximity_detector)        → front vehicle proximity
    C3 (occupancy_checker)         → adjacent lane occupancy
    C4 (guidance_decision)         → guidance state decision
    C7 (guidance_hold)             → hysteresis hold (8 frames)

Public API
----------
    from module_c import GuidancePipeline, GuidanceResult

    pipeline = GuidancePipeline(frame_width=1920, frame_height=1080)

    # yolo_result = model(frame)[0]
    # lane_result = LanePipeline.process(frame, yolo_result)
    result = pipeline.process(yolo_result, lane_result)

    # result.guidance        → e.g. "GUIDE_LEFT"
    # result.message         → e.g. "◄◄ MOVE LEFT — LEFT LANE IS CLEAR"
    # result.front_proximity → "NONE" | "CLOSE" | "VERY_CLOSE"
    # result.left_clear      → True / False
    # result.right_clear     → True / False
    # result.zone_left_x     → float (pixel x)
    # result.zone_right_x    → float (pixel x)

Guidance States
---------------
    GUIDE_NONE    No front vehicle — no banner
    GUIDE_LEFT    Move left  — left lane clear + left boundary dashed
    GUIDE_RIGHT   Move right — right lane clear + right boundary dashed
    GUIDE_BOTH    Both lanes clear + both dashed → prefer left
    GUIDE_SLOW    Reduce speed — no safe change available
    GUIDE_URGENT  Brake — very close vehicle (overrides departure warnings)

Model Constraints
-----------------
    Only vehicle classes (0=Car, 1=bus, 2=truck) are used by Module C.
    Line classes (3=white line, 4=yellow line) are filtered out in the
    pre-processing step and never reach C1–C4.
    Zone boundaries come from the boundary LINE polynomials (Module A),
    NOT from a lane area mask (the model does not output one).
"""

from .guidance_pipeline import GuidancePipeline, GuidanceResult, VEHICLE_CLASSES
from .guidance_states   import (
    GUIDE_NONE, GUIDE_LEFT, GUIDE_RIGHT, GUIDE_BOTH, GUIDE_SLOW, GUIDE_URGENT,
    ACTIVE_GUIDE_STATES, GUIDE_MESSAGES,
    PROX_NONE, PROX_CLOSE, PROX_VERY_CLOSE,
)
from .zone_definer      import compute_zone_dividers, assign_zone, ZONE_REF_Y_FRAC
from .proximity_detector import detect_front_proximity, FRONT_GATE_Y_FRAC, PROXIMITY_CLOSE, PROXIMITY_VERY_CLOSE
from .occupancy_checker  import check_adjacent_occupancy, ADJACENT_GATE_Y_FRAC
from .guidance_decision  import decide_guidance
from .guidance_hold      import GuidanceHoldLogic, GUIDE_HOLD_FRAMES

__all__ = [
    # Pipeline (main entry point)
    "GuidancePipeline",
    "GuidanceResult",

    # Sub-components (for testing or direct use)
    "compute_zone_dividers",
    "assign_zone",
    "detect_front_proximity",
    "check_adjacent_occupancy",
    "decide_guidance",
    "GuidanceHoldLogic",

    # Guidance state constants
    "GUIDE_NONE",
    "GUIDE_LEFT",
    "GUIDE_RIGHT",
    "GUIDE_BOTH",
    "GUIDE_SLOW",
    "GUIDE_URGENT",
    "ACTIVE_GUIDE_STATES",
    "GUIDE_MESSAGES",

    # Proximity sub-state constants
    "PROX_NONE",
    "PROX_CLOSE",
    "PROX_VERY_CLOSE",

    # Tuning constants
    "GUIDE_HOLD_FRAMES",
    "ZONE_REF_Y_FRAC",
    "FRONT_GATE_Y_FRAC",
    "ADJACENT_GATE_Y_FRAC",
    "PROXIMITY_CLOSE",
    "PROXIMITY_VERY_CLOSE",
    "VEHICLE_CLASSES",
]
