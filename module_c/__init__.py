from .guidance_pipeline import GuidancePipeline, GuidanceResult, VEHICLE_CLASSES
from .vehicle_tracker   import VehicleTracker, Track, TENTATIVE, CONFIRMED
from .camera_calibration import CameraIntrinsics, REAL_VEHICLE_WIDTHS_M
from .distance_estimator import (
    DistanceTTCEstimator,
    estimate_distance,
    pinhole_distance,
    width_distance,
)
from .guidance_states   import (
    GUIDE_NONE, GUIDE_LEFT, GUIDE_RIGHT, GUIDE_BOTH, GUIDE_SLOW, GUIDE_URGENT,
    ACTIVE_GUIDE_STATES, GUIDE_MESSAGES,
    PROX_NONE, PROX_DETECTED, PROX_CLOSE, PROX_VERY_CLOSE,
)
from .zone_definer      import compute_zone_dividers, assign_zone, ZONE_REF_Y_FRAC
from .proximity_detector import detect_front_proximity, FRONT_GATE_Y_FRAC, PROXIMITY_CLOSE, PROXIMITY_VERY_CLOSE
from .occupancy_checker  import check_adjacent_occupancy, ADJACENT_GATE_Y_FRAC
from .guidance_decision  import decide_guidance
from .guidance_hold      import GuidanceHoldLogic, GUIDE_HOLD_FRAMES

__all__ = [

    "GuidancePipeline",
    "GuidanceResult",

    "VehicleTracker",
    "Track",
    "TENTATIVE",
    "CONFIRMED",

    "CameraIntrinsics",
    "DistanceTTCEstimator",
    "estimate_distance",
    "pinhole_distance",
    "width_distance",
    "REAL_VEHICLE_WIDTHS_M",

    "compute_zone_dividers",
    "assign_zone",
    "detect_front_proximity",
    "check_adjacent_occupancy",
    "decide_guidance",
    "GuidanceHoldLogic",

    "GUIDE_NONE",
    "GUIDE_LEFT",
    "GUIDE_RIGHT",
    "GUIDE_BOTH",
    "GUIDE_SLOW",
    "GUIDE_URGENT",
    "ACTIVE_GUIDE_STATES",
    "GUIDE_MESSAGES",

    "PROX_NONE",
    "PROX_DETECTED",
    "PROX_CLOSE",
    "PROX_VERY_CLOSE",

    "GUIDE_HOLD_FRAMES",
    "ZONE_REF_Y_FRAC",
    "FRONT_GATE_Y_FRAC",
    "ADJACENT_GATE_Y_FRAC",
    "PROXIMITY_CLOSE",
    "PROXIMITY_VERY_CLOSE",
    "VEHICLE_CLASSES",
]
