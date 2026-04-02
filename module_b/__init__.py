"""
Module B — Lane Departure Warning
===================================
Analyses the ego-lane position each frame and classifies the driver's
behaviour as one of six departure states.

Pipeline:
    B1  (offset_calculator)    → raw lateral offset from lane centre
    B1b (bias_estimator)       → auto mount-bias correction (rolling median)
    B2  (ema_smoother)         → smoothed offset (noise + spike rejection)
    B3  (departure_classifier) → 6-state classification
    Hold logic (departure_pipeline) → flicker prevention (6 frames)

Public API
----------
    from module_b import DeparturePipeline, DepartureResult

    pipeline = DeparturePipeline(frame_width=1920, frame_height=1080)

    # lane_result is the LaneResult returned by module_a.LanePipeline
    result = pipeline.process(lane_result)

    # result.state           → e.g. "WARN_LEFT", "DEPART_RIGHT", "CENTERED"
    # result.smoothed_offset → float (pixels, bias-corrected + EMA smoothed)
    # result.raw_offset      → float (uncorrected, unsmoothed)
    # result.mount_bias      → float (current estimated camera mount offset)
    # result.raw_state       → state before hold logic

Departure States
----------------
    CENTERED          |offset| < 80 px
    WARN_LEFT         80 ≤ |offset| < 150 px, drifting left
    WARN_RIGHT        80 ≤ |offset| < 150 px, drifting right
    DEPART_LEFT       |offset| ≥ 150 px + left boundary solid
    DEPART_RIGHT      |offset| ≥ 150 px + right boundary solid
    LANE_CHANGE_LEFT  |offset| ≥ 150 px + left boundary dashed
    LANE_CHANGE_RIGHT |offset| ≥ 150 px + right boundary dashed

Sign Convention
---------------
    offset > 0  →  car is LEFT  of lane centre (drifting LEFT)
    offset < 0  →  car is RIGHT of lane centre (drifting RIGHT)
"""

from .departure_pipeline   import DeparturePipeline, DepartureResult
from .offset_calculator    import compute_lateral_offset, REF_Y_FRAC
from .ema_smoother         import EMASmoother, EMA_ALPHA
from .bias_estimator       import MountBiasEstimator, WARMUP_SAMPLES, WINDOW_SIZE
from .hold_logic           import DepartureHoldLogic, HOLD_FRAMES
from .departure_classifier import (
    classify_departure,
    CENTERED,
    WARN_LEFT,
    WARN_RIGHT,
    DEPART_LEFT,
    DEPART_RIGHT,
    LANE_CHANGE_LEFT,
    LANE_CHANGE_RIGHT,
    ACTIVE_STATES,
    WARN_THRESHOLD,
    DEPART_THRESHOLD,
)

__all__ = [
    # Pipeline (main entry point)
    "DeparturePipeline",
    "DepartureResult",

    # Sub-components (for testing or direct use)
    "compute_lateral_offset",
    "EMASmoother",
    "MountBiasEstimator",
    "DepartureHoldLogic",
    "classify_departure",

    # State constants
    "CENTERED",
    "WARN_LEFT",
    "WARN_RIGHT",
    "DEPART_LEFT",
    "DEPART_RIGHT",
    "LANE_CHANGE_LEFT",
    "LANE_CHANGE_RIGHT",
    "ACTIVE_STATES",

    # Tuning constants
    "WARN_THRESHOLD",
    "DEPART_THRESHOLD",
    "HOLD_FRAMES",
    "EMA_ALPHA",
    "REF_Y_FRAC",
    "WARMUP_SAMPLES",
    "WINDOW_SIZE",
]
