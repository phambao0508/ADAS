"""
Module B  —  Departure Pipeline (Orchestrator)
===============================================
TASK
----
Single entry point for Module B. Runs B1 → B2 → B3 + hold logic per frame.

PIPELINE FLOW
-------------
    Input:
      lane_result  — LaneResult from Module A (contains left_poly, right_poly,
                     left_type, right_type)
      frame_w, frame_h — video dimensions

    B1: compute_lateral_offset(left_poly, right_poly, frame_w, frame_h)
        → raw_offset (float | None)

    B2: ema_smoother.update(raw_offset)
        → smoothed_offset (float | None)

    B3: classify_departure(smoothed_offset, left_type, right_type)
        → raw_state (one of the 7 state constants)

    B4: holder.update(raw_state)                        → via hold_logic.DepartureHoldLogic
        → held_state  (what the HUD actually displays)

    Output: DepartureResult dataclass

    The DepartureResult always contains the HELD state (never the raw B3 state).
    See hold_logic.py (B4) for the full algorithm.

USAGE EXAMPLE
-------------
    from module_a import LanePipeline
    from module_b import DeparturePipeline

    lane_pipe = LanePipeline(frame_width=1920, frame_height=1080)
    dept_pipe = DeparturePipeline(frame_width=1920, frame_height=1080)

    while cap.isOpened():
        ret, frame = cap.read()
        yolo_result = model(frame)[0]

        lane_result = lane_pipe.process(frame, yolo_result)
        dept_result = dept_pipe.process(lane_result)

        # dept_result.state          → current departure state string
        # dept_result.smoothed_offset → smoothed pixel offset
        # dept_result.raw_offset      → raw (unsmoothed) pixel offset
"""

from dataclasses import dataclass
from typing import Optional

from .offset_calculator    import compute_lateral_offset
from .ema_smoother         import EMASmoother
from .bias_estimator       import MountBiasEstimator
from .departure_classifier import (
    classify_departure,
    CENTERED, ACTIVE_STATES,
)
from .hold_logic           import DepartureHoldLogic, HOLD_FRAMES


@dataclass
class DepartureResult:
    """
    All Module B outputs for a single frame.
    Passed directly to Module D (HUD renderer).
    """
    state:           str   = CENTERED   # held state (used by HUD)
    raw_offset:      Optional[float] = None  # B1 output (unsmoothed, uncorrected)
    smoothed_offset: Optional[float] = None  # B2 output (EMA smoothed)
    raw_state:       str   = CENTERED   # B3 output before hold logic
    mount_bias:      float = 0.0        # current auto-estimated mount bias (px)


class DeparturePipeline:
    """
    Module B orchestrator — runs B1 → B2 → B3 + hold logic per frame.

    Create ONE instance per video. Call process() on every frame,
    passing the LaneResult produced by Module A's LanePipeline.

    Parameters
    ----------
    frame_width  : int
    frame_height : int
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.w = frame_width
        self.h = frame_height

        # B1b: Mount bias auto-estimator (stateful — builds rolling median)
        self._bias_est = MountBiasEstimator()

        # B2: EMA smoother (stateful — must persist across frames)
        self._smoother = EMASmoother()

        # B4: Hold logic (stateful — delegates entirely to DepartureHoldLogic)
        self._holder = DepartureHoldLogic()

    # ─────────────────────────────────────────────────────────────────────
    def process(self, lane_result) -> DepartureResult:
        """
        Run the full Module B pipeline on one video frame.

        Parameters
        ----------
        lane_result : LaneResult  (from module_a.LanePipeline.process)
            Must provide:
              .left_poly   : np.ndarray [a,b,c] or None
              .right_poly  : np.ndarray [a,b,c] or None
              .left_type   : 'solid' or 'dashed'
              .right_type  : 'solid' or 'dashed'

        Returns
        -------
        DepartureResult
        """
        result = DepartureResult()

        # ── B1: Compute raw lateral offset ───────────────────────────────
        raw_offset = compute_lateral_offset(
            lane_result.left_poly,
            lane_result.right_poly,
            self.w,
            self.h,
        )
        result.raw_offset = raw_offset

        # ── B1b: Auto mount-bias correction ─────────────────────────────
        # Subtract the rolling-median mount bias so a consistently off-centre
        # dashcam does not produce permanent WARN_LEFT / WARN_RIGHT.
        # During the first ~3 s (warmup), the correction is 0 (pass-through).
        corrected_offset = self._bias_est.update(raw_offset)
        result.mount_bias = self._bias_est.current_bias

        # ── B2: EMA smoothing ────────────────────────────────────────────
        smoothed_offset = self._smoother.update(corrected_offset)
        result.smoothed_offset = smoothed_offset

        # ── B3: 6-state classification ───────────────────────────────────
        raw_state = classify_departure(
            smoothed_offset,
            lane_result.left_type,
            lane_result.right_type,
        )
        result.raw_state = raw_state

        # ── B4: Hold logic ───────────────────────────────────────────────
        result.state = self._holder.update(raw_state)
        return result

    # ─────────────────────────────────────────────────────────────────────
    def reset(self):
        """
        Full reset (e.g. when switching video clips or after a hard lane loss).
        Clears bias estimator, EMA history and hold state.
        """
        self._bias_est.reset()
        self._smoother.reset()
        self._holder.reset()
