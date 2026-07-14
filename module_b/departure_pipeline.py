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

    state:           str   = CENTERED
    raw_offset:      Optional[float] = None
    smoothed_offset: Optional[float] = None
    raw_state:       str   = CENTERED
    mount_bias:      float = 0.0

class DeparturePipeline:

    def __init__(self, frame_width: int, frame_height: int):
        self.w = frame_width
        self.h = frame_height

        self._bias_est = MountBiasEstimator()

        self._smoother = EMASmoother()

        self._holder = DepartureHoldLogic()

    def process(self, lane_result) -> DepartureResult:

        result = DepartureResult()

        if getattr(lane_result, "valid", True):
            raw_offset = compute_lateral_offset(
                lane_result.left_poly,
                lane_result.right_poly,
                self.w,
                self.h,
            )
        else:
            raw_offset = None
        result.raw_offset = raw_offset

        corrected_offset = self._bias_est.update(raw_offset)
        result.mount_bias = self._bias_est.current_bias

        smoothed_offset = self._smoother.update(corrected_offset)
        result.smoothed_offset = smoothed_offset

        raw_state = classify_departure(
            smoothed_offset,
            lane_result.left_type,
            lane_result.right_type,
        )
        result.raw_state = raw_state

        result.state = self._holder.update(raw_state)
        return result

    def reset(self):

        self._bias_est.reset()
        self._smoother.reset()
        self._holder.reset()
