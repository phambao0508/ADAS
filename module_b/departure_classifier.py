from typing import Optional

from states import DepartureState, LineType, ACTIVE_DEPARTURE_STATES

CENTERED          = DepartureState.CENTERED
WARN_LEFT         = DepartureState.WARN_LEFT
WARN_RIGHT        = DepartureState.WARN_RIGHT
DEPART_LEFT       = DepartureState.DEPART_LEFT
DEPART_RIGHT      = DepartureState.DEPART_RIGHT
LANE_CHANGE_LEFT  = DepartureState.LANE_CHANGE_LEFT
LANE_CHANGE_RIGHT = DepartureState.LANE_CHANGE_RIGHT

ACTIVE_STATES = ACTIVE_DEPARTURE_STATES

WARN_THRESHOLD   = 80
DEPART_THRESHOLD = 150

def classify_departure(
    smoothed_offset: Optional[float],
    left_type:       str = "solid",
    right_type:      str = "solid",
) -> DepartureState:

    if smoothed_offset is None:
        return CENTERED

    abs_offset = abs(smoothed_offset)

    if abs_offset < WARN_THRESHOLD:
        return CENTERED

    if abs_offset < DEPART_THRESHOLD:

        if smoothed_offset > 0:
            return WARN_LEFT
        else:
            return WARN_RIGHT

    if smoothed_offset > 0:

        if left_type == "dashed":
            return LANE_CHANGE_LEFT
        else:
            return DEPART_LEFT
    else:

        if right_type == "dashed":
            return LANE_CHANGE_RIGHT
        else:
            return DEPART_RIGHT
