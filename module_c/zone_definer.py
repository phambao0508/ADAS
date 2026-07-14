import numpy as np
from typing import Optional, Tuple

ZONE_REF_Y_FRAC = 0.80

FALLBACK_LEFT_FRAC  = 0.35
FALLBACK_RIGHT_FRAC = 0.65

def compute_zone_dividers(
    left_poly:  Optional[np.ndarray],
    right_poly: Optional[np.ndarray],
    frame_w:    int,
    frame_h:    int,
) -> Tuple[float, float]:

    ref_y = ZONE_REF_Y_FRAC * frame_h

    if left_poly is not None and right_poly is not None:
        zone_left_x  = float(np.polyval(left_poly,  ref_y))
        zone_right_x = float(np.polyval(right_poly, ref_y))
    else:

        zone_left_x  = FALLBACK_LEFT_FRAC  * frame_w
        zone_right_x = FALLBACK_RIGHT_FRAC * frame_w

    return zone_left_x, zone_right_x

def assign_zone(cx: float, zone_left_x: float, zone_right_x: float) -> str:

    if cx < zone_left_x:
        return "LEFT"
    elif cx > zone_right_x:
        return "RIGHT"
    else:
        return "EGO"
