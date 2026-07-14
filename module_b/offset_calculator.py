import numpy as np
from typing import Optional

# We can calculate the offset = median (xleft(yi) + xright(yi)/2) - W/2 - camera bias
# At four different row 0,68, 0,74, 0,80, 0,86, take the median of the four line center from y
# Take median of four row be the middle lane, offset = middle lane - middle line - camera bias
REF_Y_FRACS = (0.68, 0.74, 0.80, 0.86)
REF_Y_FRAC = REF_Y_FRACS[-1]

CAMERA_MOUNT_BIAS_PX: float = 0.0

def compute_lateral_offset(
    left_poly:  Optional[np.ndarray],
    right_poly: Optional[np.ndarray],
    frame_w:    int,
    frame_h:    int,
) -> Optional[float]:

    if left_poly is None and right_poly is None:
        return None

    if left_poly is None or right_poly is None:

        return None

    frame_center = frame_w / 2.0

    ref_ys = np.array([frac * frame_h for frac in REF_Y_FRACS], dtype=np.float64)
    left_xs = np.polyval(left_poly, ref_ys).astype(np.float64)
    right_xs = np.polyval(right_poly, ref_ys).astype(np.float64)
    widths = right_xs - left_xs
    centers = (left_xs + right_xs) * 0.5

    MIN_VALID_LANE_WIDTH_PX = max(120.0, frame_w * 0.07)
    MAX_VALID_LANE_WIDTH_PX = frame_w * 0.68

    valid = (
        (widths >= MIN_VALID_LANE_WIDTH_PX)
        & (widths <= MAX_VALID_LANE_WIDTH_PX)
        & (left_xs >= 0)
        & (right_xs <= frame_w)
        & (centers >= frame_w * 0.20)
        & (centers <= frame_w * 0.80)
    )
    if valid.sum() < 2:
        return None

    lane_center_x = float(np.median(centers[valid]))
    return (lane_center_x - frame_center) - CAMERA_MOUNT_BIAS_PX
