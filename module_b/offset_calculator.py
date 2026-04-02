"""
Module B  —  Step B1: Lateral Offset Calculator
================================================
TASK
----
Determine how far left or right the car is positioned within its lane.

HOW IT WORKS
------------
The boundary polynomials from Module A give us:
    left_x  = polyval(left_poly,  y)   ← x-pos of left  boundary at row y
    right_x = polyval(right_poly, y)   ← x-pos of right boundary at row y

We evaluate both at a reference row near the bottom of the frame
(y = 0.85 × H), where the lane geometry is most reliable.

    lane_center_x = (left_x + right_x) / 2
    frame_center  = frame_width / 2        ← camera is centred on the car

    lateral_offset = lane_center_x − frame_center

SIGN CONVENTION
---------------
    offset > 0  →  lane centre is to the RIGHT of the car
                →  the car is LEFT  of lane centre  (drifting LEFT)
    offset < 0  →  lane centre is to the LEFT  of the car
                →  the car is RIGHT of lane centre  (drifting RIGHT)
    offset = 0  →  car is perfectly centred

WHY y = 0.85 × H?
-----------------
The A5 note in the implementation plan states:
    "Module B measures lateral offset by evaluating the boundary
     polynomials at a reference row (y = 0.85 × H) in camera-view
     coordinates, which is simpler and equally accurate for the
     offset measurement."

This point is close to the car (reliable, low distortion), inside the
warp region's lower edge (98% H), and well below the horizon (40% H).

SINGLE-BOUNDARY FALLBACK
-------------------------
If only one polynomial is available (e.g. one line not detected),
the function returns None — we cannot compute a meaningful offset
without both boundaries. The EMA smoother and classifier will hold
the last known value via their own fallback logic.

INPUTS
------
  left_poly  : np.ndarray [a, b, c] or None
  right_poly : np.ndarray [a, b, c] or None
  frame_w    : frame width  in pixels
  frame_h    : frame height in pixels

OUTPUT
------
  float — lateral offset in pixels, or None if both polys missing
"""

import numpy as np
from typing import Optional


# Reference row for polynomial evaluation — bottom 15% of the frame.
# Matches the A5 plan note: y = 0.85 × H in camera-view coordinates.
REF_Y_FRAC = 0.85

# Camera mounting bias correction (pixels).
# If the dashcam is not perfectly centred on the car, it will produce
# a systematic offset that causes constant WARN_LEFT or WARN_RIGHT.
#
#   CAMERA_MOUNT_BIAS_PX > 0  →  camera is mounted LEFT  of centre
#   CAMERA_MOUNT_BIAS_PX < 0  →  camera is mounted RIGHT of centre
#   CAMERA_MOUNT_BIAS_PX = 0  →  camera is assumed perfectly centred
#
# HOW TO CALIBRATE:
#   1. Drive for ~30 s on a straight road with the car properly centred.
#   2. Print the average raw_offset from DepartureResult.raw_offset.
#   3. Set CAMERA_MOUNT_BIAS_PX = that average value.
# The computed offset is then: (lane_center_x − frame_center) − bias
CAMERA_MOUNT_BIAS_PX: float = 0.0


def compute_lateral_offset(
    left_poly:  Optional[np.ndarray],
    right_poly: Optional[np.ndarray],
    frame_w:    int,
    frame_h:    int,
) -> Optional[float]:
    """
    Compute the car's lateral offset from the lane centre.

    Parameters
    ----------
    left_poly : np.ndarray shape (3,) or None
        Quadratic polynomial [a, b, c] for the left boundary: x = f(y).
    right_poly : np.ndarray shape (3,) or None
        Quadratic polynomial [a, b, c] for the right boundary: x = f(y).
    frame_w : int
        Frame width in pixels.
    frame_h : int
        Frame height in pixels.

    Returns
    -------
    float : lateral_offset = lane_center_x − frame_center
        Positive → car is LEFT  of lane centre.
        Negative → car is RIGHT of lane centre.
    None  : if both polynomials are missing (no lane data this frame).
    """
    if left_poly is None and right_poly is None:
        return None

    if left_poly is None or right_poly is None:
        # Only one boundary detected — offset not reliable enough to use.
        # Caller (EMA smoother) will hold the last smoothed value instead.
        return None

    ref_y = REF_Y_FRAC * frame_h
    frame_center = frame_w / 2.0

    left_x  = float(np.polyval(left_poly,  ref_y))
    right_x = float(np.polyval(right_poly, ref_y))

    # ── Plausibility guard: lane width sanity check ─────────────────────────
    # If Module A mistakes a road edge, barrier, or adjacent lane for
    # the ego-lane boundary, left_x and right_x will be wildly wrong.
    # A real ego lane at y ≈ 0.85H spans 120–900 px on a 1080p dashcam.
    # Outside this range: reject the measurement and let the EMA smoother
    # hold its last good value rather than injecting garbage.
    MIN_VALID_LANE_WIDTH_PX = 120
    MAX_VALID_LANE_WIDTH_PX = int(frame_w * 0.85)   # ~1632 px at 1920

    lane_width = right_x - left_x
    if lane_width < MIN_VALID_LANE_WIDTH_PX or lane_width > MAX_VALID_LANE_WIDTH_PX:
        return None   # implausible — discard this frame's measurement

    # Also reject if either boundary is outside the frame (extrapolation gone wrong)
    if left_x < 0 or right_x > frame_w:
        return None

    lane_center_x = (left_x + right_x) / 2.0
    return (lane_center_x - frame_center) - CAMERA_MOUNT_BIAS_PX
