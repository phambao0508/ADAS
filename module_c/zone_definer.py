"""
Module C  —  Step C1: Zone Definer
===================================
TASK
----
Partition the camera frame into three horizontal zones by computing
two x-position dividers from the ego-lane boundary polynomials.

    LEFT ZONE  │  EGO LANE ZONE  │  RIGHT ZONE
    x < left_x │ left_x ≤ x ≤ right_x │ x > right_x

Every YOLO vehicle bounding box is then assigned to exactly one zone
based on its centre x-coordinate (cx).

HOW THE DIVIDERS ARE COMPUTED
-----------------------------
The boundary polynomials from Module A represent the INNER EDGE of the
left and right lane LINE MARKINGS (not a filled lane area mask).

They are evaluated at a fixed reference row y = 0.80 × H:

    zone_divider_left  = polyval(left_poly,  0.80 × H)
    zone_divider_right = polyval(right_poly, 0.80 × H)

Why y = 0.80 × H?
  - Close to the car → accurate lane geometry
  - Far enough ahead → gives driver reaction time
  - Above the car bonnet region (y > 0.95 × H)

WHY NOT BEV OR MASK?
--------------------
The model detects lane LINE markings (class 3: white line, class 4:
yellow line) — it does NOT output a filled lane area mask.
The inner-edge polynomials from Module A are the only source of
lane boundary position. BEV is used for visualisation only.

FALLBACK
--------
If either polynomial is None (Module A failed this frame), fixed
fractional dividers are used:
    zone_divider_left  = 0.35 × W
    zone_divider_right = 0.65 × W

INPUTS
------
  left_poly  : np.ndarray [a, b, c] or None   — left boundary polynomial
  right_poly : np.ndarray [a, b, c] or None   — right boundary polynomial
  frame_w    : int
  frame_h    : int

OUTPUTS
-------
  (zone_left_x, zone_right_x) : Tuple[float, float]
    x-coordinates of the left and right lane boundaries at y = 0.80 × H.
"""

import numpy as np
from typing import Optional, Tuple


# Reference row for zone divider evaluation — matches C1 plan specification.
ZONE_REF_Y_FRAC = 0.80

# Fallback dividers when polynomials are unavailable.
# These are rough approximations for a centred dashcam.
FALLBACK_LEFT_FRAC  = 0.35
FALLBACK_RIGHT_FRAC = 0.65


def compute_zone_dividers(
    left_poly:  Optional[np.ndarray],
    right_poly: Optional[np.ndarray],
    frame_w:    int,
    frame_h:    int,
) -> Tuple[float, float]:
    """
    Compute the left and right x-dividers that define the three road zones.

    Parameters
    ----------
    left_poly : np.ndarray shape (3,) or None
        Quadratic polynomial [a, b, c] for the LEFT boundary inner edge.
        Evaluates as: x = a·y² + b·y + c
    right_poly : np.ndarray shape (3,) or None
        Quadratic polynomial [a, b, c] for the RIGHT boundary inner edge.
    frame_w : int
        Frame width in pixels.
    frame_h : int
        Frame height in pixels.

    Returns
    -------
    (zone_left_x, zone_right_x) : Tuple[float, float]
        x-pixel positions of the ego-lane left and right walls
        at y = 0.80 × H.

        A vehicle bounding box with centre cx is assigned to:
          LEFT  zone  if  cx < zone_left_x
          EGO   zone  if  zone_left_x ≤ cx ≤ zone_right_x
          RIGHT zone  if  cx > zone_right_x
    """
    ref_y = ZONE_REF_Y_FRAC * frame_h

    if left_poly is not None and right_poly is not None:
        zone_left_x  = float(np.polyval(left_poly,  ref_y))
        zone_right_x = float(np.polyval(right_poly, ref_y))
    else:
        # Fallback: polynomial unavailable this frame
        zone_left_x  = FALLBACK_LEFT_FRAC  * frame_w
        zone_right_x = FALLBACK_RIGHT_FRAC * frame_w

    return zone_left_x, zone_right_x


def assign_zone(cx: float, zone_left_x: float, zone_right_x: float) -> str:
    """
    Assign a vehicle centre x-coordinate to a zone label.

    Parameters
    ----------
    cx : float
        Horizontal centre of the vehicle bounding box.
    zone_left_x : float
        Left divider from compute_zone_dividers().
    zone_right_x : float
        Right divider from compute_zone_dividers().

    Returns
    -------
    str : "LEFT", "EGO", or "RIGHT"
    """
    if cx < zone_left_x:
        return "LEFT"
    elif cx > zone_right_x:
        return "RIGHT"
    else:
        return "EGO"
