"""
Module B  —  Step B3: Departure State Classifier
=================================================
TASK
----
Map the smoothed lateral offset + boundary types → one of 6 named states.

THE 6 STATES
------------
    CENTERED          Car is well within the lane        (green)
    WARN_LEFT         Drifting toward the LEFT  boundary (yellow)
    WARN_RIGHT        Drifting toward the RIGHT boundary (yellow)
    DEPART_LEFT       Crossing a SOLID left boundary     (red)
    DEPART_RIGHT      Crossing a SOLID right boundary    (red)
    LANE_CHANGE_LEFT  Crossing a DASHED left boundary    (blue)
    LANE_CHANGE_RIGHT Crossing a DASHED right boundary   (blue)

SIGN CONVENTION  (inherited from B1)
-------------------------------------
    offset = lane_center_x − frame_center

    offset > 0  →  lane centre is RIGHT of car  →  car drifted LEFT
    offset < 0  →  lane centre is LEFT  of car  →  car drifted RIGHT
    offset = 0  →  centred

    ⚠️ BUG FIXED: The implementation plan B3 lists the signs REVERSED
    (it says "offset < 0 → WARN_LEFT").  This contradicts B1's own sign
    definition and the physical reality:

        If the car drifts LEFT, the road scene moves RIGHT in the camera.
        → Both boundary x-values increase  → lane_center_x increases
        → offset = lane_center_x − frame_centre becomes POSITIVE.
        → A positive offset means the car went LEFT  → WARN_LEFT ✓

    The code below implements the CORRECT sign convention from B1.

CLASSIFICATION LOGIC
--------------------
    |offset| < WARN_THRESHOLD (50 px)
        → CENTERED

    WARN_THRESHOLD ≤ |offset| < DEPART_THRESHOLD (100 px)
        offset > 0  →  WARN_LEFT   (car crept left)
        offset < 0  →  WARN_RIGHT  (car crept right)
        offset = 0  →  CENTERED    (exact boundary — shouldn't happen)

    |offset| ≥ DEPART_THRESHOLD (100 px)   AND   drifting LEFT  (offset > 0):
        left_type == 'dashed'  →  LANE_CHANGE_LEFT   (safe to cross)
        left_type == 'solid'   →  DEPART_LEFT        (danger!)

    |offset| ≥ DEPART_THRESHOLD (100 px)   AND   drifting RIGHT (offset < 0):
        right_type == 'dashed' →  LANE_CHANGE_RIGHT  (safe to cross)
        right_type == 'solid'  →  DEPART_RIGHT       (danger!)

INPUTS
------
  smoothed_offset : float | None
  left_type       : 'solid' | 'dashed'
  right_type      : 'solid' | 'dashed'

OUTPUT
------
  one of the STATE_* string constants defined below
"""

from typing import Optional


# ── Departure state constants ─────────────────────────────────────────────
CENTERED          = "CENTERED"
WARN_LEFT         = "WARN_LEFT"
WARN_RIGHT        = "WARN_RIGHT"
DEPART_LEFT       = "DEPART_LEFT"
DEPART_RIGHT      = "DEPART_RIGHT"
LANE_CHANGE_LEFT  = "LANE_CHANGE_LEFT"
LANE_CHANGE_RIGHT = "LANE_CHANGE_RIGHT"

# All non-centred states that the HUD should display actively
ACTIVE_STATES = {
    WARN_LEFT, WARN_RIGHT,
    DEPART_LEFT, DEPART_RIGHT,
    LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT,
}

# ── Offset thresholds (pixels) ────────────────────────────────────────────
# Tuning reference:
#   WARN_THRESHOLD   = 80  px  (raised from 50 — dashcam mounts are rarely
#                               perfectly centred; 50 px caused constant false
#                               WARN_RIGHT on off-centre cameras)
#   DEPART_THRESHOLD = 150 px  (raised from 100 — gives clean separation
#                               between genuine lane departure and camera bias)
WARN_THRESHOLD   = 80    # px — warning zone begins here
DEPART_THRESHOLD = 150   # px — crossing zone begins here


def classify_departure(
    smoothed_offset: Optional[float],
    left_type:       str = "solid",
    right_type:      str = "solid",
) -> str:
    """
    Map a smoothed lateral offset to one of 6 departure states.

    Parameters
    ----------
    smoothed_offset : float or None
        EMA-smoothed lateral offset from B2, in pixels.
        Positive → car is LEFT of lane centre.
        Negative → car is RIGHT of lane centre.
        None     → no lane data available; defaults to CENTERED.
    left_type : str
        Type of the left boundary: 'solid' or 'dashed'.
    right_type : str
        Type of the right boundary: 'solid' or 'dashed'.

    Returns
    -------
    str : one of the module-level state constants.
    """
    # No lane data available this frame → treat as centred (safe default)
    if smoothed_offset is None:
        return CENTERED

    abs_offset = abs(smoothed_offset)

    # ── Zone 1: Well within the lane ─────────────────────────────────────
    if abs_offset < WARN_THRESHOLD:
        return CENTERED

    # ── Zone 2: Drifting (50–100 px) ─────────────────────────────────────
    if abs_offset < DEPART_THRESHOLD:
        # offset > 0 → car went LEFT  (lane moved right in camera view)
        # offset < 0 → car went RIGHT (lane moved left  in camera view)
        if smoothed_offset > 0:
            return WARN_LEFT
        else:
            return WARN_RIGHT

    # ── Zone 3: Crossing (≥ 100 px) ──────────────────────────────────────
    if smoothed_offset > 0:
        # Drifting LEFT — check the left boundary type
        if left_type == "dashed":
            return LANE_CHANGE_LEFT    # intentional lane change (safe)
        else:
            return DEPART_LEFT         # crossing a solid line (DANGER)
    else:
        # Drifting RIGHT — check the right boundary type
        if right_type == "dashed":
            return LANE_CHANGE_RIGHT   # intentional lane change (safe)
        else:
            return DEPART_RIGHT        # crossing a solid line (DANGER)
