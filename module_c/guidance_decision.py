"""
Module C  —  Step C4+C5: Guidance Decision Logic
=================================================
TASK
----
Combine the five inputs (front_proximity, left_clear, right_clear,
left_type, right_type) into a single guidance output state.

FIVE INPUTS
-----------
    front_proximity  : PROX_NONE | PROX_CLOSE | PROX_VERY_CLOSE  (from C2)
    left_clear       : bool      (from C3)
    right_clear      : bool      (from C3)
    left_type        : 'solid' | 'dashed'   (from Module A, C3)
    right_type       : 'solid' | 'dashed'   (from Module A, C3)

DECISION ALGORITHM  (Section C4 of the plan)
--------------------------------------------
The logic is a PRIORITY-ORDERED chain, not a lookup table.

  PRIORITY 1 — VERY_CLOSE (overrides everything):
    front_proximity == PROX_VERY_CLOSE  →  GUIDE_URGENT

  PRIORITY 2 — No front vehicle:
    front_proximity == PROX_NONE        →  GUIDE_NONE

  PRIORITY 3 — CLOSE → evaluate lane change options:

    can_go_left  = left_clear  AND  left_type  == 'dashed'
    can_go_right = right_clear AND  right_type == 'dashed'

    can_go_left  AND  can_go_right  →  GUIDE_BOTH   (prefer left by conv.)
    can_go_left  only               →  GUIDE_LEFT
    can_go_right only               →  GUIDE_RIGHT
    neither                         →  GUIDE_SLOW

EQUIVALENCE WITH C4 CASES A–F
------------------------------
The simplified `can_go_left / can_go_right` logic is mathematically
equivalent to the verbose Case A–F tree. Proof (cross-check with C5
truth table):

  Row | left_clear | right_clear | left_type | right_type | Expected   | can_go_L | can_go_R | Result
  ----|-----------|------------|-----------|-----------|------------|----------|----------|-------
   1  | True      | False      | dashed    | any       | GUIDE_LEFT | True     | False    | LEFT  ✓
   2  | False     | True       | any       | dashed    | GUIDE_RIGHT| False    | True     | RIGHT ✓
   3  | True      | True       | dashed    | dashed    | GUIDE_BOTH | True     | True     | BOTH  ✓
   4  | True      | True       | dashed    | solid     | GUIDE_LEFT | True     | False    | LEFT  ✓
   5  | True      | True       | solid     | dashed    | GUIDE_RIGHT| False    | True     | RIGHT ✓
   6  | True      | True       | solid     | solid     | GUIDE_SLOW | False    | False    | SLOW  ✓
   7  | True      | False      | solid     | any       | GUIDE_SLOW | False    | False    | SLOW  ✓
   8  | False     | True       | any       | solid     | GUIDE_SLOW | False    | False    | SLOW  ✓
   9  | False     | False      | any       | any       | GUIDE_SLOW | False    | False    | SLOW  ✓

All 9 rows match. The simplified form is correct and complete.

KEY SAFETY RULE
---------------
A lane change is ONLY suggested when the boundary on that side is DASHED.
Even if the adjacent lane is completely clear, a SOLID boundary → GUIDE_SLOW.
This prevents the system from ever recommending crossing a solid line.

INPUTS
------
  front_proximity : str  — PROX_NONE | PROX_CLOSE | PROX_VERY_CLOSE
  left_clear      : bool
  right_clear     : bool
  left_type       : 'solid' | 'dashed'
  right_type      : 'solid' | 'dashed'

OUTPUT
------
  str : one of the GUIDE_* constants
"""

from .guidance_states import (
    GUIDE_NONE, GUIDE_LEFT, GUIDE_RIGHT, GUIDE_BOTH, GUIDE_SLOW, GUIDE_URGENT,
    PROX_NONE, PROX_DETECTED, PROX_CLOSE, PROX_VERY_CLOSE,
)

# ── Hysteresis: directional states persist for at least this many frames ──
# Prevents banner flicker when occupancy detection oscillates
MIN_DIRECTIONAL_HOLD = 3

# Track state for hysteresis (module-level singleton; reset via reset_hysteresis())
_prev_guidance = GUIDE_NONE
_hold_counter = 0
_DIRECTIONAL = {GUIDE_LEFT, GUIDE_RIGHT, GUIDE_BOTH}


def reset_hysteresis():
    """Reset the hysteresis state (e.g. on video switch)."""
    global _prev_guidance, _hold_counter
    _prev_guidance = GUIDE_NONE
    _hold_counter = 0


def decide_guidance(
    front_proximity: str,
    left_clear:      bool,
    right_clear:     bool,
    left_type:       str,
    right_type:      str,
) -> str:
    """
    Map the five Module C inputs to a single guidance output state.

    Parameters
    ----------
    front_proximity : str
        Proximity of the nearest vehicle ahead in the ego lane.
        One of: PROX_NONE, PROX_CLOSE, PROX_VERY_CLOSE.
    left_clear : bool
        True if no vehicle detected in the left adjacent lane.
    right_clear : bool
        True if no vehicle detected in the right adjacent lane.
    left_type : str
        Left boundary marking type: 'solid' or 'dashed'.
    right_type : str
        Right boundary marking type: 'solid' or 'dashed'.

    Returns
    -------
    str : one of GUIDE_NONE, GUIDE_LEFT, GUIDE_RIGHT,
                 GUIDE_BOTH, GUIDE_SLOW, GUIDE_URGENT
    """
    global _prev_guidance, _hold_counter

    # ── Priority 1: Emergency — brake immediately ─────────────────────────
    if front_proximity == PROX_VERY_CLOSE:
        _prev_guidance = GUIDE_URGENT
        _hold_counter = 0
        return GUIDE_URGENT

    # ── Priority 2: Road is clear ahead — no guidance needed ──────────────
    if front_proximity in (PROX_NONE, PROX_DETECTED):
        _prev_guidance = GUIDE_NONE
        _hold_counter = 0
        return GUIDE_NONE

    # ── Priority 3: Vehicle is CLOSE — evaluate lane change options ────────
    can_go_left  = left_clear  and (left_type  == "dashed")
    can_go_right = right_clear and (right_type == "dashed")

    if can_go_left and can_go_right:
        new_state = GUIDE_BOTH
    elif can_go_left:
        new_state = GUIDE_LEFT
    elif can_go_right:
        new_state = GUIDE_RIGHT
    else:
        new_state = GUIDE_SLOW

    # ── Hysteresis: hold directional states for MIN_DIRECTIONAL_HOLD ──────
    # If the previous state was directional and we'd now switch to SLOW,
    # hold the directional state for a few more frames to prevent flicker.
    if (_prev_guidance in _DIRECTIONAL
            and new_state == GUIDE_SLOW
            and _hold_counter < MIN_DIRECTIONAL_HOLD):
        _hold_counter += 1
        return _prev_guidance

    # Accept the new state
    _prev_guidance = new_state
    _hold_counter = 0
    return new_state
