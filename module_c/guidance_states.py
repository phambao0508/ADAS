"""
Module C  —  Guidance State Constants  (C6)
===========================================
Defines the six guidance output states and their priority order.

States
------
    GUIDE_NONE    Road is clear ahead — no guidance banner shown.
    GUIDE_LEFT    Move left  — left lane clear + left boundary dashed.
    GUIDE_RIGHT   Move right — right lane clear + right boundary dashed.
    GUIDE_BOTH    Both lanes clear + both boundaries dashed → prefer left.
    GUIDE_SLOW    Reduce speed — no safe lane change available.
    GUIDE_URGENT  Brake immediately — front vehicle is VERY CLOSE.

Priority
--------
    GUIDE_URGENT  overrides everything (highest)
    GUIDE_SLOW    high priority
    GUIDE_LEFT / GUIDE_RIGHT / GUIDE_BOTH  normal
    GUIDE_NONE    lowest (no banner)

GUIDE_URGENT overrides any concurrent departure warning in the HUD.

Proximity sub-states (internal — not displayed)
-----------------------------------------------
    PROX_NONE        No vehicle in ego zone (or vehicle is FAR > 40 m)
    PROX_CLOSE       Vehicle detected at ~20–40 m  (relative area > 2%)
    PROX_VERY_CLOSE  Vehicle detected at ~10–20 m  (relative area > 6%)
"""

# ── Guidance output states ────────────────────────────────────────────────
GUIDE_NONE    = "GUIDE_NONE"      # No front vehicle — no guidance
GUIDE_LEFT    = "GUIDE_LEFT"      # Move left
GUIDE_RIGHT   = "GUIDE_RIGHT"     # Move right
GUIDE_BOTH    = "GUIDE_BOTH"      # Both clear — prefer left (overtaking)
GUIDE_SLOW    = "GUIDE_SLOW"      # Reduce speed — no safe path
GUIDE_URGENT  = "GUIDE_URGENT"    # Brake — very close vehicle

# All states that trigger a visible HUD banner
ACTIVE_GUIDE_STATES = {
    GUIDE_LEFT,
    GUIDE_RIGHT,
    GUIDE_BOTH,
    GUIDE_SLOW,
    GUIDE_URGENT,
}

# ── Front proximity sub-states (internal use only) ────────────────────────
PROX_NONE       = "NONE"         # No vehicle in ego zone (or FAR > 40 m)
PROX_CLOSE      = "CLOSE"        # ~20–40 m ahead
PROX_VERY_CLOSE = "VERY_CLOSE"   # ~10–20 m ahead (emergency)

# ── HUD banner text for each guidance state ────────────────────────────────
GUIDE_MESSAGES = {
    GUIDE_NONE:    "",
    GUIDE_LEFT:    "\u25c4\u25c4 MOVE LEFT \u2014 LEFT LANE IS CLEAR",
    GUIDE_RIGHT:   "MOVE RIGHT \u2014 RIGHT LANE IS CLEAR \u25ba\u25ba",
    GUIDE_BOTH:    "\u25c4 MOVE LEFT (PREFERRED)",
    GUIDE_SLOW:    "\u26a0 REDUCE SPEED",
    GUIDE_URGENT:  "!! BRAKE \u2014 VEHICLE VERY CLOSE !!",
}
