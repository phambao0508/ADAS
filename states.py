"""
ADAS — Centralised State Enums
================================
All typed state constants for the entire project.

Uses ``StrEnum`` (Python 3.11+) so enum members compare equal to their
string values:  ``DepartureState.CENTERED == "CENTERED"``  → True.

This means existing string comparisons throughout the codebase continue
to work without modification, while gaining type-safety and IDE
autocomplete.

Modules should import from here:
    from states import DepartureState, GuidanceState, ProximityState, LineType
"""

from enum import StrEnum


# ── Module B: Lane Departure Warning ──────────────────────────────────────
class DepartureState(StrEnum):
    """One of seven departure states produced by Module B."""
    CENTERED          = "CENTERED"
    WARN_LEFT         = "WARN_LEFT"
    WARN_RIGHT        = "WARN_RIGHT"
    DEPART_LEFT       = "DEPART_LEFT"
    DEPART_RIGHT      = "DEPART_RIGHT"
    LANE_CHANGE_LEFT  = "LANE_CHANGE_LEFT"
    LANE_CHANGE_RIGHT = "LANE_CHANGE_RIGHT"


# All non-centred states that trigger an active HUD display
ACTIVE_DEPARTURE_STATES = {
    DepartureState.WARN_LEFT, DepartureState.WARN_RIGHT,
    DepartureState.DEPART_LEFT, DepartureState.DEPART_RIGHT,
    DepartureState.LANE_CHANGE_LEFT, DepartureState.LANE_CHANGE_RIGHT,
}


# ── Module C: Guidance ────────────────────────────────────────────────────
class GuidanceState(StrEnum):
    """One of six guidance output states produced by Module C."""
    GUIDE_NONE    = "GUIDE_NONE"
    GUIDE_LEFT    = "GUIDE_LEFT"
    GUIDE_RIGHT   = "GUIDE_RIGHT"
    GUIDE_BOTH    = "GUIDE_BOTH"
    GUIDE_SLOW    = "GUIDE_SLOW"
    GUIDE_URGENT  = "GUIDE_URGENT"


ACTIVE_GUIDANCE_STATES = {
    GuidanceState.GUIDE_LEFT,
    GuidanceState.GUIDE_RIGHT,
    GuidanceState.GUIDE_BOTH,
    GuidanceState.GUIDE_SLOW,
    GuidanceState.GUIDE_URGENT,
}


# ── Module C: Vehicle Proximity (internal sub-state) ──────────────────────
class ProximityState(StrEnum):
    """Front vehicle proximity — internal to Module C."""
    NONE       = "NONE"
    DETECTED   = "DETECTED"
    CLOSE      = "CLOSE"
    VERY_CLOSE = "VERY_CLOSE"


# ── Module A: Lane Line Type ─────────────────────────────────────────────
class LineType(StrEnum):
    """Classification of a lane boundary marking."""
    SOLID  = "solid"
    DASHED = "dashed"


# ── HUD Banner Messages ──────────────────────────────────────────────────
GUIDE_MESSAGES = {
    GuidanceState.GUIDE_NONE:    "",
    GuidanceState.GUIDE_LEFT:    "\u25c4\u25c4 MOVE LEFT \u2014 LEFT LANE IS CLEAR",
    GuidanceState.GUIDE_RIGHT:   "MOVE RIGHT \u2014 RIGHT LANE IS CLEAR \u25ba\u25ba",
    GuidanceState.GUIDE_BOTH:    "\u25c4 MOVE LEFT (PREFERRED)",
    GuidanceState.GUIDE_SLOW:    "\u26a0 REDUCE SPEED",
    GuidanceState.GUIDE_URGENT:  "!! BRAKE \u2014 VEHICLE VERY CLOSE !!",
}

# ── HUD Text Labels for Departure States ─────────────────────────────────
STATE_LABELS = {
    DepartureState.CENTERED:          "  LANE CENTERED  ",
    DepartureState.WARN_LEFT:         "<< DRIFTING LEFT",
    DepartureState.WARN_RIGHT:        "DRIFTING RIGHT >>",
    DepartureState.DEPART_LEFT:       "!! LANE DEPARTURE <<",
    DepartureState.DEPART_RIGHT:      "LANE DEPARTURE!! >>",
    DepartureState.LANE_CHANGE_LEFT:  "<< LANE CHANGE",
    DepartureState.LANE_CHANGE_RIGHT: "LANE CHANGE >>",
}
