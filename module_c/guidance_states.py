"""
Module C  —  Guidance State Constants  (C6)
===========================================
Re-exports typed enum constants from the centralised ``states`` module.
All existing string comparisons continue to work (StrEnum == str).
"""

from states import (
    GuidanceState,
    ProximityState,
    ACTIVE_GUIDANCE_STATES,
    GUIDE_MESSAGES,
)

# ── Re-export for backward compatibility ──────────────────────────────────
GUIDE_NONE    = GuidanceState.GUIDE_NONE
GUIDE_LEFT    = GuidanceState.GUIDE_LEFT
GUIDE_RIGHT   = GuidanceState.GUIDE_RIGHT
GUIDE_BOTH    = GuidanceState.GUIDE_BOTH
GUIDE_SLOW    = GuidanceState.GUIDE_SLOW
GUIDE_URGENT  = GuidanceState.GUIDE_URGENT

# All states that trigger a visible HUD banner
ACTIVE_GUIDE_STATES = ACTIVE_GUIDANCE_STATES

# ── Front proximity sub-states (internal use only) ────────────────────────
PROX_NONE       = ProximityState.NONE
PROX_DETECTED   = ProximityState.DETECTED
PROX_CLOSE      = ProximityState.CLOSE
PROX_VERY_CLOSE = ProximityState.VERY_CLOSE
