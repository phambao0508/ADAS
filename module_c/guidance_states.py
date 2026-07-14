from states import (
    GuidanceState,
    ProximityState,
    ACTIVE_GUIDANCE_STATES,
    GUIDE_MESSAGES,
)

GUIDE_NONE    = GuidanceState.GUIDE_NONE
GUIDE_LEFT    = GuidanceState.GUIDE_LEFT
GUIDE_RIGHT   = GuidanceState.GUIDE_RIGHT
GUIDE_BOTH    = GuidanceState.GUIDE_BOTH
GUIDE_SLOW    = GuidanceState.GUIDE_SLOW
GUIDE_URGENT  = GuidanceState.GUIDE_URGENT

ACTIVE_GUIDE_STATES = ACTIVE_GUIDANCE_STATES

PROX_NONE       = ProximityState.NONE
PROX_DETECTED   = ProximityState.DETECTED
PROX_CLOSE      = ProximityState.CLOSE
PROX_VERY_CLOSE = ProximityState.VERY_CLOSE
