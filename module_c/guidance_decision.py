from .guidance_states import (
    GUIDE_NONE, GUIDE_LEFT, GUIDE_RIGHT, GUIDE_BOTH, GUIDE_SLOW, GUIDE_URGENT,
    PROX_NONE, PROX_DETECTED, PROX_CLOSE, PROX_VERY_CLOSE,
)

MIN_DIRECTIONAL_HOLD = 3

_prev_guidance = GUIDE_NONE
_hold_counter = 0
_DIRECTIONAL = {GUIDE_LEFT, GUIDE_RIGHT, GUIDE_BOTH}

def reset_hysteresis():

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

    global _prev_guidance, _hold_counter

    if front_proximity == PROX_VERY_CLOSE:
        _prev_guidance = GUIDE_URGENT
        _hold_counter = 0
        return GUIDE_URGENT

    if front_proximity in (PROX_NONE, PROX_DETECTED):
        _prev_guidance = GUIDE_NONE
        _hold_counter = 0
        return GUIDE_NONE

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

    if (_prev_guidance in _DIRECTIONAL
            and new_state == GUIDE_SLOW
            and _hold_counter < MIN_DIRECTIONAL_HOLD):
        _hold_counter += 1
        return _prev_guidance

    _prev_guidance = new_state
    _hold_counter = 0
    return new_state
