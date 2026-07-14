from typing import List, Tuple

from .guidance_states import PROX_NONE, PROX_DETECTED, PROX_CLOSE, PROX_VERY_CLOSE

FRONT_GATE_Y_FRAC = 0.75

PROXIMITY_CLOSE       = 0.02
PROXIMITY_VERY_CLOSE  = 0.06

def detect_front_proximity(
    vehicle_boxes: List[Tuple[float, float, float, float]],
    zone_left_x:   float,
    zone_right_x:  float,
    frame_w:       int,
    frame_h:       int,
) -> str:

    frame_area    = frame_w * frame_h
    front_gate_y  = FRONT_GATE_Y_FRAC * frame_h
    proximity     = PROX_NONE

    for (cx, cy, w, h) in vehicle_boxes:

        if not (zone_left_x <= cx <= zone_right_x):
            continue

        if cy >= front_gate_y:
            continue

        rel_area = (w * h) / frame_area

        if rel_area > PROXIMITY_VERY_CLOSE:

            return PROX_VERY_CLOSE

        elif rel_area > PROXIMITY_CLOSE:

            proximity = PROX_CLOSE

        elif proximity not in (PROX_CLOSE, PROX_VERY_CLOSE):

            proximity = PROX_DETECTED

    return proximity
