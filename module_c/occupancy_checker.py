from typing import List, Tuple

ADJACENT_GATE_Y_FRAC = 0.80

def check_adjacent_occupancy(
    vehicle_boxes: List[Tuple[float, float, float, float]],
    zone_left_x:   float,
    zone_right_x:  float,
    frame_h:       int,
) -> Tuple[bool, bool]:

    adjacent_gate_y = ADJACENT_GATE_Y_FRAC * frame_h

    left_occupied  = False
    right_occupied = False

    for (cx, cy, w, h) in vehicle_boxes:

        if cy >= adjacent_gate_y:
            continue

        if cx < zone_left_x:
            left_occupied = True

        if cx > zone_right_x:
            right_occupied = True

        if left_occupied and right_occupied:
            break

    return (not left_occupied), (not right_occupied)
