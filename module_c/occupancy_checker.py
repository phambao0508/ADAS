"""
Module C  —  Step C3: Adjacent Lane Occupancy Checker
======================================================
TASK
----
Determine whether the LEFT and RIGHT adjacent lanes are clear for a
potential lane change.

HOW IT WORKS
------------
Once the zone dividers are established (C1), every YOLO vehicle detection
is tested to see if it sits in the left or right zone AND is
geometrically ahead (close enough to matter for a merge).

    for each vehicle (cx, cy, w, h):

        LEFT ZONE:   cx < zone_left_x  AND  cy < 0.80 × H  → left_occupied
        RIGHT ZONE:  cx > zone_right_x AND  cy < 0.80 × H  → right_occupied

    left_clear  = not left_occupied
    right_clear = not right_occupied

WHY 0.80 × H (not 0.75 × H like the front vehicle check)?
-----------------------------------------------------------
The front-vehicle gate uses 0.75 × H because we need to react to
vehicles that are farther ahead (more lead time).

The adjacent-lane gate uses 0.80 × H because:
  - A vehicle in a neighbouring lane that is level with our rear bumper
    (y > 0.80 × H) is NOT in our merge path — we are already ahead of it.
  - We only need to block a lane change if the adjacent vehicle will
    actually be in our path during the merge, i.e. it must be visible
    in the upper 80% of the frame (far to medium distance ahead/alongside).

CENTRE POINT ASSIGNMENT
-----------------------
Vehicles are assigned by their CENTRE point (cx), not by box edges.
A large truck whose box straddles the ego/right boundary is assigned to
whichever zone contains its cx. This prevents double-counting.

INPUTS
------
  vehicle_boxes : List of (cx, cy, w, h) — vehicle detections, centre fmt
  zone_left_x   : float — left zone divider from C1
  zone_right_x  : float — right zone divider from C1
  frame_h       : int

OUTPUTS
-------
  (left_clear, right_clear) : Tuple[bool, bool]
    True  → that adjacent lane has no detected vehicle in the merge path
    False → a vehicle is present in that lane — lane change blocked
"""

from typing import List, Tuple


# ── Tuning constant ───────────────────────────────────────────────────────
# Only vehicles in the upper 80% of the frame are considered for adjacent
# lane occupancy. Vehicles below this row are alongside or behind us.
ADJACENT_GATE_Y_FRAC = 0.80
# ──────────────────────────────────────────────────────────────────────────


def check_adjacent_occupancy(
    vehicle_boxes: List[Tuple[float, float, float, float]],
    zone_left_x:   float,
    zone_right_x:  float,
    frame_h:       int,
) -> Tuple[bool, bool]:
    """
    Check whether the left and right adjacent lanes are clear.

    Parameters
    ----------
    vehicle_boxes : List of (cx, cy, w, h)
        Vehicle bounding boxes in centre format, vehicle classes only.
    zone_left_x : float
        Left zone divider x-position (from zone_definer).
    zone_right_x : float
        Right zone divider x-position (from zone_definer).
    frame_h : int
        Frame height in pixels.

    Returns
    -------
    (left_clear, right_clear) : Tuple[bool, bool]
        left_clear  = True  → no vehicle detected in left  adjacent lane
        right_clear = True  → no vehicle detected in right adjacent lane
    """
    adjacent_gate_y = ADJACENT_GATE_Y_FRAC * frame_h

    left_occupied  = False
    right_occupied = False

    for (cx, cy, w, h) in vehicle_boxes:

        # Only consider vehicles in the upper 80% of the frame
        if cy >= adjacent_gate_y:
            continue   # vehicle is alongside or behind — not in merge path

        # ── Left adjacent lane ────────────────────────────────────────────
        if cx < zone_left_x:
            left_occupied = True

        # ── Right adjacent lane ───────────────────────────────────────────
        if cx > zone_right_x:
            right_occupied = True

        # Short-circuit: no need to keep iterating if both lanes occupied
        if left_occupied and right_occupied:
            break

    return (not left_occupied), (not right_occupied)
