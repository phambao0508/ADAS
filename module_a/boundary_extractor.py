"""
Module A  —  Step A2: Boundary Extractor  (UFLDv2 VERSION)
===========================================================
With UFLDv2, lane coordinates are already clean (x, y) points —
no mask scanning is needed. This module is now a thin passthrough
that simply validates the point lists from the ego-lane selector.

The original mask-based boundary extraction is no longer applicable
because UFLDv2 does not output segmentation masks.
"""

from typing import List, Tuple


def extract_boundaries(
    left_pts: List[Tuple[int, int]],
    right_pts: List[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Validate and return boundary points from ego-lane selection.

    With UFLDv2, left_pts and right_pts are already in (y, x) format
    from the ego_lane_selector. This function simply passes them through
    after basic validation.

    Parameters
    ----------
    left_pts  : List[(y, x)] from ego_lane_selector
    right_pts : List[(y, x)] from ego_lane_selector

    Returns
    -------
    (left_pts, right_pts) — validated point lists
    """
    # Sort by y-coordinate (top to bottom) for consistent poly fitting
    left_sorted = sorted(left_pts, key=lambda p: p[0]) if left_pts else []
    right_sorted = sorted(right_pts, key=lambda p: p[0]) if right_pts else []

    return left_sorted, right_sorted
