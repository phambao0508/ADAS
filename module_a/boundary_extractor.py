"""
Module A  —  Step A2: Boundary Extractor  (SEGMENTATION MASK VERSION)
======================================================================
TASK
----
Given the SEGMENTATION MASKS of the left and right lane lines, scan each
mask row-by-row and pick the INNER EDGE pixel on each row.

WHY SEGMENTATION MASKS (not bounding box edges)
------------------------------------------------
The model is a SEGMENTATION model — every detected line comes with a
pixel-precise mask of the painted marking. Using the mask inner edge
gives us exact sub-pixel accuracy for the boundary position, instead of
the rough rectangular box edge (x1 or x2) we had before.

WHICH EDGE TO USE
-----------------
The ego lane sits BETWEEN the two detected lines:

    LEFT line mask                        RIGHT line mask
    ┌──────────────┐   ego lane space   ┌──────────────┐
    │ ██████████ ← │ ◄───────────────► │ → ██████████ │
    └──────────────┘                   └──────────────┘
        x_rightmost of left mask           x_leftmost of right mask
        = left wall of ego lane            = right wall of ego lane

So:
  left boundary  ← rightmost filled pixel per row of the LEFT  line mask
  right boundary ← leftmost  filled pixel per row of the RIGHT line mask

ALGORITHM
---------
For every row y in the mask (sampled every SAMPLE_STEP rows):
    filled_xs = indices of all non-zero pixels in that row

    left mask:   if filled_xs not empty → record (y, max(filled_xs))  ← rightmost
    right mask:  if filled_xs not empty → record (y, min(filled_xs))  ← leftmost

SAMPLE_STEP = 5  (every 5th row)
  - Matches the plan's SAMPLE_STEP=5 specification
  - Keeps point count ~160 max for 1080p (vs. ~800 without stride)
  - Still gives poly_fitter far more than the 10-point minimum it needs

OUTPUTS
-------
  left_pts  : List of (y, x) tuples — left boundary, top to bottom
  right_pts : List of (y, x) tuples — right boundary, top to bottom

  These are the SAME FORMAT as before, so poly_fitter and
  line_type_classifier receive identical input types.
"""

import numpy as np
from typing import List, Optional, Tuple


BoundaryPoints = List[Tuple[int, int]]   # list of (row_y, col_x)

# Sample every Nth row of the mask — matches plan's SAMPLE_STEP=5.
# Keeps point lists small without losing polynomial fitting accuracy.
SAMPLE_STEP = 5


def extract_boundaries(
    left_mask:  Optional[np.ndarray],
    right_mask: Optional[np.ndarray],
) -> Tuple[BoundaryPoints, BoundaryPoints]:
    """
    Extract left and right lane boundary pixel coordinates from
    segmentation masks of the two bounding lane lines.

    Parameters
    ----------
    left_mask : np.ndarray (H, W) or None
        Segmentation mask of the LEFT boundary line.
        Non-zero pixels = painted marking.
    right_mask : np.ndarray (H, W) or None
        Segmentation mask of the RIGHT boundary line.
        Non-zero pixels = painted marking.

    Returns
    -------
    left_pts  : List of (y, x) — rightmost pixel per sampled row of left mask
    right_pts : List of (y, x) — leftmost  pixel per sampled row of right mask
    """
    left_pts:  BoundaryPoints = []
    right_pts: BoundaryPoints = []

    # ── LEFT boundary: rightmost pixel per sampled row ────────────────────
    if left_mask is not None:
        h = left_mask.shape[0]
        for y in range(0, h, SAMPLE_STEP):
            xs = np.where(left_mask[y] > 0)[0]
            if len(xs) > 0:
                left_pts.append((y, int(xs[-1])))   # rightmost = inner edge

    # ── RIGHT boundary: leftmost pixel per sampled row ────────────────────
    if right_mask is not None:
        h = right_mask.shape[0]
        for y in range(0, h, SAMPLE_STEP):
            xs = np.where(right_mask[y] > 0)[0]
            if len(xs) > 0:
                right_pts.append((y, int(xs[0])))   # leftmost = inner edge

    return left_pts, right_pts
