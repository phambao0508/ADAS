"""
Module A  —  Step A1: Ego-Lane Selector  (UFLDv2 VERSION)
==========================================================
BACKGROUND
----------
UFLDv2 outputs up to 4 lanes as lists of (x, y) pixel coordinates.
Unlike the previous YOLO-based selector, there are NO bounding boxes,
NO segmentation masks, and NO colour labels (white/yellow).

This step identifies which detected lanes form the left and right
boundaries of the ego lane (the lane the car is driving in).

HOW "CLOSEST" IS MEASURED
--------------------------
The car is assumed to be at x = frame_width / 2 (dashcam centre).

For each lane, we evaluate the lane's x-position at the BOTTOM of the
frame (where perspective separation is maximal). Lanes whose bottom
x-position is left of centre are LEFT candidates; right of centre are
RIGHT candidates. The closest on each side becomes the ego-lane boundary.

INPUTS
------
  lanes    : List[List[Tuple[int, int]]]
             Up to 4 lanes from UFLDv2. Each lane is a list of (x, y) tuples.
  frame_w  : frame width in pixels
  frame_h  : frame height in pixels

OUTPUTS
-------
  EgoLaneLines namedtuple:
    left_pts    : List[(y, x)] — left boundary points (NOTE: y,x order for poly_fitter)
    right_pts   : List[(y, x)] — right boundary points
    left_label  : always None (UFLDv2 doesn't classify colour)
    right_label : always None
    found       : True if at least one boundary was detected
"""

from typing import List, Optional, NamedTuple, Tuple

import numpy as np


class EgoLaneLines(NamedTuple):
    """Result of ego-lane selection from UFLDv2 lane coordinates."""
    left_pts:       List[Tuple[int, int]]     # (y, x) points or empty
    right_pts:      List[Tuple[int, int]]     # (y, x) points or empty
    left_label:     Optional[str]             # always None (no colour info)
    right_label:    Optional[str]             # always None
    found:          bool                      # True if at least one boundary found


# Ignore lane points above this fraction of frame height (skip sky/horizon)
HORIZON_Y_FRAC = 0.30

# Rows used to score ego-lane candidates. The lower rows dominate, but the
# mid-row term helps the selector follow bends instead of only the bonnet line.
SCORE_Y_FRACS = (0.60, 0.78, 0.92)


def _lane_x_at_y(filtered: List[Tuple[int, int]], target_y: float) -> float:
    """Interpolate a lane's x-position at target_y from sorted (x, y) points."""
    pts = sorted(filtered, key=lambda p: p[1])
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    return float(np.interp(target_y, ys, xs))


def select_ego_lane(
    lanes: List[List[Tuple[int, int]]],
    frame_w: int,
    frame_h: int,
) -> EgoLaneLines:
    """
    Identify the left and right ego-lane boundary lanes from UFLDv2 output.

    Parameters
    ----------
    lanes : List of lane coordinate lists from UFLDv2.
            Each lane is List[(x, y)] in pixel coordinates.
    frame_w : int  — video frame width in pixels
    frame_h : int  — video frame height in pixels

    Returns
    -------
    EgoLaneLines
    """
    cx_frame = frame_w / 2.0
    upper_limit_y = frame_h * HORIZON_Y_FRAC

    # ── Evaluate each lane's bottom-x to determine side ───────────────────
    left_candidates = []   # (score, lane_pts_yx)
    right_candidates = []

    for lane in lanes:
        if not lane or len(lane) < 2:
            continue

        # Filter points above horizon
        filtered = [(x, y) for (x, y) in lane if y >= upper_limit_y]
        if len(filtered) < 2:
            continue

        # Use the lower visible row for side assignment, but rank candidates
        # across several rows so curves are not reduced to one bottom point.
        ys_filtered = [p[1] for p in filtered]
        top_y = min(ys_filtered)
        bottom_y = max(ys_filtered)
        bottom_x = _lane_x_at_y(filtered, bottom_y)

        sample_dists = []
        for frac in SCORE_Y_FRACS:
            target_y = frame_h * frac
            if top_y <= target_y <= bottom_y:
                sample_x = _lane_x_at_y(filtered, target_y)
                sample_dists.append(abs(sample_x - cx_frame))

        if not sample_dists:
            sample_dists = [abs(bottom_x - cx_frame)]

        score = 0.45 * abs(bottom_x - cx_frame) + 0.55 * float(np.mean(sample_dists))

        # Convert (x, y) → (y, x) for downstream poly_fitter compatibility
        pts_yx = [(y, x) for (x, y) in filtered]

        if bottom_x < cx_frame:
            left_candidates.append((score, pts_yx))
        else:
            right_candidates.append((score, pts_yx))

    # ── Pick closest lane on each side ────────────────────────────────────
    left_pts = []
    right_pts = []

    if left_candidates:
        left_candidates.sort(key=lambda t: t[0])  # closest first
        left_pts = left_candidates[0][1]

    if right_candidates:
        right_candidates.sort(key=lambda t: t[0])  # closest first
        right_pts = right_candidates[0][1]

    found = bool(left_pts) or bool(right_pts)

    return EgoLaneLines(
        left_pts=left_pts,
        right_pts=right_pts,
        left_label=None,
        right_label=None,
        found=found,
    )
