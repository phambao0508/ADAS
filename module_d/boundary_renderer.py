"""
Module D  —  Step D2: Boundary Line Renderer
==============================================
TASK
----
Draw the left and right lane boundary lines on the frame.

  - SOLID boundary  → one continuous thick line (4 px)
  - DASHED boundary → alternating drawn/gap segments (mimics road markings)

The boundary colour adapts to the departure state:
  - Left  boundary turns RED when drifting/departing LEFT
  - Right boundary turns RED when drifting/departing RIGHT
  - Otherwise: Left = Cyan, Right = Orange

DRAW SOURCE
-----------
Boundary lines are drawn from the inner-edge sample POINTS collected
by Module A's boundary extractor (left_pts, right_pts).
Each is a List of (y, x) tuples. These are converted to (x, y) for OpenCV.

Alternatively, the polynomial can be evaluated at every row to get a
smoother curve — this module uses the POLYNOMIAL for smoother rendering
(the raw pts are noisy; poly gives a clean fitted curve).

DASHED LINE ALGORITHM
---------------------
Iterate along the boundary polynomial from bottom to top.
Alternate between drawing segments and skipping gaps:

    state = DRAW
    run   = 0

    for y from y_bottom to y_top (step -1):
        x = polyval(poly, y)
        if state == DRAW:
            add point to current segment
            run += 1
            if run >= DASH_LENGTH:
                draw segment, clear buffer, state = GAP, run = 0
        else:  # GAP
            run += 1
            if run >= GAP_LENGTH:
                state = DRAW, run = 0

INPUTS (from LaneResult + DepartureResult)
------------------------------------------
    frame           : np.ndarray (H, W, 3)
    left_poly       : np.ndarray [a,b,c] or None
    right_poly      : np.ndarray [a,b,c] or None
    left_pts        : List[(y,x)]   — for y_range estimation
    right_pts       : List[(y,x)]
    left_type       : 'solid' | 'dashed'
    right_type      : 'solid' | 'dashed'
    departure_state : str

OUTPUT
------
    np.ndarray : frame with boundary lines drawn
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple

from .hud_colours import (
    boundary_colour_left,
    boundary_colour_right,
    BOUNDARY_THICKNESS_SOLID,
    BOUNDARY_THICKNESS_DASHED,
    BOUNDARY_DASH_LENGTH,
    BOUNDARY_GAP_LENGTH,
)


def draw_boundaries(
    frame:           np.ndarray,
    left_poly:       Optional[np.ndarray],
    right_poly:      Optional[np.ndarray],
    left_pts:        List[Tuple[int, int]],
    right_pts:       List[Tuple[int, int]],
    left_type:       str,
    right_type:      str,
    departure_state: str,
) -> np.ndarray:
    """
    Draw left and right boundary lines on the frame.

    Parameters
    ----------
    frame : np.ndarray (H, W, 3), BGR
    left_poly / right_poly : np.ndarray [a, b, c] or None
    left_pts / right_pts : List of (y, x) inner-edge sample points
    left_type / right_type : 'solid' or 'dashed'
    departure_state : str  — from Module B DepartureResult.state

    Returns
    -------
    np.ndarray : frame with boundaries drawn
    """
    H, W = frame.shape[:2]

    col_left  = boundary_colour_left(departure_state)
    col_right = boundary_colour_right(departure_state)

    frame = _draw_one_boundary(
        frame, left_poly,  left_pts,  left_type,  col_left,  H, W
    )
    frame = _draw_one_boundary(
        frame, right_poly, right_pts, right_type, col_right, H, W
    )
    return frame


def _draw_one_boundary(
    frame:   np.ndarray,
    poly:    Optional[np.ndarray],
    pts:     List[Tuple[int, int]],
    btype:   str,
    colour:  Tuple[int, int, int],
    H:       int,
    W:       int,
) -> np.ndarray:
    """Draw a single boundary (left or right) onto the frame."""
    if poly is None:
        return frame   # no polynomial for this boundary this frame

    # ── Determine y range from raw points ─────────────────────────────────
    # Upward extrapolation (smaller y) is SAFE — polynomial converges.
    # Downward extrapolation (larger y) is DANGEROUS — diverges rapidly.
    # So: extend to the full top of detected pts; limit downward to data + margin.
    y_values = [y for (y, x) in pts]
    if y_values:
        y_top    = min(y_values)                             # full upward extent
        y_bottom = min(max(y_values) + 30, int(H * 0.95))   # slight downward margin
        y_bottom = min(y_bottom, H - 1)                     # never exceed frame
    else:
        y_top    = int(0.40 * H)
        y_bottom = int(0.95 * H)

    if btype == "solid":
        frame = _draw_solid(frame, poly, y_top, y_bottom, colour, W)
    else:
        frame = _draw_dashed(frame, poly, y_top, y_bottom, colour, W)

    return frame


def _draw_solid(
    frame:    np.ndarray,
    poly:     np.ndarray,
    y_top:    int,
    y_bottom: int,
    colour:   Tuple[int, int, int],
    W:        int,
) -> np.ndarray:
    """Draw a continuous thick polyline."""
    y_range = np.arange(y_top, y_bottom + 1)
    xs      = np.clip(np.polyval(poly, y_range).astype(np.int32), 0, W - 1)

    pts_cv  = np.column_stack([xs, y_range]).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts_cv], isClosed=False,
                  color=colour, thickness=BOUNDARY_THICKNESS_SOLID)
    return frame


def _draw_dashed(
    frame:    np.ndarray,
    poly:     np.ndarray,
    y_top:    int,
    y_bottom: int,
    colour:   Tuple[int, int, int],
    W:        int,
) -> np.ndarray:
    """Draw alternating draw/gap segments to mimic dashed road markings."""
    drawing  = True    # start with a drawn segment
    run      = 0
    segment  = []

    for y in range(y_bottom, y_top - 1, -1):   # iterate bottom → top
        x = int(np.clip(np.polyval(poly, y), 0, W - 1))

        if drawing:
            segment.append([x, y])
            run += 1
            if run >= BOUNDARY_DASH_LENGTH:
                # Flush the current segment
                if len(segment) >= 2:
                    pts_cv = np.array(segment, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(frame, [pts_cv], isClosed=False,
                                  color=colour, thickness=BOUNDARY_THICKNESS_DASHED)
                segment = []
                drawing = False
                run     = 0
        else:
            # Gap — skip drawing
            run += 1
            if run >= BOUNDARY_GAP_LENGTH:
                drawing = True
                run     = 0

    # Flush any remaining segment
    if drawing and len(segment) >= 2:
        pts_cv = np.array(segment, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts_cv], isClosed=False,
                      color=colour, thickness=BOUNDARY_THICKNESS_DASHED)

    return frame
