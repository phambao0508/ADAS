"""
Module D  —  Step D1: Lane Fill Overlay
=========================================
TASK
----
Draw a semi-transparent coloured fill over the ego-lane area on the frame.
The colour changes based on the departure state from Module B.

⚠️ CRITICAL: MODEL DOES NOT OUTPUT A LANE AREA MASK
----------------------------------------------------
The YOLO model detects LANE LINE MARKINGS (white line, yellow line) as
segmentation masks. It does NOT produce a filled lane-area mask.

Therefore the ego-lane fill polygon is constructed from the boundary
POLYNOMIALS produced by Module A:
  - Evaluate left_poly  at each row y from y_top to y_bottom
  - Evaluate right_poly at each row y from y_top to y_bottom
  - Connect these points into a closed polygon
  - Fill with cv2.fillPoly()

HOW THE FILL WORKS
------------------
    For y in range(y_top, y_bottom, step=1):
        left_x  = polyval(left_poly,  y)
        right_x = polyval(right_poly, y)

    Polygon = left-column (top-to-bottom) + right-column (bottom-to-top)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], colour)
    output  = cv2.addWeighted(overlay, LANE_FILL_ALPHA,
                               frame, 1.0 - LANE_FILL_ALPHA, 0)

    → weights sum to 1.0, maintaining image brightness.
      LANE_FILL_ALPHA = 0.38 → 38% fill colour + 62% original frame.
      (The plan formula 'frame×1.0 + colour×0.38' is incorrect — would
       overexpose to 1.38. The code is the correct implementation.)

y_top and y_bottom
------------------
    y_top    = min(top end of left_pts, top end of right_pts)
             = the highest reliable point in the visible lane
    y_bottom = frame_height - 1   (car bonnet row)

    If polynomials are None (lane not detected this frame), skip.

INPUTS (from Module A LaneResult)
-----------------------------------
    frame      : np.ndarray (H, W, 3) BGR
    left_poly  : np.ndarray [a,b,c] or None
    right_poly : np.ndarray [a,b,c] or None
    left_pts   : List of (y, x) inner-edge points
    right_pts  : List of (y, x) inner-edge points
    departure_state : str  (from Module B DepartureResult.state)
    frame_h, frame_w : int

OUTPUT
------
    np.ndarray (H, W, 3) — frame with lane fill overlaid
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple

from .hud_colours import lane_fill_colour, LANE_FILL_ALPHA


def draw_lane_fill(
    frame:           np.ndarray,
    left_poly:       Optional[np.ndarray],
    right_poly:      Optional[np.ndarray],
    left_pts:        List[Tuple[int, int]],   # may include synthetic pts (for display only)
    right_pts:       List[Tuple[int, int]],
    departure_state: str,
    real_left_pts:   Optional[List[Tuple[int, int]]] = None,  # YOLO-only: used for y-range
    real_right_pts:  Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    """
    Draw a semi-transparent coloured polygon over the ego lane area.

    Parameters
    ----------
    frame           : BGR video frame
    left_poly       : [a,b,c] left boundary polynomial, or None
    right_poly      : [a,b,c] right boundary polynomial, or None
    left_pts        : (y,x) points for left boundary — may be synthetic
    right_pts       : (y,x) points for right boundary — may be synthetic
    departure_state : held departure state string from Module B
    real_left_pts   : YOLO-detected left pts only (no synthetic). Used for
                      y-range so synthetic full-frame points do not push
                      the fill outside the actual detected data range.
    real_right_pts  : same, for right boundary
    """
    if left_poly is None or right_poly is None:
        return frame   # cannot fill without both boundaries

    H, W = frame.shape[:2]

    # ── y-range: use REAL (YOLO-detected) points only ─────────────────────
    # Synthetic points span 0.35H → H and would push y_top far above the
    # actual detection range, causing the polynomial to be evaluated in a
    # region where it was never fitted → diverges, balloons, or floods.
    rl_pts = real_left_pts  if real_left_pts  is not None else left_pts
    rr_pts = real_right_pts if real_right_pts is not None else right_pts

    yl = [y for (y, x) in rl_pts]
    yr = [y for (y, x) in rr_pts]

    UPWARD_MARGIN_PX   = 40
    DOWNWARD_MARGIN_PX = 20

    if yl and yr:
        # Both sides detected this frame — use real data range
        y_top    = max(max(min(yl), min(yr)) - UPWARD_MARGIN_PX, 0)
        y_bottom = min(max(max(yl), max(yr)) + DOWNWARD_MARGIN_PX, int(H * 0.95))
    elif yl:
        # Only left detected — anchor to left, conservative right
        y_top    = max(min(yl) - UPWARD_MARGIN_PX, 0)
        y_bottom = min(max(yl) + DOWNWARD_MARGIN_PX, int(H * 0.95))
    elif yr:
        # Only right detected — anchor to right
        y_top    = max(min(yr) - UPWARD_MARGIN_PX, 0)
        y_bottom = min(max(yr) + DOWNWARD_MARGIN_PX, int(H * 0.95))
    else:
        # Neither side detected this frame (both using previous-frame polys).
        # Use a default range in the lower half of the frame where lane lines
        # are reliably within the polynomial's fitted region.
        # The reference-row width check below will still reject a bad poly.
        y_top    = int(0.45 * H)
        y_bottom = int(0.90 * H)

    y_bottom = min(y_bottom, H - 1)

    # ── Hard width check at reference row (y = 0.85 × H) ──────────────────
    # This is the most reliable point: always within the detection range,
    # same row used by Module B for offset. Check BEFORE evaluating full range.
    REF_Y        = int(0.85 * H)
    MAX_FILL_PX  = int(0.50 * W)   # single lane ≤ 50% of frame width
    MIN_FILL_PX  = 80
    ref_left_x   = int(np.clip(np.polyval(left_poly,  REF_Y), 0, W - 1))
    ref_right_x  = int(np.clip(np.polyval(right_poly, REF_Y), 0, W - 1))
    ref_width    = ref_right_x - ref_left_x
    if ref_width <= 0 or ref_width < MIN_FILL_PX or ref_width > MAX_FILL_PX:
        return frame   # 2-lane span, inverted, or phantom — reject

    # ── Evaluate polynomials at every row ──────────────────────────────────
    y_range  = np.arange(y_top, y_bottom + 1)
    if y_range.size == 0:
        return frame
    left_xs  = np.clip(np.polyval(left_poly,  y_range).astype(np.int32), 0, W - 1)
    right_xs = np.clip(np.polyval(right_poly, y_range).astype(np.int32), 0, W - 1)

    # ── Final sanity checks on the full polygon ────────────────────────────
    lane_widths = right_xs - left_xs
    if lane_widths.size == 0 or lane_widths.max() <= 0:
        return frame
    if lane_widths.max() < MIN_FILL_PX or lane_widths.max() > MAX_FILL_PX:
        return frame

    # ── Build closed polygon ───────────────────────────────────────────────
    # Left column: top → bottom  (x, y)
    left_col  = np.column_stack([left_xs,           y_range  ])
    # Right column: bottom → top (x, y) — to close the polygon
    right_col = np.column_stack([right_xs[::-1],    y_range[::-1]])

    polygon = np.vstack([left_col, right_col]).astype(np.int32)

    # ── Alpha blend: draw fill onto a copy, then blend ────────────────────
    colour  = lane_fill_colour(departure_state)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], colour)

    blended = cv2.addWeighted(overlay, LANE_FILL_ALPHA, frame, 1.0 - LANE_FILL_ALPHA, 0)
    return blended

