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

    if poly is None:
        return frame

    y_values = [y for (y, x) in pts]
    if y_values:
        y_top    = min(y_values)
        y_bottom = min(max(y_values) + 30, int(H * 0.95))
        y_bottom = min(y_bottom, H - 1)
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

    drawing  = True
    run      = 0
    segment  = []

    for y in range(y_bottom, y_top - 1, -1):
        x = int(np.clip(np.polyval(poly, y), 0, W - 1))

        if drawing:
            segment.append([x, y])
            run += 1
            if run >= BOUNDARY_DASH_LENGTH:

                if len(segment) >= 2:
                    pts_cv = np.array(segment, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(frame, [pts_cv], isClosed=False,
                                  color=colour, thickness=BOUNDARY_THICKNESS_DASHED)
                segment = []
                drawing = False
                run     = 0
        else:

            run += 1
            if run >= BOUNDARY_GAP_LENGTH:
                drawing = True
                run     = 0

    if drawing and len(segment) >= 2:
        pts_cv = np.array(segment, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts_cv], isClosed=False,
                      color=colour, thickness=BOUNDARY_THICKNESS_DASHED)

    return frame
