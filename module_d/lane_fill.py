import numpy as np
import cv2
from typing import List, Optional, Tuple

from .hud_colours import lane_fill_colour

FILL_ALPHA         = 0.75
DRAW_EDGE_LINES    = False
EDGE_LINE_ALPHA    = 0.70
EDGE_LINE_THICK    = 3
FEATHER_KSIZE      = 11
GRAD_TOP_ALPHA     = 0.05
GRAD_BOTTOM_ALPHA  = 1.0
POLY_STEP          = 2
Y_MARGIN_BOTTOM    = 0
Y_MARGIN_TOP       = 0
MIN_LANE_WIDTH_PX  = 50
MAX_LANE_WIDTH_FRAC = 0.78
MAX_EDGE_STEP_PX   = 24
MIN_RENDER_ROWS    = 18

def _compute_y_range(
    left_pts:  List[Tuple[int, int]],
    right_pts: List[Tuple[int, int]],
    frame_h:   int,
) -> Tuple[int, int]:

    all_ys = [y for (y, x) in left_pts] + [y for (y, x) in right_pts]
    if not all_ys:
        return int(frame_h * 0.40), int(frame_h * 0.95)

    y_top    = max(0, min(all_ys) - Y_MARGIN_TOP)
    y_bottom = min(frame_h - 1, max(all_ys) + Y_MARGIN_BOTTOM)
    return y_top, y_bottom

def _trim_unstable_rows(
    y_range: np.ndarray,
    left_xs: np.ndarray,
    right_xs: np.ndarray,
    frame_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    widths = right_xs.astype(np.float32) - left_xs.astype(np.float32)
    valid = (widths >= MIN_LANE_WIDTH_PX) & (widths <= frame_w * MAX_LANE_WIDTH_FRAC)

    if len(y_range) > 2:
        left_step = np.r_[0, np.abs(np.diff(left_xs))]
        right_step = np.r_[0, np.abs(np.diff(right_xs))]
        valid &= (left_step <= MAX_EDGE_STEP_PX) & (right_step <= MAX_EDGE_STEP_PX)

    if not np.any(valid):
        return y_range[:0], left_xs[:0], right_xs[:0]

    best_start = 0
    best_end = 0
    start = None
    for idx, ok in enumerate(valid):
        if ok and start is None:
            start = idx
        elif not ok and start is not None:
            if idx - start > best_end - best_start:
                best_start, best_end = start, idx
            start = None

    if start is not None and len(valid) - start > best_end - best_start:
        best_start, best_end = start, len(valid)

    trimmed = (
        y_range[best_start:best_end],
        left_xs[best_start:best_end],
        right_xs[best_start:best_end],
    )
    if len(trimmed[0]) >= MIN_RENDER_ROWS:
        return trimmed

    width_only = (widths >= MIN_LANE_WIDTH_PX) & (widths <= frame_w * MAX_LANE_WIDTH_FRAC)
    if not np.any(width_only):
        return trimmed

    best_start = 0
    best_end = 0
    start = None
    for idx, ok in enumerate(width_only):
        if ok and start is None:
            start = idx
        elif not ok and start is not None:
            if idx - start > best_end - best_start:
                best_start, best_end = start, idx
            start = None

    if start is not None and len(width_only) - start > best_end - best_start:
        best_start, best_end = start, len(width_only)

    return (
        y_range[best_start:best_end],
        left_xs[best_start:best_end],
        right_xs[best_start:best_end],
    )

def draw_lane_fill(
    frame:           np.ndarray,
    left_poly:       Optional[np.ndarray],
    right_poly:      Optional[np.ndarray],
    left_pts:        List[Tuple[int, int]],
    right_pts:       List[Tuple[int, int]],
    departure_state: str,
    fill_progress:   float = 1.0,
) -> np.ndarray:

    if left_poly is None or right_poly is None:
        return frame
    if fill_progress <= 0.0:
        return frame

    H, W = frame.shape[:2]
    colour = lane_fill_colour(departure_state)

    y_top, y_bottom = _compute_y_range(left_pts, right_pts, H)
    full_span = y_bottom - y_top
    if full_span <= 0:
        return frame

    y_sweep_top = y_top

    y_range = np.arange(y_sweep_top, y_bottom + 1, POLY_STEP)
    if len(y_range) < 2:
        return frame

    left_xs  = np.clip(np.polyval(left_poly,  y_range), 0, W - 1).astype(np.int32)
    right_xs = np.clip(np.polyval(right_poly, y_range), 0, W - 1).astype(np.int32)

    y_range, left_xs, right_xs = _trim_unstable_rows(y_range, left_xs, right_xs, W)
    if len(y_range) < MIN_RENDER_ROWS:
        return frame

    y_sweep_top = int(y_range[0])
    y_bottom = int(y_range[-1])

    if left_xs[-1] >= right_xs[-1]:
        return frame

    left_contour  = np.column_stack([left_xs, y_range])
    right_contour = np.column_stack([right_xs, y_range])[::-1]
    polygon = np.vstack([left_contour, right_contour])

    fill_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(fill_mask, [polygon], 255)

    soft_mask = cv2.GaussianBlur(
        fill_mask.astype(np.float32) / 255.0,
        (FEATHER_KSIZE, FEATHER_KSIZE), 0
    )

    region_h = y_bottom - y_sweep_top + 1
    gradient = np.linspace(GRAD_TOP_ALPHA, GRAD_BOTTOM_ALPHA, region_h, dtype=np.float32)

    alpha_map = np.zeros(H, dtype=np.float32)
    alpha_map[y_sweep_top:y_bottom + 1] = gradient
    alpha_2d = alpha_map[:, np.newaxis] * soft_mask

    alpha_2d *= FILL_ALPHA

    region_slice = slice(y_sweep_top, y_bottom + 1)
    region_frame = frame[region_slice].astype(np.float32)
    region_alpha = alpha_2d[region_slice]

    colour_arr = np.array(colour, dtype=np.float32)
    alpha_3d   = region_alpha[:, :, np.newaxis]

    blended = region_frame * (1.0 - alpha_3d) + colour_arr * alpha_3d

    out = frame.copy()
    out[region_slice] = np.clip(blended, 0, 255).astype(np.uint8)

    if DRAW_EDGE_LINES:
        bright = tuple(min(255, int(c * 0.5 + 128)) for c in colour)

        left_line  = left_contour.reshape(-1, 1, 2)
        right_line = np.column_stack([right_xs, y_range]).reshape(-1, 1, 2)

        overlay = out.copy()
        cv2.polylines(overlay, [left_line],  False, bright, EDGE_LINE_THICK, cv2.LINE_AA)
        cv2.polylines(overlay, [right_line], False, bright, EDGE_LINE_THICK, cv2.LINE_AA)
        cv2.addWeighted(overlay, EDGE_LINE_ALPHA, out, 1.0 - EDGE_LINE_ALPHA, 0, out)

    return out
