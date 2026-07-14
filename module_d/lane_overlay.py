import numpy as np
import cv2
from typing import Optional

from .hud_colours import lane_fill_colour

CORE_ALPHA        = 0.55
CONTOUR_THICKNESS = 3
CONTOUR_GLOW_SIZE = 15
CONTOUR_GLOW_ALPHA= 0.50
FEATHER_KSIZE     = 7
TRAIL_FADE_FRAC   = 0.20
GLOW_ROWS         = 8
GLOW_BRIGHTNESS   = 1.6

def _ease_in_out(t: float) -> float:

    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4.0 * t * t * t
    else:
        return 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0

def _render_single_side(
    frame:    np.ndarray,
    mask:     Optional[np.ndarray],
    colour:   tuple,
    progress: float,
) -> np.ndarray:

    if mask is None or progress <= 0.0:
        return frame

    H, W = frame.shape[:2]

    side_mask = mask.copy()

    rows_with_mask = np.where(side_mask.max(axis=1) > 0)[0]
    if len(rows_with_mask) == 0:
        return frame

    y_top_mask    = int(rows_with_mask[0])
    y_bottom_mask = int(rows_with_mask[-1])
    mask_span     = y_bottom_mask - y_top_mask

    if mask_span <= 0:
        return frame

    eased = _ease_in_out(max(0.0, min(1.0, progress)))

    y_threshold = int(y_bottom_mask - eased * mask_span)

    side_mask[:y_threshold, :] = 0

    if side_mask.max() == 0:
        return frame

    soft_mask = cv2.GaussianBlur(
        side_mask.astype(np.float32) / 255.0,
        (FEATHER_KSIZE, FEATHER_KSIZE), 0
    )

    fade_rows = max(1, int(mask_span * TRAIL_FADE_FRAC))
    region_h = y_bottom_mask - y_threshold + 1

    alpha_1d = np.ones(region_h, dtype=np.float32)
    fade_end = min(fade_rows, region_h)
    if fade_end > 0:
        alpha_1d[:fade_end] = np.linspace(0.0, 1.0, fade_end, dtype=np.float32)
    alpha_1d *= CORE_ALPHA

    glow_end = min(GLOW_ROWS, region_h)
    if glow_end > 0:
        for i in range(glow_end):
            glow_t = 1.0 - (i / glow_end)
            glow_factor = 1.0 + (GLOW_BRIGHTNESS - 1.0) * glow_t * glow_t
            alpha_1d[i] = min(1.0, alpha_1d[i] * glow_factor)

    region_slice = slice(y_threshold, y_bottom_mask + 1)
    region_frame = frame[region_slice].astype(np.float32)
    region_soft  = soft_mask[region_slice]

    alpha_2d = alpha_1d[:, np.newaxis] * region_soft

    colour_arr = np.array(colour, dtype=np.float32)
    alpha_3d = alpha_2d[:, :, np.newaxis]

    blended = region_frame * (1.0 - alpha_3d) + colour_arr * alpha_3d

    contour_mask = side_mask[region_slice]
    contours, _ = cv2.findContours(
        contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        bright_colour = tuple(min(255, int(c * 0.4 + 153)) for c in colour)
        contour_layer = np.zeros_like(blended, dtype=np.float32)
        cv2.drawContours(
            contour_layer,
            contours, -1,
            color=[float(c) for c in bright_colour],
            thickness=CONTOUR_THICKNESS,
            lineType=cv2.LINE_AA,
        )

        glow_layer = cv2.GaussianBlur(
            contour_layer,
            (CONTOUR_GLOW_SIZE, CONTOUR_GLOW_SIZE), 0
        )

        glow_visible = (glow_layer.sum(axis=2, keepdims=True) > 0).astype(np.float32)
        blended = blended + glow_layer * CONTOUR_GLOW_ALPHA * glow_visible

        contour_visible = (contour_layer.sum(axis=2, keepdims=True) > 0).astype(np.float32)
        contour_alpha = 0.85
        blended = blended * (1.0 - contour_visible * contour_alpha) + contour_layer * contour_visible * contour_alpha

    frame = frame.copy()
    frame[region_slice] = np.clip(blended, 0, 255).astype(np.uint8)

    return frame

def draw_lane_lines(
    frame:               np.ndarray,
    left_mask:           Optional[np.ndarray],
    right_mask:          Optional[np.ndarray],
    departure_state:     str,
    left_fill_progress:  float = 1.0,
    right_fill_progress: float = 1.0,
) -> np.ndarray:

    if left_mask is None and right_mask is None:
        return frame

    colour = lane_fill_colour(departure_state)

    frame = _render_single_side(frame, left_mask,  colour, left_fill_progress)
    frame = _render_single_side(frame, right_mask, colour, right_fill_progress)

    return frame
