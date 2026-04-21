"""
Module D  —  Step D1: Lane Line Mask Overlay  (SWEEPING TRAIL)
===============================================================
APPROACH
--------
Render each YOLO segmentation mask segment with a continuous
bottom-to-top sweeping animation — the overlay progressively
reveals from near the car upward, then resets and sweeps again.

The sweep creates a "car painting the road" effect: only the
portion of the lane markings near the car's current position
is strongly visible, with a gradient fade at the leading edge.

Only the exact YOLO segmentation mask is rendered — no dilation
or glow, so the overlay sits precisely on the lane markings.
"""

import numpy as np
import cv2
from typing import Optional

from .hud_colours import lane_fill_colour


# ── Tuning constants ──────────────────────────────────────────────────────────
CORE_ALPHA       = 0.70    # peak opacity of the lane overlay
TRAIL_FADE_FRAC  = 0.25    # fraction of the revealed zone that has a gradient fade


def draw_lane_lines(
    frame:           np.ndarray,
    left_mask:       Optional[np.ndarray],
    right_mask:      Optional[np.ndarray],
    departure_state: str,
    fill_progress:   float = 1.0,
) -> np.ndarray:
    """
    Draw lane-line mask segments with a sweeping bottom-to-top trail.

    Parameters
    ----------
    frame           : BGR video frame (H, W, 3)
    left_mask       : (H, W) uint8 or None — ego-lane left boundary mask
    right_mask      : (H, W) uint8 or None — ego-lane right boundary mask
    departure_state : str — controls fill colour
    fill_progress   : 0.0 → 1.0 — sweep position (0 = nothing, 1 = fully revealed)
                      This value is managed by HUDPipeline and cycles continuously.

    Returns
    -------
    np.ndarray — frame with lane-line mask overlay
    """
    if fill_progress <= 0.0:
        return frame
    if left_mask is None and right_mask is None:
        return frame

    H, W = frame.shape[:2]
    colour = lane_fill_colour(departure_state)

    # ── Combine both masks into one canvas ────────────────────────────────
    combined = np.zeros((H, W), dtype=np.uint8)
    if left_mask is not None:
        combined = cv2.bitwise_or(combined, left_mask)
    if right_mask is not None:
        combined = cv2.bitwise_or(combined, right_mask)

    if combined.max() == 0:
        return frame

    # ── Find the y-range of the mask ──────────────────────────────────────
    rows_with_mask = np.where(combined.max(axis=1) > 0)[0]
    if len(rows_with_mask) == 0:
        return frame

    y_top_mask    = int(rows_with_mask[0])
    y_bottom_mask = int(rows_with_mask[-1])
    mask_span     = y_bottom_mask - y_top_mask

    if mask_span <= 0:
        return frame

    progress = max(0.0, min(1.0, fill_progress))

    # ── Compute the sweep threshold ───────────────────────────────────────
    # y_threshold: the top-most y that is currently revealed
    # Moves from y_bottom_mask (progress=0) up to y_top_mask (progress=1)
    y_threshold = int(y_bottom_mask - progress * mask_span)

    # Zero out everything above the threshold (not yet revealed)
    combined[:y_threshold, :] = 0

    if combined.max() == 0:
        return frame

    # ── Build a per-row alpha gradient for smooth leading-edge fade ────────
    # Rows near y_threshold (the leading edge) fade in gradually
    fade_rows = int(mask_span * TRAIL_FADE_FRAC)
    if fade_rows < 1:
        fade_rows = 1

    alpha_map = np.ones((H,), dtype=np.float32)
    for y in range(y_threshold, min(y_threshold + fade_rows, H)):
        # Linear fade: 0 at threshold → 1 at threshold + fade_rows
        t = (y - y_threshold) / fade_rows
        alpha_map[y] = t

    # ── Render with per-row alpha ─────────────────────────────────────────
    mask_pixels = combined > 0
    colour_arr = np.array(colour, dtype=np.float32)

    # Build the tinted overlay
    overlay = frame.astype(np.float32)
    for y in range(y_threshold, y_bottom_mask + 1):
        row_mask = mask_pixels[y]
        if not row_mask.any():
            continue
        a = alpha_map[y] * CORE_ALPHA
        overlay[y, row_mask] = (
            overlay[y, row_mask] * (1.0 - a) + colour_arr * a
        )

    frame = np.clip(overlay, 0, 255).astype(np.uint8)

    return frame
