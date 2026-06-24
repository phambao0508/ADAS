"""
Module D  —  Step D1: Lane Line Mask Overlay  (ENHANCED GLOW FILL)
===================================================================
APPROACH
--------
Render each YOLO segmentation mask with an attractive polished look:

  1. FEATHERED EDGES   — Gaussian-blurred mask for soft anti-aliased edges
  2. CONTOUR GLOW      — bright outlined edges with bloom for a neon border
  3. INNER FILL        — semi-transparent colour fill within the mask
  4. VERTICAL GRADIENT  — bottom=opaque → top=transparent for depth
  5. SWEEP ANIMATION   — bottom-to-top reveal with ease-in-out
"""

import numpy as np
import cv2
from typing import Optional

from .hud_colours import lane_fill_colour


# ── Tuning constants ──────────────────────────────────────────────────────────
CORE_ALPHA        = 0.55    # peak opacity of the inner fill
CONTOUR_THICKNESS = 3       # px — bright edge outline
CONTOUR_GLOW_SIZE = 15      # Gaussian blur kernel for glow (must be odd)
CONTOUR_GLOW_ALPHA= 0.50    # opacity of the glow bloom layer
FEATHER_KSIZE     = 7       # Gaussian kernel to soften mask edges
TRAIL_FADE_FRAC   = 0.20    # fraction of revealed zone with gradient fade
GLOW_ROWS         = 8       # rows of bright glow at the leading edge
GLOW_BRIGHTNESS   = 1.6     # multiplier for glow intensity (>1 = brighter)


def _ease_in_out(t: float) -> float:
    """
    Smooth ease-in-out curve (cubic).
    Maps linear [0,1] → [0,1] with smooth acceleration/deceleration.
    """
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
    """
    Render one side's lane mask with feathered edges and contour glow.

    Parameters
    ----------
    frame    : BGR video frame (H, W, 3)
    mask     : (H, W) uint8 or None — lane boundary mask for this side
    colour   : BGR tuple for the lane fill
    progress : 0.0 → 1.0 — raw fill progress from HUDPipeline

    Returns
    -------
    np.ndarray — frame with this side's overlay applied
    """
    if mask is None or progress <= 0.0:
        return frame

    H, W = frame.shape[:2]

    # Work on a copy of the mask so we don't mutate the original
    side_mask = mask.copy()

    # ── Find the y-range of the mask ──────────────────────────────────
    rows_with_mask = np.where(side_mask.max(axis=1) > 0)[0]
    if len(rows_with_mask) == 0:
        return frame

    y_top_mask    = int(rows_with_mask[0])
    y_bottom_mask = int(rows_with_mask[-1])
    mask_span     = y_bottom_mask - y_top_mask

    if mask_span <= 0:
        return frame

    # ── Apply ease-in-out to the progress ─────────────────────────────
    eased = _ease_in_out(max(0.0, min(1.0, progress)))

    # ── Compute the sweep threshold ───────────────────────────────────
    y_threshold = int(y_bottom_mask - eased * mask_span)

    # Zero out everything above the threshold (not yet revealed)
    side_mask[:y_threshold, :] = 0

    if side_mask.max() == 0:
        return frame

    # ── Feather the mask edges for anti-aliasing ──────────────────────
    # Convert binary mask → soft float mask with blurred edges
    soft_mask = cv2.GaussianBlur(
        side_mask.astype(np.float32) / 255.0,
        (FEATHER_KSIZE, FEATHER_KSIZE), 0
    )

    # ── Build per-row alpha gradient ──────────────────────────────────
    fade_rows = max(1, int(mask_span * TRAIL_FADE_FRAC))
    region_h = y_bottom_mask - y_threshold + 1

    alpha_1d = np.ones(region_h, dtype=np.float32)
    fade_end = min(fade_rows, region_h)
    if fade_end > 0:
        alpha_1d[:fade_end] = np.linspace(0.0, 1.0, fade_end, dtype=np.float32)
    alpha_1d *= CORE_ALPHA

    # ── Leading-edge GLOW: boost first few rows ──────────────────────
    glow_end = min(GLOW_ROWS, region_h)
    if glow_end > 0:
        for i in range(glow_end):
            glow_t = 1.0 - (i / glow_end)
            glow_factor = 1.0 + (GLOW_BRIGHTNESS - 1.0) * glow_t * glow_t
            alpha_1d[i] = min(1.0, alpha_1d[i] * glow_factor)

    # ── VECTORISED fill blending ──────────────────────────────────────
    region_slice = slice(y_threshold, y_bottom_mask + 1)
    region_frame = frame[region_slice].astype(np.float32)
    region_soft  = soft_mask[region_slice]  # (region_h, W) float [0,1]

    # Build 2-D alpha: per-row alpha × per-pixel soft mask
    alpha_2d = alpha_1d[:, np.newaxis] * region_soft  # (region_h, W)

    # Blend: frame * (1-a) + colour * a
    colour_arr = np.array(colour, dtype=np.float32)
    alpha_3d = alpha_2d[:, :, np.newaxis]  # (region_h, W, 1)

    blended = region_frame * (1.0 - alpha_3d) + colour_arr * alpha_3d

    # ── Contour glow — bright edge outline with bloom ─────────────────
    # Find contours of the revealed mask
    contour_mask = side_mask[region_slice]
    contours, _ = cv2.findContours(
        contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # Bright contour line
        bright_colour = tuple(min(255, int(c * 0.4 + 153)) for c in colour)
        contour_layer = np.zeros_like(blended, dtype=np.float32)
        cv2.drawContours(
            contour_layer,
            contours, -1,
            color=[float(c) for c in bright_colour],
            thickness=CONTOUR_THICKNESS,
            lineType=cv2.LINE_AA,
        )

        # Glow bloom: blur the contour and composite
        glow_layer = cv2.GaussianBlur(
            contour_layer,
            (CONTOUR_GLOW_SIZE, CONTOUR_GLOW_SIZE), 0
        )

        # Composite glow (additive-ish blend)
        glow_visible = (glow_layer.sum(axis=2, keepdims=True) > 0).astype(np.float32)
        blended = blended + glow_layer * CONTOUR_GLOW_ALPHA * glow_visible

        # Composite sharp contour on top
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
    """
    Draw lane-line mask segments with feathered edges and contour glow.

    Parameters
    ----------
    frame               : BGR video frame (H, W, 3)
    left_mask           : (H, W) uint8 or None — left boundary mask
    right_mask          : (H, W) uint8 or None — right boundary mask
    departure_state     : str — controls fill colour
    left_fill_progress  : 0.0 → 1.0 — left side sweep progress
    right_fill_progress : 0.0 → 1.0 — right side sweep progress

    Returns
    -------
    np.ndarray — frame with lane-line mask overlay
    """
    if left_mask is None and right_mask is None:
        return frame

    colour = lane_fill_colour(departure_state)

    # ── Render each side independently ────────────────────────────────
    frame = _render_single_side(frame, left_mask,  colour, left_fill_progress)
    frame = _render_single_side(frame, right_mask, colour, right_fill_progress)

    return frame
