"""
Module D  —  Step D6: Telemetry Info Panel (Bottom Right)  [Lane Status]
==========================================================================
Draw a compact data panel in the bottom-right corner of the frame
showing live telemetry matching the HUD demo aesthetic.

LAYOUT
------
    ┌──────────────────────────┐
    │     LANE STATUS          │
    ├──────────────────────────┤
    │ FRONT   CLOSE            │
    │ LEFT    CLEAR    ×2      │
    │ RIGHT   OCCUPIED ×1      │
    ├──────────────────────────┤
    │ L-LINE  DASHED           │
    │ R-LINE  SOLID            │
    └──────────────────────────┘

Vehicle counts show how many vehicles are detected in the left/right
adjacent lanes. Values ≥ 1 are highlighted to draw the driver's
attention.

INPUTS
------
    frame           : np.ndarray (H, W, 3)
    front_proximity : str   — NONE | DETECTED | CLOSE | VERY_CLOSE
    left_clear      : bool
    right_clear     : bool
    left_type       : str   — 'solid' | 'dashed'
    right_type      : str
    left_count      : int   — vehicles in left lane
    right_count     : int   — vehicles in right lane

OUTPUT
------
    np.ndarray : frame with telemetry panel drawn
"""

import numpy as np
import cv2
from typing import Optional

from .hud_colours import (
    HUD_BG_ALPHA,
    PANEL_BORDER, PANEL_SEPARATOR, PANEL_ACCENT,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    STATUS_OK, STATUS_DETECTED, STATUS_CLOSE, STATUS_URGENT, STATUS_OCCUPIED,
    LINE_TYPE_DASHED, LINE_TYPE_SOLID,
    COUNT_ZERO, COUNT_LOW, COUNT_HIGH,
)
from .hud_effects import draw_glass_panel, draw_status_dot


# ── Layout ────────────────────────────────────────────────────────────────
PANEL_MARGIN_RIGHT  = 14
PANEL_MARGIN_BOTTOM = 14
PANEL_W             = 320   # wider to accommodate live/seen count badge
ROW_HEIGHT          = 26
PADDING_X           = 12
PADDING_Y           = 10
TITLE_HEIGHT        = 28    # height for the "LANE STATUS" title row

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.50
FONT_THICK = 1
FONT_SCALE_TITLE = 0.46
FONT_SCALE_COUNT = 0.42

COL_KEY          = TEXT_DIM
COL_VAL_DEFAULT  = TEXT_PRIMARY
COL_VAL_OK       = STATUS_OK
COL_VAL_DETECTED = STATUS_DETECTED
COL_VAL_CLOSE    = STATUS_CLOSE
COL_VAL_URGENT   = STATUS_URGENT
COL_VAL_OCCUPIED = STATUS_OCCUPIED
COL_VAL_DASHED   = LINE_TYPE_DASHED
COL_VAL_SOLID    = LINE_TYPE_SOLID
COL_BORDER       = PANEL_BORDER
COL_SEPARATOR    = PANEL_SEPARATOR
COL_TITLE        = TEXT_SECONDARY
COL_COUNT_ZERO   = COUNT_ZERO
COL_COUNT_LOW    = COUNT_LOW
COL_COUNT_HIGH   = COUNT_HIGH
COL_ACCENT       = PANEL_ACCENT


def draw_telemetry_panel(
    frame:            np.ndarray,
    front_proximity:  str,
    left_clear:       bool,
    right_clear:      bool,
    left_type:        str,
    right_type:       str,
    left_count:       int   = 0,
    right_count:      int   = 0,
    total_seen:       int   = 0,
    front_distance_m: Optional[float] = None,
    front_ttc_s:      Optional[float] = None,
) -> np.ndarray:
    """
    Draw the Lane Status telemetry panel in the bottom-right corner.

    Parameters
    ----------
    frame : np.ndarray (H, W, 3), BGR
    left_count, right_count : int — *live* confirmed-track counts per lane
    total_seen : int — cumulative unique vehicles seen since pipeline start
        (shown as a small "seen" total alongside the live badge).

    Returns
    -------
    np.ndarray : frame with panel drawn
    """
    H, W = frame.shape[:2]

    # ── Build rows: (key_label, value_label, value_colour, count) ────────
    prox_val   = front_proximity
    prox_col   = (COL_VAL_URGENT   if prox_val == "VERY_CLOSE" else
                  COL_VAL_CLOSE    if prox_val == "CLOSE"      else
                  COL_VAL_DETECTED if prox_val == "DETECTED"   else
                  COL_VAL_OK)

    left_str   = "CLEAR" if left_clear  else "OCCUPIED"
    left_col   = COL_VAL_OK if left_clear else COL_VAL_OCCUPIED
    right_str  = "CLEAR" if right_clear else "OCCUPIED"
    right_col  = COL_VAL_OK if right_clear else COL_VAL_OCCUPIED

    # Main status rows: (key, value, colour, vehicle_count_or_None)
    rows = [
        ("FRONT",  prox_val,  prox_col,  None),
        ("LEFT",   left_str,  left_col,  left_count),
        ("RIGHT",  right_str, right_col, right_count),
    ]

    # Append distance / TTC rows when a front vehicle has been measured.
    if front_distance_m is not None:
        dist_text = f"{front_distance_m:.0f} m"
        dist_col = (COL_VAL_URGENT if front_distance_m < 10 else
                    COL_VAL_CLOSE  if front_distance_m < 25 else
                    COL_VAL_OK)
        rows.append(("DIST",  dist_text, dist_col, None))

    if front_ttc_s is not None:
        ttc_text = f"{front_ttc_s:.1f} s"
        ttc_col = (COL_VAL_URGENT if front_ttc_s < 1.5 else
                   COL_VAL_CLOSE  if front_ttc_s < 3.0 else
                   COL_VAL_OK)
        rows.append(("TTC",   ttc_text, ttc_col, None))

    n_rows = len(rows)
    panel_h = TITLE_HEIGHT + 2 * PADDING_Y + n_rows * ROW_HEIGHT

    x2 = W - PANEL_MARGIN_RIGHT
    y2 = H - PANEL_MARGIN_BOTTOM
    x1 = x2 - PANEL_W
    y1 = y2 - panel_h

    # ── Background (glassmorphism) ────────────────────────────────────────
    frame = draw_glass_panel(frame, x1, y1, x2, y2,
                             tint_bgr=(8, 12, 20), blur_k=21, alpha=0.70,
                             border_colour=COL_BORDER)

    # ── Title bar: "LANE STATUS" ──────────────────────────────────────────
    title_y = y1 + TITLE_HEIGHT
    # Accent bar under title
    cv2.line(frame, (x1 + PADDING_X, title_y),
             (x2 - PADDING_X, title_y), COL_SEPARATOR, 1)
    # Left accent strip
    cv2.rectangle(frame, (x1, y1), (x1 + 4, y2), COL_ACCENT, cv2.FILLED)

    title_text = "LANE STATUS"
    cv2.putText(frame, title_text, (x1 + PADDING_X + 6, y1 + TITLE_HEIGHT - 8),
                FONT, FONT_SCALE_TITLE, COL_TITLE, 1, cv2.LINE_AA)

    # ── Live + cumulative count badge (right side of title) ───────────────
    # "live" = vehicles tracked in adjacent lanes right now.
    # "seen" = unique vehicles ever observed since the pipeline started.
    live = left_count + right_count
    if live > 0 or total_seen > 0:
        badge_text = f"{live} LIVE"
        if total_seen > 0:
            badge_text += f"  /  {total_seen} SEEN"
        ts = cv2.getTextSize(badge_text, FONT, FONT_SCALE_COUNT, 1)[0]
        badge_x = x2 - PADDING_X - ts[0] - 8
        badge_y = y1 + TITLE_HEIGHT - 10
        # Badge background
        cv2.rectangle(frame,
                      (badge_x - 4, badge_y - ts[1] - 3),
                      (badge_x + ts[0] + 4, badge_y + 4),
                      (30, 40, 55), cv2.FILLED)
        cv2.rectangle(frame,
                      (badge_x - 4, badge_y - ts[1] - 3),
                      (badge_x + ts[0] + 4, badge_y + 4),
                      COL_ACCENT, 1)
        badge_col = COL_COUNT_HIGH if live >= 3 else (
            COL_COUNT_LOW if live > 0 else COL_VAL_DETECTED
        )
        cv2.putText(frame, badge_text, (badge_x, badge_y),
                    FONT, FONT_SCALE_COUNT, badge_col, 1, cv2.LINE_AA)

    # ── Data rows ─────────────────────────────────────────────────────────
    col_val_x = x1 + PADDING_X + 70   # Value column x offset
    dot_x = col_val_x - 10            # Status dot position
    col_count_x = x2 - PADDING_X - 50  # Count column x offset

    for i, (key, val, val_clr, count) in enumerate(rows):
        row_y = title_y + PADDING_Y + i * ROW_HEIGHT + ROW_HEIGHT - 4

        # Key (dim)
        cv2.putText(frame, key, (x1 + PADDING_X + 6, row_y),
                    FONT, FONT_SCALE, COL_KEY, FONT_THICK, cv2.LINE_AA)

        # Status dot (colour-coded)
        draw_status_dot(frame, dot_x, row_y - 5, val_clr, radius=3)

        # Value (colour coded)
        cv2.putText(frame, val, (col_val_x, row_y),
                    FONT, FONT_SCALE, val_clr, FONT_THICK, cv2.LINE_AA)

        # Vehicle count (for LEFT and RIGHT rows)
        if count is not None:
            if count == 0:
                count_text = "x0"
                count_col = COL_COUNT_ZERO
            elif count <= 2:
                count_text = f"x{count}"
                count_col = COL_COUNT_LOW
            else:
                count_text = f"x{count}"
                count_col = COL_COUNT_HIGH

            cv2.putText(frame, count_text, (col_count_x, row_y),
                        FONT, FONT_SCALE_COUNT, count_col, 1, cv2.LINE_AA)

        # Separator line between rows (except after last)
        if i < n_rows - 1:
            sep_y = title_y + PADDING_Y + (i + 1) * ROW_HEIGHT - 2
            cv2.line(frame, (x1 + PADDING_X, sep_y),
                     (x2 - PADDING_X, sep_y), COL_SEPARATOR, 1)

    return frame
