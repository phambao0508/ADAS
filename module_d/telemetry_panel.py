"""
Module D  —  Step D6: Telemetry Info Panel (Bottom Right)  [NEW]
=================================================================
Draw a compact data panel in the bottom-right corner of the frame
showing live telemetry matching the HUD demo aesthetic.

LAYOUT
------
    ┌─────────────────────┐
    │ FRONT   CLOSE       │
    │ LEFT    CLEAR       │
    │ RIGHT   OCCUPIED    │
    │ L-LINE  DASHED      │
    │ R-LINE  SOLID       │
    └─────────────────────┘

Value colours:
    NONE / CLEAR / DASHED  → dim white
    CLOSE / OCCUPIED       → orange
    VERY_CLOSE             → red
    DASHED                 → cyan
    SOLID                  → dim red-orange

INPUTS
------
    frame           : np.ndarray (H, W, 3)
    front_proximity : str   — NONE | CLOSE | VERY_CLOSE
    left_clear      : bool
    right_clear     : bool
    left_type       : str   — 'solid' | 'dashed'
    right_type      : str

OUTPUT
------
    np.ndarray : frame with telemetry panel drawn
"""

import numpy as np
import cv2
from typing import Optional

from .hud_colours import HUD_BG_ALPHA


# ── Layout ────────────────────────────────────────────────────────────────
PANEL_MARGIN_RIGHT  = 14
PANEL_MARGIN_BOTTOM = 14
PANEL_W             = 210   # was 150
ROW_HEIGHT          = 26    # was 18
PADDING_X           = 12
PADDING_Y           = 10

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.50   # was 0.37
FONT_THICK = 1

COL_KEY          = ( 80, 100, 120)   # dim blue-grey key labels
COL_VAL_DEFAULT  = (190, 200, 210)   # off-white values
COL_VAL_OK       = ( 74, 200,   0)   # #00c84a green
COL_VAL_CLOSE    = (  0, 136, 255)   # orange
COL_VAL_URGENT   = ( 51,  51, 255)   # red
COL_VAL_OCCUPIED = ( 51,  51, 255)   # red for occupied
COL_VAL_DASHED   = (238, 204,   0)   # cyan for dashed
COL_VAL_SOLID    = (100, 100, 200)   # muted red-orange for solid
COL_BORDER       = ( 55,  70,  90)


def draw_telemetry_panel(
    frame:           np.ndarray,
    front_proximity: str,
    left_clear:      bool,
    right_clear:     bool,
    left_type:       str,
    right_type:      str,
) -> np.ndarray:
    """
    Draw the telemetry info panel in the bottom-right corner.

    Parameters
    ----------
    frame : np.ndarray (H, W, 3), BGR

    Returns
    -------
    np.ndarray : frame with panel drawn
    """
    H, W = frame.shape[:2]

    # ── Build rows: (key_label, value_label, value_colour) ──────────────
    prox_val   = front_proximity
    prox_col   = (COL_VAL_URGENT  if prox_val == "VERY_CLOSE" else
                  COL_VAL_CLOSE   if prox_val == "CLOSE"      else
                  COL_VAL_OK)

    left_str   = "CLEAR" if left_clear  else "OCCUPIED"
    left_col   = COL_VAL_OK if left_clear else COL_VAL_OCCUPIED
    right_str  = "CLEAR" if right_clear else "OCCUPIED"
    right_col  = COL_VAL_OK if right_clear else COL_VAL_OCCUPIED

    ltype_str  = left_type.upper()
    ltype_col  = COL_VAL_DASHED if left_type == "dashed" else COL_VAL_SOLID
    rtype_str  = right_type.upper()
    rtype_col  = COL_VAL_DASHED if right_type == "dashed" else COL_VAL_SOLID

    rows = [
        ("FRONT",  prox_val,  prox_col),
        ("LEFT",   left_str,  left_col),
        ("RIGHT",  right_str, right_col),
        ("L-LINE", ltype_str, ltype_col),
        ("R-LINE", rtype_str, rtype_col),
    ]

    n_rows = len(rows)
    panel_h = 2 * PADDING_Y + n_rows * ROW_HEIGHT

    x2 = W - PANEL_MARGIN_RIGHT
    y2 = H - PANEL_MARGIN_BOTTOM
    x1 = x2 - PANEL_W
    y1 = y2 - panel_h

    # ── Background ────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (8, 12, 20), cv2.FILLED)
    frame = cv2.addWeighted(overlay, HUD_BG_ALPHA, frame, 1.0 - HUD_BG_ALPHA, 0)

    # ── Border ────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (x1, y1), (x2, y2), COL_BORDER, 1)

    # ── Rows ──────────────────────────────────────────────────────────────
    col_val_x = x1 + PADDING_X + 70   # Value column x offset (was 52)

    for i, (key, val, val_clr) in enumerate(rows):
        row_y = y1 + PADDING_Y + i * ROW_HEIGHT + ROW_HEIGHT - 4

        # Key (dim)
        cv2.putText(frame, key, (x1 + PADDING_X, row_y),
                    FONT, FONT_SCALE, COL_KEY, FONT_THICK, cv2.LINE_AA)

        # Value (colour coded)
        cv2.putText(frame, val, (col_val_x, row_y),
                    FONT, FONT_SCALE, val_clr, FONT_THICK, cv2.LINE_AA)

    return frame
