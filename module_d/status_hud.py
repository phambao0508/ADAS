"""
Module D  —  Step D3: Departure Status HUD Bar (Top Right)
============================================================
Redesigned to match the HUD demo aesthetic:
  - Semi-transparent dark background
  - Glowing coloured LEFT accent bar (4–6 px thick)
  - Right outer border matching the state colour
  - State label with glow-like rendering (drawn twice: shadow pass + bright pass)
  - Offset sub-text in dim monospaced style
  - Corner bracket decorations on the panel

LAYOUT
------
    ┌─[colour]──────────────────────────────────┐
    │  DRIFTING RIGHT >>                        │
    │  offset: +75 px                           │
    └────────────────────────────────────────────┘

INPUTS
------
    frame             : np.ndarray (H, W, 3)
    departure_state   : str  — from DepartureResult.state
    smoothed_offset   : float | None  — from DepartureResult.smoothed_offset

OUTPUT
------
    np.ndarray : frame with status HUD drawn
"""

import numpy as np
import cv2
from typing import Optional

from .hud_colours import lane_fill_colour, HUD_BG_ALPHA, DEPARTURE_COLOURS


# ── HUD text per state ────────────────────────────────────────────────────
STATE_LABELS = {
    "CENTERED":          "  LANE CENTERED  ",
    "WARN_LEFT":         "<< DRIFTING LEFT",
    "WARN_RIGHT":        "DRIFTING RIGHT >>",
    "DEPART_LEFT":       "!! LANE DEPARTURE <<",
    "DEPART_RIGHT":      "LANE DEPARTURE!! >>",
    "LANE_CHANGE_LEFT":  "<< LANE CHANGE",
    "LANE_CHANGE_RIGHT": "LANE CHANGE >>",
}

# ── Layout constants ──────────────────────────────────────────────────────
HUD_WIDTH        = 360   # px wide  (was 240)
HUD_HEIGHT       = 85    # px tall  (was 58)
HUD_MARGIN_RIGHT = 18    # px from right edge
HUD_MARGIN_TOP   = 14    # px from top edge
ACCENT_BAR_W     = 6     # px — coloured bar on left of panel
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_BIG   = 0.72  # was 0.52
FONT_SCALE_SMALL = 0.50  # was 0.38
FONT_THICKNESS_MAIN = 2  # was 1
TEXT_COLOUR      = (230, 230, 230)   # off-white
TEXT_COLOUR_DIM  = (130, 150, 170)   # dim blue-grey for sub-line


def draw_status_hud(
    frame:           np.ndarray,
    departure_state: str,
    smoothed_offset: Optional[float],
) -> np.ndarray:
    """
    Draw the departure status HUD panel in the top-right corner.

    Parameters
    ----------
    frame : np.ndarray (H, W, 3), BGR
    departure_state : str — from DepartureResult.state
    smoothed_offset : float or None — from DepartureResult.smoothed_offset

    Returns
    -------
    np.ndarray : frame with HUD panel drawn
    """
    H, W = frame.shape[:2]

    # ── Panel position (top-right) ─────────────────────────────────────────
    x1 = W - HUD_WIDTH - HUD_MARGIN_RIGHT
    y1 = HUD_MARGIN_TOP
    x2 = W - HUD_MARGIN_RIGHT
    y2 = y1 + HUD_HEIGHT

    state_colour = DEPARTURE_COLOURS.get(departure_state, (74, 200, 0))

    # ── 1. Semi-transparent dark background ───────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (12, 16, 22), cv2.FILLED)
    frame = cv2.addWeighted(overlay, HUD_BG_ALPHA, frame, 1.0 - HUD_BG_ALPHA, 0)

    # ── 2. Outer border matching state colour (dim version) ───────────────
    b, g, r = state_colour
    dim_border = (max(0, b - 140), max(0, g - 140), max(0, r - 140))
    cv2.rectangle(frame, (x1, y1), (x2, y2), dim_border, 1)

    # ── 3. Coloured left accent bar ────────────────────────────────────────
    cv2.rectangle(frame, (x1, y1), (x1 + ACCENT_BAR_W, y2), state_colour, cv2.FILLED)

    # ── 4. Corner bracket details (top-left and bottom-right only) ─────────
    brk = 8   # bracket length
    brk_clr = (120, 150, 200)
    # top-right corner of panel
    cv2.line(frame, (x2 - brk, y1), (x2, y1), brk_clr, 1)
    cv2.line(frame, (x2, y1), (x2, y1 + brk), brk_clr, 1)
    # bottom-right corner
    cv2.line(frame, (x2 - brk, y2), (x2, y2), brk_clr, 1)
    cv2.line(frame, (x2, y2 - brk), (x2, y2), brk_clr, 1)

    # ── 5. State text (shadow pass then bright pass for glow look) ─────────
    label   = STATE_LABELS.get(departure_state, departure_state)
    text_x  = x1 + ACCENT_BAR_W + 12
    text_y1 = y1 + 34

    # Shadow in state colour (dim, offset 1px)
    b2, g2, r2 = state_colour
    shadow_clr = (int(b2 * 0.4), int(g2 * 0.4), int(r2 * 0.4))
    cv2.putText(frame, label, (text_x + 1, text_y1 + 1),
                FONT, FONT_SCALE_BIG, shadow_clr, FONT_THICKNESS_MAIN, cv2.LINE_AA)
    # Main text in state colour
    cv2.putText(frame, label, (text_x, text_y1),
                FONT, FONT_SCALE_BIG, state_colour, FONT_THICKNESS_MAIN, cv2.LINE_AA)

    # ── 6. Numeric offset (line 2) — dim sub-text ─────────────────────────
    if smoothed_offset is not None:
        offset_text = f"offset: {smoothed_offset:+.1f} px"
    else:
        offset_text = "offset: -- px"

    # Append state label in a compact form
    state_short = {
        "CENTERED": "CENTERED", "WARN_LEFT": "WARN_L", "WARN_RIGHT": "WARN_R",
        "DEPART_LEFT": "DEPART_L", "DEPART_RIGHT": "DEPART_R",
        "LANE_CHANGE_LEFT": "LC_L", "LANE_CHANGE_RIGHT": "LC_R",
    }.get(departure_state, departure_state)
    offset_text += f"  |  {state_short}"

    text_y2 = y1 + 65
    cv2.putText(frame, offset_text, (text_x, text_y2),
                FONT, FONT_SCALE_SMALL, TEXT_COLOUR_DIM, 1, cv2.LINE_AA)

    return frame
