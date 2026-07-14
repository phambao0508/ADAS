import numpy as np
import cv2
from typing import Optional

from .hud_colours import (
    lane_fill_colour, HUD_BG_ALPHA, DEPARTURE_COLOURS,
    COLOUR_CENTERED, PANEL_TINT_DEFAULT, BRACKET_COLOUR,
    TEXT_PRIMARY, TEXT_SECONDARY,
)
from .hud_effects import draw_glass_panel, ColourSmoother

_state_colour_smoother = ColourSmoother(speed=0.18, initial=COLOUR_CENTERED)

STATE_LABELS = {

}

HUD_WIDTH        = 360
HUD_HEIGHT       = 85
HUD_MARGIN_RIGHT = 18
HUD_MARGIN_TOP   = 14
ACCENT_BAR_W     = 6
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_BIG   = 0.72
FONT_SCALE_SMALL = 0.50
FONT_THICKNESS_MAIN = 2
TEXT_COLOUR      = TEXT_PRIMARY
TEXT_COLOUR_DIM  = TEXT_SECONDARY

def draw_status_hud(
    frame:           np.ndarray,
    departure_state: str,
    smoothed_offset: Optional[float],
) -> np.ndarray:

    H, W = frame.shape[:2]

    x1 = W - HUD_WIDTH - HUD_MARGIN_RIGHT
    y1 = HUD_MARGIN_TOP
    x2 = W - HUD_MARGIN_RIGHT
    y2 = y1 + HUD_HEIGHT

    target_colour = DEPARTURE_COLOURS.get(departure_state, COLOUR_CENTERED)
    state_colour = _state_colour_smoother.update(target_colour)

    frame = draw_glass_panel(frame, x1, y1, x2, y2,
                             tint_bgr=PANEL_TINT_DEFAULT, blur_k=21, alpha=0.70)

    b, g, r = state_colour
    dim_border = (max(0, b - 140), max(0, g - 140), max(0, r - 140))
    cv2.rectangle(frame, (x1, y1), (x2, y2), dim_border, 1)

    cv2.rectangle(frame, (x1, y1), (x1 + ACCENT_BAR_W, y2), state_colour, cv2.FILLED)

    brk = 8
    brk_clr = BRACKET_COLOUR

    cv2.line(frame, (x2 - brk, y1), (x2, y1), brk_clr, 1)
    cv2.line(frame, (x2, y1), (x2, y1 + brk), brk_clr, 1)

    cv2.line(frame, (x2 - brk, y2), (x2, y2), brk_clr, 1)
    cv2.line(frame, (x2, y2 - brk), (x2, y2), brk_clr, 1)

    label   = STATE_LABELS.get(departure_state, departure_state)
    text_x  = x1 + ACCENT_BAR_W + 12
    text_y1 = y1 + 34

    b2, g2, r2 = state_colour
    shadow_clr = (int(b2 * 0.4), int(g2 * 0.4), int(r2 * 0.4))
    cv2.putText(frame, label, (text_x + 1, text_y1 + 1),
                FONT, FONT_SCALE_BIG, shadow_clr, FONT_THICKNESS_MAIN, cv2.LINE_AA)

    cv2.putText(frame, label, (text_x, text_y1),
                FONT, FONT_SCALE_BIG, state_colour, FONT_THICKNESS_MAIN, cv2.LINE_AA)

    if smoothed_offset is not None:
        offset_text = f"offset: {smoothed_offset:+.1f} px"
    else:
        offset_text = "offset: -- px"

    state_short = {

    }.get(departure_state, departure_state)
    offset_text += f"  |  {state_short}"

    text_y2 = y1 + 65
    cv2.putText(frame, offset_text, (text_x, text_y2),
                FONT, FONT_SCALE_SMALL, TEXT_COLOUR_DIM, 1, cv2.LINE_AA)

    return frame
