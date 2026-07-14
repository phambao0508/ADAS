import numpy as np
import cv2

from .hud_colours import GUIDANCE_COLOURS, HUD_BG_ALPHA
from .hud_effects import draw_glass_panel, pulse_brightness

_frame_counter = 0

BANNER_HEIGHT       = 60
BANNER_MARGIN_TOP   = 14
BANNER_SIDE_PADDING = 30
FONT                = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_NORMAL   = 0.75
FONT_SCALE_URGENT   = 0.95
FONT_THICKNESS      = 2
FONT_THICKNESS_URGENT = 2
TEXT_COLOUR         = (240, 245, 250)

def draw_guidance_banner(
    frame:          np.ndarray,
    guidance_state: str,
    message:        str,
) -> np.ndarray:

    global _frame_counter
    _frame_counter += 1

    if guidance_state == "GUIDE_NONE" or not message:
        return frame

    H, W = frame.shape[:2]
    is_urgent   = (guidance_state == "GUIDE_URGENT")
    font_scale  = FONT_SCALE_URGENT  if is_urgent else FONT_SCALE_NORMAL
    font_thick  = FONT_THICKNESS_URGENT if is_urgent else FONT_THICKNESS

    (text_w, text_h), baseline = cv2.getTextSize(
        message, FONT, font_scale, font_thick
    )

    banner_w = text_w + 2 * BANNER_SIDE_PADDING
    banner_w = max(banner_w, W // 3)

    x1 = (W - banner_w) // 2
    y1 = BANNER_MARGIN_TOP
    x2 = x1 + banner_w
    y2 = y1 + BANNER_HEIGHT

    state_colour = GUIDANCE_COLOURS.get(guidance_state, (180, 180, 180))
    b, g, r = state_colour

    if is_urgent:
        frame = draw_glass_panel(frame, x1, y1, x2, y2,
                                 tint_bgr=(20, 10, 40), blur_k=21, alpha=0.85)
    else:
        frame = draw_glass_panel(frame, x1, y1, x2, y2,
                                 tint_bgr=(10, 14, 22), blur_k=21, alpha=0.70)

    glow_clr = (int(b * 0.35), int(g * 0.35), int(r * 0.35))
    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), glow_clr, 1)

    border_thick = 2 if is_urgent else 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), state_colour, border_thick)

    brk    = 10
    bright = (min(255, b + 60), min(255, g + 60), min(255, r + 60))

    cv2.line(frame, (x1, y1 + brk), (x1, y1), bright, 2)
    cv2.line(frame, (x1, y1), (x1 + brk, y1), bright, 2)

    cv2.line(frame, (x2 - brk, y1), (x2, y1), bright, 2)
    cv2.line(frame, (x2, y1), (x2, y1 + brk), bright, 2)

    cv2.line(frame, (x1, y2 - brk), (x1, y2), bright, 2)
    cv2.line(frame, (x1, y2), (x1 + brk, y2), bright, 2)

    cv2.line(frame, (x2 - brk, y2), (x2, y2), bright, 2)
    cv2.line(frame, (x2, y2 - brk), (x2, y2), bright, 2)

    text_x = x1 + (banner_w - text_w) // 2
    text_y = y1 + BANNER_HEIGHT // 2 + text_h // 2

    shadow_clr = (int(b * 0.3), int(g * 0.3), int(r * 0.3))
    cv2.putText(frame, message, (text_x + 1, text_y + 1),
                FONT, font_scale, shadow_clr, font_thick, cv2.LINE_AA)

    cv2.putText(frame, message, (text_x, text_y),
                FONT, font_scale, state_colour, font_thick, cv2.LINE_AA)

    if is_urgent:
        highlight = pulse_brightness(state_colour, _frame_counter,
                                     amplitude=50, period_frames=15)
        cv2.putText(frame, message, (text_x, text_y),
                    FONT, font_scale, highlight, 1, cv2.LINE_AA)

    return frame
