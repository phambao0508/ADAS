import numpy as np
import cv2

from .hud_colours import BRACKET_COLOUR
from .hud_effects import draw_vignette

BRACKET_LEN     = 22
BRACKET_THICK   = 2
BRACKET_MARGIN  = 10

def draw_frame_decorations(frame: np.ndarray) -> np.ndarray:

    H, W = frame.shape[:2]
    m  = BRACKET_MARGIN
    L  = BRACKET_LEN
    t  = BRACKET_THICK
    c  = BRACKET_COLOUR

    cv2.line(frame, (m,     m),     (m + L, m),     c, t)
    cv2.line(frame, (m,     m),     (m,     m + L), c, t)

    cv2.line(frame, (W - m,     m),     (W - m - L, m),     c, t)
    cv2.line(frame, (W - m,     m),     (W - m,     m + L), c, t)

    cv2.line(frame, (m,     H - m),     (m + L, H - m),     c, t)
    cv2.line(frame, (m,     H - m),     (m,     H - m - L), c, t)

    cv2.line(frame, (W - m,     H - m),     (W - m - L, H - m),     c, t)
    cv2.line(frame, (W - m,     H - m),     (W - m,     H - m - L), c, t)

    frame = draw_vignette(frame, strength=0.25)

    return frame
