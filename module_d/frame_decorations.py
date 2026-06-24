"""
Module D  —  Step D7: Frame Decorations  [NEW]
================================================
Draw visual chrome decorations onto the frame to give it the
HUD/targeting reticle feel of the demo:

  - Blue corner brackets (all 4 corners) — same as demo
  - Thin vignette darkening at the frame edges (using alpha blend)

INPUTS
------
    frame : np.ndarray (H, W, 3)

OUTPUT
------
    np.ndarray : frame with corner brackets drawn
"""

import numpy as np
import cv2

from .hud_colours import BRACKET_COLOUR
from .hud_effects import draw_vignette


# ── Bracket style ─────────────────────────────────────────────────────────
BRACKET_LEN     = 22     # px — length of each bracket arm
BRACKET_THICK   = 2      # line thickness
BRACKET_MARGIN  = 10     # px inset from frame edge


def draw_frame_decorations(frame: np.ndarray) -> np.ndarray:
    """
    Draw HUD-style corner bracket decorations + vignette on the frame.

    Parameters
    ----------
    frame : np.ndarray (H, W, 3), BGR

    Returns
    -------
    np.ndarray : frame with decorations drawn
    """
    H, W = frame.shape[:2]
    m  = BRACKET_MARGIN
    L  = BRACKET_LEN
    t  = BRACKET_THICK
    c  = BRACKET_COLOUR

    # ── Top-Left ──────────────────────────────────────────────────────────
    cv2.line(frame, (m,     m),     (m + L, m),     c, t)
    cv2.line(frame, (m,     m),     (m,     m + L), c, t)

    # ── Top-Right ─────────────────────────────────────────────────────────
    cv2.line(frame, (W - m,     m),     (W - m - L, m),     c, t)
    cv2.line(frame, (W - m,     m),     (W - m,     m + L), c, t)

    # ── Bottom-Left ───────────────────────────────────────────────────────
    cv2.line(frame, (m,     H - m),     (m + L, H - m),     c, t)
    cv2.line(frame, (m,     H - m),     (m,     H - m - L), c, t)

    # ── Bottom-Right ──────────────────────────────────────────────────────
    cv2.line(frame, (W - m,     H - m),     (W - m - L, H - m),     c, t)
    cv2.line(frame, (W - m,     H - m),     (W - m,     H - m - L), c, t)

    # ── Vignette darkening at edges (cached per resolution) ───────────────
    frame = draw_vignette(frame, strength=0.25)

    return frame
