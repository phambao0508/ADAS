"""
Module D  —  Step D4: Guidance Banner (Center Top)
====================================================
Redesigned to match HUD demo aesthetic:
  - Glowing coloured border and text (drawn twice for glow effect)
  - Dark semi-transparent background
  - Monospaced HUD-style font
  - Pulsing effect simulated by semi-opaque secondary border ring
  - GUIDE_URGENT: larger font + brighter glow + full red background

INPUTS
------
    frame           : np.ndarray (H, W, 3)
    guidance_state  : str  — from GuidanceResult.guidance
    message         : str  — from GuidanceResult.message

OUTPUT
------
    np.ndarray : frame with guidance banner drawn (or unchanged if GUIDE_NONE)
"""

import numpy as np
import cv2

from .hud_colours import GUIDANCE_COLOURS, HUD_BG_ALPHA


# ── Layout constants ────────────────────────────────────────────
BANNER_HEIGHT       = 60    # px  (was 40)
BANNER_MARGIN_TOP   = 14    # px from top edge
BANNER_SIDE_PADDING = 30    # px left/right padding inside banner
FONT                = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_NORMAL   = 0.75  # was 0.58
FONT_SCALE_URGENT   = 0.95  # was 0.75
FONT_THICKNESS      = 2     # was 1
FONT_THICKNESS_URGENT = 2
TEXT_COLOUR         = (240, 245, 250)   # near-white


def draw_guidance_banner(
    frame:          np.ndarray,
    guidance_state: str,
    message:        str,
) -> np.ndarray:
    """
    Draw the guidance banner centred at the top of the frame.

    Parameters
    ----------
    frame : np.ndarray (H, W, 3), BGR
    guidance_state : str — from GuidanceResult.guidance
    message : str — HUD text for the current guidance state

    Returns
    -------
    np.ndarray : frame with banner (or unchanged if GUIDE_NONE)
    """
    if guidance_state == "GUIDE_NONE" or not message:
        return frame   # nothing to draw

    H, W = frame.shape[:2]
    is_urgent   = (guidance_state == "GUIDE_URGENT")
    font_scale  = FONT_SCALE_URGENT  if is_urgent else FONT_SCALE_NORMAL
    font_thick  = FONT_THICKNESS_URGENT if is_urgent else FONT_THICKNESS

    # ── Measure text to centre the banner ─────────────────────────────────
    (text_w, text_h), baseline = cv2.getTextSize(
        message, FONT, font_scale, font_thick
    )

    banner_w = text_w + 2 * BANNER_SIDE_PADDING
    banner_w = max(banner_w, W // 3)   # min 1/3 frame width

    x1 = (W - banner_w) // 2
    y1 = BANNER_MARGIN_TOP
    x2 = x1 + banner_w
    y2 = y1 + BANNER_HEIGHT

    state_colour = GUIDANCE_COLOURS.get(guidance_state, (180, 180, 180))
    b, g, r = state_colour

    # ── 1. Background: dark semi-transparent rect ──────────────────────────
    overlay = frame.copy()
    if is_urgent:
        # Urgent: slightly reddish background fill
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 10, 40), cv2.FILLED)
        alpha_bg = 0.88
    else:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 14, 22), cv2.FILLED)
        alpha_bg = HUD_BG_ALPHA
    frame = cv2.addWeighted(overlay, alpha_bg, frame, 1.0 - alpha_bg, 0)

    # ── 2. Outer glow ring (dim, 1px wider) ───────────────────────────────
    glow_clr = (int(b * 0.35), int(g * 0.35), int(r * 0.35))
    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), glow_clr, 1)

    # ── 3. Main border (state colour) ─────────────────────────────────────
    border_thick = 2 if is_urgent else 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), state_colour, border_thick)

    # ── 4. Corner bracket accents ─────────────────────────────────────────
    brk    = 10
    bright = (min(255, b + 60), min(255, g + 60), min(255, r + 60))
    # top-left
    cv2.line(frame, (x1, y1 + brk), (x1, y1), bright, 2)
    cv2.line(frame, (x1, y1), (x1 + brk, y1), bright, 2)
    # top-right
    cv2.line(frame, (x2 - brk, y1), (x2, y1), bright, 2)
    cv2.line(frame, (x2, y1), (x2, y1 + brk), bright, 2)
    # bottom-left
    cv2.line(frame, (x1, y2 - brk), (x1, y2), bright, 2)
    cv2.line(frame, (x1, y2), (x1 + brk, y2), bright, 2)
    # bottom-right
    cv2.line(frame, (x2 - brk, y2), (x2, y2), bright, 2)
    cv2.line(frame, (x2, y2 - brk), (x2, y2), bright, 2)

    # ── 5. Text — shadow pass (state colour, dim) then main pass ──────────
    text_x = x1 + (banner_w - text_w) // 2
    text_y = y1 + BANNER_HEIGHT // 2 + text_h // 2

    # Shadow offset
    shadow_clr = (int(b * 0.3), int(g * 0.3), int(r * 0.3))
    cv2.putText(frame, message, (text_x + 1, text_y + 1),
                FONT, font_scale, shadow_clr, font_thick, cv2.LINE_AA)

    # Main text in state colour
    cv2.putText(frame, message, (text_x, text_y),
                FONT, font_scale, state_colour, font_thick, cv2.LINE_AA)

    # Extra bright highlight for urgent (second pass, slightly lighter)
    if is_urgent:
        highlight = (min(255, b + 80), min(255, g + 80), min(255, r + 80))
        cv2.putText(frame, message, (text_x, text_y),
                    FONT, font_scale, highlight, 1, cv2.LINE_AA)

    return frame
