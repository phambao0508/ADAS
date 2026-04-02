"""
Module D  —  Step D5: Mini-Map Schematic (Bottom Left)  [Redesigned]
=====================================================================
Redesigned to match the HUD demo aesthetic:
  - Glowing ego-vehicle triangle in state colour
  - Adjacent lanes shaded clear (blue-tint) or occupied (red-tint)
  - Guidance directional arrows drawn in adjacent lane when GUIDE_LEFT/RIGHT
  - Front vehicle box with colour coded by proximity (orange=CLOSE, red=VERY_CLOSE)
  - Lateral offset dot above ego triangle
  - "BIRD EYE" label at top with dim monospaced style
  - Dark panel with subtle border

MAP COORDINATE SYSTEM
---------------------
    Left  20% = left adjacent lane
    Middle 60% = ego lane
    Right 20% = right adjacent lane

SIGN CONVENTION (from B1)
--------------------------
    offset > 0 → car is LEFT  of lane centre → dot moves left
    offset < 0 → car is RIGHT of lane centre → dot moves right
    We NEGATE the raw offset to get physical screen direction.

INPUTS
------
    frame           : np.ndarray (H, W, 3)
    departure_state : str       — Module B
    smoothed_offset : float|None — Module B
    front_proximity : str       — Module C (NONE|CLOSE|VERY_CLOSE)
    left_clear      : bool      — Module C
    right_clear     : bool      — Module C
    left_type       : str       — Module A ('solid'|'dashed')
    right_type      : str       — Module A ('solid'|'dashed')

OUTPUT
------
    np.ndarray : frame with mini-map drawn
"""

import numpy as np
import cv2
from typing import Optional

from .hud_colours import lane_fill_colour, DEPARTURE_COLOURS, GUIDANCE_COLOURS

_PROX_CLOSE       = "CLOSE"
_PROX_VERY_CLOSE  = "VERY_CLOSE"


# ── Mini-map panel dimensions ────────────────────────────────────────────
MAP_WIDTH   = 230    # px  (was 155)
MAP_HEIGHT  = 185    # px  (was 125)
MAP_MARGIN  = 14     # px from bottom-left corner

ADJ_LANE_W_FRAC = 0.20     # each adjacent lane = 20% of map width
EGO_LANE_W_FRAC = 0.60     # ego lane            = 60% of map width

# ── Colours (BGR) ─────────────────────────────────────────────────────────
COL_MAP_BG       = (  8,  12,  20)   # #06090f  very dark blue
COL_MAP_BORDER   = ( 55,  70,  90)   # dim blue-grey border — #1e2d45 approx
COL_ADJ_CLEAR    = ( 50,  40,  10)   # dim blue tint (clear)
COL_ADJ_OCCUPIED = ( 30,  30, 110)   # red tint (occupied) → BGR
COL_DIVIDER_L    = (238, 204,   0)   # cyan  #00ccee
COL_DIVIDER_R    = (  0, 136, 255)   # orange #ff8800
COL_DIVIDER_DANGER=(51,  51, 255)    # red when departing
COL_OUTER_DIV    = ( 90, 100, 120)   # outer dividers
COL_LABEL        = ( 90, 110, 130)   # dim label text
COL_OFFSET_DOT   = (210, 210, 210)   # white dot
COL_FRONT_CLOSE  = (  0, 136, 255)   # orange — CLOSE
COL_FRONT_URGENT = ( 51,  51, 255)   # red    — VERY_CLOSE
MAX_OFFSET_DISPLAY = 150             # px offset → edge of ego lane


def draw_mini_map(
    frame:           np.ndarray,
    departure_state: str,
    smoothed_offset: Optional[float],
    front_proximity: str,
    left_clear:      bool,
    right_clear:     bool,
    left_type:       str,
    right_type:      str,
    guidance_state:  str = "GUIDE_NONE",
) -> np.ndarray:
    """Draw the top-down schematic mini-map in the bottom-left corner."""
    H, W = frame.shape[:2]

    mw, mh = MAP_WIDTH, MAP_HEIGHT
    canvas  = np.zeros((mh, mw, 3), dtype=np.uint8)
    canvas[:] = COL_MAP_BG

    left_w   = int(mw * ADJ_LANE_W_FRAC)
    right_w  = int(mw * ADJ_LANE_W_FRAC)
    ego_w    = mw - left_w - right_w
    ego_cx   = left_w + ego_w // 2

    # ── Adjacent lane backgrounds ─────────────────────────────────────────
    l_col = COL_ADJ_CLEAR    if left_clear  else COL_ADJ_OCCUPIED
    r_col = COL_ADJ_CLEAR    if right_clear else COL_ADJ_OCCUPIED
    cv2.rectangle(canvas, (0, 0),                (left_w - 1,  mh - 1), l_col, cv2.FILLED)
    cv2.rectangle(canvas, (mw - right_w, 0),     (mw - 1,      mh - 1), r_col, cv2.FILLED)

    # ── Ego lane background (state colour, very dim) ──────────────────────
    lane_col = lane_fill_colour(departure_state)
    b, g, r  = lane_col
    dash_ego = (max(0, b - 100), max(0, g - 100), max(0, r - 100))
    cv2.rectangle(canvas, (left_w, 0), (left_w + ego_w - 1, mh - 1), dash_ego, cv2.FILLED)

    # ── Lane dividers ─────────────────────────────────────────────────────
    left_div_clr  = COL_DIVIDER_DANGER if departure_state in (
        "DEPART_LEFT", "LANE_CHANGE_LEFT", "WARN_LEFT") else COL_DIVIDER_L
    right_div_clr = COL_DIVIDER_DANGER if departure_state in (
        "DEPART_RIGHT", "LANE_CHANGE_RIGHT", "WARN_RIGHT") else COL_DIVIDER_R

    _draw_lane_divider(canvas, left_w,         left_type,  mh, left_div_clr)
    _draw_lane_divider(canvas, left_w + ego_w, right_type, mh, right_div_clr)

    # Outer dividers (dim)
    _draw_lane_divider(canvas, 1,      "dashed", mh, COL_OUTER_DIV)
    _draw_lane_divider(canvas, mw - 2, "dashed", mh, COL_OUTER_DIV)

    # ── Guidance arrows in adjacent lanes ─────────────────────────────────
    if guidance_state in ("GUIDE_LEFT", "GUIDE_BOTH"):
        _draw_arrow(canvas, left_w // 2, mh // 2, "left",  COL_DIVIDER_L)
    if guidance_state in ("GUIDE_RIGHT", "GUIDE_BOTH"):
        _draw_arrow(canvas, left_w + ego_w + right_w // 2, mh // 2, "right", COL_DIVIDER_R)

    # ── Ego vehicle marker (▲) with glow ──────────────────────────────────
    ego_veh_y = mh - 28
    _draw_triangle(canvas, ego_cx, ego_veh_y, 14, lane_col)

    # ── Lateral offset dot ────────────────────────────────────────────────
    if smoothed_offset is not None:
        half_ego = ego_w // 2
        # Negate: offset>0 → car left → dot left; offset<0 → car right → dot right
        shift = -int(smoothed_offset / MAX_OFFSET_DISPLAY * half_ego)
        shift = max(-half_ego, min(half_ego, shift))
        dot_x = ego_cx + shift
        dot_y = ego_veh_y - 20
        cv2.circle(canvas, (dot_x, dot_y), 4, COL_OFFSET_DOT, cv2.FILLED)

    # ── Front vehicle box ─────────────────────────────────────────────────
    if front_proximity in (_PROX_CLOSE, _PROX_VERY_CLOSE):
        fv_col = COL_FRONT_URGENT if front_proximity == _PROX_VERY_CLOSE else COL_FRONT_CLOSE
        fv_y   = 24 if front_proximity == _PROX_VERY_CLOSE else 40
        fv_h   = 14
        fv_x1  = left_w + ego_w // 5
        fv_x2  = left_w + 4 * ego_w // 5
        cv2.rectangle(canvas, (fv_x1, fv_y), (fv_x2, fv_y + fv_h), fv_col, 1)
        if front_proximity == _PROX_VERY_CLOSE:
            # Fill for urgent
            fill = (int(fv_col[0] * 0.35), int(fv_col[1] * 0.35), int(fv_col[2] * 0.35))
            cv2.rectangle(canvas, (fv_x1 + 1, fv_y + 1),
                          (fv_x2 - 1, fv_y + fv_h - 1), fill, cv2.FILLED)

    # ── Adjacent vehicle boxes (left / right) ─────────────────────────────
    if not left_clear:
        lv_w = int(left_w * 0.55)
        lv_cx = left_w // 2
        cv2.rectangle(canvas, (lv_cx - lv_w // 2, mh // 2 - 5),
                      (lv_cx + lv_w // 2, mh // 2 + 5), (100, 100, 220), 1)
    if not right_clear:
        rv_w  = int(right_w * 0.55)
        rv_cx = left_w + ego_w + right_w // 2
        cv2.rectangle(canvas, (rv_cx - rv_w // 2, mh // 2 - 5),
                      (rv_cx + rv_w // 2, mh // 2 + 5), (100, 100, 220), 1)

    # ── "BIRD EYE" label ──────────────────────────────────────────────────
    cv2.putText(canvas, "BIRD EYE", (5, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_LABEL, 1, cv2.LINE_AA)

    # ── Border ────────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (0, 0), (mw - 1, mh - 1), COL_MAP_BORDER, 1)

    # ── Paste onto main frame (bottom-left) ───────────────────────────────
    px = MAP_MARGIN
    py = H - MAP_HEIGHT - MAP_MARGIN
    frame[py:py + mh, px:px + mw] = canvas

    return frame


def _draw_lane_divider(canvas, x: int, line_type: str, mh: int, colour):
    """Draw a vertical lane divider line (solid or dashed)."""
    if line_type == "solid":
        cv2.line(canvas, (x, 0), (x, mh - 1), colour, 1)
    else:
        y = 0
        draw = True
        while y < mh:
            seg_end = min(y + 7, mh - 1)
            if draw:
                cv2.line(canvas, (x, y), (x, seg_end), colour, 1)
            y   += (7 if draw else 5)
            draw = not draw


def _draw_triangle(canvas, cx: int, cy: int, size: int, colour):
    """Draw an upward-pointing filled triangle (ego vehicle marker)."""
    pts = np.array([
        [cx,          cy - size],        # apex
        [cx - size,   cy + size // 2],   # bottom-left
        [cx + size,   cy + size // 2],   # bottom-right
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], colour)
    # Bright outline for glow feel
    b, g, r = colour
    bright = (min(255, b + 80), min(255, g + 80), min(255, r + 80))
    cv2.polylines(canvas, [pts], True, bright, 1)


def _draw_arrow(canvas, cx: int, cy: int, direction: str, colour):
    """Draw a small directional arrow (← or →) in the adjacent lane."""
    d  = -1 if direction == "left" else 1
    x1 = cx - d * 7
    x2 = cx + d * 7
    # Shaft
    cv2.line(canvas, (x1, cy), (x2, cy), colour, 1)
    # Arrowhead
    pts = np.array([
        [x2,         cy],
        [x2 - d * 5, cy - 4],
        [x2 - d * 5, cy + 4],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], colour)
