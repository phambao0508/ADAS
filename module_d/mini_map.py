"""
Module D  —  Step D5: Mini-Map Schematic (Bottom Left)  [UTOUR-Style Redesign]
===============================================================================
Redesigned to match the UTOUR automotive HUD aesthetic:
  - Light blue-gray background panel with rounded corners
  - Perspective road with converging lane lines
  - Stylized ego-vehicle (top-down car silhouette) at bottom
  - Teal/cyan detection cone fanning out ahead
  - Object silhouettes for front/side detected vehicles
  - Status display at top (proximity indicator)
  - Soft, modern, minimal design

MAP COORDINATE SYSTEM
---------------------
    The road is drawn in perspective: lane lines converge toward
    a vanishing point at the top-centre of the panel.

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

from .hud_colours import (
    lane_fill_colour,
    DEPARTURE_COLOURS,
    MAP_BG_TOP, MAP_BG_BOT, MAP_ROAD_FILL,
    MAP_LANE_LINE, MAP_LANE_SOLID, MAP_EGO_LANE_FILL, MAP_CENTRELINE,
    MAP_CONE_FILL, MAP_CONE_EDGE,
    MAP_EGO_BODY, MAP_EGO_ACCENT, MAP_EGO_WINDOW, MAP_EGO_OUTLINE,
    MAP_OBJ_FRONT, MAP_OBJ_SIDE,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_BRAND,
    COLOUR_DEPART, COLOUR_WARN,
)

_PROX_DETECTED    = "DETECTED"
_PROX_CLOSE       = "CLOSE"
_PROX_VERY_CLOSE  = "VERY_CLOSE"


# ── Mini-map panel dimensions ────────────────────────────────────────────
MAP_WIDTH   = 230    # px
MAP_HEIGHT  = 320    # px
MAP_MARGIN  = 14     # px from bottom-left corner
CORNER_RAD  = 18     # rounded corner radius

# ── Local aliases onto the central theme ─────────────────────────────────
COL_BG_TOP        = MAP_BG_TOP
COL_BG_BOT        = MAP_BG_BOT
COL_ROAD_FILL     = MAP_ROAD_FILL
COL_LANE_LINE     = MAP_LANE_LINE
COL_LANE_SOLID    = MAP_LANE_SOLID
COL_EGO_LANE_FILL = MAP_EGO_LANE_FILL
COL_CONE_FILL     = MAP_CONE_FILL
COL_CONE_EDGE     = MAP_CONE_EDGE
COL_EGO_BODY      = MAP_EGO_BODY
COL_EGO_ACCENT    = MAP_EGO_ACCENT
COL_EGO_WINDOW    = MAP_EGO_WINDOW
COL_EGO_OUTLINE   = MAP_EGO_OUTLINE
COL_OBJ_CAR       = MAP_OBJ_FRONT
COL_OBJ_SIDE      = MAP_OBJ_SIDE
COL_TEXT_MAIN     = TEXT_PRIMARY
COL_TEXT_DIM      = TEXT_SECONDARY
COL_TEXT_BRAND    = TEXT_BRAND
COL_WARNING       = COLOUR_DEPART
COL_CAUTION       = COLOUR_WARN


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
    """Draw the UTOUR-style top-down schematic mini-map in the bottom-left corner."""
    H, W = frame.shape[:2]

    mw, mh = MAP_WIDTH, MAP_HEIGHT
    canvas = np.zeros((mh, mw, 3), dtype=np.uint8)

    # ── Gradient background ───────────────────────────────────────────────
    _draw_gradient_bg(canvas, mw, mh)

    # ── Perspective road geometry ─────────────────────────────────────────
    vp_x = mw // 2          # vanishing point x
    vp_y = int(mh * 0.14)   # vanishing point y (near top)
    road_bot_y = mh - 28    # bottom of road

    # Road edges at bottom (wide) and converge at VP
    road_bl_x = int(mw * 0.12)   # bottom-left road edge
    road_br_x = int(mw * 0.88)   # bottom-right road edge

    # Draw road surface
    road_pts = np.array([
        [vp_x - 3, vp_y],
        [vp_x + 3, vp_y],
        [road_br_x, road_bot_y],
        [road_bl_x, road_bot_y],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [road_pts], COL_ROAD_FILL)

    # ── Lane markings in perspective ──────────────────────────────────────
    road_cx_bot = (road_bl_x + road_br_x) // 2
    road_w_bot = road_br_x - road_bl_x

    # Ego lane boundaries (inner pair — thicker, more prominent)
    left_ego_x  = road_bl_x + int(road_w_bot * 0.33)
    right_ego_x = road_bl_x + int(road_w_bot * 0.67)

    _draw_ego_lane_fill(
        canvas,
        vp_x,
        vp_y,
        left_ego_x,
        right_ego_x,
        road_bot_y,
        departure_state,
    )

    _draw_persp_line(canvas, vp_x, vp_y, left_ego_x,  road_bot_y, left_type,
                     COL_LANE_SOLID, departure_state, "left",  thickness=3)
    _draw_persp_line(canvas, vp_x, vp_y, right_ego_x, road_bot_y, right_type,
                     COL_LANE_SOLID, departure_state, "right", thickness=3)

    # Outer lane lines (dimmer, thinner)
    outer_col = tuple(min(255, c + 40) for c in COL_LANE_LINE)
    _draw_persp_line(canvas, vp_x, vp_y, road_bl_x + int(road_w_bot * 0.08), road_bot_y,
                     "dashed", outer_col, departure_state, "left", thickness=1)
    _draw_persp_line(canvas, vp_x, vp_y, road_bl_x + int(road_w_bot * 0.92), road_bot_y,
                     "dashed", outer_col, departure_state, "right", thickness=1)

    # ── Detection cone (teal/cyan fan) ────────────────────────────────────
    _draw_detection_cone(canvas, mw, mh)

    # ── Detected objects ──────────────────────────────────────────────────
    # Front vehicle
    if front_proximity in (_PROX_DETECTED, _PROX_CLOSE, _PROX_VERY_CLOSE):
        if front_proximity == _PROX_VERY_CLOSE:
            fv_y = int(mh * 0.35)
            col = COL_WARNING
        elif front_proximity == _PROX_CLOSE:
            fv_y = int(mh * 0.26)
            col = COL_OBJ_CAR
        else:  # DETECTED - far away, near the top of the cone
            fv_y = int(mh * 0.19)
            col = COL_TEXT_DIM
        _draw_front_car(canvas, vp_x, fv_y, colour=col)

    # Side objects (vehicles in adjacent lanes)
    if not left_clear:
        lv_x = road_bl_x + int(road_w_bot * 0.18)
        _draw_side_car(canvas, lv_x, int(mh * 0.55), COL_OBJ_SIDE)

    if not right_clear:
        rv_x = road_bl_x + int(road_w_bot * 0.82)
        _draw_side_car(canvas, rv_x, int(mh * 0.55), COL_OBJ_SIDE)

    # ── Ego vehicle ──────────────────────────────────────────────────────
    ego_x = mw // 2
    if smoothed_offset is not None:
        max_shift = (right_ego_x - left_ego_x) // 4
        shift = -int(smoothed_offset / 150.0 * max_shift)
        shift = max(-max_shift, min(max_shift, shift))
        ego_x += shift

    _draw_ego_car(canvas, ego_x, int(mh * 0.76))

    # ── Status indicator at top ───────────────────────────────────────────
    _draw_status_top(canvas, mw, front_proximity, departure_state)

    # ── Branding ──────────────────────────────────────────────────────────
    brand = "ADAS"
    ts = cv2.getTextSize(brand, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
    cv2.putText(canvas, brand, ((mw - ts[0]) // 2, mh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_TEXT_BRAND, 1, cv2.LINE_AA)

    # ── Rounded-corner mask ───────────────────────────────────────────────
    mask = _rounded_rect_mask(mw, mh, CORNER_RAD)

    # ── Paste onto frame (bottom-left) with blending ──────────────────────
    px = MAP_MARGIN
    py = H - MAP_HEIGHT - MAP_MARGIN
    py = max(0, py)
    paste_h = min(mh, H - py)
    paste_w = min(mw, W - px)

    roi = frame[py:py + paste_h, px:px + paste_w]
    mask_roi = mask[:paste_h, :paste_w]
    canvas_roi = canvas[:paste_h, :paste_w]

    alpha = 0.93
    blended = cv2.addWeighted(canvas_roi, alpha, roi, 1.0 - alpha, 0)
    np.copyto(roi, blended, where=(mask_roi[:, :, None] > 0))

    return frame


# ═══════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _draw_gradient_bg(canvas, mw, mh):
    """Vertical gradient from light to slightly less light — VECTORISED."""
    top = np.array(COL_BG_TOP, dtype=np.float32)
    bot = np.array(COL_BG_BOT, dtype=np.float32)
    # Build (mh, 1, 3) gradient in one shot via linspace
    t = np.linspace(0.0, 1.0, mh, dtype=np.float32).reshape(mh, 1, 1)
    gradient = (top * (1.0 - t) + bot * t).astype(np.uint8)  # (mh, 1, 3)
    canvas[:] = gradient  # broadcasts across width


def _rounded_rect_mask(w, h, r):
    """Create a binary mask with rounded corners."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (r, 0), (w - r - 1, h - 1), 255, cv2.FILLED)
    cv2.rectangle(mask, (0, r), (w - 1, h - r - 1), 255, cv2.FILLED)
    cv2.circle(mask, (r, r), r, 255, cv2.FILLED)
    cv2.circle(mask, (w - r - 1, r), r, 255, cv2.FILLED)
    cv2.circle(mask, (r, h - r - 1), r, 255, cv2.FILLED)
    cv2.circle(mask, (w - r - 1, h - r - 1), r, 255, cv2.FILLED)
    return mask


def _draw_ego_lane_fill(canvas, vp_x, vp_y, left_bot_x, right_bot_x, bot_y, departure_state):
    """Fill the ego lane area in the minimap."""
    colour = lane_fill_colour(departure_state)
    fill_colour = tuple(int(0.55 * COL_EGO_LANE_FILL[i] + 0.45 * colour[i]) for i in range(3))
    top_y = vp_y + 18
    left_top_x = int(vp_x + (left_bot_x - vp_x) * 0.12)
    right_top_x = int(vp_x + (right_bot_x - vp_x) * 0.12)
    pts = np.array([
        [left_top_x, top_y],
        [right_top_x, top_y],
        [right_bot_x, bot_y],
        [left_bot_x, bot_y],
    ], dtype=np.int32)

    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts], fill_colour)
    cv2.addWeighted(overlay, 0.36, canvas, 0.64, 0, canvas)

    center_pts = np.array([
        [vp_x, top_y],
        [(left_bot_x + right_bot_x) // 2, bot_y],
    ], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [center_pts], False, MAP_CENTRELINE, 1, cv2.LINE_AA)


def _draw_persp_line(canvas, vp_x, vp_y, bot_x, bot_y,
                     line_type, colour, departure_state, side, thickness=1):
    """Draw a lane line from vanishing point to bottom in perspective."""
    if side == "left" and departure_state in ("DEPART_LEFT", "LANE_CHANGE_LEFT", "WARN_LEFT"):
        colour = COL_WARNING
    elif side == "right" and departure_state in ("DEPART_RIGHT", "LANE_CHANGE_RIGHT", "WARN_RIGHT"):
        colour = COL_WARNING

    if line_type == "solid":
        cv2.line(canvas, (vp_x, vp_y), (bot_x, bot_y), colour, thickness, cv2.LINE_AA)
    else:
        # Dashed — segments get larger toward the bottom (perspective)
        for t1, t2 in ((0.18, 0.25), (0.32, 0.42), (0.50, 0.63), (0.72, 0.90)):
            x1 = int(vp_x + (bot_x - vp_x) * t1)
            y1 = int(vp_y + (bot_y - vp_y) * t1)
            x2 = int(vp_x + (bot_x - vp_x) * t2)
            y2 = int(vp_y + (bot_y - vp_y) * t2)
            cv2.line(canvas, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)


def _draw_detection_cone(canvas, mw, mh):
    """Draw the teal detection cone fanning from the ego car forward."""
    cx = mw // 2
    ego_y = int(mh * 0.70)
    cone_top_y = int(mh * 0.28)

    # Narrow at car, wide at detection extent
    bot_hw = 18
    top_hw = 52

    pts = np.array([
        [cx - bot_hw, ego_y],
        [cx - top_hw, cone_top_y],
        [cx + top_hw, cone_top_y],
        [cx + bot_hw, ego_y],
    ], dtype=np.int32)

    # Semi-transparent fill
    overlay = canvas.copy()
    # Soft teal-cyan   (BGR ≈ 200, 210, 170)
    cv2.fillPoly(overlay, [pts], COL_CONE_FILL)
    cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0, canvas)

    # Edge lines
    edge = COL_CONE_EDGE
    cv2.line(canvas, (cx - bot_hw, ego_y), (cx - top_hw, cone_top_y), edge, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx + bot_hw, ego_y), (cx + top_hw, cone_top_y), edge, 1, cv2.LINE_AA)
    # Top arc
    cv2.line(canvas, (cx - top_hw, cone_top_y), (cx + top_hw, cone_top_y), edge, 1, cv2.LINE_AA)


def _draw_ego_car(canvas, cx, cy):
    """
    Draw a refined top-down car silhouette closely matching the UTOUR reference.
    The car is viewed from directly above: sleek, aerodynamic shape.
    """
    # ── Overall proportions ───────────────────────────────────────────────
    car_w = 38   # total width
    car_h = 64   # total length (front to back)
    hw = car_w // 2
    hh = car_h // 2

    # ── Car body outline (smooth, aerodynamic) ────────────────────────────
    body = np.array([
        # Front nose (aerodynamic point)
        [cx,           cy - hh - 4],
        # Front-left fender
        [cx - hw + 4,  cy - hh + 10],
        [cx - hw,      cy - hh + 20],
        # Left side
        [cx - hw - 1,  cy - 5],
        [cx - hw,      cy + hh - 12],
        # Rear-left
        [cx - hw + 2,  cy + hh - 2],
        [cx - hw + 5,  cy + hh + 2],
        # Rear (flat)
        [cx + hw - 5,  cy + hh + 2],
        [cx + hw - 2,  cy + hh - 2],
        # Right side
        [cx + hw,      cy + hh - 12],
        [cx + hw + 1,  cy - 5],
        # Front-right fender
        [cx + hw,      cy - hh + 20],
        [cx + hw - 4,  cy - hh + 10],
    ], dtype=np.int32)

    # Main body fill (white/silver)
    cv2.fillPoly(canvas, [body], COL_EGO_BODY)
    # Subtle outline
    cv2.polylines(canvas, [body], True, COL_EGO_OUTLINE, 1, cv2.LINE_AA)

    # ── Front windshield ──────────────────────────────────────────────────
    wind = np.array([
        [cx - hw + 7,  cy - hh + 16],
        [cx,           cy - hh + 2],
        [cx + hw - 7,  cy - hh + 16],
        [cx + hw - 6,  cy - hh + 25],
        [cx - hw + 6,  cy - hh + 25],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [wind], COL_EGO_WINDOW)

    # ── Roof / cabin ──────────────────────────────────────────────────────
    roof = np.array([
        [cx - hw + 6,  cy - hh + 27],
        [cx + hw - 6,  cy - hh + 27],
        [cx + hw - 5,  cy + 4],
        [cx - hw + 5,  cy + 4],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [roof], COL_EGO_ACCENT)

    # ── Rear window ───────────────────────────────────────────────────────
    rw = np.array([
        [cx - hw + 6,  cy + 6],
        [cx + hw - 6,  cy + 6],
        [cx + hw - 4,  cy + 18],
        [cx - hw + 4,  cy + 18],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [rw], COL_EGO_WINDOW)

    # ── Side mirrors (small notches) ──────────────────────────────────────
    cv2.line(canvas, (cx - hw - 2, cy - 10), (cx - hw - 5, cy - 8),
             COL_EGO_OUTLINE, 2, cv2.LINE_AA)
    cv2.line(canvas, (cx + hw + 2, cy - 10), (cx + hw + 5, cy - 8),
             COL_EGO_OUTLINE, 2, cv2.LINE_AA)


def _draw_front_car(canvas, cx, cy, colour=(80, 85, 165)):
    """Draw a simple top-down car silhouette for a front-detected vehicle."""
    w, h = 26, 18
    hw, hh = w // 2, h // 2

    # Rounded rectangle body
    pts = np.array([
        [cx - hw + 3, cy - hh],
        [cx + hw - 3, cy - hh],
        [cx + hw, cy - hh + 3],
        [cx + hw, cy + hh - 3],
        [cx + hw - 3, cy + hh],
        [cx - hw + 3, cy + hh],
        [cx - hw, cy + hh - 3],
        [cx - hw, cy - hh + 3],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], colour)

    # Window strip
    rw = int(hw * 0.55)
    cv2.rectangle(canvas, (cx - rw, cy - hh + 3), (cx + rw, cy + hh - 3),
                  (max(0, colour[0] - 25), max(0, colour[1] - 25), max(0, colour[2] - 35)),
                  cv2.FILLED)


def _draw_side_car(canvas, cx, cy, colour=(105, 105, 115)):
    """Draw a small top-down car silhouette for a side-detected vehicle."""
    w, h = 22, 16
    hw, hh = w // 2, h // 2

    # Rounded body
    pts = np.array([
        [cx - hw + 2, cy - hh],
        [cx + hw - 2, cy - hh],
        [cx + hw, cy - hh + 2],
        [cx + hw, cy + hh - 2],
        [cx + hw - 2, cy + hh],
        [cx - hw + 2, cy + hh],
        [cx - hw, cy + hh - 2],
        [cx - hw, cy - hh + 2],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], colour)

    # Window strip
    rw = int(hw * 0.5)
    darker = (max(0, colour[0] - 25), max(0, colour[1] - 25), max(0, colour[2] - 25))
    cv2.rectangle(canvas, (cx - rw, cy - hh + 3), (cx + rw, cy + hh - 3),
                  darker, cv2.FILLED)


def _draw_status_top(canvas, mw, front_proximity, departure_state):
    """Draw proximity status icon at the top of the minimap panel."""
    if front_proximity == _PROX_VERY_CLOSE:
        label = "!!"
        col = COL_WARNING
        font_scale = 1.1
    elif front_proximity == _PROX_CLOSE:
        label = "!"
        col = COL_CAUTION
        font_scale = 1.1
    elif front_proximity == _PROX_DETECTED:
        label = ".."
        col = COL_TEXT_DIM
        font_scale = 0.9
    else:
        label = "--"
        col = COL_TEXT_MAIN
        font_scale = 0.9

    ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
    tx = (mw - ts[0]) // 2
    cv2.putText(canvas, label, (tx, 38),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, 2, cv2.LINE_AA)

    # Small label
    unit = "PROXIMITY"
    us = cv2.getTextSize(unit, cv2.FONT_HERSHEY_SIMPLEX, 0.28, 1)[0]
    cv2.putText(canvas, unit, ((mw - us[0]) // 2, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, COL_TEXT_DIM, 1, cv2.LINE_AA)
