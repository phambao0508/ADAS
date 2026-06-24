"""
Module D — HUD Theme & Colour System
=====================================
A single cohesive automotive-HUD palette. All renderers import from here so
the theme can be swapped in one place.

Aesthetic
---------
    Background : near-black navy with a subtle blue cast
    Accents    : electric cyan (calm) → amber (caution) → coral red (danger)
    Text       : cool off-white for primary, muted blue-grey for secondary
    Lane lines : cyan / coral pair with strong contrast for the danger swap

All colours are **OpenCV BGR** ``(B, G, R)`` tuples.

Layout
------
    1. Brand neutrals & panel chrome    — used by every panel
    2. Lane state palette               — DEPARTURE_COLOURS
    3. Lane boundary palette            — left/right line colours
    4. Guidance state palette           — GUIDANCE_COLOURS
    5. Mini-map palette                 — schematic-specific tones
    6. Vehicle / count palette          — telemetry + object-box colours
    7. Tunable rendering constants      — alphas, thicknesses, bracket sizes
"""

from typing import Tuple

import numpy as np

BGR = Tuple[int, int, int]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Brand neutrals & panel chrome
# ─────────────────────────────────────────────────────────────────────────────
# Glass-panel tints (deep navy, near-black). Used as the tint colour for
# `draw_glass_panel`. The blur stays the same — only the wash changes.
PANEL_TINT_DEFAULT  = (10, 14, 22)      # dark slate-navy
PANEL_TINT_URGENT   = (18, 14, 38)      # red-tinged for urgent banners

# Panel borders / dividers. Darker = inner separators, brighter = outer chrome.
PANEL_BORDER        = (70,  88, 118)    # cool steel-blue chrome
PANEL_SEPARATOR     = (38,  46,  60)    # dim row divider
PANEL_ACCENT        = (180, 150,  90)   # subtle teal-gold accent strip

# Text. Primary is a cool off-white; secondary is a muted slate-blue.
TEXT_PRIMARY        = (240, 245, 250)   # near-white
TEXT_SECONDARY      = (170, 185, 205)   # muted blue-grey
TEXT_DIM            = (110, 130, 155)   # faded label
TEXT_BRAND          = (150, 168, 188)   # branding watermark

# Corner-bracket chrome (frame_decorations + per-panel brackets)
BRACKET_COLOUR      = (210, 175, 100)   # soft cyan-steel — distinct from any state


# ─────────────────────────────────────────────────────────────────────────────
# 2. Lane departure state palette
# ─────────────────────────────────────────────────────────────────────────────
# These flow through DEPARTURE_COLOURS into the lane fill, the status HUD
# accent bar, the boundary danger swap, and the mini-map highlight.
COLOUR_CENTERED     = (170, 230,  80)   # electric mint — calmer than pure green
COLOUR_WARN         = ( 40, 190, 255)   # rich amber — high contrast on dark
COLOUR_DEPART       = ( 90,  72, 255)   # saturated coral red
COLOUR_LANE_CHANGE  = (255, 195,  90)   # cool azure for safe lane change
COLOUR_NEUTRAL      = (160, 160, 160)   # fallback grey
COLOUR_NONE         = (  0,   0,   0)

DEPARTURE_COLOURS = {
    "CENTERED":          COLOUR_CENTERED,
    "WARN_LEFT":         COLOUR_WARN,
    "WARN_RIGHT":        COLOUR_WARN,
    "DEPART_LEFT":       COLOUR_DEPART,
    "DEPART_RIGHT":      COLOUR_DEPART,
    "LANE_CHANGE_LEFT":  COLOUR_LANE_CHANGE,
    "LANE_CHANGE_RIGHT": COLOUR_LANE_CHANGE,
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Lane boundary palette (the painted lines on the road overlay)
# ─────────────────────────────────────────────────────────────────────────────
BOUNDARY_COLOUR_LEFT_NORMAL  = (245, 200,  80)   # electric cyan-teal
BOUNDARY_COLOUR_RIGHT_NORMAL = ( 80, 165, 255)   # warm coral-orange
BOUNDARY_COLOUR_DANGER       = COLOUR_DEPART     # both sides go coral when departing

BOUNDARY_THICKNESS_SOLID     = 7
BOUNDARY_THICKNESS_DASHED    = 5
BOUNDARY_DASH_LENGTH         = 30
BOUNDARY_GAP_LENGTH          = 20


# ─────────────────────────────────────────────────────────────────────────────
# 4. Guidance state palette (banner + cues)
# ─────────────────────────────────────────────────────────────────────────────
COLOUR_GUIDE_OK      = (255, 220,  60)   # bright cyan — "you have an option"
COLOUR_GUIDE_SLOW    = ( 40, 165, 255)   # amber — "no safe move, slow"
COLOUR_GUIDE_URGENT  = ( 70,  46, 255)   # bright coral — "BRAKE NOW"

GUIDANCE_COLOURS = {
    "GUIDE_NONE":    COLOUR_NONE,
    "GUIDE_LEFT":    COLOUR_GUIDE_OK,
    "GUIDE_RIGHT":   COLOUR_GUIDE_OK,
    "GUIDE_BOTH":    COLOUR_GUIDE_OK,
    "GUIDE_SLOW":    COLOUR_GUIDE_SLOW,
    "GUIDE_URGENT":  COLOUR_GUIDE_URGENT,
}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Mini-map palette (schematic top-down view)
# ─────────────────────────────────────────────────────────────────────────────
# Background gradient: top is brighter than bottom, both very dark navy.
MAP_BG_TOP          = (32, 38, 48)
MAP_BG_BOT          = (16, 20, 28)
MAP_ROAD_FILL       = (44, 50, 60)
MAP_LANE_LINE       = (220, 220, 230)
MAP_LANE_SOLID      = (245, 245, 250)
MAP_EGO_LANE_FILL   = ( 70, 110,  60)   # base tint (gets blended with state colour)
MAP_CENTRELINE      = (110, 130, 150)

# Detection cone — soft cyan
MAP_CONE_FILL       = (180, 200, 130)
MAP_CONE_EDGE       = (220, 240, 165)

# Ego car silhouette
MAP_EGO_BODY        = (245, 246, 250)
MAP_EGO_ACCENT      = (200, 210, 220)
MAP_EGO_WINDOW      = (180, 175, 145)
MAP_EGO_OUTLINE     = (170, 178, 185)

# Detected objects in the mini-map
MAP_OBJ_FRONT       = ( 90,  90, 215)   # red-ish silhouette for front car
MAP_OBJ_SIDE        = (110, 115, 130)   # neutral grey for side vehicles


# ─────────────────────────────────────────────────────────────────────────────
# 6. Vehicle / per-zone count palette
# ─────────────────────────────────────────────────────────────────────────────
# Object-box colours (per zone)
BOX_EGO             = COLOUR_DEPART                       # red — vehicle ahead in our lane
BOX_LEFT            = BOUNDARY_COLOUR_LEFT_NORMAL         # matches left lane line
BOX_RIGHT           = BOUNDARY_COLOUR_RIGHT_NORMAL        # matches right lane line

# Telemetry status dots / value colours
STATUS_OK           = COLOUR_CENTERED
STATUS_DETECTED     = (180, 220, 200)
STATUS_CLOSE        = ( 40, 165, 255)   # amber
STATUS_URGENT       = COLOUR_DEPART
STATUS_OCCUPIED     = COLOUR_DEPART

LINE_TYPE_DASHED    = BOUNDARY_COLOUR_LEFT_NORMAL    # cyan
LINE_TYPE_SOLID     = (120, 150, 220)                # muted coral

# Count badge (LEFT/RIGHT vehicle counters in the telemetry panel)
COUNT_ZERO          = ( 95, 110, 130)
COUNT_LOW           = ( 60, 200, 255)   # 1–2 vehicles : amber
COUNT_HIGH          = COLOUR_DEPART     # 3+ vehicles  : red


# ─────────────────────────────────────────────────────────────────────────────
# 7. Rendering constants (shared)
# ─────────────────────────────────────────────────────────────────────────────
LANE_FILL_ALPHA     = 0.35
HUD_BG_ALPHA        = 0.72


# ── Compatibility aliases (kept so older imports still resolve) ──────────────
COLOUR_GUIDE        = COLOUR_GUIDE_OK
HUD_ACCENT          = DEPARTURE_COLOURS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def lane_fill_colour(departure_state: str) -> BGR:
    """Return the BGR fill colour for the ego-lane fill given a departure state."""
    return DEPARTURE_COLOURS.get(departure_state, COLOUR_CENTERED)


def boundary_colour_left(departure_state: str) -> BGR:
    """Return the BGR colour for the LEFT boundary line."""
    if departure_state in ("DEPART_LEFT", "LANE_CHANGE_LEFT", "WARN_LEFT"):
        return BOUNDARY_COLOUR_DANGER
    return BOUNDARY_COLOUR_LEFT_NORMAL


def boundary_colour_right(departure_state: str) -> BGR:
    """Return the BGR colour for the RIGHT boundary line."""
    if departure_state in ("DEPART_RIGHT", "LANE_CHANGE_RIGHT", "WARN_RIGHT"):
        return BOUNDARY_COLOUR_DANGER
    return BOUNDARY_COLOUR_RIGHT_NORMAL


def shade(colour: BGR, factor: float) -> BGR:
    """Multiplicatively darken (factor<1) or brighten (factor>1) a BGR colour."""
    return tuple(int(np.clip(c * factor, 0, 255)) for c in colour)


def with_alpha_over(colour: BGR, alpha: float, base: BGR = (0, 0, 0)) -> BGR:
    """Composite a BGR colour over a base, returning the resulting BGR."""
    return tuple(int(np.clip(c * alpha + b * (1 - alpha), 0, 255))
                 for c, b in zip(colour, base))
