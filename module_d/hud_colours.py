"""
Module D  —  HUD Colour Constants  (updated to match HUD demo aesthetic)
=========================================================================
All colours are in OpenCV BGR format: (Blue, Green, Red).

Demo palette reference
----------------------
    CENTERED          (#00c84a)  → BGR (74, 200, 0)   bright green
    WARN              (#ffc200)  → BGR (0, 194, 255)   amber/yellow
    DEPART            (#ff3333)  → BGR (51, 51, 255)   red
    LANE_CHANGE       (#2299ff)  → BGR (255, 153, 34)  blue
    GUIDE_LEFT/RIGHT  (#00e5ff)  → BGR (255, 229, 0)   cyan
    GUIDE_SLOW        (#ff9900)  → BGR (0, 153, 255)   orange
    GUIDE_URGENT      (#ff2020)  → BGR (32, 32, 255)   bright red
"""

import numpy as np

# ── State → lane fill colour (BGR) ───────────────────────────────────────
COLOUR_CENTERED    = ( 74, 200,   0)   # #00c84a  bright green
COLOUR_WARN        = (  0, 194, 255)   # #ffc200  amber
COLOUR_DEPART      = ( 51,  51, 255)   # #ff3333  red
COLOUR_LANE_CHANGE = (255, 153,  34)   # #2299ff  blue
COLOUR_GUIDE       = (255, 229,   0)   # #00e5ff  cyan (guidance active)
COLOUR_GUIDE_SLOW  = (  0, 153, 255)   # #ff9900  orange
COLOUR_GUIDE_URGENT= ( 32,  32, 255)   # #ff2020  bright red
COLOUR_NONE        = (  0,   0,   0)   # black

# Map of each Module B departure state → BGR fill colour
DEPARTURE_COLOURS = {
    "CENTERED":          COLOUR_CENTERED,
    "WARN_LEFT":         COLOUR_WARN,
    "WARN_RIGHT":        COLOUR_WARN,
    "DEPART_LEFT":       COLOUR_DEPART,
    "DEPART_RIGHT":      COLOUR_DEPART,
    "LANE_CHANGE_LEFT":  COLOUR_LANE_CHANGE,
    "LANE_CHANGE_RIGHT": COLOUR_LANE_CHANGE,
}

# ── Boundary line colours (BGR) ───────────────────────────────────────────
# Left  → cyan  #00ccee → BGR (238, 204, 0)
# Right → orange #ff8800 → BGR (0, 136, 255)
BOUNDARY_COLOUR_LEFT_NORMAL  = (238, 204,   0)   # cyan
BOUNDARY_COLOUR_RIGHT_NORMAL = (  0, 136, 255)   # orange
BOUNDARY_COLOUR_DANGER       = ( 51,  51, 255)   # red — departing on that side
BOUNDARY_THICKNESS_SOLID     = 7    # px  (was 4)
BOUNDARY_THICKNESS_DASHED    = 5    # px  (was 3)
BOUNDARY_DASH_LENGTH         = 30   # px of drawn segment
BOUNDARY_GAP_LENGTH          = 20   # px of gap between dashes

# ── Guidance state → banner colours (BGR) ─────────────────────────────────
GUIDANCE_COLOURS = {
    "GUIDE_NONE":    (  0,   0,   0),
    "GUIDE_LEFT":    (255, 229,   0),   # cyan   #00e5ff
    "GUIDE_RIGHT":   (255, 229,   0),   # cyan
    "GUIDE_BOTH":    (255, 229,   0),   # cyan
    "GUIDE_SLOW":    (  0, 153, 255),   # orange #ff9900
    "GUIDE_URGENT":  ( 32,  32, 255),   # red    #ff2020
}

# ── Lane fill transparency ────────────────────────────────────────────────
LANE_FILL_ALPHA   = 0.35   # more visible fill (was 0.22)

# ── HUD panel transparency ────────────────────────────────────────────────
HUD_BG_ALPHA      = 0.72   # matches demo semi-transparent bg

# ── Accent colours for departure HUD bar (same as departure colours) ──────
HUD_ACCENT        = DEPARTURE_COLOURS


def lane_fill_colour(departure_state: str):
    """Return the BGR fill colour for the ego-lane fill given a departure state."""
    return DEPARTURE_COLOURS.get(departure_state, COLOUR_CENTERED)


def boundary_colour_left(departure_state: str):
    """Return the BGR colour for the LEFT boundary line."""
    if departure_state in ("DEPART_LEFT", "LANE_CHANGE_LEFT", "WARN_LEFT"):
        return BOUNDARY_COLOUR_DANGER
    return BOUNDARY_COLOUR_LEFT_NORMAL


def boundary_colour_right(departure_state: str):
    """Return the BGR colour for the RIGHT boundary line."""
    if departure_state in ("DEPART_RIGHT", "LANE_CHANGE_RIGHT", "WARN_RIGHT"):
        return BOUNDARY_COLOUR_DANGER
    return BOUNDARY_COLOUR_RIGHT_NORMAL
