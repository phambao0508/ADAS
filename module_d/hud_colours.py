from typing import Tuple

import numpy as np

BGR = Tuple[int, int, int]

PANEL_TINT_DEFAULT  = (10, 14, 22)
PANEL_TINT_URGENT   = (18, 14, 38)

PANEL_BORDER        = (70,  88, 118)
PANEL_SEPARATOR     = (38,  46,  60)
PANEL_ACCENT        = (180, 150,  90)

TEXT_PRIMARY        = (240, 245, 250)
TEXT_SECONDARY      = (170, 185, 205)
TEXT_DIM            = (110, 130, 155)
TEXT_BRAND          = (150, 168, 188)

BRACKET_COLOUR      = (210, 175, 100)

COLOUR_CENTERED     = (170, 230,  80)
COLOUR_WARN         = ( 40, 190, 255)
COLOUR_DEPART       = ( 90,  72, 255)
COLOUR_LANE_CHANGE  = (255, 195,  90)
COLOUR_NEUTRAL      = (160, 160, 160)
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

BOUNDARY_COLOUR_LEFT_NORMAL  = (245, 200,  80)
BOUNDARY_COLOUR_RIGHT_NORMAL = ( 80, 165, 255)
BOUNDARY_COLOUR_DANGER       = COLOUR_DEPART

BOUNDARY_THICKNESS_SOLID     = 7
BOUNDARY_THICKNESS_DASHED    = 5
BOUNDARY_DASH_LENGTH         = 30
BOUNDARY_GAP_LENGTH          = 20

COLOUR_GUIDE_OK      = (255, 220,  60)
COLOUR_GUIDE_SLOW    = ( 40, 165, 255)
COLOUR_GUIDE_URGENT  = ( 70,  46, 255)

GUIDANCE_COLOURS = {
    "GUIDE_NONE":    COLOUR_NONE,
    "GUIDE_LEFT":    COLOUR_GUIDE_OK,
    "GUIDE_RIGHT":   COLOUR_GUIDE_OK,
    "GUIDE_BOTH":    COLOUR_GUIDE_OK,
    "GUIDE_SLOW":    COLOUR_GUIDE_SLOW,
    "GUIDE_URGENT":  COLOUR_GUIDE_URGENT,
}

MAP_BG_TOP          = (32, 38, 48)
MAP_BG_BOT          = (16, 20, 28)
MAP_ROAD_FILL       = (44, 50, 60)
MAP_LANE_LINE       = (220, 220, 230)
MAP_LANE_SOLID      = (245, 245, 250)
MAP_EGO_LANE_FILL   = ( 70, 110,  60)
MAP_CENTRELINE      = (110, 130, 150)

MAP_CONE_FILL       = (180, 200, 130)
MAP_CONE_EDGE       = (220, 240, 165)

MAP_EGO_BODY        = (245, 246, 250)
MAP_EGO_ACCENT      = (200, 210, 220)
MAP_EGO_WINDOW      = (180, 175, 145)
MAP_EGO_OUTLINE     = (170, 178, 185)

MAP_OBJ_FRONT       = ( 90,  90, 215)
MAP_OBJ_SIDE        = (110, 115, 130)

BOX_EGO             = COLOUR_DEPART
BOX_LEFT            = BOUNDARY_COLOUR_LEFT_NORMAL
BOX_RIGHT           = BOUNDARY_COLOUR_RIGHT_NORMAL

STATUS_OK           = COLOUR_CENTERED
STATUS_DETECTED     = (180, 220, 200)
STATUS_CLOSE        = ( 40, 165, 255)
STATUS_URGENT       = COLOUR_DEPART
STATUS_OCCUPIED     = COLOUR_DEPART

LINE_TYPE_DASHED    = BOUNDARY_COLOUR_LEFT_NORMAL
LINE_TYPE_SOLID     = (120, 150, 220)

COUNT_ZERO          = ( 95, 110, 130)
COUNT_LOW           = ( 60, 200, 255)
COUNT_HIGH          = COLOUR_DEPART

LANE_FILL_ALPHA     = 0.35
HUD_BG_ALPHA        = 0.72

COLOUR_GUIDE        = COLOUR_GUIDE_OK
HUD_ACCENT          = DEPARTURE_COLOURS

def lane_fill_colour(departure_state: str) -> BGR:

    return DEPARTURE_COLOURS.get(departure_state, COLOUR_CENTERED)

def boundary_colour_left(departure_state: str) -> BGR:

    if departure_state in ("DEPART_LEFT", "LANE_CHANGE_LEFT", "WARN_LEFT"):
        return BOUNDARY_COLOUR_DANGER
    return BOUNDARY_COLOUR_LEFT_NORMAL

def boundary_colour_right(departure_state: str) -> BGR:

    if departure_state in ("DEPART_RIGHT", "LANE_CHANGE_RIGHT", "WARN_RIGHT"):
        return BOUNDARY_COLOUR_DANGER
    return BOUNDARY_COLOUR_RIGHT_NORMAL

def shade(colour: BGR, factor: float) -> BGR:

    return tuple(int(np.clip(c * factor, 0, 255)) for c in colour)

def with_alpha_over(colour: BGR, alpha: float, base: BGR = (0, 0, 0)) -> BGR:

    return tuple(int(np.clip(c * alpha + b * (1 - alpha), 0, 255))
                 for c, b in zip(colour, base))
