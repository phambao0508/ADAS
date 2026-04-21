"""
Module D — HUD Rendering & Display
=====================================
Composes all visual overlay layers onto the video frame and returns
the final annotated image.

Rendering order:
    D1 (lane_overlay)       → lane fill polygon from boundary polynomials
    D2 (boundary_renderer)  → solid/dashed boundary lines
    D3 (status_hud)         → top-right departure status panel
    D4 (guidance_banner)    → center-top guidance banner (when active)
    D5 (mini_map)           → bottom-left schematic top-down view
    D6 (telemetry_panel)    → bottom-right live telemetry data panel
    D7 (frame_decorations)  → corner bracket chrome decorations

Public API
----------
    from module_d import HUDPipeline

    hud = HUDPipeline()
    output_frame = hud.render(frame, lane_result, dept_result, guid_result)

Notes
-----
- Module D is STATELESS — no persistent state between frames.
- The input frame is never modified; a working copy is made internally.
- All visual decisions are driven by the result dataclasses from
  Modules A, B, and C. Module D has zero model/YOLO dependency.
- The ego-lane fill (D1) is generated from boundary polynomials,
  NOT from a YOLO mask (the model has no lane-area mask output).
"""

from .hud_pipeline       import HUDPipeline
from .hud_colours        import (
    lane_fill_colour,
    boundary_colour_left,
    boundary_colour_right,
    DEPARTURE_COLOURS,
    GUIDANCE_COLOURS,
    LANE_FILL_ALPHA,
    BOUNDARY_THICKNESS_SOLID,
    BOUNDARY_THICKNESS_DASHED,
    BOUNDARY_DASH_LENGTH,
    BOUNDARY_GAP_LENGTH,
)
from .lane_overlay       import draw_lane_lines
from .boundary_renderer  import draw_boundaries
from .status_hud         import draw_status_hud, STATE_LABELS
from .guidance_banner    import draw_guidance_banner
from .mini_map           import draw_mini_map
from .telemetry_panel    import draw_telemetry_panel
from .frame_decorations  import draw_frame_decorations

__all__ = [
    # Main entry point
    "HUDPipeline",

    # Individual layer functions (for testing or custom pipelines)
    "draw_lane_lines",
    "draw_boundaries",
    "draw_status_hud",
    "draw_guidance_banner",
    "draw_mini_map",
    "draw_telemetry_panel",
    "draw_frame_decorations",

    # Colour helpers
    "lane_fill_colour",
    "boundary_colour_left",
    "boundary_colour_right",

    # Colour maps
    "DEPARTURE_COLOURS",
    "GUIDANCE_COLOURS",

    # HUD text
    "STATE_LABELS",

    # Tuning constants
    "LANE_FILL_ALPHA",
    "BOUNDARY_THICKNESS_SOLID",
    "BOUNDARY_THICKNESS_DASHED",
    "BOUNDARY_DASH_LENGTH",
    "BOUNDARY_GAP_LENGTH",
]
