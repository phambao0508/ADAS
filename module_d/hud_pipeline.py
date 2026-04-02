"""
Module D  —  HUD Pipeline (Orchestrator)
=========================================
Single entry point for Module D. Composites all rendering layers onto the
frame in the correct order and returns the final annotated frame.

RENDERING ORDER (D1 → D2 → D3 → D4 → D5 → D6 → D7)
------------------------------------------------------
    D1  Lane fill overlay        — bottom-most layer
    D2  Boundary lines           — drawn ON TOP of the fill
    D3  Status HUD bar           — top-right text panel
    D4  Guidance banner          — center-top (only when guidance active)
    D5  Mini-map                 — bottom-left schematic
    D6  Telemetry panel          — bottom-right info panel  [NEW]
    D7  Frame corner decorations — chrome corner brackets   [NEW]

INPUTS
------
    frame        : np.ndarray (H, W, 3) — original BGR video frame
    lane_result  : LaneResult      (from module_a)
    dept_result  : DepartureResult (from module_b)
    guid_result  : GuidanceResult  (from module_c)

OUTPUT
------
    np.ndarray (H, W, 3) — fully annotated frame ready to write to MP4
"""

import numpy as np

from .lane_overlay       import draw_lane_fill
from .boundary_renderer  import draw_boundaries
from .status_hud         import draw_status_hud
from .guidance_banner    import draw_guidance_banner
from .mini_map           import draw_mini_map
from .telemetry_panel    import draw_telemetry_panel
from .frame_decorations  import draw_frame_decorations


class HUDPipeline:
    """
    Module D orchestrator — composites all HUD layers onto a video frame.

    Stateless: no persistent state between frames.
    Create ONE instance and call render() on every frame.
    """

    def render(
        self,
        frame:       np.ndarray,
        lane_result,    # module_a.LaneResult
        dept_result,    # module_b.DepartureResult
        guid_result,    # module_c.GuidanceResult
    ) -> np.ndarray:
        """
        Composite all HUD layers and return the annotated frame.

        Parameters
        ----------
        frame : np.ndarray (H, W, 3), BGR
            Original video frame (not modified — a working copy is made).
        lane_result : LaneResult from module_a
        dept_result : DepartureResult from module_b
        guid_result : GuidanceResult from module_c

        Returns
        -------
        np.ndarray (H, W, 3) — fully rendered frame.
        """
        # Work on a copy so the original frame remains unmodified
        out = frame.copy()

        # ── D1: Lane fill (bottom-most layer) ────────────────────────────
        out = draw_lane_fill(
            out,
            lane_result.left_poly,
            lane_result.right_poly,
            lane_result.left_pts,
            lane_result.right_pts,
            dept_result.state,
            real_left_pts  = lane_result.real_left_pts,
            real_right_pts = lane_result.real_right_pts,
        )

        # ── D2: Boundary lines ────────────────────────────────────────────
        out = draw_boundaries(
            out,
            lane_result.left_poly,
            lane_result.right_poly,
            lane_result.left_pts,
            lane_result.right_pts,
            lane_result.left_type,
            lane_result.right_type,
            dept_result.state,
        )

        # ── D3: Departure status HUD (top-right) ──────────────────────────
        out = draw_status_hud(
            out,
            dept_result.state,
            dept_result.smoothed_offset,
        )

        # ── D4: Guidance banner (center-top, only when active) ───────────
        out = draw_guidance_banner(
            out,
            guid_result.guidance,
            guid_result.message,
        )

        # ── D5: Mini-map (bottom-left) ────────────────────────────────────
        out = draw_mini_map(
            out,
            dept_result.state,
            dept_result.smoothed_offset,
            guid_result.front_proximity,
            guid_result.left_clear,
            guid_result.right_clear,
            lane_result.left_type,
            lane_result.right_type,
            guid_result.guidance,          # NEW: for guidance arrows
        )

        # ── D6: Telemetry panel (bottom-right) ────────────────────────────
        out = draw_telemetry_panel(
            out,
            guid_result.front_proximity,
            guid_result.left_clear,
            guid_result.right_clear,
            lane_result.left_type,
            lane_result.right_type,
        )

        # ── D7: Frame corner decorations (top layer) ──────────────────────
        out = draw_frame_decorations(out)

        return out
