"""
Module D  —  HUD Pipeline (Orchestrator)
=========================================
Single entry point for Module D.
"""

import numpy as np

from .lane_overlay       import draw_lane_lines
from .boundary_renderer  import draw_boundaries
from .status_hud         import draw_status_hud
from .guidance_banner    import draw_guidance_banner
from .mini_map           import draw_mini_map
from .telemetry_panel    import draw_telemetry_panel
from .frame_decorations  import draw_frame_decorations


class HUDPipeline:
    """
    Module D orchestrator — composites all HUD layers onto a video frame.
    Tracks fill_progress for the bottom-to-top dash reveal animation.
    """

    # Sweep speed: progress per frame
    # 0.015 → ~67 frames → ~2.2 sec at 30fps for a full-height line
    # Short dashed lines fill faster because their y-span is smaller
    FILL_GROW_RATE   = 0.015
    FILL_DECAY_RATE  = 0.03   # slow fade when lane lost

    def __init__(self):
        self._fill_progress = 0.0

    def render(
        self,
        frame:       np.ndarray,
        lane_result,    # module_a.LaneResult
        dept_result,    # module_b.DepartureResult
        guid_result,    # module_c.GuidanceResult
    ) -> np.ndarray:
        """Composite all HUD layers and return the annotated frame."""
        out = frame.copy()

        # ── Update fill animation (grow and stay) ─────────────────────────
        lane_detected = (lane_result.left_mask is not None or
                         lane_result.right_mask is not None)
        if lane_detected:
            # Grow toward 1.0 and clamp — no reset
            # Long lines take the full ~2.2s to reveal; short lines
            # appear faster because their mask y-span is smaller
            self._fill_progress = min(1.0, self._fill_progress + self.FILL_GROW_RATE)
        else:
            # Slowly fade out when lanes are lost
            self._fill_progress = max(0.0, self._fill_progress - self.FILL_DECAY_RATE)

        # ── D1: Lane line dashes (bottom-most layer) ──────────────────────
        out = draw_lane_lines(
            out,
            lane_result.left_mask,
            lane_result.right_mask,
            dept_result.state,
            fill_progress = self._fill_progress,
        )

        # ── D2: Boundary lines — DISABLED ─────────────────────────────────

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
            guid_result.guidance,
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
