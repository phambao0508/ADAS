"""
Module D - HUD Pipeline (Orchestrator)
======================================
Single entry point for Module D.
"""

import numpy as np

from .lane_fill import draw_lane_fill
from .status_hud import draw_status_hud
from .guidance_banner import draw_guidance_banner
from .mini_map import draw_mini_map
from .object_boxes import draw_object_boxes
from .telemetry_panel import draw_telemetry_panel
from .frame_decorations import draw_frame_decorations


class HUDPipeline:
    """Composite all HUD layers onto a video frame."""

    # Keep the last stable lane for several seconds when the detector drops
    # out. This avoids visible 3-5 s flicker during short occlusions, glare,
    # shadows, or road-paint gaps while still allowing the pipeline to reset
    # after sustained loss.
    MAX_RENDER_HOLD_FRAMES = 180
    POLY_SMOOTH_ALPHA = 0.28
    MAX_CENTER_JUMP_PX = 100.0
    MAX_WIDTH_JUMP_PX = 180.0
    MAX_REJECT_STREAK = 12

    def __init__(self):
        self._stable_left_poly = None
        self._stable_right_poly = None
        self._stable_left_pts = []
        self._stable_right_pts = []
        self._stable_measure = None
        self._hold_frames = 0
        self._reject_streak = 0

    def _measure_pair(self, left_poly, right_poly, frame_w: int, frame_h: int):
        if left_poly is None or right_poly is None:
            return None

        ref_y = frame_h * 0.82
        left_x = float(np.polyval(left_poly, ref_y))
        right_x = float(np.polyval(right_poly, ref_y))
        width = right_x - left_x
        center = (left_x + right_x) * 0.5

        min_width = max(70.0, frame_w * 0.06)
        max_width = frame_w * 0.72
        if width < min_width or width > max_width:
            return None
        if right_x <= 0 or left_x >= frame_w:
            return None
        return center, width

    def _stable_render_lane(self, lane_result, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        left_poly = lane_result.left_poly
        right_poly = lane_result.right_poly
        measure = self._measure_pair(left_poly, right_poly, frame_w, frame_h)

        # Only treat as a fresh detection when Module A actually found both
        # boundaries this frame. Without this gate, held polys (forwarded by
        # lane_pipeline when ego.found is False) pass `measure` and overwrite
        # the stable pts with []  — which makes draw_lane_fill fall back to
        # its default y-range and drop the whole polygon for that frame.
        accepts_new_lane = measure is not None and getattr(lane_result, "valid", False)
        if accepts_new_lane and self._stable_measure is not None:
            prev_center, prev_width = self._stable_measure
            center, width = measure
            if (
                abs(center - prev_center) > self.MAX_CENTER_JUMP_PX
                or abs(width - prev_width) > self.MAX_WIDTH_JUMP_PX
            ) and self._reject_streak < self.MAX_REJECT_STREAK:
                accepts_new_lane = False

        if accepts_new_lane:
            left_pts = lane_result.real_left_pts or lane_result.left_pts
            right_pts = lane_result.real_right_pts or lane_result.right_pts
            if self._stable_left_poly is None or self._stable_right_poly is None:
                self._stable_left_poly = left_poly.copy()
                self._stable_right_poly = right_poly.copy()
            else:
                alpha = self.POLY_SMOOTH_ALPHA
                self._stable_left_poly = (
                    alpha * left_poly + (1.0 - alpha) * self._stable_left_poly
                )
                self._stable_right_poly = (
                    alpha * right_poly + (1.0 - alpha) * self._stable_right_poly
                )

            self._stable_left_pts = list(left_pts)
            self._stable_right_pts = list(right_pts)
            self._stable_measure = self._measure_pair(
                self._stable_left_poly, self._stable_right_poly, frame_w, frame_h
            )
            self._hold_frames = 0
            self._reject_streak = 0
            return (
                self._stable_left_poly,
                self._stable_right_poly,
                self._stable_left_pts,
                self._stable_right_pts,
                True,
            )

        if measure is not None and getattr(lane_result, "valid", False):
            self._reject_streak += 1

        if (
            self._stable_left_poly is not None
            and self._stable_right_poly is not None
            and self._hold_frames < self.MAX_RENDER_HOLD_FRAMES
        ):
            self._hold_frames += 1
            return (
                self._stable_left_poly,
                self._stable_right_poly,
                self._stable_left_pts,
                self._stable_right_pts,
                True,
            )

        return None, None, [], [], False

    def render(
        self,
        frame: np.ndarray,
        lane_result,
        dept_result,
        guid_result,
    ) -> np.ndarray:
        """Composite all HUD layers and return the annotated frame."""
        out = frame.copy()

        left_poly, right_poly, left_fill_pts, right_fill_pts, lane_detected = (
            self._stable_render_lane(lane_result, out.shape)
        )

        # D1: Ego-lane fill polygon. This is a true lane-area fill between the
        # two fitted boundaries, with no grow/reveal/breathing animation.
        out = draw_lane_fill(
            out,
            left_poly,
            right_poly,
            left_fill_pts,
            right_fill_pts,
            dept_result.state,
            fill_progress=1.0 if lane_detected else 0.0,
        )

        # D2: Vehicle object boxes. Ego-lane objects are red; other-lane
        # objects are blue.
        out = draw_object_boxes(
            out,
            getattr(guid_result, "vehicle_detections", []),
            left_poly,
            right_poly,
        )

        # D3: Departure status HUD (top-right)
        out = draw_status_hud(
            out,
            dept_result.state,
            dept_result.smoothed_offset,
        )

        # D4: Guidance banner (center-top, only when active)
        out = draw_guidance_banner(
            out,
            guid_result.guidance,
            guid_result.message,
        )

        # D5: Mini-map (bottom-left)
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

        # D6: Telemetry panel (bottom-right)
        out = draw_telemetry_panel(
            out,
            guid_result.front_proximity,
            guid_result.left_clear,
            guid_result.right_clear,
            lane_result.left_type,
            lane_result.right_type,
            left_count=guid_result.left_count,
            right_count=guid_result.right_count,
            total_seen=getattr(guid_result, "total_seen", 0),
            front_distance_m=getattr(guid_result, "front_distance_m", None),
            front_ttc_s=getattr(guid_result, "front_ttc_s", None),
        )

        # D7: Frame corner decorations (top layer)
        out = draw_frame_decorations(out)

        return out
