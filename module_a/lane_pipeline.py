"""
Module A  —  Lane Pipeline (Orchestrator)  — UFLDv2 VERSION
=============================================================
TASK
----
Single entry point for Module A. Runs A1 → A4 per frame using
UFLDv2 lane coordinates instead of YOLO segmentation masks.

PIPELINE FLOW
-------------
    Input:
      frame       — raw BGR dashcam frame (H×W×3)
      ufld_lanes  — List[List[Tuple[int, int]]] from UFLDv2Wrapper.detect_lanes()
                    Each lane is a list of (x, y) pixel coordinates.

    A1: select_ego_lane(ufld_lanes, W, H)
        → EgoLaneLines(left_pts, right_pts, found)

    A2: extract_boundaries(left_pts, right_pts)
        → left_pts, right_pts  (validated and sorted)

    A3: (removed — lane type always defaults to "solid")

    A4: fit_boundary_polynomial(left_pts, prev_left_poly)
        fit_boundary_polynomial(right_pts, prev_right_poly)
        → left_poly, right_poly  (np.ndarray [a, b, c])

    Output: LaneResult dataclass

USAGE EXAMPLE
-------------
    from ufld_wrapper import UFLDv2Wrapper
    from module_a import LanePipeline

    ufld = UFLDv2Wrapper(model_path="culane_res18.pth")
    pipeline = LanePipeline(frame_width=1920, frame_height=1080)

    while cap.isOpened():
        ret, frame = cap.read()
        ufld_lanes = ufld.detect_lanes(frame)

        result = pipeline.process(frame, ufld_lanes)

        # result.left_poly    → [a,b,c] for left boundary curve
        # result.right_poly   → [a,b,c] for right boundary curve
        # result.left_type    → always "solid" (UFLDv2 default)
        # result.right_type   → always "solid"
        # result.left_pts     → (y,x) inner-edge points, left line
        # result.right_pts    → (y,x) inner-edge points, right line
        # result.valid        → False if no lane lines found this frame
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .ego_lane_selector    import select_ego_lane, EgoLaneLines
from .boundary_extractor   import extract_boundaries

from .poly_fitter          import fit_boundary_polynomial


def _synth_pts_from_poly(
    poly: np.ndarray,
    frame_h: int,
    frame_w: int,
    step: int = 30,
) -> List[Tuple[int, int]]:
    """
    Generate sparse synthetic (y, x) boundary points by evaluating a
    polynomial across the frame height.

    Used when UFLDv2 misses a boundary for one or more frames: the renderer
    gets a full y-range from these synthetic points, so the boundary line
    and lane fill both extend correctly without gaps.

    Parameters
    ----------
    poly : np.ndarray [a, b, c]
    frame_h : int — frame height in pixels
    frame_w : int — frame width  in pixels
    step : int    — evaluate every `step` rows (default 30 → ~36 pts for 1080p)

    Returns
    -------
    List of (y, x) tuples spanning 0 → frame_h-1
    """
    pts = []
    start_y = int(frame_h * 0.35)   # skip sky rows
    for y in range(start_y, frame_h, step):
        x = int(np.clip(np.polyval(poly, y), 0, frame_w - 1))
        pts.append((y, x))
    return pts


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class LaneResult:
    """
    All Module A outputs for a single frame.
    Passed directly to Module B (departure warning) and Module C (guidance).
    """
    valid:       bool = False

    # Raw boundary pixel lists: List of (y, x) tuples
    left_pts:    List[Tuple[int, int]] = field(default_factory=list)
    right_pts:   List[Tuple[int, int]] = field(default_factory=list)

    # YOLO-detected-only points (no synthetic fill-in).
    # With UFLDv2, these are the raw UFLDv2 points before synthetic fill.
    real_left_pts:  List[Tuple[int, int]] = field(default_factory=list)
    real_right_pts: List[Tuple[int, int]] = field(default_factory=list)

    # Raw segmentation masks — always None with UFLDv2 (no masks available)
    left_mask:   Optional[np.ndarray] = None
    right_mask:  Optional[np.ndarray] = None

    # Polynomial coefficients [a, b, c] for: x = a·y² + b·y + c
    left_poly:   Optional[np.ndarray] = None
    right_poly:  Optional[np.ndarray] = None

    # Lane marking type for each boundary (always "solid" with UFLDv2)
    left_type:   str = "solid"
    right_type:  str = "solid"

    # Colour label from YOLO class — always None with UFLDv2
    left_label:  Optional[str] = None
    right_label: Optional[str] = None
# ─────────────────────────────────────────────────────────────────────────────


class LanePipeline:
    """
    Module A orchestrator — runs A1 → A4 per frame using UFLDv2 input.

    Create ONE instance per video.
    Call process() on every frame.

    Parameters
    ----------
    frame_width  : int
    frame_height : int
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.w = frame_width
        self.h = frame_height

        # A4: remember last good polynomials for fallback
        self._prev_left_poly:  Optional[np.ndarray] = None
        self._prev_right_poly: Optional[np.ndarray] = None
        self._prev_measure: Optional[Tuple[float, float]] = None
        self._reject_streak = 0
        self._max_reject_streak = 12

    def _has_curve_signal(self, pts: List[Tuple[int, int]]) -> bool:
        """Return True when points show a real bend rather than a near-line."""
        if len(pts) < 6:
            return False

        ys = np.array([p[0] for p in pts], dtype=np.float64)
        xs = np.array([p[1] for p in pts], dtype=np.float64)
        if float(np.ptp(ys)) < self.h * 0.12:
            return False

        try:
            poly = np.polyfit(ys, xs, 2)
        except (np.linalg.LinAlgError, ValueError):
            return False

        y_top = float(np.min(ys))
        y_bottom = float(np.max(ys))
        y_mid = (y_top + y_bottom) * 0.5

        x_top = float(np.polyval(poly, y_top))
        x_mid = float(np.polyval(poly, y_mid))
        x_bottom = float(np.polyval(poly, y_bottom))
        x_mid_linear = (x_top + x_bottom) * 0.5

        bow_px = abs(x_mid - x_mid_linear)
        curvature_px = abs(float(poly[0])) * (y_bottom - y_top) ** 2
        threshold = max(18.0, self.w * 0.012)
        return bow_px > threshold or curvature_px > threshold

    def _measure_pair(
        self,
        left_poly: Optional[np.ndarray],
        right_poly: Optional[np.ndarray],
    ) -> Optional[Tuple[float, float]]:
        """Return lane centre/width for plausibility checks."""
        if left_poly is None or right_poly is None:
            return None

        ref_y = self.h * 0.78
        left_x = float(np.polyval(left_poly, ref_y))
        right_x = float(np.polyval(right_poly, ref_y))
        width = right_x - left_x
        center = (left_x + right_x) * 0.5

        min_width = max(120.0, self.w * 0.08)
        max_width = self.w * 0.62
        if width < min_width or width > max_width:
            return None
        if center < self.w * 0.22 or center > self.w * 0.78:
            return None
        return center, width

    def _accept_measure(
        self,
        measure: Optional[Tuple[float, float]],
        curve_responsive: bool = False,
    ) -> bool:
        """Reject sudden ego-lane jumps caused by a bad mask track."""
        if measure is None:
            return False
        if self._prev_measure is None:
            return True
        if self._reject_streak >= self._max_reject_streak:
            return True

        center, width = measure
        prev_center, prev_width = self._prev_measure
        center_limit = 260.0 if curve_responsive else 140.0
        width_limit = 320.0 if curve_responsive else 220.0
        if abs(center - prev_center) > center_limit:
            return False
        if abs(width - prev_width) > width_limit:
            return False
        return True

    # ─────────────────────────────────────────────────────────────────────
    def process(
        self,
        frame: np.ndarray,
        ufld_lanes: List[List[Tuple[int, int]]],
    ) -> LaneResult:
        """
        Run the full Module A pipeline on one video frame.

        Parameters
        ----------
        frame : np.ndarray (H, W, 3)
            Original BGR dashcam frame.
        ufld_lanes : List[List[Tuple[int, int]]]
            Lane coordinates from UFLDv2Wrapper.detect_lanes().
            Each lane is a list of (x, y) pixel coordinates.

        Returns
        -------
        LaneResult
        """
        result = LaneResult()

        if not ufld_lanes:
            result.left_poly  = self._prev_left_poly
            result.right_poly = self._prev_right_poly
            return result

        # ── A1: Find left/right ego-lane boundaries ───────────────────────
        ego: EgoLaneLines = select_ego_lane(ufld_lanes, self.w, self.h)

        result.left_label  = ego.left_label
        result.right_label = ego.right_label

        if not ego.found:
            result.left_poly  = self._prev_left_poly
            result.right_poly = self._prev_right_poly
            return result

        result.valid = True

        # ── A2: Validate and sort boundary points ─────────────────────────
        left_pts, right_pts = extract_boundaries(ego.left_pts, ego.right_pts)
        result.left_pts  = left_pts
        result.right_pts = right_pts
        # Store real points before synthetic substitution
        result.real_left_pts  = list(left_pts)
        result.real_right_pts = list(right_pts)

        # ── A3: Lane type — always "solid" (U-Net/UFLDv2 cannot classify) ─
        # LaneResult defaults to "solid" already, no action needed.

        # ── A4: Fit smooth quadratic polynomials ──────────────────────────
        curve_responsive = (
            self._has_curve_signal(left_pts) or
            self._has_curve_signal(right_pts)
        )
        left_poly = fit_boundary_polynomial(
            left_pts,
            self._prev_left_poly,
            curve_responsive=curve_responsive,
        )
        right_poly = fit_boundary_polynomial(
            right_pts,
            self._prev_right_poly,
            curve_responsive=curve_responsive,
        )

        result.left_poly  = left_poly
        result.right_poly = right_poly

        measure = self._measure_pair(left_poly, right_poly)
        if (
            left_poly is not None and
            right_poly is not None and
            not self._accept_measure(measure, curve_responsive=curve_responsive)
        ):
            self._reject_streak += 1
            result.valid = False
            result.left_poly = self._prev_left_poly
            result.right_poly = self._prev_right_poly
            return result

        if left_poly is not None:
            self._prev_left_poly = left_poly
        if right_poly is not None:
            self._prev_right_poly = right_poly
        if measure is not None:
            self._prev_measure = measure
            self._reject_streak = 0

        # ── A4b: Synthesise pts when UFLDv2 missed a boundary ─────────────
        if not left_pts and left_poly is not None:
            result.left_pts = _synth_pts_from_poly(left_poly, self.h, self.w)
        if not right_pts and right_poly is not None:
            result.right_pts = _synth_pts_from_poly(right_poly, self.h, self.w)

        return result
