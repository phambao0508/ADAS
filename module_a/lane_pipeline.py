"""
Module A  —  Lane Pipeline (Orchestrator)  — SEGMENTATION MODEL VERSION
========================================================================
TASK
----
Single entry point for Module A. Runs A1 → A5 per frame.

PIPELINE FLOW
-------------
    Input:
      frame      — raw BGR dashcam frame (H×W×3)
      yolo_result — ultralytics Results object from model(frame)[0]

    Pre-processing (inside process()):
      Extract boxes:  [x1, y1, x2, y2, conf, cls] per detection
      Extract masks:  (H, W) uint8 binary mask per detection
                      (resized from YOLO's internal resolution to frame size)

    A1: select_ego_lane(detections, masks, W, H)
        → EgoLaneLines(left_det, right_det, left_mask, right_mask,
                        left_label, right_label, found)

    A2: extract_boundaries(left_mask, right_mask)
        → left_pts, right_pts  (inner-edge pixels from segmentation masks)

    A3: classify_line_type(left_pts,  frame, H, left_label)
        classify_line_type(right_pts, frame, H, right_label)
        → left_type, right_type  ("solid" or "dashed")

    A4: fit_boundary_polynomial(left_pts,  prev_left_poly)
        fit_boundary_polynomial(right_pts, prev_right_poly)
        → left_poly, right_poly  (np.ndarray [a, b, c])

    A5: bev.warp(frame)
        → bev_frame  (top-down warped frame, for visualisation)

    Output: LaneResult dataclass

HOW YOLO SEGMENTATION MASKS ARE EXTRACTED
------------------------------------------
    results = model(frame)[0]

    boxes_data = results.boxes.data.cpu().numpy()      # (N, 6)
    masks_data = results.masks.data.cpu().numpy()      # (N, mask_h, mask_w)

    masks_data is at a lower internal resolution and must be resized to
    the original frame size (H, W) using cv2.resize with INTER_NEAREST
    so pixel values stay binary (0 or 1).

USAGE EXAMPLE
-------------
    from module_a import LanePipeline

    pipeline = LanePipeline(frame_width=1920, frame_height=1080)

    while cap.isOpened():
        ret, frame = cap.read()
        yolo_result = yolo_model(frame)[0]

        result = pipeline.process(frame, yolo_result)

        # result.left_poly    → [a,b,c] for left boundary curve
        # result.right_poly   → [a,b,c] for right boundary curve
        # result.left_type    → "solid" or "dashed"
        # result.right_type   → "solid" or "dashed"
        # result.left_pts     → raw (y,x) inner-edge points, left line
        # result.right_pts    → raw (y,x) inner-edge points, right line
        # result.valid        → False if no lane lines found this frame
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import warnings

import cv2
import numpy as np

from .ego_lane_selector    import select_ego_lane, EgoLaneLines
from .boundary_extractor   import extract_boundaries
from .line_type_classifier import classify_line_type
from .poly_fitter          import fit_boundary_polynomial
from .bev_transformer      import BEVTransformer


def _synth_pts_from_poly(
    poly: np.ndarray,
    frame_h: int,
    frame_w: int,
    step: int = 30,
) -> List[Tuple[int, int]]:
    """
    Generate sparse synthetic (y, x) boundary points by evaluating a
    polynomial across the frame height.

    Used when YOLO misses a boundary for one or more frames: the renderer
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

    # Raw boundary pixel lists: List of (y, x) tuples (inner edge of each line)
    left_pts:    List[Tuple[int, int]] = field(default_factory=list)
    right_pts:   List[Tuple[int, int]] = field(default_factory=list)

    # YOLO-detected-only points (no synthetic fill-in).
    # Used for y-range calculation in the fill overlay so synthetic pts
    # (which span the full frame) don't push y_top/y_bottom out of range.
    real_left_pts:  List[Tuple[int, int]] = field(default_factory=list)
    real_right_pts: List[Tuple[int, int]] = field(default_factory=list)

    # Polynomial coefficients [a, b, c] for: x = a·y² + b·y + c
    left_poly:   Optional[np.ndarray] = None
    right_poly:  Optional[np.ndarray] = None

    # Lane marking type for each boundary
    left_type:   str = "solid"    # "solid" or "dashed"
    right_type:  str = "solid"

    # Colour label from YOLO class
    left_label:  Optional[str] = None    # "yellow" or "white"
    right_label: Optional[str] = None

    # Bird's Eye View warped frame (for debugging / visualisation)
    bev_frame:   Optional[np.ndarray] = None
# ─────────────────────────────────────────────────────────────────────────────


class LanePipeline:
    """
    Module A orchestrator — runs A1 → A5 per frame.

    Create ONE instance per video (BEV matrix pre-computed for frame size).
    Call process() on every frame.

    Parameters
    ----------
    frame_width  : int
    frame_height : int
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.w = frame_width
        self.h = frame_height

        # A5: BEV transformer — computed once
        self.bev = BEVTransformer(frame_width, frame_height)

        # A4: remember last good polynomials for fallback
        self._prev_left_poly:  Optional[np.ndarray] = None
        self._prev_right_poly: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────────────────────────────
    def process(self, frame: np.ndarray, yolo_result) -> LaneResult:
        """
        Run the full Module A pipeline on one video frame.

        Parameters
        ----------
        frame : np.ndarray (H, W, 3)
            Original BGR dashcam frame.
        yolo_result : ultralytics Results
            Direct output of model(frame)[0].
            Must contain both .boxes and .masks (segmentation model output).

        Returns
        -------
        LaneResult
        """
        result = LaneResult()

        # ── Extract boxes and masks from YOLO result ──────────────────────
        detections, masks = self._extract_from_yolo(yolo_result)

        if not detections:
            result.left_poly  = self._prev_left_poly
            result.right_poly = self._prev_right_poly
            return result

        # ── A1: Find left/right ego-lane boundary detections + masks ──────
        ego: EgoLaneLines = select_ego_lane(detections, masks, self.w, self.h)

        result.left_label  = ego.left_label
        result.right_label = ego.right_label

        if not ego.found:
            result.left_poly  = self._prev_left_poly
            result.right_poly = self._prev_right_poly
            return result

        result.valid = True

        # ── A2: Scan segmentation masks for precise inner-edge points ─────
        left_pts, right_pts = extract_boundaries(ego.left_mask, ego.right_mask)
        result.left_pts  = left_pts
        result.right_pts = right_pts
        # Store real/YOLO-only points before synthetic substitution
        result.real_left_pts  = list(left_pts)
        result.real_right_pts = list(right_pts)

        # ── A3: Classify solid vs. dashed ─────────────────────────────────
        result.left_type  = classify_line_type(
            left_pts,  frame, self.h, ego.left_label
        )
        result.right_type = classify_line_type(
            right_pts, frame, self.h, ego.right_label
        )

        # ── A4: Fit smooth quadratic polynomials ──────────────────────────
        left_poly  = fit_boundary_polynomial(left_pts,  self._prev_left_poly)
        right_poly = fit_boundary_polynomial(right_pts, self._prev_right_poly)

        result.left_poly  = left_poly
        result.right_poly = right_poly

        if left_poly  is not None: self._prev_left_poly  = left_poly
        if right_poly is not None: self._prev_right_poly = right_poly

        # ── A4b: Synthesise pts for renderer when mask was empty ──────────
        # If YOLO missed a boundary this frame but we have a poly from a
        # previous frame, generate sparse synthetic pts from that poly so
        # the renderer knows the correct y-range and draws the full line.
        # This is what makes the boundary "connect" across gap frames.
        if not left_pts and left_poly is not None:
            result.left_pts = _synth_pts_from_poly(left_poly, self.h, self.w)
        if not right_pts and right_poly is not None:
            result.right_pts = _synth_pts_from_poly(right_poly, self.h, self.w)

        # ── A5: Warp frame to BEV ─────────────────────────────────────────
        result.bev_frame = self.bev.warp(frame)

        return result

    # ─────────────────────────────────────────────────────────────────────
    def _extract_from_yolo(self, yolo_result):
        """
        Extract bounding boxes and segmentation masks from a YOLO result.

        Parameters
        ----------
        yolo_result : ultralytics Results object (model(frame)[0])

        Returns
        -------
        detections : List of [x1, y1, x2, y2, conf, cls]  — one per detection
        masks      : List of np.ndarray (H, W) uint8       — one per detection
                     masks[i] corresponds to detections[i]
                     Returns empty lists if no detections or no masks.
        """
        # ── Boxes ─────────────────────────────────────────────────────────
        if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
            return [], []

        detections = yolo_result.boxes.data.cpu().numpy().tolist()

        # ── Segmentation masks ────────────────────────────────────────────
        if yolo_result.masks is None:
            # No masks returned — this should never happen with a segmentation
            # model. If you see this warning, you may have deployed a detection-
            # only model (no seg head). All lines will fall back to "solid".
            warnings.warn(
                "[Module A] yolo_result.masks is None — expected a segmentation "
                "model output. Check that the correct model weights are loaded. "
                "Falling back to empty masks (all lines classified as solid).",
                RuntimeWarning,
                stacklevel=2,
            )
            empty_mask = np.zeros((self.h, self.w), dtype=np.uint8)
            masks = [empty_mask] * len(detections)
            return detections, masks

        # masks.data shape: (N, mask_h, mask_w)  — values 0.0 or 1.0
        raw_masks = yolo_result.masks.data.cpu().numpy()

        masks = []
        for i in range(raw_masks.shape[0]):
            # Resize from YOLO's internal resolution to original frame size
            m = cv2.resize(
                raw_masks[i],
                (self.w, self.h),
                interpolation=cv2.INTER_NEAREST,
            )
            # Convert float 0/1 → uint8 0/255
            masks.append((m * 255).astype(np.uint8))

        return detections, masks
