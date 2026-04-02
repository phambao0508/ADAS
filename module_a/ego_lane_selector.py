"""
Module A  —  Step A1: Ego-Lane Selector  (SEGMENTATION MODEL VERSION)
======================================================================
BACKGROUND
----------
Our YOLO model is a SEGMENTATION model with 5 classes:
    {0: 'Car', 1: 'bus', 2: 'truck', 3: 'white line', 4: 'yellow line'}

For each detected line it outputs:
    - A bounding box  [x1, y1, x2, y2, conf, cls]
    - A pixel-level segmentation MASK  (H × W binary image)

This step uses the BOUNDING BOX CENTRE to decide which line is the
left boundary and which is the right boundary of the ego lane.
It then returns the corresponding MASK so that A2 can scan it precisely.

HOW "CLOSEST" IS MEASURED
--------------------------
The car is assumed to be at x = frame_width / 2 (dashcam centre).

    cx < frame_centre  →  line is LEFT  of car
    cx ≥ frame_centre  →  line is RIGHT of car

    Closest left  = smallest (frame_centre − cx)  →  left boundary
    Closest right = smallest (cx − frame_centre)  →  right boundary

HORIZON FILTER — Why we ignore the top 40% of the frame
---------------------------------------------------------
In a dashcam perspective image, the vertical position (y) of an object
corresponds directly to its real-world DISTANCE from the car:

    y = 0%   (top of frame)
    │  ░░░ sky / far road ░░░░  road >60 m away  ← lines here are
    │  ░░░░░░░░░░░░░░░░░░  road 40–60 m away       tiny, unreliable,
    │  ░░░░░░░░░░░░░░░░░░  road 20–40 m away       NOT our lane bounds
    ├─ horizon ≈ y=40% of H ──────────────────
    │  ███ near road ███████  road 10–20 m away  ← our candidates ✔
    │  ██████████████████  road 0–10 m away   ← our immediate lane ✔
    y = 100%  (bottom of frame = car bonnet)

The road horizon sits at roughly y = 40–45% of frame height for a
dashcam mounted at 1–1.5 m on the windscreen.

INPUTS
------
  detections : List of [x1, y1, x2, y2, conf, class_id]
               All YOLO detections (vehicles + lines). A1 filters internally.
  masks      : List of np.ndarray (H, W) uint8
               masks[i] is the segmentation mask for detections[i].
               Both lists must have the same length and same ordering.
  frame_w    : frame width in pixels
  frame_h    : frame height in pixels

OUTPUTS
-------
  EgoLaneLines namedtuple:
    left_det    : [x1,y1,x2,y2,conf,cls] of the left boundary line (or None)
    right_det   : [x1,y1,x2,y2,conf,cls] of the right boundary line (or None)
    left_mask   : np.ndarray (H, W) segmentation mask of the left line (or None)
    right_mask  : np.ndarray (H, W) segmentation mask of the right line (or None)
    left_label  : 'yellow' | 'white' | None
    right_label : 'yellow' | 'white' | None
    found       : True if at least one boundary was detected

  LABEL MEANINGS:
    'yellow' → TWO-WAY ROAD CENTRE LINE (oncoming traffic on the other side)
               Always classified as "solid" — never cross.
    'white'  → SAME-DIRECTION lane divider
               May be solid or dashed — A3 decides via brightness analysis.
"""

from typing import List, Optional, NamedTuple

import numpy as np


# ── Class IDs from the YOLO model ─────────────────────────────────────────
CLASS_CAR         = 0
CLASS_BUS         = 1
CLASS_TRUCK       = 2
CLASS_WHITE_LINE  = 3
CLASS_YELLOW_LINE = 4
LINE_CLASSES      = {CLASS_WHITE_LINE, CLASS_YELLOW_LINE}

# Horizon filter: ignore line detections whose centre-y is above this fraction.
# Tuned for dashcam video where sky occupies ~50% of the frame — the actual
# road surface starts at y ≈ 50-55% of H. Setting 0.30 is permissive enough
# to keep all lane detections while still rejecting spurious far-field marks.
HORIZON_Y_FRAC = 0.30   # was 0.40 — lowered because video horizon is higher
# ──────────────────────────────────────────────────────────────────────────


class EgoLaneLines(NamedTuple):
    """Result of ego-lane selection. Contains both box and mask for each side."""
    left_det:    Optional[List[float]]    # [x1,y1,x2,y2,conf,cls] or None
    right_det:   Optional[List[float]]    # [x1,y1,x2,y2,conf,cls] or None
    left_mask:   Optional[np.ndarray]     # segmentation mask (H,W) or None
    right_mask:  Optional[np.ndarray]     # segmentation mask (H,W) or None
    left_label:  Optional[str]            # 'white' or 'yellow' or None
    right_label: Optional[str]            # 'white' or 'yellow' or None
    found:       bool                     # True if at least one boundary found


def select_ego_lane(
    detections: List[List[float]],
    masks:      List[np.ndarray],
    frame_w:    int,
    frame_h:    int,
) -> EgoLaneLines:
    """
    Identify the left and right ego-lane boundary lines from YOLO outputs.

    Parameters
    ----------
    detections : List of [x1, y1, x2, y2, confidence, class_id]
        All YOLO detections for this frame (vehicles + lines).
    masks : List of np.ndarray (H, W)
        Segmentation masks, one per detection, same order as detections.
    frame_w : int  — video frame width in pixels
    frame_h : int  — video frame height in pixels

    Returns
    -------
    EgoLaneLines
    """
    cx_frame      = frame_w / 2.0
    upper_limit_y = frame_h * HORIZON_Y_FRAC

    best_left_det    = None
    best_right_det   = None
    best_left_mask   = None
    best_right_mask  = None
    best_left_dist   = float('inf')
    best_right_dist  = float('inf')
    best_left_inner  = None   # x2 of selected left  line (for span check only)
    best_right_inner = None   # x1 of selected right line (for span check only)

    for idx, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls_id = det[:6]
        cls_id = int(cls_id)

        # ── Filter 1: lane line classes only ──────────────────────────────
        if cls_id not in LINE_CLASSES:
            continue

        # ── Filter 2: horizon gate ─────────────────────────────────────────
        cy = (y1 + y2) / 2.0
        if cy < upper_limit_y:
            continue

        # ── Side assignment: use box CENTRE (cx) ──────────────────────────
        # cx correctly reflects which side of the car the line is on.
        # Inner-edge assignment caused dividers near frame_center to be
        # mis-classified as opposite-side candidates.
        cx   = (x1 + x2) / 2.0
        mask = masks[idx] if idx < len(masks) else None

        if cx < cx_frame:
            dist = cx_frame - cx
            if dist < best_left_dist:
                best_left_dist   = dist
                best_left_det    = det
                best_left_mask   = mask
                best_left_inner  = x2   # inner wall (faces right / ego lane)
        else:
            dist = cx - cx_frame
            if dist < best_right_dist:
                best_right_dist  = dist
                best_right_det   = det
                best_right_mask  = mask
                best_right_inner = x1   # inner wall (faces left / ego lane)

    # ── Post-selection: reject 2-lane span ────────────────────────────────
    # If the inner walls of the two selected lines span > 45% of frame width
    # the selector has grabbed lines from separate lanes (road outer edges).
    # Drop the farther boundary so only one confirmed side is used;
    # lane_overlay.py's fill-width guard then prevents a bad fill.
    MAX_EGO_LANE_FRAC = 0.50    # must match MAX_LANE_FILL_FRAC in lane_overlay.py
    if best_left_inner is not None and best_right_inner is not None:
        span = best_right_inner - best_left_inner
        if span <= 0 or span > MAX_EGO_LANE_FRAC * frame_w:
            if best_left_dist >= best_right_dist:
                best_left_det   = None
                best_left_mask  = None
                best_left_inner = None
            else:
                best_right_det   = None
                best_right_mask  = None
                best_right_inner = None

    # ── Determine colour label from class id ──────────────────────────────
    def _label(det):
        if det is None:
            return None
        return 'yellow' if int(det[5]) == CLASS_YELLOW_LINE else 'white'

    found = (best_left_det is not None) or (best_right_det is not None)

    return EgoLaneLines(
        left_det    = best_left_det,
        right_det   = best_right_det,
        left_mask   = best_left_mask,
        right_mask  = best_right_mask,
        left_label  = _label(best_left_det),
        right_label = _label(best_right_det),
        found       = found,
    )
