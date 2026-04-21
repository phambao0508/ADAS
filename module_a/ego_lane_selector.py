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
import cv2


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
    left_det:       Optional[List[float]]    # [x1,y1,x2,y2,conf,cls] or None
    right_det:      Optional[List[float]]    # [x1,y1,x2,y2,conf,cls] or None
    left_mask:      Optional[np.ndarray]     # segmentation mask (H,W) or None
    right_mask:     Optional[np.ndarray]     # segmentation mask (H,W) or None
    left_label:     Optional[str]            # 'white' or 'yellow' or None
    right_label:    Optional[str]            # 'white' or 'yellow' or None
    found:          bool                     # True if at least one boundary found
    left_det_count:  int = 1                 # how many detections merged on left
    right_det_count: int = 1                 # how many detections merged on right


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

    # ── Collect ALL lane-line detections per side ─────────────────────────
    # For dashed lines each dash is a separate YOLO detection.  We must
    # merge every dash on the same side into ONE combined mask so the
    # boundary extractor and poly fitter see the full extent of the line.
    left_dets:   list = []   # list of (det, mask, dist)
    right_dets:  list = []

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
        cx   = (x1 + x2) / 2.0
        mask = masks[idx] if idx < len(masks) else None

        if cx < cx_frame:
            dist = cx_frame - cx
            left_dets.append((det, mask, dist))
        else:
            dist = cx - cx_frame
            right_dets.append((det, mask, dist))

    # ── Merge masks: only dashes belonging to the SAME lane line ────────────
    # On multi-lane roads, multiple lines appear on each side. We must only
    # merge dashes that belong to the ego-lane boundary (the closest line).
    # Strategy: pick the closest detection, get its cx, then merge only
    # detections whose cx is within CX_TOLERANCE of that reference.
    CX_TOLERANCE = frame_w * 0.08   # ~8% of frame width

    def _merge(det_list, frame_h, frame_w):
        """Merge only dashes belonging to the closest lane line.
        Returns (best_det, combined_mask, merged_count)."""
        if not det_list:
            return None, None, 0
        # Sort by distance so det_list[0] is the closest to frame center
        det_list.sort(key=lambda t: t[2])
        best_det = det_list[0][0]
        ref_cx   = (best_det[0] + best_det[2]) / 2.0   # cx of ego boundary

        # OR only masks whose cx is close to the reference (same lane line)
        combined = np.zeros((frame_h, frame_w), dtype=np.uint8)
        merged_count = 0
        for _det, _mask, _dist in det_list:
            det_cx = (_det[0] + _det[2]) / 2.0
            if abs(det_cx - ref_cx) <= CX_TOLERANCE:
                merged_count += 1
                if _mask is not None:
                    combined = cv2.bitwise_or(combined, _mask)

        return best_det, combined, merged_count

    best_left_det,  best_left_mask,  left_count  = _merge(left_dets,  frame_h, frame_w)
    best_right_det, best_right_mask, right_count = _merge(right_dets, frame_h, frame_w)

    # Inner wall x-coordinates for span check
    best_left_inner  = best_left_det[2]  if best_left_det  is not None else None  # x2
    best_right_inner = best_right_det[0] if best_right_det is not None else None  # x1

    # ── Post-selection: reject 2-lane span ────────────────────────────────
    MAX_EGO_LANE_FRAC = 0.50
    if best_left_inner is not None and best_right_inner is not None:
        span = best_right_inner - best_left_inner
        if span <= 0 or span > MAX_EGO_LANE_FRAC * frame_w:
            # Determine which side to drop based on closest distance
            left_min_dist  = min(t[2] for t in left_dets)  if left_dets  else float('inf')
            right_min_dist = min(t[2] for t in right_dets) if right_dets else float('inf')
            if left_min_dist >= right_min_dist:
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
        left_det        = best_left_det,
        right_det       = best_right_det,
        left_mask       = best_left_mask,
        right_mask      = best_right_mask,
        left_label      = _label(best_left_det),
        right_label     = _label(best_right_det),
        found           = found,
        left_det_count  = left_count  if best_left_det  is not None else 0,
        right_det_count = right_count if best_right_det is not None else 0,
    )
