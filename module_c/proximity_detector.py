"""
Module C  —  Step C2: Front Vehicle Proximity Detector
=======================================================
TASK
----
For all YOLO vehicle detections, find the vehicle AHEAD in the ego lane
and estimate how close it is.

INPUT FORMAT
------------
YOLO (ultralytics segmentation) outputs bounding boxes in CORNER format:
    [x1, y1, x2, y2, conf, cls_id]

This module receives pre-converted CENTRE format tuples (cx, cy, w, h)
from the pipeline's pre-processing step.

THREE-GATE CHECK  (Section C2 of the implementation plan)
---------------------------------------------------------
Each vehicle box is tested through three gates IN ORDER:

  GATE 1 — Zone Gate (is it in our lane?)
    PASS if:  zone_left_x ≤ cx ≤ zone_right_x
    FAIL → not in ego lane, skip immediately

  GATE 2 — Direction Gate (is it AHEAD of us?)
    PASS if:  cy < 0.75 × H
    FAIL → in lower 25% of frame (bonnet region / rear mirror), skip

    Why 75%?  Vehicles ahead appear in the upper part of the frame.
    The bottom 25% is either the car bonnet or shows vehicles that are
    already behind or alongside — not relevant for a front collision.

  GATE 3 — Proximity Estimation (how far away?)
    relative_area = (w × h) / (W × H)   ← box area as fraction of frame

    relative_area > 0.06  →  PROX_VERY_CLOSE  (~10–20 m, emergency)
    relative_area > 0.02  →  PROX_CLOSE       (~20–40 m, guidance range)
    relative_area ≤ 0.02  →  ignored          (FAR, no action needed)

MULTI-VEHICLE RULE
------------------
If more than one vehicle passes all three gates (multiple vehicles ahead
in our lane), the LARGEST relative_area wins. This is equivalent to
choosing the nearest vehicle and prevents a distant car from masking
an urgent nearby one.

PROX_VERY_CLOSE always overrides PROX_CLOSE — once "VERY_CLOSE" is
set, it cannot be downgraded even if a farther vehicle is processed
after a closer one.

DISTANCE PROXY TABLE (sedan-sized vehicle, 1080p, ~70° FOV)
------------------------------------------------------------
  relative_area ≈ 0.01  →  ~60 m ahead  (ignored)
  relative_area ≈ 0.02  →  ~40 m ahead  (CLOSE threshold)
  relative_area ≈ 0.04  →  ~25 m ahead  (CLOSE)
  relative_area ≈ 0.06  →  ~15 m ahead  (VERY_CLOSE threshold)
  relative_area ≈ 0.10  →  ~10 m ahead  (critical)

INPUTS
------
  vehicle_boxes  : List of (cx, cy, w, h) — vehicle detections, centre fmt
  zone_left_x    : float — left zone divider from C1
  zone_right_x   : float — right zone divider from C1
  frame_w, frame_h : int

OUTPUT
------
  str : PROX_NONE | PROX_CLOSE | PROX_VERY_CLOSE
"""

from typing import List, Tuple

from .guidance_states import PROX_NONE, PROX_CLOSE, PROX_VERY_CLOSE


# ── Tuning constants ───────────────────────────────────────────────────────
# Gate 2 — direction gate threshold
FRONT_GATE_Y_FRAC = 0.75      # vehicles below this row are NOT ahead of us

# Gate 3 — proximity thresholds (relative box area = w*h / W*H)
PROXIMITY_CLOSE       = 0.02  # box area > 2%  of frame → CLOSE  (~40 m)
PROXIMITY_VERY_CLOSE  = 0.06  # box area > 6%  of frame → VERY_CLOSE (~15 m)
# ──────────────────────────────────────────────────────────────────────────


def detect_front_proximity(
    vehicle_boxes: List[Tuple[float, float, float, float]],
    zone_left_x:   float,
    zone_right_x:  float,
    frame_w:       int,
    frame_h:       int,
) -> str:
    """
    Determine how close the nearest ego-lane vehicle ahead is.

    Parameters
    ----------
    vehicle_boxes : List of (cx, cy, w, h)
        Vehicle bounding boxes in centre format, VEHICLE CLASSES ONLY.
        (white/yellow line detections must already be filtered out upstream)
    zone_left_x : float
        Left zone divider x-position (from zone_definer.compute_zone_dividers).
    zone_right_x : float
        Right zone divider x-position.
    frame_w : int
    frame_h : int

    Returns
    -------
    str : one of PROX_NONE, PROX_CLOSE, PROX_VERY_CLOSE
    """
    frame_area    = frame_w * frame_h
    front_gate_y  = FRONT_GATE_Y_FRAC * frame_h
    proximity     = PROX_NONE

    for (cx, cy, w, h) in vehicle_boxes:

        # ── Gate 1: must be in the ego lane zone ─────────────────────────
        if not (zone_left_x <= cx <= zone_right_x):
            continue

        # ── Gate 2: must be geometrically AHEAD of the car ───────────────
        if cy >= front_gate_y:
            continue   # in the lower 25% → bonnet region / not ahead

        # ── Gate 3: proximity estimation by relative box area ─────────────
        rel_area = (w * h) / frame_area

        if rel_area > PROXIMITY_VERY_CLOSE:
            # Emergency range — immediately return (nothing overrides this)
            return PROX_VERY_CLOSE

        elif rel_area > PROXIMITY_CLOSE:
            # Guidance range — record but keep searching for a closer vehicle
            proximity = PROX_CLOSE
            # (PROX_VERY_CLOSE check above already handles the upgrade)

    return proximity
