"""
Module A  —  Step A3: Line Type Classifier  (LINE-DETECTION VERSION)
=====================================================================
TASK
----
Decide whether each lane boundary is SOLID or DASHED.
This is used by:
  - Module B: to distinguish  DEPART (solid) from LANE_CHANGE (dashed)
  - Module C: to decide whether to recommend a lane change

TWO-LAYER APPROACH
------------------
With our model's 5 classes we know the LINE COLOUR (white or yellow).
The colour tells us something critical about road geometry:

  LAYER 1 — Yellow line shortcut (fast, no pixel analysis):
    yellow_line (class 4) → TWO-WAY ROAD CENTRE LINE.
                             Oncoming traffic is on the other side.
                             Crossing a yellow line = head-on collision risk.
                             → Always return "solid" to block any lane-change
                               recommendation toward the yellow side.
                             We do NOT run brightness analysis — the result
                             must always be "solid" regardless of appearance.

  LAYER 2 — Brightness continuity analysis (for white lines only):
    white_line (class 3) → Separates lanes going the SAME direction.
                           Could be EITHER solid (no-cross edge line)
                           OR dashed (lane divider, safe to cross).
                           → Run the original brightness continuity method.

LAYER 2 ALGORITHM (unchanged from original design)
---------------------------------------------------
  Step 1: Sample brightness along the boundary in the bottom 50% of frame.
  Step 2: Compute adaptive threshold = min + (max − min) × 0.45
  Step 3: Binarise: is_bright[y] = 1 if brightness > threshold, else 0
  Step 4: Compute:
    continuity  = fraction of bright rows (solid ≈ 0.95, dashed ≈ 0.55)
    transitions = number of 0↔1 flips (solid ≈ 2, dashed ≈ 8+)
  Step 5: IF continuity < 0.70 AND transitions ≥ 6 → "dashed"
          ELSE                                      → "solid"

INPUTS
------
  boundary_pts : list of (y, x) boundary pixel coordinates
  frame        : the original BGR colour frame (H, W, 3)
  frame_h      : total frame height
  line_label   : 'yellow' or 'white' (from ego_lane_selector)

OUTPUT
------
  "solid"  or  "dashed"
"""

import numpy as np
from typing import List, Optional, Tuple


# ── Tuning constants ───────────────────────────────────────────────────────
SAMPLE_STRIP_HALF_WIDTH  = 8      # half-width of brightness sampling strip (px)
SAMPLE_LOWER_FRAC        = 0.60   # sample bottom 60% of frame (was 50%)
                                   # More samples → more reliable gap detection
ADAPTIVE_THRESHOLD_RATIO = 0.40   # threshold placed lower in the brightness range
                                   # (was 0.45) — helps in low-contrast/sunset scenes
CONTINUITY_THRESHOLD     = 0.60   # below → not solid (was 0.70)
                                   # Lowered because road reflection at sunset
                                   # raises gap-row brightness, shrinking the
                                   # bright/dark contrast. A dashed line here
                                   # may score 0.65 continuity falsely as solid.
TRANSITION_THRESHOLD     = 4      # at or above → dashed (was 6)
                                   # Dashes at mid-distance (perspective compression)
                                   # produces fewer visible gaps → fewer 0↔1 flips.
MIN_BRIGHTNESS_RANGE     = 5.0    # minimum b_max - b_min to attempt analysis
                                   # (was 1.0 — too tight; now a bit more lenient)
# ──────────────────────────────────────────────────────────────────────────


def classify_line_type(
    boundary_pts: List[Tuple[int, int]],
    frame: np.ndarray,
    frame_h: int,
    line_label: Optional[str] = None,   # 'yellow', 'white', or None
) -> str:
    """
    Classify a lane boundary as 'solid' or 'dashed'.

    Parameters
    ----------
    boundary_pts : List[Tuple[int, int]]
        (y, x) coordinates along one boundary edge (from boundary_extractor).
    frame : np.ndarray
        Original BGR video frame, shape (H, W, 3).
    frame_h : int
        Frame height in pixels.
    line_label : str or None
        Colour label from the YOLO class: 'yellow', 'white', or None.
        If 'yellow' → returns "solid" immediately (no pixel analysis needed).
        If 'white' or None → runs brightness continuity analysis.

    Returns
    -------
    str : "solid" or "dashed"
    """
    # ── LAYER 1: Yellow line = two-way road centre line ──────────────────
    if line_label == 'yellow':
        return "solid"
        # Yellow line = the boundary between YOUR direction and ONCOMING traffic.
        # Crossing it risks a head-on collision.
        # We always return "solid" to ensure the system NEVER recommends
        # a lane change across a yellow line, regardless of its visual appearance.

    # ── LAYER 2: Brightness continuity analysis for white lines ──────────
    lower_start_y = int(frame_h * (1.0 - SAMPLE_LOWER_FRAC))
    h_frame, w_frame = frame.shape[:2]

    brightness_values: List[float] = []

    for (y, x) in boundary_pts:
        if y < lower_start_y:
            continue   # skip the far (upper) half of the frame

        x_left  = max(0,       x - SAMPLE_STRIP_HALF_WIDTH)
        x_right = min(w_frame, x + SAMPLE_STRIP_HALF_WIDTH)

        # Guard: skip if the strip has zero width (x is at the frame edge)
        # Without this, np.mean of an empty array returns nan, which silently
        # corrupts the adaptive threshold and produces wrong classifications.
        if x_left >= x_right:
            continue

        strip       = frame[y, x_left:x_right]   # shape: (strip_width, 3) BGR
        mean_bright = float(np.mean(strip))       # mean across all pixels & channels
        brightness_values.append(mean_bright)

    # Fallback: not enough samples → assume solid (safe default)
    if len(brightness_values) < 5:
        return "solid"

    # Adaptive threshold
    b_min = float(np.min(brightness_values))
    b_max = float(np.max(brightness_values))
    if b_max - b_min < MIN_BRIGHTNESS_RANGE:
        return "solid"   # uniform brightness — can't distinguish — default solid

    threshold = b_min + (b_max - b_min) * ADAPTIVE_THRESHOLD_RATIO

    # Binarise
    is_bright = [1 if b > threshold else 0 for b in brightness_values]

    # Measure continuity and transitions
    total       = len(is_bright)
    bright_rows = sum(is_bright)
    continuity  = bright_rows / total

    transitions = sum(
        1 for i in range(1, total) if is_bright[i] != is_bright[i - 1]
    )

    # Classify
    if continuity < CONTINUITY_THRESHOLD and transitions >= TRANSITION_THRESHOLD:
        return "dashed"
    else:
        return "solid"
