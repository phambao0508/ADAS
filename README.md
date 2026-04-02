# 🚗 ADAS — Vision-Based Lane Departure & Directional Guidance System

A real-time, vision-only Advanced Driver Assistance System (ADAS) that processes dashcam video frame-by-frame using a YOLO segmentation model. The system simultaneously detects lane lines, classifies their type, tracks the ego lane, warns about lane departures, and advises on safe overtaking manoeuvres — all rendered as a live heads-up display (HUD) overlay on the output video.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Model & Detections](#model--detections)
4. [Module A — Lane Detection & Ego-Lane Tracking](#module-a--lane-detection--ego-lane-tracking)
5. [Module B — Lane Departure Warning](#module-b--lane-departure-warning)
6. [Module C — Directional Guidance (Overtaking Assist)](#module-c--directional-guidance-overtaking-assist)
7. [Module D — HUD Rendering & Display](#module-d--hud-rendering--display)
8. [Project Structure](#project-structure)
9. [Requirements](#requirements)
10. [Usage](#usage)
11. [Tuning Reference](#tuning-reference)

---

## Overview

```
INPUT: raw dashcam frame
    │
    ├─ [A] Lane Detection     → "Which lane am I in? Where are its edges?"
    │
    ├─ [B] Departure Warning  → "Am I drifting or crossing an unsafe line?"
    │
    ├─ [C] Guidance           → "Is there a car ahead? Should I change lanes?"
    │
    └─ [D] HUD Display        → "Show all of the above clearly on screen"

OUTPUT: annotated frame saved to MP4
```

All four modules run on **every frame** in a single pipeline. The system is purely vision-based — no GPS, no HD maps, no radar.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ADAS Pipeline                            │
│                                                                 │
│  dashcam frame                                                  │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────┐   boxes + masks    ┌──────────────────────────┐   │
│  │  YOLO   │ ──────────────────►│  Module A: Lane Pipeline  │   │
│  │ (seg)   │                    │  A1 ego-lane selector     │   │
│  └─────────┘                    │  A2 boundary extractor    │   │
│       │                         │  A3 solid/dashed classify │   │
│       │ vehicle boxes           │  A4 polynomial fitter     │   │
│       │                         │  A5 BEV transformer       │   │
│       ▼                         └────────────┬─────────────┘   │
│  ┌──────────────────────────┐                │ LaneResult       │
│  │  Module B: Departure     │◄───────────────┤                  │
│  │  B1 lateral offset       │                │                  │
│  │  B2 EMA smoother         │                ▼                  │
│  │  B3 6-state classifier   │   ┌──────────────────────────┐   │
│  │  B4 hold logic           │   │  Module C: Guidance       │   │
│  └──────────┬───────────────┘   │  C1 zone definer          │   │
│             │                   │  C2 proximity detector    │   │
│             │ DepartureState    │  C3 occupancy checker     │   │
│             │                   │  C4 decision logic        │   │
│             │                   │  C5 hold logic            │   │
│             │                   └────────────┬─────────────┘   │
│             │                                │ GuidanceState    │
│             ▼                                ▼                  │
│       ┌──────────────────────────────────────────────────┐     │
│       │           Module D: HUD Renderer                 │     │
│       │   lane overlay · boundaries · guidance banner    │     │
│       │   status HUD · mini-map · telemetry panel        │     │
│       └──────────────────────────────────────────────────┘     │
│                             │                                   │
│                             ▼                                   │
│                    annotated output frame                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model & Detections

The system uses a **single YOLO segmentation model** (`best (1).pt`) that produces both bounding boxes and pixel-level masks. All five classes are detected in a single inference pass:

| Class ID | Label | Used By |
|----------|-------|---------|
| 0 | Car | Module C |
| 1 | bus | Module C |
| 2 | truck | Module C |
| 3 | white line | Module A |
| 4 | yellow line | Module A |

> ⚠️ **Critical distinction:** A **yellow line** marks the centre of a two-way road — oncoming traffic is on the other side. The system **never** recommends crossing toward a yellow line, regardless of its visual appearance. A **white line** separates lanes going the same direction and may be either solid (no-cross) or dashed (safe to cross).

---

## Module A — Lane Detection & Ego-Lane Tracking

**Files:** `module_a/`

| Step | File | Description |
|------|------|-------------|
| A1 | `ego_lane_selector.py` | Identifies the left and right boundaries of the ego lane from all YOLO detections using the **inner-edge** method |
| A2 | `boundary_extractor.py` | Extracts dense `(y, x)` point lists from segmentation masks along the inner wall of each lane line |
| A3 | `line_type_classifier.py` | Classifies each boundary as **solid** or **dashed** via brightness continuity analysis (yellow lines are always solid by rule) |
| A4 | `poly_fitter.py` | Fits a quadratic polynomial `x = a·y² + b·y + c` to raw boundary points for smooth, noise-robust curves |
| A5 | `bev_transformer.py` | Warps each frame to a top-down Bird's Eye View (BEV) for visualisation and debugging |
| — | `lane_pipeline.py` | **Orchestrator** — runs A1→A5 per frame, returns a `LaneResult` dataclass |

### Ego-Lane Selection (A1)

The car is assumed at `x = frame_width / 2`. The system selects:
- **Left boundary** → the closest line whose **right edge (x2)** is left of frame centre
- **Right boundary** → the closest line whose **left edge (x1)** is right of frame centre

A 2-lane-span guard rejects boundary pairs where `inner_width > 0.60 × frame_width`.

### Solid vs. Dashed Classification (A3)

```
Layer 1 — Yellow line shortcut:
  yellow line  →  always "solid"  (oncoming traffic on other side)

Layer 2 — Brightness continuity (white lines only):
  Sample brightness strip every row in bottom 60% of frame
  Compute: continuity = bright_rows / total_rows
           transitions = number of bright↔dark changes

  IF continuity < 0.60 AND transitions ≥ 4  →  "dashed"
  ELSE                                        →  "solid"
```

---

## Module B — Lane Departure Warning

**Files:** `module_b/`

| Step | File | Description |
|------|------|-------------|
| B1 | `offset_calculator.py` | Measures lateral offset of car from lane centre using polynomials evaluated at `y = 0.85 × H` |
| B2 | `ema_smoother.py` | Exponential Moving Average smoother with spike rejection (`MAX_JUMP_PX = 120`) |
| B3 | `departure_classifier.py` | Maps smoothed offset + boundary types to one of 6 states |
| B4 | `hold_logic.py` | Holds active warning states for `HOLD_FRAMES = 6` frames to prevent HUD flicker |
| — | `bias_estimator.py` | Estimates systematic camera mounting offset |
| — | `departure_pipeline.py` | **Orchestrator** — runs B1→B4 per frame |

### 6 Departure States

| State | Colour | Trigger |
|-------|--------|---------|
| `CENTERED` | 🟢 | `|offset| < 50 px` |
| `WARN_LEFT` | 🟡 | `50 ≤ |offset| < 100`, drifting left |
| `WARN_RIGHT` | 🟡 | `50 ≤ |offset| < 100`, drifting right |
| `LANE_CHANGE_LEFT` | 🔵 | `|offset| ≥ 100`, left boundary is **dashed** |
| `LANE_CHANGE_RIGHT` | 🔵 | `|offset| ≥ 100`, right boundary is **dashed** |
| `DEPART_LEFT` | 🔴 | `|offset| ≥ 100`, left boundary is **solid** |
| `DEPART_RIGHT` | 🔴 | `|offset| ≥ 100`, right boundary is **solid** |

---

## Module C — Directional Guidance (Overtaking Assist)

**Files:** `module_c/`

| Step | File | Description |
|------|------|-------------|
| C1 | `zone_definer.py` | Partitions the frame into Left / Ego / Right zones using evaluated boundary polynomials |
| C2 | `proximity_detector.py` | Three-gate front vehicle check (zone → direction → bounding box area proxy) |
| C3 | `occupancy_checker.py` | Determines if each adjacent lane is clear for a potential merge |
| C4 | `guidance_decision.py` | Decision tree combining proximity + occupancy + boundary type into a guidance state |
| C5 | `guidance_hold.py` | Hold logic identical to Module B to prevent guidance banner flicker |
| — | `guidance_states.py` | Enum definitions for all guidance states |
| — | `guidance_pipeline.py` | **Orchestrator** — runs C1→C5 per frame |

### Proximity Estimation

Proximity is estimated from the relative area of the detected vehicle bounding box:

| Relative Area | State | Approximate Distance |
|---------------|-------|----------------------|
| > 6% of frame | `PROX_VERY_CLOSE` | ≈ 10–20 m |
| > 2% of frame | `PROX_CLOSE` | ≈ 20–40 m |
| ≤ 2% of frame | `PROX_NONE` | > 40 m (ignored) |

### Guidance Decision (Priority-Ordered)

```
Priority 1 — VERY_CLOSE front vehicle  →  GUIDE_URGENT   "!! BRAKE !!"
Priority 2 — No front vehicle          →  GUIDE_NONE     (silent)
Priority 3 — CLOSE front vehicle:
  left clear  + left dashed            →  GUIDE_LEFT
  right clear + right dashed           →  GUIDE_RIGHT
  both clear  + both dashed            →  GUIDE_BOTH     (prefer left)
  any clear   but solid boundary       →  GUIDE_SLOW
  neither clear                        →  GUIDE_SLOW
```

> **Safety rule:** A lane change is **only suggested** when the boundary on that side is **dashed**. An empty adjacent lane with a solid boundary still produces `GUIDE_SLOW`.

---

## Module D — HUD Rendering & Display

**Files:** `module_d/`

| File | Description |
|------|-------------|
| `lane_overlay.py` | Draws filled lane area polygon and coloured boundary lines on the frame |
| `boundary_renderer.py` | Renders the fitted polynomial curves for left and right boundaries |
| `guidance_banner.py` | Displays the guidance message banner (colour-coded by urgency) |
| `status_hud.py` | Top-left status panel showing departure state and line types |
| `mini_map.py` | Simplified top-down mini-map of lane and vehicles |
| `telemetry_panel.py` | Lateral offset value and other numeric telemetry |
| `frame_decorations.py` | Watermark, frame counter, and other decorations |
| `hud_colours.py` | Centralised colour palette for all HUD elements |
| `hud_pipeline.py` | **Orchestrator** — composites all overlays onto the output frame |

---

## Project Structure

```
ADAS/
├── best (1).pt             ← YOLO segmentation model weights
├── implementation_plan.md  ← Detailed per-module logic specification
├── README.md
│
├── module_a/               ← Lane Detection & Ego-Lane Tracking
│   ├── __init__.py
│   ├── lane_pipeline.py        (orchestrator)
│   ├── ego_lane_selector.py    (A1)
│   ├── boundary_extractor.py   (A2)
│   ├── line_type_classifier.py (A3)
│   ├── poly_fitter.py          (A4)
│   └── bev_transformer.py      (A5)
│
├── module_b/               ← Lane Departure Warning
│   ├── __init__.py
│   ├── departure_pipeline.py   (orchestrator)
│   ├── offset_calculator.py    (B1)
│   ├── ema_smoother.py         (B2)
│   ├── departure_classifier.py (B3)
│   ├── hold_logic.py           (B4)
│   └── bias_estimator.py
│
├── module_c/               ← Directional Guidance (Overtaking Assist)
│   ├── __init__.py
│   ├── guidance_pipeline.py    (orchestrator)
│   ├── zone_definer.py         (C1)
│   ├── proximity_detector.py   (C2)
│   ├── occupancy_checker.py    (C3)
│   ├── guidance_decision.py    (C4)
│   ├── guidance_hold.py        (C5)
│   └── guidance_states.py
│
└── module_d/               ← HUD Rendering & Display
    ├── __init__.py
    ├── hud_pipeline.py         (orchestrator)
    ├── lane_overlay.py
    ├── boundary_renderer.py
    ├── guidance_banner.py
    ├── status_hud.py
    ├── mini_map.py
    ├── telemetry_panel.py
    ├── frame_decorations.py
    └── hud_colours.py
```

---

## Requirements

```
python >= 3.9
opencv-python
numpy
ultralytics   # for YOLO inference
```

Install with:

```bash
pip install opencv-python numpy ultralytics
```

---

## Usage

```python
import cv2
from ultralytics import YOLO

from module_a import LanePipeline
from module_b import DeparturePipeline
from module_c import GuidancePipeline
from module_d import HudPipeline

# Load model
model = YOLO("best (1).pt")

# Open video
cap = cv2.VideoCapture("dashcam_video.mp4")
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialise pipelines (one instance per video)
lane_pipe     = LanePipeline(frame_width=W, frame_height=H)
depart_pipe   = DeparturePipeline(frame_width=W, frame_height=H)
guidance_pipe = GuidancePipeline(frame_width=W, frame_height=H)
hud_pipe      = HudPipeline(frame_width=W, frame_height=H)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Single YOLO inference — all classes in one pass
    yolo_result = model(frame)[0]

    # Module A — lane detection
    lane_result = lane_pipe.process(frame, yolo_result)

    # Module B — departure warning
    depart_state = depart_pipe.process(lane_result)

    # Module C — guidance
    guidance_state = guidance_pipe.process(lane_result, yolo_result)

    # Module D — HUD rendering
    output_frame = hud_pipe.render(frame, lane_result, depart_state, guidance_state)

    cv2.imshow("ADAS", output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Tuning Reference

### Module A — Line Type Classifier

| Parameter | Value | Effect |
|-----------|-------|--------|
| `SAMPLE_LOWER_FRAC` | 0.60 | Sample bottom 60% of frame for more data points |
| `ADAPTIVE_THRESHOLD_RATIO` | 0.40 | Lower threshold → better gap detection at sunset |
| `MIN_BRIGHTNESS_RANGE` | 5.0 | Minimum contrast to attempt classification |
| `CONTINUITY_THRESHOLD` | 0.60 | Max bright-row fraction to still call "dashed" |
| `TRANSITION_THRESHOLD` | 4 | Min bright↔dark changes to call "dashed" |

### Module B — EMA Smoother & Classifier

| Parameter | Value | Effect |
|-----------|-------|--------|
| `EMA_ALPHA` | 0.25 | Weight for each new offset measurement |
| `SPIKE_ALPHA` | 0.05 | Weight when frame-to-frame jump > MAX_JUMP_PX |
| `MAX_JUMP_PX` | 120 | Jump threshold for spike rejection |
| `CENTERED_THRESHOLD` | 50 px | Max offset for CENTERED state |
| `WARN_THRESHOLD` | 100 px | Offset at which WARN escalates to DEPART |
| `HOLD_FRAMES` | 6 | Frames to hold an active warning |

### Module C — Proximity & Zone

| Parameter | Value | Effect |
|-----------|-------|--------|
| `VERY_CLOSE_AREA` | 0.06 | Box area fraction → PROX_VERY_CLOSE |
| `CLOSE_AREA` | 0.02 | Box area fraction → PROX_CLOSE |
| `DIRECTION_GATE_Y` | 0.75 × H | Max cy for front vehicle (above = too far back) |
| `ZONE_REFERENCE_Y` | 0.80 × H | Row at which polynomials define zone dividers |

---

## Key Design Decisions

- **Inner-edge boundary extraction:** Using the inner wall of each detected line box (not the box centre) prevents a 2-lane span bug where a wide lane divider is misclassified as an ego-lane boundary.
- **Yellow = always solid:** The system hard-codes yellow lines as solid, not from visual analysis but from road meaning — oncoming traffic is on the other side. No lane change is ever recommended toward a yellow line.
- **Segmentation over detection:** The YOLO model returns segmentation masks (not just boxes), enabling precise pixel-level boundary reconstruction — a significant accuracy improvement over box-centre sampling.
- **Polynomial fallback:** When YOLO misses a boundary for a frame, the last good polynomial is reused and synthetic `(y, x)` points are generated from it so the HUD fill remains visually continuous.
- **EMA spike rejection:** Outlier frames (caused by partial occlusion, sun glare, or model miss) are given a 5% weight instead of 25%, preventing a single bad frame from crossing a departure threshold.
