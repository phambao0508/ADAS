from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

# Pinhole distance: Z = fy * h_cam/(y2 - y horizon). If far, y2 near horizon -> Big Z, if near, y2 far horizon -> Small Z
# Width- based distance: Z = fx * W_real/(x2-x1). Far, bbox narrow, near, bbox wide

# Vehicle near the car -> using pinhole
# Vehicle far from the car -> using width

#TTC = distance at the moment / speed Ex: Frame 100: 20m, frame 107: 17m, FPS 30 => 7/30 = 0.233s => TTC = 3/0.233
from .camera_calibration import (
    CameraIntrinsics,
    REAL_VEHICLE_WIDTHS_M,
    DEFAULT_VEHICLE_WIDTH_M,
)

TTC_HISTORY_LEN          = 8
MIN_SAMPLES_FOR_TTC      = 3
MIN_CLOSING_RATE_MPS     = 0.5
MAX_REASONABLE_TTC_S     = 30.0
TTC_SMOOTHING_ALPHA      = 0.35
DISTANCE_SMOOTHING_ALPHA = 0.50

PINHOLE_HORIZON_GUARD_PX = 1.0
BOTTOM_TRUNCATION_GUARD  = 4.0

def pinhole_distance(
    y_bot: float,
    intrinsics: CameraIntrinsics,
) -> Optional[float]:

    dy = y_bot - intrinsics.horizon_y
    if dy <= PINHOLE_HORIZON_GUARD_PX:
        return None
    return intrinsics.fy * intrinsics.h_camera / dy + intrinsics.distance_offset_m

def width_distance(
    box_w_px: float,
    cls_id: int,
    intrinsics: CameraIntrinsics,
) -> Optional[float]:

    if box_w_px <= 1.0:
        return None
    real_w = REAL_VEHICLE_WIDTHS_M.get(int(cls_id), DEFAULT_VEHICLE_WIDTH_M)
    return intrinsics.fx * real_w / box_w_px + intrinsics.distance_offset_m

def estimate_distance(
    x1: float, y1: float, x2: float, y2: float,
    cls_id: int,
    intrinsics: CameraIntrinsics,
) -> Optional[float]:

    box_w_px = x2 - x1
    truncated = (intrinsics.frame_h - y2) < BOTTOM_TRUNCATION_GUARD

    if truncated:
        return width_distance(box_w_px, cls_id, intrinsics)

    d_pin = pinhole_distance(y2, intrinsics)
    d_wid = width_distance(box_w_px, cls_id, intrinsics)

    if d_pin is None and d_wid is None:
        return None
    if d_pin is None:
        return d_wid
    if d_wid is None:
        return d_pin

    weight_pin = 1.0 / (1.0 + (d_pin / 25.0) ** 2)
    weight_pin = max(0.30, min(0.90, weight_pin))
    return weight_pin * d_pin + (1.0 - weight_pin) * d_wid

@dataclass
class _TrackHistory:
    distance_history: Deque[Tuple[int, float]]
    smoothed_distance: Optional[float] = None
    smoothed_ttc:      Optional[float] = None

class DistanceTTCEstimator:

    def __init__(self, intrinsics: CameraIntrinsics, fps: float = 30.0):
        self._intrinsics = intrinsics
        self._dt = 1.0 / max(1.0, float(fps))
        self._frame_idx = 0
        self._histories: Dict[int, _TrackHistory] = {}

    def update(self, tracks: Iterable) -> None:

        self._frame_idx += 1

        seen_ids: set = set()
        for t in tracks:
            seen_ids.add(t.track_id)

            d_raw = estimate_distance(
                t.x1, t.y1, t.x2, t.y2, t.cls_id, self._intrinsics,
            )
            if d_raw is None:
                t.distance_m = None
                t.ttc_s = None
                continue

            history = self._histories.get(t.track_id)
            if history is None:
                history = _TrackHistory(
                    distance_history=deque(maxlen=TTC_HISTORY_LEN),
                    smoothed_distance=d_raw,
                )
                self._histories[t.track_id] = history

            a = DISTANCE_SMOOTHING_ALPHA
            if history.smoothed_distance is None:
                history.smoothed_distance = d_raw
            else:
                history.smoothed_distance = (
                    a * d_raw + (1.0 - a) * history.smoothed_distance
                )

            history.distance_history.append(
                (self._frame_idx, history.smoothed_distance)
            )
            t.distance_m = history.smoothed_distance

            ttc = self._estimate_ttc(history)
            if ttc is not None:
                if history.smoothed_ttc is None:
                    history.smoothed_ttc = ttc
                else:
                    s = TTC_SMOOTHING_ALPHA
                    history.smoothed_ttc = s * ttc + (1.0 - s) * history.smoothed_ttc
                t.ttc_s = min(history.smoothed_ttc, MAX_REASONABLE_TTC_S)
            else:
                history.smoothed_ttc = None
                t.ttc_s = None

        stale = [tid for tid in self._histories if tid not in seen_ids]
        for tid in stale:
            self._histories.pop(tid, None)

    def _estimate_ttc(self, history: _TrackHistory) -> Optional[float]:

        h = history.distance_history
        if len(h) < MIN_SAMPLES_FOR_TTC:
            return None

        f0, d0 = h[0]
        f1, d1 = h[-1]
        n_frames = max(1, f1 - f0)

        closing = (d0 - d1) / (n_frames * self._dt)

        if closing < MIN_CLOSING_RATE_MPS:
            return None
        return d1 / closing

    def reset(self) -> None:

        self._histories.clear()
        self._frame_idx = 0
