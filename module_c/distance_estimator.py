"""
Module C — Monocular Distance & TTC Estimator
==============================================
Per-frame distance and per-track time-to-collision (TTC) for confirmed
vehicle tracks.

Pipeline
--------
    confirmed tracks  ─►  estimate_distance(track)              # metres
                       ─►  TTCHistory.update(track_id, distance) # seconds
                       ─►  writes back into Track.distance_m / Track.ttc_s

Two distance methods, blended:

    1. Pinhole ground-plane (primary)
        Z = fy * h_camera / (y_bot - y_horizon)
        Robust on flat roads with a level mount. Fails when:
          - the bbox foot is at or above the horizon row (y_bot ≤ horizon_y)
          - the bbox is truncated at the bottom edge of the frame
        Both produce ``None`` in those cases.

    2. Class-width fallback (secondary)
        Z = fx * W_real / w_box
        Works regardless of bbox truncation, but depends on knowing the
        real-world width per class. We use ~1.85 m for cars, ~2.55 m for
        bus / truck. Real distances on motorbikes / unusual vehicles will
        be off by the width-ratio.

Blend rule: if both are available, prefer pinhole near (≤ ~30 m), width
fallback far. A single weighted blend keeps the result continuous as a
vehicle approaches.

TTC
---
TTC is computed from a short rolling history of (frame_idx, distance) per
track. We fit a linear closing rate ``v = -dZ/dt`` over the last ~8
samples; when ``v > 0.5 m/s`` (vehicle approaching) we report
``Z / v``, otherwise ``None`` (the vehicle is stationary or pulling away).

Light EMA smoothing on TTC prevents jumpy values when distance is noisy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from .camera_calibration import (
    CameraIntrinsics,
    REAL_VEHICLE_WIDTHS_M,
    DEFAULT_VEHICLE_WIDTH_M,
)


# ── Tunables ─────────────────────────────────────────────────────────────────
TTC_HISTORY_LEN          = 8     # frames of distance history per track
MIN_SAMPLES_FOR_TTC      = 3     # need at least N points before fitting a rate
MIN_CLOSING_RATE_MPS     = 0.5   # below this, we don't report a TTC
MAX_REASONABLE_TTC_S     = 30.0  # clamp display; >30 s ≈ "no risk"
TTC_SMOOTHING_ALPHA      = 0.35  # EMA on TTC output
DISTANCE_SMOOTHING_ALPHA = 0.50  # EMA on raw distance per track

PINHOLE_HORIZON_GUARD_PX = 1.0   # min pixels below horizon for pinhole
BOTTOM_TRUNCATION_GUARD  = 4.0   # if y2 within N px of frame edge, prefer width


# ─────────────────────────────────────────────────────────────────────────────
# Pure-function distance estimators
# ─────────────────────────────────────────────────────────────────────────────
def pinhole_distance(
    y_bot: float,
    intrinsics: CameraIntrinsics,
) -> Optional[float]:
    """
    Ground-plane pinhole distance from the bbox foot row to the horizon.

    Returns ``None`` if the foot is at or above the horizon (would imply
    infinite or negative distance).
    """
    dy = y_bot - intrinsics.horizon_y
    if dy <= PINHOLE_HORIZON_GUARD_PX:
        return None
    return intrinsics.fy * intrinsics.h_camera / dy + intrinsics.distance_offset_m


def width_distance(
    box_w_px: float,
    cls_id: int,
    intrinsics: CameraIntrinsics,
) -> Optional[float]:
    """
    Distance from class-specific real-world width and the bbox pixel width.

    Returns ``None`` if box width is degenerate.
    """
    if box_w_px <= 1.0:
        return None
    real_w = REAL_VEHICLE_WIDTHS_M.get(int(cls_id), DEFAULT_VEHICLE_WIDTH_M)
    return intrinsics.fx * real_w / box_w_px + intrinsics.distance_offset_m


def estimate_distance(
    x1: float, y1: float, x2: float, y2: float,
    cls_id: int,
    intrinsics: CameraIntrinsics,
) -> Optional[float]:
    """
    Best-available distance for a single bbox.

    Strategy:
      * if the bbox is truncated at the bottom of the frame, prefer width
        (pinhole would map to a foot that doesn't exist)
      * otherwise blend pinhole (good near) with width (good far) by a
        smooth weight that follows the pinhole reading.
    """
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

    # Smooth weighted blend: trust pinhole more for short distances, width
    # more for far ones. Crossover ≈ 25 m.
    weight_pin = 1.0 / (1.0 + (d_pin / 25.0) ** 2)
    weight_pin = max(0.30, min(0.90, weight_pin))
    return weight_pin * d_pin + (1.0 - weight_pin) * d_wid


# ─────────────────────────────────────────────────────────────────────────────
# Per-track distance + TTC history
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _TrackHistory:
    distance_history: Deque[Tuple[int, float]]   # (frame_idx, distance_m)
    smoothed_distance: Optional[float] = None
    smoothed_ttc:      Optional[float] = None


class DistanceTTCEstimator:
    """
    Holds per-track distance/TTC history and writes results back into
    each ``Track``'s ``distance_m`` and ``ttc_s`` fields each frame.

    Parameters
    ----------
    intrinsics : CameraIntrinsics
    fps : float
        Video frame rate; used to convert frame steps to seconds.
    """

    def __init__(self, intrinsics: CameraIntrinsics, fps: float = 30.0):
        self._intrinsics = intrinsics
        self._dt = 1.0 / max(1.0, float(fps))
        self._frame_idx = 0
        self._histories: Dict[int, _TrackHistory] = {}

    # ─────────────────────────────────────────────────────────────────────────
    def update(self, tracks: Iterable) -> None:
        """
        Compute distance and TTC for every confirmed track and write the
        results into each track's ``.distance_m`` and ``.ttc_s`` fields.

        Tracks that disappear from the input are garbage-collected from
        history so the estimator's memory follows the tracker's lifecycle.
        """
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

            # Smooth raw distance to suppress per-frame jitter (~5%).
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

            # ── TTC from closing rate over the rolling window ───────────────
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

        # Drop history for tracks that vanished this frame.
        stale = [tid for tid in self._histories if tid not in seen_ids]
        for tid in stale:
            self._histories.pop(tid, None)

    # ─────────────────────────────────────────────────────────────────────────
    def _estimate_ttc(self, history: _TrackHistory) -> Optional[float]:
        """
        Linear closing-rate fit over the recent distance history.
        Returns TTC in seconds, or None if not approaching.
        """
        h = history.distance_history
        if len(h) < MIN_SAMPLES_FOR_TTC:
            return None

        # Endpoint slope (oldest → newest) is more responsive than full
        # least-squares for this short window and avoids the matrix work.
        f0, d0 = h[0]
        f1, d1 = h[-1]
        n_frames = max(1, f1 - f0)
        # closing rate (m/s) — positive = vehicle approaching
        closing = (d0 - d1) / (n_frames * self._dt)

        if closing < MIN_CLOSING_RATE_MPS:
            return None
        return d1 / closing

    # ─────────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Clear all per-track history (e.g. on video switch)."""
        self._histories.clear()
        self._frame_idx = 0
