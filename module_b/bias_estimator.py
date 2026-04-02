"""
Module B  —  Step B1b: Auto-Calibrating Mount Bias Estimator
=============================================================
PROBLEM
-------
The dashcam is rarely mounted perfectly centred on the car's bonnet.
Even a 1–2 cm offset shifts the apparent lane centre by 50–150 px,
causing a persistent negative raw_offset → constant WARN_RIGHT (or
WARN_LEFT for a left-mounted camera) even when the car is centred.

This is clearly visible in the output log as hundreds of consecutive
WARN_RIGHT frames while the driver reports the car IS centred.

SOLUTION — Rolling Median Auto-Calibration
------------------------------------------
1. Collect valid (non-None) raw_offset values in a rolling window.
2. Once at least WARMUP_SAMPLES have been collected, compute the
   rolling MEDIAN as the estimated mount bias.
3. Subtract this bias from every subsequent raw_offset BEFORE it
   enters the EMA smoother.

WHY MEDIAN, NOT MEAN?
---------------------
The mean would be skewed by frames where the car genuinely drifts
or by occasional bad polynomial frames. The MEDIAN is robust to
those outliers — as long as the car drives mostly straight (which
is true for highway/road dashcam footage), the median of many
offset samples equals the mount bias.

WHY ROLLING WINDOW?
-------------------
The camera may be repositioned between recording sessions, or the
road surface may have a consistent cross-slope (bank). A rolling
window (last N frames) allows the bias estimate to slowly adapt
rather than locking to a stale from the start of the video.

PARAMETERS
----------
  WARMUP_SAMPLES  = 90   — frames before bias correction activates
                           (3 s at 30 fps). During warmup: no correction.
  WINDOW_SIZE     = 300  — rolling window length (10 s at 30 fps).
                           Bias is recomputed each frame from this window.
  MAX_BIAS_PX     = 200  — safety clamp: never subtract more than this.
                           Prevents correction from going haywire on
                           roads with extreme cross-slopes.

INPUTS / OUTPUTS
----------------
  update(raw_offset) → corrected_offset (float | None)
    raw_offset : float or None
    returns    : raw_offset − current_bias, or None if raw_offset is None
"""

from collections import deque
from typing import Optional

import numpy as np


# ── Tuning constants ───────────────────────────────────────────────────────
WARMUP_SAMPLES = 90     # frames before correction activates  (~3 s @ 30 fps)
WINDOW_SIZE    = 300    # rolling window size                 (~10 s @ 30 fps)
MAX_BIAS_PX    = 200    # maximum bias correction (px) — safety clamp
# ──────────────────────────────────────────────────────────────────────────


class MountBiasEstimator:
    """
    Auto-calibrating camera mount bias corrector.

    Maintains a rolling window of valid raw offsets, computes their
    median as the estimated mount bias, and subtracts it from each
    new raw offset before it enters the EMA smoother.

    Usage
    -----
        estimator = MountBiasEstimator()

        for frame in video:
            raw_offset  = compute_lateral_offset(...)
            corrected   = estimator.update(raw_offset)
            smoothed    = ema_smoother.update(corrected)
    """

    def __init__(
        self,
        warmup_samples: int = WARMUP_SAMPLES,
        window_size:    int = WINDOW_SIZE,
        max_bias_px:    float = MAX_BIAS_PX,
    ):
        self._warmup   = warmup_samples
        self._window   = deque(maxlen=window_size)
        self._max_bias = max_bias_px
        self._bias: float = 0.0          # current estimated mount bias (px)
        self._ready: bool = False        # True once warmup samples collected

    # ─────────────────────────────────────────────────────────────────────
    def update(self, raw_offset: Optional[float]) -> Optional[float]:
        """
        Feed a raw offset, receive the bias-corrected offset.

        Parameters
        ----------
        raw_offset : float or None
            Raw lateral offset from B1 (compute_lateral_offset).
            None means lane was not detected this frame.

        Returns
        -------
        float : raw_offset − current_bias  (corrected offset)
        None  : if raw_offset is None (pass-through)
        """
        if raw_offset is None:
            return None

        # Add to rolling window
        self._window.append(raw_offset)

        # Update bias estimate once we have enough samples
        if len(self._window) >= self._warmup:
            self._ready = True
            estimated = float(np.median(self._window))
            # Safety clamp: never over-correct
            self._bias = float(np.clip(estimated, -self._max_bias, self._max_bias))

        if not self._ready:
            # Warmup phase: no correction yet — pass through raw
            return raw_offset

        return raw_offset - self._bias

    # ─────────────────────────────────────────────────────────────────────
    def reset(self):
        """Clear history and restart warmup (e.g. when switching clips)."""
        self._window.clear()
        self._bias  = 0.0
        self._ready = False

    # ─────────────────────────────────────────────────────────────────────
    @property
    def current_bias(self) -> float:
        """Return the current estimated mount bias in pixels."""
        return self._bias

    @property
    def is_calibrated(self) -> bool:
        """True once warmup samples have been collected."""
        return self._ready

    @property
    def samples_collected(self) -> int:
        """Number of valid samples in the current rolling window."""
        return len(self._window)
