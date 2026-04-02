"""
Module B  —  Step B2: Exponential Moving Average (EMA) Smoother
================================================================
TASK
----
Smooth the raw lateral offset values to remove frame-to-frame noise
from YOLO mask flickering without losing track of genuine slow drift.

WHY THIS IS NEEDED
------------------
The raw offset from B1 can jump several pixels between consecutive frames
even when the car is perfectly still. This is caused by:
  - Sub-pixel mask edge variation in the segmentation output
  - Polynomial fit sensitivity to a few noisy boundary points
  - YOLO confidence oscillation causing slightly different mask shapes

Without smoothing, the 6-state classifier would flicker between
CENTERED and WARN every few frames — confusing and unsafe.

ALGORITHM  (Section B2 of the implementation plan)
---------------------------------------------------
    smoothed = ALPHA × raw + (1 − ALPHA) × previous_smoothed

    ALPHA = 0.25   ← weight given to the NEW measurement
    1 − ALPHA = 0.75   ← weight given to the HISTORY

    Effect:
      - A sudden 1-frame spike of 50 px only shifts the smoothed value
        by 0.25 × 50 = 12.5 px in that frame.
      - A genuine slow drift of 5 px/frame accumulates steadily.
      - The lag is approximately 1/(1−ALPHA) ≈ 4–6 frames.

NONE HANDLING
-------------
If the raw offset is None (both polynomials missing this frame), the
smoother holds its previous value — the car does not suddenly appear
centred just because one frame had no lane data.

If many frames in a row return None (complete lane loss), the smoother
eventually has to be reset externally (via reset()).

INPUTS / OUTPUTS
----------------
  update(raw) → float | None
    raw : float (new offset) or None (no data this frame)
    returns : smoothed offset (float) or None on first call with no data

TUNING
------
  From the plan's Tuning Reference:
    SMOOTHING_ALPHA = 0.25  → Faster response (but more flicker above 0.25)
                              Slower but smoother below 0.25
"""

from typing import Optional


# Weight given to the new measurement each frame.
# Lower = smoother, higher = more responsive.
# Plan value: 0.25
EMA_ALPHA = 0.25

# If a new raw offset jumps more than this many pixels from the
# current smoothed value, treat it as a spike / bad polynomial frame
# and blend it in at a much lower weight instead of the normal alpha.
# This is the last line of defense against single-frame WARN_L/WARN_R
# caused by a mis-fitted polynomial.
MAX_JUMP_PX     = 120   # px jump threshold
SPIKE_ALPHA     = 0.05  # weight for spike frames (vs normal EMA_ALPHA)


class EMASmoother:
    """
    Single-variable Exponential Moving Average smoother.

    Usage
    -----
        smoother = EMASmoother()

        for frame in video:
            raw_offset = compute_lateral_offset(...)
            smoothed   = smoother.update(raw_offset)
    """

    def __init__(self, alpha: float = EMA_ALPHA):
        """
        Parameters
        ----------
        alpha : float in (0, 1]
            Smoothing factor. 1.0 = no smoothing (pass-through).
            Plan default: 0.25.
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._prev: Optional[float] = None

    # ─────────────────────────────────────────────────────────────────────
    def update(self, raw: Optional[float]) -> Optional[float]:
        """
        Feed a new raw offset and return the smoothed value.

        Parameters
        ----------
        raw : float or None
            New raw lateral offset from B1.
            None means no lane data was available this frame.

        Returns
        -------
        float : the smoothed offset (EMA of all previous raw values)
        None  : only on the very first call if raw is also None
                (no history and no current data)
        """
        if raw is None:
            # Hold: no new data — return last known smoothed value
            return self._prev

        if self._prev is None:
            # First ever measurement — initialise (no history to blend)
            self._prev = raw
            return self._prev

        # Spike rejection: if the new measurement jumps too far from the
        # current smoothed value, blend it in at a much lower weight.
        # This prevents a single bad polynomial frame from flipping state.
        jump = abs(raw - self._prev)
        alpha = SPIKE_ALPHA if jump > MAX_JUMP_PX else self.alpha

        # Standard EMA update (with adaptive alpha)
        smoothed   = alpha * raw + (1.0 - alpha) * self._prev
        self._prev = smoothed
        return smoothed

    # ─────────────────────────────────────────────────────────────────────
    def reset(self):
        """
        Clear the smoother's history (e.g. after a hard lane loss).
        The next call to update() will reinitialise from scratch.
        """
        self._prev = None

    # ─────────────────────────────────────────────────────────────────────
    @property
    def current(self) -> Optional[float]:
        """Return the last smoothed value without updating."""
        return self._prev
