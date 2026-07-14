from typing import Optional

EMA_ALPHA = 0.25

MAX_JUMP_PX     = 120
SPIKE_ALPHA     = 0.05

class EMASmoother:

    def __init__(self, alpha: float = EMA_ALPHA):

        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._prev: Optional[float] = None

    def update(self, raw: Optional[float]) -> Optional[float]:

        if raw is None:

            return self._prev

        if self._prev is None:

            self._prev = raw
            return self._prev

        jump = abs(raw - self._prev)
        alpha = SPIKE_ALPHA if jump > MAX_JUMP_PX else self.alpha

        smoothed   = alpha * raw + (1.0 - alpha) * self._prev
        self._prev = smoothed
        return smoothed

    def reset(self):

        self._prev = None

    @property
    def current(self) -> Optional[float]:

        return self._prev
