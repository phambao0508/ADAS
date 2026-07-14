from collections import deque
from typing import Optional

import numpy as np

# Remove symetric error for unwanted camera position
# Using sliding window, for example like median is 18, the raw offset is 19 so the corrected offset will be 19-18=1
#ax bias = 200, min bias = -200, raw offset <260 adding into sliding window and remove old offset
WARMUP_SAMPLES = 15
WINDOW_SIZE    = 300
MAX_BIAS_PX    = 200
MAX_SAMPLE_ABS_PX = 260
MAX_CORRECTED_ABS_PX = 120

class MountBiasEstimator:

    def __init__(
        self,
        warmup_samples: int = WARMUP_SAMPLES,
        window_size:    int = WINDOW_SIZE,
        max_bias_px:    float = MAX_BIAS_PX,
        max_sample_abs_px: float = MAX_SAMPLE_ABS_PX,
        max_corrected_abs_px: float = MAX_CORRECTED_ABS_PX,
    ):
        self._warmup   = warmup_samples
        self._window   = deque(maxlen=window_size)
        self._max_bias = max_bias_px
        self._max_sample_abs = max_sample_abs_px
        self._max_corrected_abs = max_corrected_abs_px
        self._bias: float = 0.0
        self._ready: bool = False

    def update(self, raw_offset: Optional[float]) -> Optional[float]:

        if raw_offset is None:
            return None

        if abs(raw_offset) <= self._max_sample_abs:
            self._window.append(raw_offset)

        if len(self._window) >= self._warmup:
            self._ready = True
            estimated = float(np.median(self._window))

            self._bias = float(np.clip(estimated, -self._max_bias, self._max_bias))

        if not self._ready:

            return raw_offset

        corrected = raw_offset - self._bias
        if abs(corrected) > self._max_corrected_abs:
            return None
        return corrected

    def reset(self):

        self._window.clear()
        self._bias  = 0.0
        self._ready = False

    @property
    def current_bias(self) -> float:

        return self._bias

    @property
    def is_calibrated(self) -> bool:

        return self._ready

    @property
    def samples_collected(self) -> int:

        return len(self._window)
