import math
import numpy as np
import cv2
from typing import Tuple

def draw_glass_panel(
    frame:    np.ndarray,
    x1: int, y1: int,
    x2: int, y2: int,
    tint_bgr: Tuple[int, int, int] = (12, 16, 22),
    blur_k:   int = 21,
    alpha:    float = 0.65,
    border_colour: Tuple[int, int, int] = (55, 70, 90),
    border_width:  int = 1,
) -> np.ndarray:

    H, W = frame.shape[:2]

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return frame

    roi = frame[y1:y2, x1:x2].copy()
    if blur_k > 1:
        blur_k = blur_k | 1
        roi = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)

    tint = np.full_like(roi, tint_bgr, dtype=np.uint8)
    blended = cv2.addWeighted(roi, 1.0 - alpha * 0.6, tint, alpha * 0.6, 0)

    frame[y1:y2, x1:x2] = blended

    if border_width > 0:
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_colour, border_width)

    return frame

class ColourSmoother:

    def __init__(self, speed: float = 0.15, initial: Tuple[int, int, int] = (74, 200, 0)):
        self._current = np.array(initial, dtype=np.float64)
        self._speed = speed

    def update(self, target_bgr: Tuple[int, int, int]) -> Tuple[int, ...]:

        target = np.array(target_bgr, dtype=np.float64)
        self._current += (target - self._current) * self._speed
        return tuple(int(v) for v in np.clip(self._current, 0, 255))

    @property
    def current_bgr(self) -> Tuple[int, ...]:
        return tuple(int(v) for v in np.clip(self._current, 0, 255))

def pulse_brightness(
    colour_bgr: Tuple[int, int, int],
    frame_count: int,
    amplitude: int = 40,
    period_frames: int = 20,
) -> Tuple[int, int, int]:

    t = math.sin(2 * math.pi * frame_count / period_frames) * amplitude
    b, g, r = colour_bgr
    return (
        max(0, min(255, int(b + t))),
        max(0, min(255, int(g + t))),
        max(0, min(255, int(r + t))),
    )

_vignette_cache = {}

def draw_vignette(
    frame: np.ndarray,
    strength: float = 0.3,
) -> np.ndarray:

    H, W = frame.shape[:2]
    key = (H, W, round(strength, 2))

    if key not in _vignette_cache:

        Y, X = np.ogrid[:H, :W]
        cx, cy = W / 2, H / 2

        dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
        dist = np.clip(dist, 0, 1)

        mask = (1.0 - strength * dist ** 1.5).astype(np.float32)
        _vignette_cache[key] = mask

    mask = _vignette_cache[key]

    result = (frame.astype(np.float32) * mask[:, :, np.newaxis]).astype(np.uint8)
    return result

def draw_status_dot(
    frame: np.ndarray,
    cx: int, cy: int,
    colour: Tuple[int, int, int],
    radius: int = 4,
) -> np.ndarray:

    cv2.circle(frame, (cx, cy), radius, colour, cv2.FILLED, cv2.LINE_AA)

    glow = tuple(max(0, c - 80) for c in colour)
    cv2.circle(frame, (cx, cy), radius + 2, glow, 1, cv2.LINE_AA)
    return frame
