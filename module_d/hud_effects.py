"""
Module D  —  HUD Effects Library
==================================
Reusable visual effects for the HUD system:

  - ``draw_glass_panel``  — Glassmorphism panel with blur-behind
  - ``ColourSmoother``    — Lerps between state colours over time
  - ``draw_vignette``     — Radial darkening at frame edges
  - ``pulse_brightness``  — Sinusoidal brightness oscillation

Design philosophy: *Design Spells* skill — every panel should feel
premium, translucent, and alive with subtle micro-interactions.
"""

import math
import numpy as np
import cv2
from typing import Tuple


# ── Glassmorphism Panel ───────────────────────────────────────────────────

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
    """
    Draw a glassmorphism panel: blurred background + tinted overlay.

    Parameters
    ----------
    frame : np.ndarray (H, W, 3) BGR
    x1, y1, x2, y2 : int — panel rectangle (clipped to frame bounds)
    tint_bgr : tuple — colour tint applied over the blur
    blur_k : int — Gaussian blur kernel size (must be odd)
    alpha : float — blend factor (0 = fully transparent, 1 = fully opaque)
    border_colour : tuple — panel border colour
    border_width : int — border thickness (0 = no border)

    Returns
    -------
    np.ndarray — frame with glass panel drawn
    """
    H, W = frame.shape[:2]
    # Clamp to frame bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return frame

    # 1. Extract ROI and blur it
    roi = frame[y1:y2, x1:x2].copy()
    if blur_k > 1:
        blur_k = blur_k | 1  # ensure odd
        roi = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)

    # 2. Tint the blurred ROI
    tint = np.full_like(roi, tint_bgr, dtype=np.uint8)
    blended = cv2.addWeighted(roi, 1.0 - alpha * 0.6, tint, alpha * 0.6, 0)

    # 3. Paste back
    frame[y1:y2, x1:x2] = blended

    # 4. Border
    if border_width > 0:
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_colour, border_width)

    return frame


# ── Colour Smoother ───────────────────────────────────────────────────────

class ColourSmoother:
    """
    Lerps between previous and current state colours at a configurable
    speed, creating smooth colour transitions instead of instant snapping.

    Usage
    -----
        smoother = ColourSmoother(speed=0.15)
        # Each frame:
        bgr = smoother.update((74, 200, 0))  # target green
    """

    def __init__(self, speed: float = 0.15, initial: Tuple[int, int, int] = (74, 200, 0)):
        self._current = np.array(initial, dtype=np.float64)
        self._speed = speed

    def update(self, target_bgr: Tuple[int, int, int]) -> Tuple[int, ...]:
        """Advance toward target colour by ``speed`` fraction."""
        target = np.array(target_bgr, dtype=np.float64)
        self._current += (target - self._current) * self._speed
        return tuple(int(v) for v in np.clip(self._current, 0, 255))

    @property
    def current_bgr(self) -> Tuple[int, ...]:
        return tuple(int(v) for v in np.clip(self._current, 0, 255))


# ── Pulse Brightness ─────────────────────────────────────────────────────

def pulse_brightness(
    colour_bgr: Tuple[int, int, int],
    frame_count: int,
    amplitude: int = 40,
    period_frames: int = 20,
) -> Tuple[int, int, int]:
    """
    Oscillate a colour's brightness using a sine wave.

    Parameters
    ----------
    colour_bgr : base colour
    frame_count : current frame number (monotonically increasing)
    amplitude : max brightness delta
    period_frames : number of frames for one full cycle

    Returns
    -------
    Tuple[int, int, int] — pulsed BGR colour
    """
    t = math.sin(2 * math.pi * frame_count / period_frames) * amplitude
    b, g, r = colour_bgr
    return (
        max(0, min(255, int(b + t))),
        max(0, min(255, int(g + t))),
        max(0, min(255, int(r + t))),
    )


# ── Vignette Effect ──────────────────────────────────────────────────────

_vignette_cache = {}

def draw_vignette(
    frame: np.ndarray,
    strength: float = 0.3,
) -> np.ndarray:
    """
    Apply a subtle radial vignette darkening at frame edges.

    Uses a cached mask to avoid recomputing every frame.
    """
    H, W = frame.shape[:2]
    key = (H, W, round(strength, 2))

    if key not in _vignette_cache:
        # Build radial gradient mask
        Y, X = np.ogrid[:H, :W]
        cx, cy = W / 2, H / 2
        # Normalised distance from centre
        dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
        dist = np.clip(dist, 0, 1)
        # Map to darkening factor: 1.0 at centre, (1-strength) at corners
        mask = (1.0 - strength * dist ** 1.5).astype(np.float32)
        _vignette_cache[key] = mask

    mask = _vignette_cache[key]
    # Apply in one vectorised operation
    result = (frame.astype(np.float32) * mask[:, :, np.newaxis]).astype(np.uint8)
    return result


# ── Status Dot ────────────────────────────────────────────────────────────

def draw_status_dot(
    frame: np.ndarray,
    cx: int, cy: int,
    colour: Tuple[int, int, int],
    radius: int = 4,
) -> np.ndarray:
    """Draw a small filled circle (status indicator dot)."""
    cv2.circle(frame, (cx, cy), radius, colour, cv2.FILLED, cv2.LINE_AA)
    # Subtle glow ring
    glow = tuple(max(0, c - 80) for c in colour)
    cv2.circle(frame, (cx, cy), radius + 2, glow, 1, cv2.LINE_AA)
    return frame
