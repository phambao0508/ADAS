"""
Module A  —  Step A4: Polynomial Boundary Fitter
==================================================
TASK
----
Take the raw left-boundary and right-boundary pixel lists (noisy, with
gaps) from the boundary extractor and fit a smooth mathematical curve
through each one.

WHY THIS IS NEEDED
------------------
The raw boundary points from A2 are NOISY:
  - YOLO masks have ragged, pixel-level edges
  - Some rows may be empty (mask gap)
  - Consecutive rows can jump by several pixels

A polynomial curve SMOOTHS this noise into a single clean equation
that can be evaluated at any y-coordinate. This is critical for:
  - Drawing nice smooth boundary lines in Module D (HUD)
  - Computing accurate lane-centre x-position for Module B
  - Computing zone-divider positions for Module C

ALGORITHM  (Section A4 of the implementation plan)
---------------------------------------------------
Model:  x = a·y² + b·y + c         (quadratic / degree-2 polynomial)

This says: "at height y on the image, the boundary is at x-position
given by the quadratic formula".

WHY quadratic (degree 2)?
  - Degree 1 (straight line): can't represent curved roads
  - Degree 2 (parabola): correctly models gentle highway curves ✓
  - Degree 3 or higher: overfits the noise → wiggly, unreliable

Fitting:  np.polyfit(ys, xs, deg=2)
  This uses least-squares regression to find [a, b, c] that minimises
  the sum of squared errors between predicted x and actual boundary x.

Example:
  Left boundary points:  [ (100, 320), (200, 340), (400, 390), (600, 420) ]
  polyfit gives:  a=-0.00002, b=0.18, c=304       (hypothetical values)
  Evaluate at y=500:  x = -0.00002·500² + 0.18·500 + 304  ≈ 389 px

FALLBACK
--------
If fitting fails (too few points, degenerate case), the function returns
the PREVIOUS frame's coefficients (passed in as `prev_poly`).
This prevents sudden jumps when a detection is temporarily lost.

INPUTS
------
  pts       : list of (y, x) tuples (from boundary_extractor)
  prev_poly : np.ndarray shape (3,) from the previous frame, or None
              It is used as a fallback if fitting fails this frame.

OUTPUTS
-------
  poly_coeffs : np.ndarray([a, b, c]) — the fitted polynomial coefficients.
                Evaluate with: np.polyval(poly_coeffs, y_value)
                Returns None only if fitting fails AND no prev_poly available.
"""

import numpy as np
from typing import List, Optional, Tuple


# Minimum number of boundary points needed to attempt a NEW fit.
# Below this → keep previous poly entirely (no new data to trust).
MIN_POINTS_FOR_FIT = 10

# Above this threshold → fully trust the new fit (no blending needed).
# Between MIN_POINTS_FOR_FIT and FULL_TRUST_POINTS → blend old & new.
FULL_TRUST_POINTS  = 40

# Polynomial degree (must be 2 per the design)
POLY_DEGREE = 2


def fit_boundary_polynomial(
    pts: List[Tuple[int, int]],
    prev_poly: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Fit a degree-2 polynomial x = f(y) through boundary pixel coordinates.

    When YOLO only detects a short fragment of the lane line (e.g. the line
    is partially hidden behind a car), the new fit is blended with the previous
    frame's polynomial rather than fully replacing it.  The blend weight is
    proportional to how many points are available:

        n < MIN_POINTS_FOR_FIT   → return prev_poly unchanged  (no new data)
        MIN ≤ n < FULL_TRUST     → weighted blend: more prev at low n
        n ≥ FULL_TRUST_POINTS    → return new fit only

    This implements the "connect last known to next detected" behaviour:
    the extrapolated previous polynomial fills in the gap, and the new
    partial detection gradually steers it back without sudden jumps.

    Parameters
    ----------
    pts : List[Tuple[int, int]]
        Raw boundary points as (y, x) pairs from boundary_extractor.
    prev_poly : np.ndarray or None
        Polynomial coefficients [a, b, c] from the previous successful frame.

    Returns
    -------
    np.ndarray of shape (3,)  — [a, b, c] for  x = a·y² + b·y + c
    or None if no fit is possible (no current points and no prev_poly).
    """
    n = len(pts)

    if n < MIN_POINTS_FOR_FIT:
        # Not enough points — hold previous polynomial as-is
        return prev_poly   # may be None on the very first frame

    # Unpack into separate y and x arrays for numpy
    ys = np.array([p[0] for p in pts], dtype=np.float64)
    xs = np.array([p[1] for p in pts], dtype=np.float64)

    try:
        new_poly = np.polyfit(ys, xs, POLY_DEGREE)
    except (np.linalg.LinAlgError, ValueError):
        return prev_poly

    # ── Blend with previous polynomial based on detection confidence ──────
    # If prev_poly is None (first frame), just use the new fit regardless.
    if prev_poly is None:
        return new_poly

    if n >= FULL_TRUST_POINTS:
        # Plenty of points — fully trust the new fit
        return new_poly

    # Partial detection: blend.
    # alpha = how much weight goes to the NEW polynomial (0 → all prev, 1 → all new)
    alpha = (n - MIN_POINTS_FOR_FIT) / (FULL_TRUST_POINTS - MIN_POINTS_FOR_FIT)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = alpha * new_poly + (1.0 - alpha) * prev_poly
    return blended


def eval_poly(coeffs: np.ndarray, y: float) -> float:
    """
    Evaluate the polynomial at a given y coordinate to get the x position.

    Parameters
    ----------
    coeffs : np.ndarray
        Polynomial coefficients [a, b, c].
    y : float
        The row (y-coordinate) at which to evaluate the boundary's x position.

    Returns
    -------
    float : the x-coordinate of the boundary at that row.

    Example
    -------
    >>> coeffs = np.array([-0.00002, 0.18, 304.0])
    >>> eval_poly(coeffs, 500)
    389.0   # hypothetical
    """
    return float(np.polyval(coeffs, y))
