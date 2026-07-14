import numpy as np
from typing import List, Optional, Tuple

# Finding function for two line of the ego-lane: x = ay^2+by+c, Finding a, b,c to goes through all the raw points of the two left and right ego lane
MIN_POINTS_FOR_FIT = 10

FULL_TRUST_POINTS  = 40

POLY_DEGREE = 2

def fit_boundary_polynomial(
    pts: List[Tuple[int, int]],
    prev_poly: Optional[np.ndarray] = None,
    curve_responsive: bool = False,
) -> Optional[np.ndarray]:

    n = len(pts)
    min_points = 6 if curve_responsive else MIN_POINTS_FOR_FIT
    full_trust_points = 24 if curve_responsive else FULL_TRUST_POINTS

    if n < min_points:
        return prev_poly

    ys = np.array([p[0] for p in pts], dtype=np.float64)
    xs = np.array([p[1] for p in pts], dtype=np.float64)

    try:
        new_poly = np.polyfit(ys, xs, POLY_DEGREE)
    except (np.linalg.LinAlgError, ValueError):
        return prev_poly

    if prev_poly is None:
        return new_poly

    if n >= full_trust_points:
        return new_poly

    alpha = (n - min_points) / max(1, full_trust_points - min_points)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if curve_responsive:
        alpha = max(alpha, 0.65)
    blended = alpha * new_poly + (1.0 - alpha) * prev_poly
    return blended

def eval_poly(coeffs: np.ndarray, y: float) -> float:

    return float(np.polyval(coeffs, y))
