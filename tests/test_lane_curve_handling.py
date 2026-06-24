import numpy as np

from module_a.ego_lane_selector import select_ego_lane
from module_a.poly_fitter import fit_boundary_polynomial


def test_curve_responsive_fit_uses_sparse_new_curve_more_strongly():
    prev_poly = np.array([0.0, 0.0, 500.0])
    pts = []
    for y in range(400, 760, 30):
        x = 0.0007 * (y - 400) ** 2 + 500.0
        pts.append((y, int(round(x))))

    normal = fit_boundary_polynomial(pts, prev_poly, curve_responsive=False)
    responsive = fit_boundary_polynomial(pts, prev_poly, curve_responsive=True)

    y_eval = 750
    raw_curve_x = np.polyval(np.polyfit(
        np.array([p[0] for p in pts], dtype=np.float64),
        np.array([p[1] for p in pts], dtype=np.float64),
        2,
    ), y_eval)

    normal_err = abs(np.polyval(normal, y_eval) - raw_curve_x)
    responsive_err = abs(np.polyval(responsive, y_eval) - raw_curve_x)
    assert responsive_err < normal_err


def test_selector_ranks_curved_candidate_with_multi_row_score():
    frame_w = 1920
    frame_h = 1080

    # Both lanes are left of center at the bottom. The curved candidate is
    # farther at the bottom but better aligned over the lower/mid road.
    straight_left = [(500, 650), (700, 800), (900, 1000)]
    curved_left = [(930, 650), (940, 800), (850, 1000)]
    right = [(1150, 650), (1180, 800), (1230, 1000)]

    ego = select_ego_lane([straight_left, curved_left, right], frame_w, frame_h)

    assert ego.left_pts == [(650, 930), (800, 940), (1000, 850)]
    assert ego.right_pts
