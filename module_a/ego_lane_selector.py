from typing import List, Optional, NamedTuple, Tuple

import numpy as np

### We can select the egolane by calculating the score at different line at 0.60, 0.78 and 0.92). Take the score of the two min from the center of the image
class EgoLaneLines(NamedTuple):

    left_pts:       List[Tuple[int, int]]
    right_pts:      List[Tuple[int, int]]
    left_label:     Optional[str]
    right_label:    Optional[str]
    found:          bool

HORIZON_Y_FRAC = 0.30

SCORE_Y_FRACS = (0.60, 0.78, 0.92)

def _lane_x_at_y(filtered: List[Tuple[int, int]], target_y: float) -> float:

    pts = sorted(filtered, key=lambda p: p[1])
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    return float(np.interp(target_y, ys, xs))

def select_ego_lane(
    lanes: List[List[Tuple[int, int]]],
    frame_w: int,
    frame_h: int,
) -> EgoLaneLines:

    cx_frame = frame_w / 2.0
    upper_limit_y = frame_h * HORIZON_Y_FRAC

    left_candidates = []
    right_candidates = []

    for lane in lanes:
        if not lane or len(lane) < 2:
            continue

        filtered = [(x, y) for (x, y) in lane if y >= upper_limit_y]
        if len(filtered) < 2:
            continue

        ys_filtered = [p[1] for p in filtered]
        top_y = min(ys_filtered)
        bottom_y = max(ys_filtered)
        bottom_x = _lane_x_at_y(filtered, bottom_y)

        sample_dists = []
        for frac in SCORE_Y_FRACS:
            target_y = frame_h * frac
            if top_y <= target_y <= bottom_y:
                sample_x = _lane_x_at_y(filtered, target_y)
                sample_dists.append(abs(sample_x - cx_frame))

        if not sample_dists:
            sample_dists = [abs(bottom_x - cx_frame)]

        score = 0.45 * abs(bottom_x - cx_frame) + 0.55 * float(np.mean(sample_dists))

        pts_yx = [(y, x) for (x, y) in filtered]

        if bottom_x < cx_frame:
            left_candidates.append((score, pts_yx))
        else:
            right_candidates.append((score, pts_yx))

    left_pts = []
    right_pts = []

    if left_candidates:
        left_candidates.sort(key=lambda t: t[0])
        left_pts = left_candidates[0][1]

    if right_candidates:
        right_candidates.sort(key=lambda t: t[0])
        right_pts = right_candidates[0][1]

    found = bool(left_pts) or bool(right_pts)

    return EgoLaneLines(
        left_pts=left_pts,
        right_pts=right_pts,
        left_label=None,
        right_label=None,
        found=found,
    )
