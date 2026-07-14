from typing import List, Tuple

# Organize all the ego points by y from lowest to highest
def extract_boundaries(
    left_pts: List[Tuple[int, int]],
    right_pts: List[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:

    left_sorted = sorted(left_pts, key=lambda p: p[0]) if left_pts else []
    right_sorted = sorted(right_pts, key=lambda p: p[0]) if right_pts else []

    return left_sorted, right_sorted
