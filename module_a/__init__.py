"""
Module A — Lane Detection & Ego-Lane Tracking (UFLDv2 VERSION)
===============================================================
Uses UFLDv2 lane coordinate output instead of YOLO segmentation masks.

UFLDv2 detects up to 4 lanes as coordinate lists.
This module selects the ego-lane boundaries, fits polynomials,
and produces a LaneResult for downstream modules.

Public API — import everything you need from here.

Usage example:
    from ufld_wrapper import UFLDv2Wrapper
    from module_a import LanePipeline

    ufld = UFLDv2Wrapper(model_path="culane_res18.pth")
    pipeline = LanePipeline(frame_width=1920, frame_height=1080)

    ufld_lanes = ufld.detect_lanes(frame)
    result = pipeline.process(frame, ufld_lanes)
"""

from .lane_pipeline        import LanePipeline, LaneResult
from .ego_lane_selector    import select_ego_lane, EgoLaneLines
from .boundary_extractor   import extract_boundaries

from .poly_fitter          import fit_boundary_polynomial, eval_poly


__all__ = [
    "LanePipeline",
    "LaneResult",
    "select_ego_lane",
    "EgoLaneLines",
    "extract_boundaries",

    "fit_boundary_polynomial",
    "eval_poly",

]
