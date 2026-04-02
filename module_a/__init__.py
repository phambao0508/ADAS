"""
Module A — Lane Detection & Ego-Lane Tracking
==============================================
Works with a YOLO model that detects lane LINES (not lane areas):
    Class 3: white line
    Class 4: yellow line

Public API — import everything you need from here.

Usage example:
    from module_a import LanePipeline

    pipeline = LanePipeline(frame_width=1920, frame_height=1080)

    # detections = list of [x1, y1, x2, y2, conf, class_id] from YOLO
    result = pipeline.process(frame, detections)
"""

from .lane_pipeline        import LanePipeline, LaneResult
from .ego_lane_selector    import select_ego_lane, EgoLaneLines
from .boundary_extractor   import extract_boundaries
from .line_type_classifier import classify_line_type
from .poly_fitter          import fit_boundary_polynomial, eval_poly
from .bev_transformer      import BEVTransformer

__all__ = [
    "LanePipeline",
    "LaneResult",
    "select_ego_lane",
    "EgoLaneLines",
    "extract_boundaries",
    "classify_line_type",
    "fit_boundary_polynomial",
    "eval_poly",
    "BEVTransformer",
]
