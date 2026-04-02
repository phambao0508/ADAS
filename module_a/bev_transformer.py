"""
Module A  —  Step A5: Bird's Eye View (BEV) Transformer
=========================================================
TASK
----
Apply a perspective warp to convert the dashcam's natural camera view
(where the road appears to converge to a vanishing point) into a
top-down "Bird's Eye View" (BEV) where the lane lines appear parallel
and vertical — much easier to measure laterally.

WHY THIS IS NEEDED
------------------
In the normal camera view, the road looks like a trapezoid:
  - Lane lines converge as they get farther away
  - 1 pixel at the top of the frame represents MUCH more real-world
    distance than 1 pixel at the bottom

This means directly measuring "how far left is the car" in pixel space
is WRONG — the scale changes with distance.

In BEV (top-down view):
  - Both lane lines are nearly parallel
  - Real-world "left/right" position is uniformly represented in pixels
  - Measuring lateral offset is now accurate and meaningful (Module B)

ALGORITHM  (Section A5 of the implementation plan)
---------------------------------------------------
A homography (perspective) transform is a 3×3 matrix M that maps
4 source points in the camera image to 4 destination points in BEV space.

  Source (camera view trapezoid):          Destination (BEV rectangle):
    Top-left    ≈ (43% W, 65% H)           Top-left    → (20% W,  0% H)
    Top-right   ≈ (57% W, 65% H)           Top-right   → (80% W,  0% H)
    Bottom-right≈ (95% W, 98% H)           Bottom-right→ (80% W,100% H)
    Bottom-left ≈  (5% W, 98% H)           Bottom-left → (20% W,100% H)

OpenCV's getPerspectiveTransform() computes the matrix M from these 8 points.
warpPerspective() then applies M to the full image.

The INVERSE matrix M_inv maps BEV coordinates BACK to camera view.
This is used by Module D to draw the lane overlay on the original frame.

DESIGN NOTE: The class is initialised ONCE (expensive transform computation),
then used every frame (fast matrix multiply).

INPUTS (to __init__)
--------------------
  frame_w, frame_h : video frame dimensions (pixels)

INPUTS (to warp())
------------------
  image : np.ndarray (H, W) or (H, W, 3)
          Any image in camera view — mask, frame, etc.

OUTPUTS (from warp())
---------------------
  warped : np.ndarray — same shape as input, in BEV space
"""

import cv2
import numpy as np


# ── Perspective warp source/destination fractions ─────────────────────────
# These define the trapezoid on the road surface and its BEV rectangle.
# Adjust them if the dashcam mounting position changes.
WRP_SRC = np.array([
    [0.43, 0.65],   # top-left     (x_frac, y_frac)
    [0.57, 0.65],   # top-right
    [0.95, 0.98],   # bottom-right
    [0.05, 0.98],   # bottom-left
], dtype=np.float32)

WRP_DST = np.array([
    [0.20, 0.00],   # top-left
    [0.80, 0.00],   # top-right
    [0.80, 1.00],   # bottom-right
    [0.20, 1.00],   # bottom-left
], dtype=np.float32)
# ──────────────────────────────────────────────────────────────────────────


class BEVTransformer:
    """
    Computes and applies a Bird's-Eye-View perspective warp.

    Create ONE instance per video (the warp matrix is fixed for a given
    frame size). Then call warp() or unwarp() every frame.

    Attributes
    ----------
    M     : np.ndarray (3, 3) — camera-to-BEV homography matrix
    M_inv : np.ndarray (3, 3) — BEV-to-camera inverse homography
    src   : np.ndarray (4, 2) — source trapezoid corners in pixel coords
    dst   : np.ndarray (4, 2) — destination rectangle corners in pixel coords
    """

    def __init__(self, frame_w: int, frame_h: int):
        """
        Pre-compute the warp matrices for a given frame resolution.

        Parameters
        ----------
        frame_w : int  — width  of each video frame in pixels
        frame_h : int  — height of each video frame in pixels
        """
        self.frame_w = frame_w
        self.frame_h = frame_h

        # Scale fractional coordinates to actual pixel positions
        self.src = WRP_SRC * np.array([frame_w, frame_h], dtype=np.float32)
        self.dst = WRP_DST * np.array([frame_w, frame_h], dtype=np.float32)

        # Compute forward matrix (camera view → BEV)
        self.M     = cv2.getPerspectiveTransform(self.src, self.dst)

        # Compute inverse matrix (BEV → camera view)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    # ──────────────────────────────────────────────────────────────────────
    def warp(self, image: np.ndarray) -> np.ndarray:
        """
        Warp a camera-view image INTO Bird's Eye View.

        Parameters
        ----------
        image : np.ndarray
            Any camera-view image — raw frame, lane mask, etc.
            Shape: (H, W) for grayscale, or (H, W, 3) for colour.

        Returns
        -------
        np.ndarray — the same image warped to top-down BEV.
        """
        return cv2.warpPerspective(
            image,
            self.M,
            (self.frame_w, self.frame_h),
            flags=cv2.INTER_LINEAR,
        )

    # ──────────────────────────────────────────────────────────────────────
    def unwarp(self, bev_image: np.ndarray) -> np.ndarray:
        """
        Warp a BEV image BACK to the original camera perspective.

        Use this in Module D to overlay BEV-drawn lane fills back onto
        the original dashcam frame.

        Parameters
        ----------
        bev_image : np.ndarray
            Any BEV-space image.

        Returns
        -------
        np.ndarray — same image projected back to camera view.
        """
        return cv2.warpPerspective(
            bev_image,
            self.M_inv,
            (self.frame_w, self.frame_h),
            flags=cv2.INTER_LINEAR,
        )

    # ──────────────────────────────────────────────────────────────────────
    def draw_warp_region(self, frame: np.ndarray) -> np.ndarray:
        """
        Debug helper: draw the source trapezoid on the camera-view frame.

        Useful for tuning the WRP_SRC fractions — verifies that the
        warp region correctly covers the road surface ahead of the car.

        Parameters
        ----------
        frame : np.ndarray (H, W, 3) — the original BGR video frame.

        Returns
        -------
        np.ndarray — frame with a green quadrilateral drawn over it.
        """
        vis = frame.copy()
        pts = self.src.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        return vis
