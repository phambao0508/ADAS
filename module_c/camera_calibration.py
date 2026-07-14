from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional
# Knowing the different between different kind of dash-cameras because the lens of the cameras may distort the image
# Without .npz, using the default fx (~HFOV 75o = 1251px), fy = fx, no undistort -> Estimate 80% - 90%
# With .npz, fx, fy real, read from camera matrix from opencv, undistort = map, calibarate using chessboard -> Higher accuracy
import cv2
import numpy as np

@dataclass
class CameraIntrinsics:

    frame_w: int
    frame_h: int

    hfov_deg: float = 75.0

    h_camera: float = 1.4

    horizon_y_frac: float = 0.50

    distance_offset_m: float = 0.0

    _fx_override: Optional[float] = field(default=None, repr=False)
    _fy_override: Optional[float] = field(default=None, repr=False)
    _cam_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    _dist_coeff: Optional[np.ndarray] = field(default=None, repr=False)
    _undistort_map1: Optional[np.ndarray] = field(default=None, repr=False)
    _undistort_map2: Optional[np.ndarray] = field(default=None, repr=False)
    _calibrated: bool = field(default=False, repr=False)

    @property
    def fx(self) -> float:

        if self._fx_override is not None:
            return self._fx_override
        return (self.frame_w * 0.5) / math.tan(math.radians(self.hfov_deg) * 0.5)

    @property
    def fy(self) -> float:

        if self._fy_override is not None:
            return self._fy_override
        return self.fx

    @property
    def horizon_y(self) -> float:

        return self.horizon_y_frac * self.frame_h

    @property
    def is_calibrated(self) -> bool:

        return self._calibrated

    @classmethod
    def default_for(cls, frame_w: int, frame_h: int) -> "CameraIntrinsics":
        return cls(frame_w=frame_w, frame_h=frame_h)

    @classmethod
    def from_calibration_file(
        cls,
        calib_path: str,
        frame_w: int,
        frame_h: int,
        h_camera: float = 1.4,
        horizon_y_frac: float = 0.50,
    ) -> "CameraIntrinsics":

        data = np.load(calib_path, allow_pickle=True)
        cam_matrix = data["cam_matrix"]
        dist_coeff = data["dist_coeff"]
        reproj_error = float(data["reproj_error"])

        fx_real = cam_matrix[0, 0]
        fy_real = cam_matrix[1, 1]

        hfov_real = 2 * math.degrees(math.atan(frame_w / (2 * fx_real)))

        new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(
            cam_matrix, dist_coeff, (frame_w, frame_h), 1, (frame_w, frame_h)
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            cam_matrix, dist_coeff, None, new_cam_matrix,
            (frame_w, frame_h), cv2.CV_16SC2
        )

        instance = cls(
            frame_w=frame_w,
            frame_h=frame_h,
            hfov_deg=hfov_real,
            h_camera=h_camera,
            horizon_y_frac=horizon_y_frac,
        )
        instance._fx_override = fx_real
        instance._fy_override = fy_real
        instance._cam_matrix = cam_matrix
        instance._dist_coeff = dist_coeff
        instance._undistort_map1 = map1
        instance._undistort_map2 = map2
        instance._calibrated = True

        print(f"[INFO] Custom calibration loaded: fx={fx_real:.1f} fy={fy_real:.1f} "
              f"HFOV={hfov_real:.1f}° reproj_err={reproj_error:.4f}px")

        return instance

    def undistort(self, frame: np.ndarray) -> np.ndarray:

        if not self._calibrated or self._undistort_map1 is None:
            return frame
        return cv2.remap(frame, self._undistort_map1, self._undistort_map2,
                         cv2.INTER_LINEAR)

REAL_VEHICLE_WIDTHS_M = {
    0: 1.85,
    1: 2.55,
    2: 2.55,
}
DEFAULT_VEHICLE_WIDTH_M = 1.85

