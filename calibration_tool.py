"""
Camera Calibration Tool
========================
Chạy 1 lần để tính thông số camera (camera matrix + distortion coefficients)
từ ảnh checkerboard. Kết quả lưu vào calibration.npz để pipeline ADAS sử dụng.

Cách dùng:
    1. Chuẩn bị ảnh checkerboard:
       - In 1 tấm checkerboard (9x6 inner corners)
       - Chụp 15-20 ảnh từ nhiều góc khác nhau bằng chính dashcam
       - Lưu vào thư mục calibration_images/

    2. Chạy calibration:
       python calibration_tool.py --images calibration_images/
       
    3. Hoặc trích frame từ video checkerboard:
       python calibration_tool.py --video checkerboard_video.mp4 --extract 20

    4. Kết quả: calibration.npz (tự động được main.py load)
"""

import os
import sys
import glob
import argparse

import cv2 as cv
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Extract frames from video (nếu không có ảnh sẵn)
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames_from_video(video_path: str, output_dir: str, num_frames: int = 20):
    """
    Trích xuất frames từ video để dùng cho calibration.
    Lấy đều num_frames frame từ toàn bộ video.
    """
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Không mở được video: {video_path}")

    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if total < num_frames:
        num_frames = total

    os.makedirs(output_dir, exist_ok=True)

    # Lấy đều các frame
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    saved = 0

    for idx in indices:
        cap.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        path = os.path.join(output_dir, f"calib_frame_{saved:03d}.jpg")
        cv.imwrite(path, frame)
        saved += 1

    cap.release()
    print(f"[INFO] Đã trích xuất {saved} frames vào: {output_dir}")
    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Camera Calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(image_dir: str, board_rows: int = 9, board_cols: int = 6,
              show: bool = True, output_path: str = None):
    """
    Calibrate camera từ ảnh checkerboard.

    Args:
        image_dir:   thư mục chứa ảnh checkerboard (.jpg/.png)
        board_rows:  số inner corners theo chiều ngang (mặc định 9)
        board_cols:  số inner corners theo chiều dọc (mặc định 6)
        show:        hiển thị ảnh phát hiện corners
        output_path: đường dẫn file output (mặc định: calibration.npz cùng thư mục)

    Returns:
        cam_matrix, dist_coeff
    """

    # Tìm tất cả ảnh
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    if len(image_paths) == 0:
        sys.exit(f"[ERROR] Không tìm thấy ảnh calibration trong: {image_dir}")

    print(f"[INFO] Tìm thấy {len(image_paths)} ảnh calibration")

    # Termination criteria cho corner refinement
    term_criteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        30, 0.001
    )

    # Chuẩn bị 3D object points: (0,0,0), (1,0,0), (2,0,0), ...
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_rows, 0:board_cols].T.reshape(-1, 2)

    obj_points = []  # 3D points
    img_points = []  # 2D points
    img_size = None

    # ── Detect chessboard corners ──────────────────────────────────────────
    for i, path in enumerate(image_paths):
        img = cv.imread(path)
        if img is None:
            print(f"  [SKIP] Không đọc được: {path}")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]  # (width, height)

        found, corners = cv.findChessboardCorners(gray, (board_rows, board_cols), None)

        if found:
            obj_points.append(objp)
            refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), term_criteria)
            img_points.append(refined)

            print(f"  [OK]   {os.path.basename(path)} — corners detected")

            if show:
                cv.drawChessboardCorners(img, (board_rows, board_cols), refined, found)
                cv.imshow("Calibration - Chessboard Corners", img)
                cv.waitKey(400)
        else:
            print(f"  [FAIL] {os.path.basename(path)} — no corners found")

    cv.destroyAllWindows()

    if len(obj_points) == 0:
        sys.exit("[ERROR] Không phát hiện được corners nào. Kiểm tra lại ảnh và board size.")

    print(f"\n[INFO] Sử dụng {len(obj_points)}/{len(image_paths)} ảnh thành công")

    # ── Calibrate ──────────────────────────────────────────────────────────
    reproj_error, cam_matrix, dist_coeff, rvecs, tvecs = cv.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )

    # ── In kết quả ─────────────────────────────────────────────────────────
    fx = cam_matrix[0, 0]
    fy = cam_matrix[1, 1]
    cx = cam_matrix[0, 2]
    cy = cam_matrix[1, 2]

    print(f"\n{'='*55}")
    print(f"  Camera Calibration Results")
    print(f"{'='*55}")
    print(f"  fx = {fx:.2f} px")
    print(f"  fy = {fy:.2f} px")
    print(f"  cx = {cx:.2f} px")
    print(f"  cy = {cy:.2f} px")
    print(f"  Distortion: {dist_coeff.ravel()}")
    print(f"  Reprojection Error: {reproj_error:.4f} px")
    print(f"{'='*55}")

    # ── Lưu file ───────────────────────────────────────────────────────────
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "calibration.npz"
        )

    np.savez(
        output_path,
        cam_matrix=cam_matrix,
        dist_coeff=dist_coeff,
        reproj_error=reproj_error,
        image_size=np.array(img_size),
        rvecs=np.array(rvecs, dtype=object),
        tvecs=np.array(tvecs, dtype=object),
    )

    print(f"\n[INFO] Đã lưu calibration vào: {output_path}")
    print(f"[INFO] Pipeline ADAS sẽ tự động load file này khi chạy main.py")

    return cam_matrix, dist_coeff


# ─────────────────────────────────────────────────────────────────────────────
# Demo: So sánh trước/sau undistort
# ─────────────────────────────────────────────────────────────────────────────

def demo_undistort(image_path: str, calib_path: str = None):
    """Hiển thị so sánh ảnh trước/sau khi undistort."""

    if calib_path is None:
        calib_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "calibration.npz"
        )

    if not os.path.exists(calib_path):
        sys.exit(f"[ERROR] Chưa có file calibration: {calib_path}\n"
                 f"        Chạy calibration trước: python calibration_tool.py --images <folder>")

    data = np.load(calib_path)
    cam_matrix = data["cam_matrix"]
    dist_coeff = data["dist_coeff"]

    img = cv.imread(image_path)
    if img is None:
        sys.exit(f"[ERROR] Không đọc được ảnh: {image_path}")

    h, w = img.shape[:2]

    new_cam_matrix, roi = cv.getOptimalNewCameraMatrix(
        cam_matrix, dist_coeff, (w, h), 1, (w, h)
    )

    undistorted = cv.undistort(img, cam_matrix, dist_coeff, None, new_cam_matrix)

    # Hiển thị
    combined = np.hstack([img, undistorted])
    cv.putText(combined, "Original", (20, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.putText(combined, "Undistorted", (w + 20, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow("Before vs After Undistortion", combined)
    print("[INFO] Nhấn phím bất kỳ để đóng...")
    cv.waitKey(0)
    cv.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Camera Calibration Tool cho ADAS Pipeline"
    )
    p.add_argument("--images", type=str, default=None,
                   help="Thư mục chứa ảnh checkerboard")
    p.add_argument("--video", type=str, default=None,
                   help="Video checkerboard (sẽ tự trích frame)")
    p.add_argument("--extract", type=int, default=20,
                   help="Số frame cần trích từ video (mặc định: 20)")
    p.add_argument("--rows", type=int, default=9,
                   help="Số inner corners theo chiều ngang (mặc định: 9)")
    p.add_argument("--cols", type=int, default=6,
                   help="Số inner corners theo chiều dọc (mặc định: 6)")
    p.add_argument("--output", type=str, default=None,
                   help="Đường dẫn file output (mặc định: calibration.npz)")
    p.add_argument("--no-show", action="store_true",
                   help="Không hiển thị ảnh corners")
    p.add_argument("--demo", type=str, default=None,
                   help="Đường dẫn ảnh để demo undistort (chạy sau calibration)")

    args = p.parse_args()

    # Demo mode
    if args.demo:
        demo_undistort(args.demo)
        return

    # Trích frame từ video nếu cần
    image_dir = args.images
    if args.video:
        image_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "calibration_images"
        )
        extract_frames_from_video(args.video, image_dir, args.extract)

    if image_dir is None:
        p.print_help()
        print("\n[ERROR] Cần chỉ định --images hoặc --video")
        sys.exit(1)

    calibrate(
        image_dir=image_dir,
        board_rows=args.rows,
        board_cols=args.cols,
        show=not args.no_show,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
