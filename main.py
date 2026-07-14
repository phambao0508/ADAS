import argparse
import sys
import time
from pathlib import Path
import os
from collections import defaultdict

# ── Suppress ultralytics progress bars ──────────────────────────────────
os.environ["YOLO_VERBOSE"] = "False"

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] ultralytics not installed. Run: pip install ultralytics")

from unet_resnet34_wrapper import UNetResNet34Wrapper
from module_a import LanePipeline
from module_b import DeparturePipeline
from module_c import GuidancePipeline, CameraIntrinsics
from module_d import HUDPipeline

# ─────────────────────────────────────────────────────────────────────────────
# ★ CONFIGURATION (Default values) ★
# ─────────────────────────────────────────────────────────────────────────────
INPUT_VIDEO   = r"D:\B3-ICT\ADAS\test_video\test.mp4"
OUTPUT_VIDEO  = r"D:\B3-ICT\ADAS\output.mp4"

# YOLO11s — 3-class object detection (Car, bus, truck)
YOLO_WEIGHTS  = r"D:\B3-ICT\ADAS\model\best_yolo.pt"

# ResNet34 Attention U-Net IoU60 — Lane detection
LANE_WEIGHTS  = r"D:\B3-ICT\ADAS\model\resnet34_attention_unet_iou60_try.pth"

CONF_THRESHOLD = 0.35
IOU_THRESHOLD  = 0.45
LOG_EVERY_N    = 30

# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="ADAS dual-model pipeline (YOLO + ResNet34 Attention U-Net IoU60)")
    p.add_argument("--input",       default=INPUT_VIDEO)
    p.add_argument("--output",      default=OUTPUT_VIDEO)
    p.add_argument("--yolo-model",  default=YOLO_WEIGHTS,
                   help="Path to YOLO11s weights (.pt) for object detection")
    p.add_argument("--lane-model",  default=LANE_WEIGHTS,
                   help="Path to ResNet34 Attention U-Net weights (.pth) for lane detection")
    p.add_argument("--conf",        type=float, default=CONF_THRESHOLD)
    p.add_argument("--iou",         type=float, default=IOU_THRESHOLD)
    p.add_argument("--benchmark-frames", type=int, default=0,
                   help="Process only the first N frames for runtime benchmarking. "
                        "Use 0 to process the full video.")
    p.add_argument("--no-output-video", action="store_true",
                   help="Skip writing the rendered output video. Useful for measuring "
                        "pipeline speed without MP4 encoding overhead.")

    # ── Monocular distance / TTC calibration ──────────────────────────────
    p.add_argument("--cam-height",  type=float, default=1.4,
                   help="Camera mount height above ground in metres "
                        "(default: 1.4 — typical sedan dashboard).")
    p.add_argument("--cam-hfov",    type=float, default=75.0,
                   help="Horizontal field of view in degrees "
                        "(default: 75 — mid-range dashcam).")
    p.add_argument("--cam-horizon", type=float, default=0.50,
                   help="Horizon row as fraction of frame height "
                        "(default: 0.50 — level mount, flat road).")

    # ── Custom calibration file ───────────────────────────────────────────
    p.add_argument("--calib-file",  type=str, default=None,
                   help="Path to calibration.npz (from calibration_tool.py). "
                        "Auto-detected if calibration.npz exists in project root.")

    args, unknown = p.parse_known_args()
    return args

def open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): sys.exit(f"[ERROR] Cannot open video: {path}")
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, w, h, fps, total

def open_writer(path: str, w: int, h: int, fps: float):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    return writer

def sync_cuda():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()

# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if not Path(args.input).exists():
        sys.exit(f"[ERROR] Missing input video: {args.input}")
    if not Path(args.yolo_model).exists():
        sys.exit(f"[ERROR] Missing YOLO weights: {args.yolo_model}")
    if not Path(args.lane_model).exists():
        sys.exit(f"[ERROR] Missing ResNet34 Attention U-Net weights: {args.lane_model}")

    # ── Load dual models ──────────────────────────────────────────────────
    print("[INFO] Loading YOLO11s (object detection)...")
    yolo_model = YOLO(args.yolo_model)

    print("[INFO] Loading ResNet34 Attention U-Net IoU60 (lane detection)...")
    lane_model = UNetResNet34Wrapper(model_path=args.lane_model)

    cap, W, H, fps, total = open_video(args.input)
    writer = None if args.no_output_video else open_writer(args.output, W, H, fps)

    print(f"[INFO] Video: {W}x{H} @ {fps:.1f} fps, {total} frames")
    print(f"[INFO] Dual-model pipeline ready.")
    if args.benchmark_frames > 0:
        print(f"[INFO] Benchmark mode: processing first {args.benchmark_frames} frames only.")
    if args.no_output_video:
        print("[INFO] Output video writing disabled for benchmark.")

    # ── Camera intrinsics: auto-detect calibration file ───────────────────
    calib_path = args.calib_file
    if calib_path is None:
        # Tự tìm calibration.npz trong thư mục project
        default_calib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.npz")
        if os.path.exists(default_calib):
            calib_path = default_calib

    if calib_path and os.path.exists(calib_path):
        # Dùng custom calibration (fx/fy thật + undistort)
        intrinsics = CameraIntrinsics.from_calibration_file(
            calib_path=calib_path,
            frame_w=W,
            frame_h=H,
            h_camera=args.cam_height,
            horizon_y_frac=args.cam_horizon,
        )
        print(f"[INFO] Camera: {W}x{H}  CUSTOM CALIBRATION  "
              f"mount={args.cam_height} m  horizon={args.cam_horizon:.2f}·H "
              f"(fx={intrinsics.fx:.0f} px, fy={intrinsics.fy:.0f} px)")
    else:
        # Fallback: ước tính từ HFOV (hành vi cũ)
        intrinsics = CameraIntrinsics(
            frame_w=W,
            frame_h=H,
            hfov_deg=args.cam_hfov,
            h_camera=args.cam_height,
            horizon_y_frac=args.cam_horizon,
        )
        print(f"[INFO] Camera: {W}x{H}  HFOV={args.cam_hfov}°  "
              f"mount={args.cam_height} m  horizon={args.cam_horizon:.2f}·H "
              f"(fx≈{intrinsics.fx:.0f} px)  [no calibration file]")

    lane_pipe = LanePipeline(frame_width=W, frame_height=H)
    dept_pipe = DeparturePipeline(frame_width=W, frame_height=H)
    guid_pipe = GuidancePipeline(
        frame_width=W,
        frame_height=H,
        intrinsics=intrinsics,
        fps=fps,
    )
    hud_pipe  = HUDPipeline()

    stats = {"frames": 0, "lane_valid": 0, "departure": {}, "guidance": {}}
    timing = defaultdict(float)
    offset_samples = []          # collect raw offsets for bias calibration
    lane_loss_streak = 0         # consecutive frames with no lane data
    LANE_LOSS_RESET_FRAMES = 45  # reset EMA after ~1.5 s of no lane data
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # ── Undistort frame nếu có custom calibration ─────────────
            t0 = time.perf_counter()
            frame = intrinsics.undistort(frame)
            timing["undistort"] += time.perf_counter() - t0

            stats["frames"] += 1

            # ── YOLO: Object detection (Car, bus, truck) ──────────────────
            sync_cuda()
            t0 = time.perf_counter()
            yolo_result = yolo_model(frame, conf=args.conf, iou=args.iou, verbose=False)[0]
            sync_cuda()
            timing["yolo_detection"] += time.perf_counter() - t0

            # ── ResNet34 Attention U-Net IoU60: Lane detection ────────────
            sync_cuda()
            t0 = time.perf_counter()
            detected_lanes = lane_model.detect_lanes(frame)
            sync_cuda()
            timing["lane_segmentation"] += time.perf_counter() - t0

            # ── Module A: Lane pipeline (uses IoU60 U-Net lanes) ──────────
            t0 = time.perf_counter()
            lane_result = lane_pipe.process(frame, detected_lanes)
            timing["module_a_lane_tracking"] += time.perf_counter() - t0
            if lane_result.valid: stats["lane_valid"] += 1

            # ── Module B: Departure warning (uses lane_result) ────────────
            t0 = time.perf_counter()
            dept_result = dept_pipe.process(lane_result)
            timing["module_b_departure"] += time.perf_counter() - t0
            stats["departure"][dept_result.state] = stats["departure"].get(dept_result.state, 0) + 1

            # ── Track raw offset for bias calibration diagnostic ──────────
            if dept_result.raw_offset is not None:
                offset_samples.append(dept_result.raw_offset)
                lane_loss_streak = 0
            else:
                lane_loss_streak += 1
                # Reset EMA after prolonged lane loss so a stale negative
                # offset doesn't cause permanent WARN_RIGHT on next detection
                if lane_loss_streak >= LANE_LOSS_RESET_FRAMES:
                    dept_pipe.reset()
                    lane_loss_streak = 0

            # ── Module C: Guidance (uses YOLO vehicles + lane polys) ──────
            t0 = time.perf_counter()
            guid_result = guid_pipe.process(yolo_result, lane_result)
            timing["module_c_guidance"] += time.perf_counter() - t0
            stats["guidance"][guid_result.guidance] = stats["guidance"].get(guid_result.guidance, 0) + 1

            # ── Module D: Render & Write ──────────────────────────────────
            t0 = time.perf_counter()
            output_frame = hud_pipe.render(frame, lane_result, dept_result, guid_result)
            timing["module_d_hud_render"] += time.perf_counter() - t0

            if writer is not None:
                t0 = time.perf_counter()
                writer.write(output_frame)
                timing["video_write"] += time.perf_counter() - t0

            if stats["frames"] % LOG_EVERY_N == 0:
                avg_off = (sum(offset_samples[-LOG_EVERY_N:]) / len(offset_samples[-LOG_EVERY_N:])
                           if offset_samples else float('nan'))
                print(f" Frame {stats['frames']:>5}/{total:<5} | "
                      f"State: {dept_result.state:<15} | "
                      f"raw_offset: {dept_result.raw_offset!s:>8} | "
                      f"avg_offset(last {LOG_EVERY_N}): {avg_off:+.1f} px | "
                      f"lanes: {len(detected_lanes)}")

            if args.benchmark_frames > 0 and stats["frames"] >= args.benchmark_frames:
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        elapsed = time.time() - t_start
        total_f = max(1, stats["frames"])

        print(f"\n{'='*65}\n  ADAS Processing Complete (Dual-Model: YOLO + ResNet34 Attention U-Net IoU60)")
        print(f"  Lane valid   : {stats['lane_valid']} / {total_f} ({100*stats['lane_valid']/total_f:.1f}%)")
        print(f"  Speed        : {total_f/max(0.1, elapsed):.1f} fps")
        print(f"  Avg latency  : {1000*elapsed/total_f:.2f} ms/frame")

        print("\n  Runtime breakdown:")
        for name, seconds in sorted(timing.items()):
            ms_per_frame = 1000.0 * seconds / total_f
            print(f"    {name:<26} {ms_per_frame:>8.2f} ms/frame")
        total_ms = 1000.0 * elapsed / total_f
        print(f"    {'total_end_to_end':<26} {total_ms:>8.2f} ms/frame")
        print(f"    {'effective_fps':<26} {1000.0 / max(0.001, total_ms):>8.2f} fps")

        print("\n  Departure state breakdown:")
        if stats["departure"]:
            for s, c in sorted(stats["departure"].items(), key=lambda x: -x[1]):
                print(f"    {s:<22} {c:>6} frames ({100*c/total_f:5.1f}%)")
        else:
            print("    No data recorded.")

        print("\n  Guidance state breakdown:")
        if stats["guidance"]:
            for g, c in sorted(stats["guidance"].items(), key=lambda x: -x[1]):
                print(f"    {g:<22} {c:>6} frames ({100*c/total_f:5.1f}%)")
        else:
            print("    No data recorded.")

        # ── Bias calibration hint ───────────────────────────────────────
        if offset_samples:
            mean_off = sum(offset_samples) / len(offset_samples)
            print(f"\n  Offset diagnostic ({len(offset_samples)} frames with lane data):")
            print(f"    mean raw_offset : {mean_off:+.1f} px")
            print(f"    min  raw_offset : {min(offset_samples):+.1f} px")
            print(f"    max  raw_offset : {max(offset_samples):+.1f} px")
            if abs(mean_off) > 30:
                print(f"\n  [HINT] Systematic offset detected ({mean_off:+.1f} px).")
                print(f"         Set CAMERA_MOUNT_BIAS_PX = {mean_off:.1f} in module_b/offset_calculator.py")
                print(f"         to correct for dashcam mounting position.")
        print('='*65)

if __name__ == "__main__":
    main()
