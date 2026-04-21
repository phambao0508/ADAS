import argparse
import sys
import os
from pathlib import Path
import cv2
import numpy as np

# ── Suppress ultralytics progress bars ──────────────────────────────────
os.environ["YOLO_VERBOSE"] = "False"

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] ultralytics not installed. Run: pip install ultralytics")

from module_a import LanePipeline
from module_b import DeparturePipeline
from module_c import GuidancePipeline
from module_d import HUDPipeline

# ─────────────────────────────────────────────────────────────────────────────
# ★ CONFIGURATION ★
# ─────────────────────────────────────────────────────────────────────────────
INPUT_IMAGE   = r"D:\B3-ICT\ADAS\Screenshot 2026-04-11 091830.png"
OUTPUT_IMAGE  = r"D:\B3-ICT\ADAS\output_result.jpg"
MODEL_WEIGHTS = r"D:\B3-ICT\ADAS\best (1).pt"

def parse_args():
    p = argparse.ArgumentParser(description="ADAS single image test")
    p.add_argument("--input",  default=INPUT_IMAGE)
    p.add_argument("--output", default=OUTPUT_IMAGE)
    p.add_argument("--model",  default=MODEL_WEIGHTS)
    p.add_argument("--conf",   type=float, default=0.35)
    p.add_argument("--iou",    type=float, default=0.45)
    args, _ = p.parse_known_args()
    return args

def main():
    args = parse_args()

    # 1. Kiểm tra file
    if not Path(args.input).exists():
        sys.exit(f"[ERROR] Không tìm thấy ảnh đầu vào: {args.input}")

    # 2. Load Model & Image
    model = YOLO(args.model)
    frame = cv2.imread(args.input)
    if frame is None:
        sys.exit(f"[ERROR] Không thể đọc ảnh: {args.input}")

    H, W, _ = frame.shape

    # 3. Khởi tạo các module ADAS
    lane_pipe = LanePipeline(frame_width=W, frame_height=H)
    dept_pipe = DeparturePipeline(frame_width=W, frame_height=H)
    guid_pipe = GuidancePipeline(frame_width=W, frame_height=H)
    hud_pipe  = HUDPipeline()

    # 4. Chạy Pipeline (Chỉ 1 lần duy nhất)
    print(f"--- Đang xử lý ảnh: {Path(args.input).name} ---")

    # Chạy YOLO
    yolo_result = model(frame, conf=args.conf, iou=args.iou, verbose=False)[0]

    # Module A: Lane detection
    lane_result = lane_pipe.process(frame, yolo_result)

    # Module B: Departure warning
    dept_result = dept_pipe.process(lane_result)

    # Module C: Guidance
    guid_result = guid_pipe.process(yolo_result, lane_result)

    # Module D: HUD Rendering
    output_frame = hud_pipe.render(frame, lane_result, dept_result, guid_result)

    # 5. Lưu kết quả
    cv2.imwrite(args.output, output_frame)
    print(f"--- Hoàn tất! Kết quả được lưu tại: {args.output} ---")

    # Hiển thị log nhanh
    print(f"\nKết quả chẩn đoán:")
    print(f" - Làn đường: {'Hợp lệ' if lane_result.valid else 'Không tìm thấy'}")
    print(f" - Trạng thái: {dept_result.state}")
    print(f" - Chỉ dẫn: {guid_result.guidance}")
    if dept_result.raw_offset is not None:
        print(f" - Độ lệch (Offset): {dept_result.raw_offset:.1f} px")

if __name__ == "__main__":
    main()