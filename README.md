# ADAS Vision Pipeline

This project implements a camera-based Advanced Driver Assistance System (ADAS) prototype for dashcam video. It combines YOLO11s vehicle detection with ResNet34 Attention U-Net lane segmentation, then converts the perception outputs into lane departure warnings, front-vehicle guidance, lane occupancy information, and a rendered HUD video.

## Objectives

- Detect surrounding vehicles: `car`, `bus`, and `truck`.
- Segment lane markings and estimate the ego lane from dashcam frames.
- Provide Lane Departure Warning (LDW) from lateral lane offset.
- Estimate front, left, and right lane occupancy for guidance decisions.
- Render an output HUD with lane fill, vehicle boxes, warning status, guidance banners, minimap, and telemetry.

## System Architecture

The end-to-end pipeline is implemented in `main.py`. Each frame is processed by two vision models followed by four rule-based modules.

| Component | Role |
| --- | --- |
| YOLO11s | Detects `car`, `bus`, and `truck` objects |
| ResNet34 Attention U-Net | Produces lane segmentation masks |
| `module_a` | Extracts lane boundaries, selects ego lane, fits lane curves, and tracks lanes |
| `module_b` | Computes lateral offset, smooths it with EMA, and classifies LDW state |
| `module_c` | Estimates distance, checks lane occupancy, and decides driving guidance |
| `module_d` | Renders lane overlays, vehicle boxes, HUD panels, banners, and minimap |

## Project Structure

```text
.
├── main.py                         # End-to-end video processing pipeline
├── calibration_tool.py             # Optional camera calibration utility
├── requirements.txt                # Python dependencies
├── states.py                       # Shared state definitions
├── unet_resnet34_wrapper.py        # ResNet34 Attention U-Net inference wrapper
├── unet_vanilla_wrapper.py         # Alternative U-Net wrapper
├── unet_wrapper.py                 # Alternative U-Net wrapper
├── model/                          # Model weights
├── module_a/                       # Lane detection and ego-lane tracking
├── module_b/                       # Lane departure warning
├── module_c/                       # Guidance and vehicle awareness
├── module_d/                       # HUD rendering
├── tests/                          # Unit tests for rule-based modules
├── report/                         # Thesis report and figures
└── test_video/                     # Sample input videos
```

## Installation

Use a Python virtual environment.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

For GPU inference, install the PyTorch build that matches your CUDA version.

## Required Runtime Files

The default `main.py` configuration expects:

```text
model/best_yolo.pt
model/resnet34_attention_unet_iou60_try.pth
test_video/test.mp4
```

The default output is:

```text
output.mp4
```

You can override all runtime paths from the command line.

## Usage

Run with default paths:

```bash
python main.py
```

Run with explicit input, output, and model paths:

```bash
python main.py ^
  --input test_video/test.mp4 ^
  --output output.mp4 ^
  --yolo-model model/best_yolo.pt ^
  --lane-model model/resnet34_attention_unet_iou60_try.pth
```

Run a short benchmark without writing an output video:

```bash
python main.py --benchmark-frames 300 --no-output-video
```

Run with a calibration file:

```bash
python main.py --calib-file calibration.npz
```

## Reported Results

YOLO11s vehicle detection results:

| Metric | Value |
| --- | ---: |
| Precision | 0.8981 |
| Recall | 0.8656 |
| F1-score | 0.8816 |
| mAP@50 | 0.9207 |
| mAP@50-95 | 0.7860 |

Best ResNet34 Attention U-Net lane segmentation results on CULane:

| Metric | Value |
| --- | ---: |
| Best Epoch | 19 |
| Validation Loss | 0.4340 |
| Validation IoU | 0.5236 |
| Validation Precision | 0.6954 |
| Validation Recall | 0.6795 |
| Validation F1-score | 0.6873 |
| Best Threshold | 0.90 |
| Lane Existence Accuracy | 0.9417 |
| Lane Count MAE | 0.231 |

End-to-end runtime measured on a 1920x1080 video with 1,562 frames using an NVIDIA GTX 1650:

| Component | Average Time |
| --- | ---: |
| YOLO11s vehicle detection | 22.12 ms/frame |
| ResNet34 Attention U-Net lane segmentation | 102.20 ms/frame |
| Module A: Lane tracking | 0.86 ms/frame |
| Module B: Lane departure warning | 0.23 ms/frame |
| Module C: Guidance decision | 1.47 ms/frame |
| Module D: HUD rendering | 123.14 ms/frame |
| Total end-to-end latency | 269.37 ms/frame |
| Effective speed | 3.71 FPS |

## Limitations

- Lane segmentation is still sensitive to night scenes, shadows, faded markings, curves, dense traffic, and missing lane markings.
- The current lane model does not classify solid and dashed lane markings, so lane-change guidance is handled conservatively.
- Monocular distance and TTC estimation are approximate and depend on calibration, bounding box quality, vehicle assumptions, and camera mounting angle.
- The current implementation is an offline video-processing prototype and is not yet optimized for embedded real-time deployment.

## Tests

Run unit tests with:

```bash
pytest
```

The tests cover rule-based components such as departure classification, EMA smoothing, guidance decision, proximity detection, lane handling, and vehicle tracking.

## Future Work

- Add solid/dashed lane marking classification.
- Improve lane robustness for night, shadow, crossroad, no-line, and crowded scenarios.
- Improve tracking, distance estimation, and TTC with road-plane estimation or optical flow.
- Optimize deployment with ONNX, TensorRT, quantization, or model compression.
