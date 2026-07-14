import os
from typing import List, Tuple, Optional, Dict
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


# ── Attention Gate ─────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 16)

        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, gate, skip):
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attention = self.psi(self.gate_proj(gate) + self.skip_proj(skip))
        return skip * attention


# ── Decoder Block with optional Attention Gate ─────────────────────────────

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True):
        super().__init__()
        self.attention = AttentionGate(in_channels, skip_channels) if use_attention and skip_channels > 0 else None
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            if self.attention is not None:
                skip = self.attention(x, skip)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# ── ResNet34 Attention UNet Architecture (must match training notebook) ────

class ResNet34AttentionUNet(nn.Module):
    """U-Net decoder with ImageNet-pretrained ResNet-34 encoder,
    attention gates, and lane-existence head."""
    def __init__(self, out_channels=1, num_lanes=4, pretrained=False):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = resnet34(weights=weights)

        self.input_block = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)  # 1/2, 64 channels
        self.maxpool = encoder.maxpool
        self.enc1 = encoder.layer1  # 1/4, 64 channels
        self.enc2 = encoder.layer2  # 1/8, 128 channels
        self.enc3 = encoder.layer3  # 1/16, 256 channels
        self.enc4 = encoder.layer4  # 1/32, 512 channels

        self.center = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)

        self.mask_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

        self.exist_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_lanes),
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        x0 = self.input_block(x)
        x1 = self.enc1(self.maxpool(x0))
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        exist_logits = self.exist_head(x4)

        x = self.center(x4)
        x = self.dec4(x, x3)
        x = self.dec3(x, x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, x0)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        mask_logits = self.mask_head(x)
        return mask_logits, exist_logits


# ── Wrapper (same API as UNetWrapper) ──────────────────────────────────────

class UNetResNet34Wrapper:
    IMG_HEIGHT = 384
    IMG_WIDTH = 1024

    # ImageNet normalization (ResNet34 encoder was pretrained with these)
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Lane configuration
    NUM_LANES = 4
    LANE_NAMES = ["left-left", "left", "right", "right-right"]

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        threshold: Optional[float] = None,
        existence_threshold: Optional[float] = None,
        min_lane_pixels: int = 35,
        num_sample_points: int = 72,
        poly_degree: int = 2,
        temporal_alpha: float = 0.5,
        temporal_history: int = 5,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_lane_pixels = min_lane_pixels
        self.num_sample_points = num_sample_points
        self.poly_degree = poly_degree
        self.temporal_alpha = temporal_alpha

        # ── Load checkpoint ───────────────────────────────────────────────
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            self.threshold = threshold if threshold is not None else float(checkpoint.get("best_threshold", 0.5))
            self.existence_threshold = (
                existence_threshold if existence_threshold is not None
                else float(checkpoint.get("existence_threshold", 0.5))
            )
            num_lanes = int(checkpoint.get("num_lanes", self.NUM_LANES))
            self.IMG_HEIGHT = int(checkpoint.get("img_height", self.IMG_HEIGHT))
            self.IMG_WIDTH = int(checkpoint.get("img_width", self.IMG_WIDTH))
        else:
            state_dict = checkpoint
            self.threshold = threshold if threshold is not None else 0.5
            self.existence_threshold = existence_threshold if existence_threshold is not None else 0.5
            num_lanes = self.NUM_LANES

        # ── Handle torch.compile _orig_mod. prefix ────────────────────────
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                clean_state_dict[k[len("_orig_mod."):]] = v
            else:
                clean_state_dict[k] = v

        # ── Build & load model ────────────────────────────────────────────
        self.net = ResNet34AttentionUNet(out_channels=1, num_lanes=num_lanes, pretrained=False)
        self.net.load_state_dict(clean_state_dict)
        self.net.to(self.device)
        self.net.eval()

        # ── Morphology kernels (reused every frame) ──────────────────────
        # Tall close kernel bridges vertical gaps in dashed markings
        self._close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 9)
        )
        self._open_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 3)
        )

        # ── Temporal smoothing state ─────────────────────────────────────
        # Each entry: list of (coeff, y_min, y_max, bottom_x) per lane
        self._prev_lanes: List[Dict] = []
        self._frame_count = 0

    # ══════════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════════════

    def detect_lanes(
        self, frame: np.ndarray
    ) -> List[List[Tuple[int, int]]]:
        ori_h, ori_w = frame.shape[:2]
        self._frame_count += 1

        # ── 1. Preprocess (ImageNet normalization) ────────────────────────
        img = cv2.resize(frame, (self.IMG_WIDTH, self.IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        # ── 2. Inference (dual-head) ─────────────────────────────────────
        with torch.no_grad():
            mask_logits, exist_logits = self.net(img_tensor)

        mask_prob = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
        exist_prob = torch.sigmoid(exist_logits).squeeze().cpu().numpy()

        mask = (mask_prob > self.threshold).astype(np.uint8)

        # ── 3. Morphological cleanup ─────────────────────────────────────
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)

        # ── 4. Row-tracking → raw lane tracks ────────────────────────────
        raw_tracks = self._track_row_clusters(mask, mask_prob)

        # ── 5. Polynomial curve fitting per track ────────────────────────
        fitted_lanes = self._fit_and_filter_tracks(raw_tracks, exist_prob)

        # ── 6. Temporal EMA smoothing ────────────────────────────────────
        smoothed_lanes = self._temporal_smooth(fitted_lanes)

        # ── 7. Resample & rescale to original resolution ─────────────────
        scale_x = ori_w / self.IMG_WIDTH
        scale_y = ori_h / self.IMG_HEIGHT

        lanes = []
        for lane_info in smoothed_lanes:
            points = self._resample_from_poly(
                lane_info["coeff"],
                lane_info["y_min"],
                lane_info["y_max"],
                scale_x, scale_y,
            )
            if len(points) >= 2:
                lanes.append(points)

        return lanes

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 4: ROW-BY-ROW TRACKING
    # ══════════════════════════════════════════════════════════════════════

    def _row_clusters(
        self, mask: np.ndarray, prob: np.ndarray, row: int
    ) -> List[float]:
        xs = np.flatnonzero(mask[row] > 0)
        if len(xs) == 0:
            return []

        breaks = np.where(np.diff(xs) > 1)[0] + 1
        centres = []
        for run in np.split(xs, breaks):
            if len(run) < 1 or len(run) > 50:
                continue
            # Probability-weighted centroid (smoother than binary median)
            weights = prob[row, run]
            w_sum = weights.sum()
            if w_sum > 1e-6:
                cx = float(np.dot(run.astype(np.float64), weights) / w_sum)
            else:
                cx = float(np.median(run))
            centres.append(cx)
        return centres

    def _predict_track_x(self, track: dict, y: int) -> float:
        pts = track["points"]
        if len(pts) < 6:
            return float(track["last_x"])

        recent = pts[-20:]
        ys = np.array([p[1] for p in recent], dtype=np.float64)
        xs = np.array([p[0] for p in recent], dtype=np.float64)
        if np.ptp(ys) < 3:
            return float(track["last_x"])

        try:
            slope, intercept = np.polyfit(ys, xs, 1)
            return float(slope * y + intercept)
        except (np.linalg.LinAlgError, ValueError):
            return float(track["last_x"])

    def _track_row_clusters(
        self, mask: np.ndarray, prob: np.ndarray
    ) -> List[dict]:
        tracks: List[dict] = []
        next_id = 0
        max_gap_rows = 28        # max vertical gap before a track is closed
        horizon_row = int(self.IMG_HEIGHT * 0.28)  # don't track above horizon

        for y in range(self.IMG_HEIGHT - 1, horizon_row - 1, -1):
            clusters = self._row_clusters(mask, prob, y)
            if not clusters:
                continue

            # Find active tracks (recently updated, above current row)
            active = [
                t for t in tracks
                if y <= t["last_y"] and (t["last_y"] - y) <= max_gap_rows
            ]

            # Greedy nearest-neighbor matching
            pairs = []
            for ti, track in enumerate(active):
                pred_x = self._predict_track_x(track, y)
                for ci, cx in enumerate(clusters):
                    pairs.append((abs(pred_x - cx), ti, ci))

            pairs.sort(key=lambda item: item[0])
            used_tracks = set()
            used_clusters = set()
            for dist, active_idx, cluster_idx in pairs:
                if active_idx in used_tracks or cluster_idx in used_clusters:
                    continue
                track = active[active_idx]
                # Tolerance increases toward top (perspective effect)
                tolerance = 18.0 + 20.0 * (1.0 - y / max(1, self.IMG_HEIGHT - 1))
                if dist > tolerance:
                    continue
                cx = clusters[cluster_idx]
                track["points"].append((cx, float(y)))
                track["last_x"] = cx
                track["last_y"] = y
                used_tracks.add(active_idx)
                used_clusters.add(cluster_idx)

            # Unmatched clusters start new tracks
            for ci, cx in enumerate(clusters):
                if ci in used_clusters:
                    continue
                tracks.append({
                    "id": next_id,
                    "points": [(cx, float(y))],
                    "last_x": cx,
                    "last_y": y,
                })
                next_id += 1

        return tracks

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 5: POLYNOMIAL FITTING + EXISTENCE FILTERING
    # ══════════════════════════════════════════════════════════════════════

    def _fit_and_filter_tracks(
        self,
        tracks: List[dict],
        exist_prob: np.ndarray,
    ) -> List[Dict]:
        min_track_points = 12
        min_y_span = 25
        max_rmse = 16.0

        candidates = []
        for track in tracks:
            pts = track["points"]
            if len(pts) < min_track_points:
                continue

            ys = np.array([p[1] for p in pts], dtype=np.float64)
            xs = np.array([p[0] for p in pts], dtype=np.float64)
            y_span = float(ys.max() - ys.min())
            if y_span < min_y_span:
                continue

            # Fit polynomial x = f(y)
            try:
                coeff = np.polyfit(ys, xs, self.poly_degree)
                fitted = np.polyval(coeff, ys)
                rmse = float(np.sqrt(np.mean((fitted - xs) ** 2)))
            except (np.linalg.LinAlgError, ValueError):
                continue

            if rmse > max_rmse:
                continue

            bottom_x = float(np.polyval(coeff, ys.max()))
            # Score: prefer long tracks with low fitting error
            score = y_span + 0.4 * len(pts) - 2.0 * rmse

            candidates.append({
                "coeff": coeff,
                "y_min": float(ys.min()),
                "y_max": float(ys.max()),
                "bottom_x": bottom_x,
                "score": score,
            })

        if not candidates:
            return []

        # ── Existence-guided lane count ──────────────────────────────────
        exist_count = int((exist_prob > self.existence_threshold).sum())
        # If mask produced tracks but existence head says 0, keep at least 1
        if exist_count == 0 and len(candidates) > 0:
            exist_count = 1
        # Use max of existence count and geometric count (existence can undercount
        # on dashcam video with partial lane visibility)
        keep_n = min(max(exist_count, 1), self.NUM_LANES, len(candidates))

        # Sort by score descending, keep top-N
        candidates.sort(key=lambda c: c["score"], reverse=True)
        selected = candidates[:keep_n]

        # Sort left-to-right by bottom_x
        selected.sort(key=lambda c: c["bottom_x"])

        return selected

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 6: TEMPORAL EMA SMOOTHING
    # ══════════════════════════════════════════════════════════════════════

    def _temporal_smooth(
        self, current_lanes: List[Dict]
    ) -> List[Dict]:
        alpha = self.temporal_alpha

        if not self._prev_lanes or not current_lanes:
            # First frame or no lanes: no smoothing possible
            self._prev_lanes = current_lanes
            return current_lanes

        # Match current lanes to previous by bottom_x
        smoothed = []
        used_prev = set()

        for cur in current_lanes:
            best_dist = float('inf')
            best_idx = -1
            for pi, prev in enumerate(self._prev_lanes):
                if pi in used_prev:
                    continue
                dist = abs(cur["bottom_x"] - prev["bottom_x"])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = pi

            # Match threshold: lanes shouldn't jump more than ~60px between frames
            if best_idx >= 0 and best_dist < 60.0:
                prev = self._prev_lanes[best_idx]
                used_prev.add(best_idx)

                # Blend polynomial coefficients
                prev_coeff = prev["coeff"]
                cur_coeff = cur["coeff"]

                # Pad to same length if degrees differ
                max_len = max(len(prev_coeff), len(cur_coeff))
                pc = np.zeros(max_len)
                cc = np.zeros(max_len)
                pc[-len(prev_coeff):] = prev_coeff
                cc[-len(cur_coeff):] = cur_coeff

                blended_coeff = alpha * cc + (1.0 - alpha) * pc

                smoothed.append({
                    "coeff": blended_coeff,
                    "y_min": alpha * cur["y_min"] + (1.0 - alpha) * prev["y_min"],
                    "y_max": alpha * cur["y_max"] + (1.0 - alpha) * prev["y_max"],
                    "bottom_x": float(np.polyval(blended_coeff, cur["y_max"])),
                    "score": cur["score"],
                })
            else:
                # New lane, no previous match → no smoothing
                smoothed.append(cur)

        self._prev_lanes = smoothed
        return smoothed

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 7: RESAMPLE FROM POLYNOMIAL
    # ══════════════════════════════════════════════════════════════════════

    def _resample_from_poly(
        self,
        coeff: np.ndarray,
        y_min: float,
        y_max: float,
        scale_x: float,
        scale_y: float,
    ) -> List[Tuple[int, int]]:
        sample_ys = np.linspace(y_min, y_max, self.num_sample_points)
        sample_xs = np.polyval(coeff, sample_ys)

        max_x = self.IMG_WIDTH * scale_x - 1
        max_y = self.IMG_HEIGHT * scale_y - 1

        points = []
        for x, y in zip(sample_xs, sample_ys):
            orig_x = int(np.clip(round(float(x) * scale_x), 0, max_x))
            orig_y = int(np.clip(round(float(y) * scale_y), 0, max_y))
            points.append((orig_x, orig_y))

        return points
