from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

from .hud_colours import BOX_EGO, BOX_LEFT, BOX_RIGHT

CLASS_NAMES = {

}

EGO_BOX_COLOUR   = BOX_EGO
LEFT_BOX_COLOUR  = BOX_LEFT
RIGHT_BOX_COLOUR = BOX_RIGHT
ZONE_COLOURS = {
    "EGO":   EGO_BOX_COLOUR,
    "LEFT":  LEFT_BOX_COLOUR,
    "RIGHT": RIGHT_BOX_COLOUR,
}
TEXT_BG_ALPHA = 0.78

def draw_object_boxes(
    frame: np.ndarray,
    vehicle_detections: Iterable[Tuple],
    left_poly: Optional[np.ndarray],
    right_poly: Optional[np.ndarray],
) -> np.ndarray:

    out = frame.copy()
    h, w = out.shape[:2]

    for det in vehicle_detections:

        track_id = None
        distance_m = None
        ttc_s = None
        if len(det) >= 9:
            (x1, y1, x2, y2, conf, cls_id, track_id, distance_m, ttc_s) = det[:9]
        elif len(det) >= 7:
            (x1, y1, x2, y2, conf, cls_id, track_id) = det[:7]
        else:
            (x1, y1, x2, y2, conf, cls_id) = det[:6]

        x1_i = int(np.clip(round(x1), 0, w - 1))
        y1_i = int(np.clip(round(y1), 0, h - 1))
        x2_i = int(np.clip(round(x2), 0, w - 1))
        y2_i = int(np.clip(round(y2), 0, h - 1))
        if x2_i <= x1_i or y2_i <= y1_i:
            continue

        zone = _classify_box_zone(x1, x2, y2, left_poly, right_poly, w, h)
        if zone == "OUT":
            continue

        colour = ZONE_COLOURS[zone]
        cls_name = CLASS_NAMES.get(int(cls_id), 'OBJ')

        head = (f"#{int(track_id):02d}  {zone} {cls_name} {conf:.2f}"
                if track_id is not None
                else f"{zone} {cls_name} {conf:.2f}")
        metric_bits = []
        if distance_m is not None:
            metric_bits.append(f"{distance_m:.0f} m")
        if ttc_s is not None:
            metric_bits.append(f"TTC {ttc_s:.1f}s")

        cv2.rectangle(out, (x1_i, y1_i), (x2_i, y2_i), colour, 2, cv2.LINE_AA)
        _draw_label(out, head, x1_i, y1_i, colour)
        if metric_bits:
            _draw_metric_label(out, "  |  ".join(metric_bits),
                               x1_i, y2_i, colour)

        foot_x = int(round((x1 + x2) * 0.5))
        foot_y = int(round(y2))
        cv2.circle(out, (int(np.clip(foot_x, 0, w - 1)), int(np.clip(foot_y, 0, h - 1))),
                   3, colour, cv2.FILLED, cv2.LINE_AA)

    return out

def _classify_box_zone(
    x1: float,
    x2: float,
    y2: float,
    left_poly: Optional[np.ndarray],
    right_poly: Optional[np.ndarray],
    frame_w: int,
    frame_h: int,
) -> str:

    if left_poly is None or right_poly is None:
        return "OUT"

    foot_x = (x1 + x2) * 0.5
    foot_y = float(np.clip(y2, 0, frame_h - 1))
    left_x  = float(np.polyval(left_poly,  foot_y))
    right_x = float(np.polyval(right_poly, foot_y))

    lane_width = right_x - left_x
    if lane_width < max(50.0, frame_w * 0.04) or lane_width > frame_w * 0.75:
        return "OUT"

    margin = min(18.0, lane_width * 0.08)
    ego_left   = left_x  - margin
    ego_right  = right_x + margin
    left_outer  = left_x  - lane_width
    right_outer = right_x + lane_width

    if ego_left <= foot_x <= ego_right:
        return "EGO"
    if left_outer <= foot_x < ego_left:
        return "LEFT"
    if ego_right < foot_x <= right_outer:
        return "RIGHT"
    return "OUT"

def _draw_label(
    frame: np.ndarray,
    label: str,
    x: int,
    y: int,
    colour: Tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    text_size, baseline = cv2.getTextSize(label, font, scale, thick)
    tw, th = text_size
    pad_x = 5
    pad_y = 4
    y_top = max(0, y - th - baseline - pad_y * 2)
    x_right = min(frame.shape[1] - 1, x + tw + pad_x * 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y_top), (x_right, y), colour, cv2.FILLED)
    cv2.addWeighted(overlay, TEXT_BG_ALPHA, frame, 1.0 - TEXT_BG_ALPHA, 0, frame)
    cv2.putText(
        frame,
        label,
        (x + pad_x, max(th + pad_y, y - baseline - pad_y)),
        font,
        scale,
        (255, 255, 255),
        thick,
        cv2.LINE_AA,
    )

def _draw_metric_label(
    frame: np.ndarray,
    label: str,
    x: int,
    y_bottom: int,
    colour: Tuple[int, int, int],
) -> None:

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thick = 1
    text_size, baseline = cv2.getTextSize(label, font, scale, thick)
    tw, th = text_size
    pad_x = 4
    pad_y = 3
    y_top = y_bottom + 1
    y_bot = min(frame.shape[0] - 1, y_top + th + baseline + pad_y * 2)
    x_right = min(frame.shape[1] - 1, x + tw + pad_x * 2)

    overlay = frame.copy()
    dark = tuple(int(c * 0.25) for c in colour)
    cv2.rectangle(overlay, (x, y_top), (x_right, y_bot), dark, cv2.FILLED)
    cv2.rectangle(overlay, (x, y_top), (x_right, y_bot), colour, 1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(
        frame,
        label,
        (x + pad_x, y_bot - baseline - pad_y + th // 2 + 1),
        font,
        scale,
        (240, 245, 250),
        thick,
        cv2.LINE_AA,
    )
