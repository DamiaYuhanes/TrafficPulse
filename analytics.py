"""
Traffic analytics: density scoring, heatmap, congestion level.
"""

import numpy as np
import cv2
from collections import deque


CONGESTION_LEVELS = [
    (0,  10, "Free Flow",    "#10B981"),
    (10, 25, "Light",        "#84CC16"),
    (25, 50, "Moderate",     "#F59E0B"),
    (50, 80, "Heavy",        "#EF4444"),
    (80, 999,"Gridlock",     "#7F1D1D"),
]


def congestion_level(vehicle_count: int) -> tuple[str, str]:
    for lo, hi, label, color in CONGESTION_LEVELS:
        if lo <= vehicle_count < hi:
            return label, color
    return "Gridlock", "#7F1D1D"


class HeatmapAccumulator:
    def __init__(self, height: int, width: int, decay: float = 0.97):
        self.heatmap = np.zeros((height, width), dtype=np.float32)
        self.decay = decay

    def update(self, detections: list[dict]):
        self.heatmap *= self.decay
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(self.heatmap, (cx, cy), 30, 1.0, -1)

    def render(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        normalized = np.clip(self.heatmap / (self.heatmap.max() + 1e-6), 0, 1)
        heat_uint8 = (normalized * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        mask = heat_uint8 > 10
        output = frame.copy()
        output[mask] = cv2.addWeighted(frame, 1 - alpha, heat_color, alpha, 0)[mask]
        return output


class TrafficTimeline:
    def __init__(self, window: int = 300):
        self.window = window
        self.counts: deque[int] = deque(maxlen=window)
        self.timestamps: deque[float] = deque(maxlen=window)

    def record(self, count: int, ts: float):
        self.counts.append(count)
        self.timestamps.append(ts)

    def as_lists(self) -> tuple[list, list]:
        return list(self.timestamps), list(self.counts)
