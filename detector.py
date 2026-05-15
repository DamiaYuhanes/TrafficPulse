"""
TrafficPulse - YOLOv8-powered vehicle detection engine.
"""

from ultralytics import YOLO
import numpy as np
import cv2

VEHICLE_CLASSES = {
    2: ("car", "#3B82F6"),
    3: ("motorcycle", "#F59E0B"),
    5: ("bus", "#EF4444"),
    7: ("truck", "#8B5CF6"),
    1: ("bicycle", "#10B981"),
    0: ("person", "#F97316"),
}

CLASS_IDS = list(VEHICLE_CLASSES.keys())


class TrafficDetector:
    def __init__(self, model_size: str = "yolov8n.pt", conf: float = 0.4):
        self.model = YOLO(model_size)
        self.conf = conf

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self.model(frame, conf=self.conf, classes=CLASS_IDS, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            label, color = VEHICLE_CLASSES[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class_id": cls_id,
                "label": label,
                "color": color,
                "conf": float(box.conf[0]),
                "bbox": (x1, y1, x2, y2),
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
            })
        return detections

    def annotate(self, frame: np.ndarray, detections: list[dict], show_conf: bool = True) -> np.ndarray:
        annotated = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            hex_color = d["color"].lstrip("#")
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            bgr = (b, g, r)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)
            label = f"{d['label']} {d['conf']:.0%}" if show_conf else d["label"]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), bgr, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return annotated
