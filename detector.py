"""
TrafficPulse - YOLOv8 detection + ByteTrack engine.
"""

from ultralytics import YOLO
import numpy as np
import cv2

VEHICLE_CLASSES = {
    2: ("car",        "#3B82F6"),
    3: ("motorcycle", "#F59E0B"),
    5: ("bus",        "#EF4444"),
    7: ("truck",      "#8B5CF6"),
    1: ("bicycle",    "#10B981"),
    0: ("person",     "#F97316"),
}

CLASS_IDS = list(VEHICLE_CLASSES.keys())

# Distinct BGR colors per class for OpenCV drawing
CLASS_BGR = {
    0: (0,   165, 249),   # person  - orange
    1: (113, 186,  16),   # bicycle - green
    2: (235, 130,  59),   # car     - blue
    3: (11,  158, 245),   # moto    - amber
    5: (63,   63, 239),   # bus     - red
    7: (177,  91, 139),   # truck   - purple
}


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement to improve detection in dark/flat areas."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


class TrafficDetector:
    def __init__(self, model_size: str = "yolov8s.pt", conf: float = 0.25,
                 imgsz: int = 1280, use_enhance: bool = True):
        self.model = YOLO(model_size)
        self.conf = conf
        self.imgsz = imgsz
        self.use_enhance = use_enhance

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run detection only (no tracking)."""
        src = enhance_frame(frame) if self.use_enhance else frame
        results = self.model(
            src,
            conf=self.conf,
            classes=CLASS_IDS,
            imgsz=self.imgsz,
            iou=0.45,
            agnostic_nms=True,
            verbose=False,
        )[0]
        return self._parse_boxes(results)

    def track(self, frame: np.ndarray) -> list[dict]:
        """Run detection + ByteTrack (assigns persistent track IDs)."""
        src = enhance_frame(frame) if self.use_enhance else frame
        results = self.model.track(
            src,
            conf=self.conf,
            classes=CLASS_IDS,
            imgsz=self.imgsz,
            iou=0.45,
            agnostic_nms=True,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )[0]
        return self._parse_boxes(results, use_track_id=True)

    def _parse_boxes(self, results, use_track_id: bool = False) -> list[dict]:
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            label, hex_color = VEHICLE_CLASSES[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if (use_track_id and box.id is not None) else None
            detections.append({
                "class_id": cls_id,
                "label":    label,
                "color":    hex_color,
                "bgr":      CLASS_BGR.get(cls_id, (255, 255, 255)),
                "conf":     float(box.conf[0]),
                "bbox":     (x1, y1, x2, y2),
                "center":   ((x1 + x2) // 2, (y1 + y2) // 2),
                "track_id": track_id,
            })
        return detections

    def annotate(self, frame: np.ndarray, detections: list[dict],
                 show_conf: bool = True, show_track_id: bool = True) -> np.ndarray:
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            bgr = d["bgr"]
            w = x2 - x1
            thickness = max(2, min(4, w // 60))

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), bgr, thickness)

            # Label text
            parts = [d["label"].upper()]
            if show_track_id and d.get("track_id") is not None:
                parts.append(f"#{d['track_id']}")
            if show_conf:
                parts.append(f"{d['conf']:.0%}")
            text = " ".join(parts)

            font_scale = max(0.45, min(0.75, w / 120))
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)

            # Label background
            pad = 4
            lx1, ly1 = x1, max(0, y1 - th - baseline - pad * 2)
            lx2, ly2 = x1 + tw + pad * 2, y1
            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), bgr, -1)

            # Dot at centroid
            cx, cy = d["center"]
            cv2.circle(out, (cx, cy), max(3, thickness + 1), bgr, -1)
            cv2.circle(out, (cx, cy), max(3, thickness + 1), (255, 255, 255), 1)

            cv2.putText(out, text, (lx1 + pad, ly2 - baseline - 1),
                        cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        return out
