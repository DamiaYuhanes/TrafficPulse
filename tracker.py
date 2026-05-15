"""
Centroid-based multi-object tracker with virtual line crossing counter.
"""

import numpy as np
from collections import defaultdict, OrderedDict


class CentroidTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: int = 80):
        self.next_id = 0
        self.objects: OrderedDict[int, np.ndarray] = OrderedDict()
        self.disappeared: dict[int, int] = {}
        self.labels: dict[int, str] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: tuple, label: str):
        self.objects[self.next_id] = np.array(centroid)
        self.disappeared[self.next_id] = 0
        self.labels[self.next_id] = label
        self.next_id += 1

    def deregister(self, obj_id: int):
        del self.objects[obj_id]
        del self.disappeared[obj_id]
        del self.labels[obj_id]

    def update(self, detections: list[dict]) -> dict[int, tuple]:
        if not detections:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = np.array([d["center"] for d in detections])
        input_labels = [d["label"] for d in detections]

        if not self.objects:
            for c, lbl in zip(input_centroids, input_labels):
                self.register(tuple(c), lbl)
        else:
            obj_ids = list(self.objects.keys())
            obj_centroids = np.array(list(self.objects.values()))
            D = np.linalg.norm(obj_centroids[:, None] - input_centroids[None, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows, used_cols = set(), set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                obj_id = obj_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            for row in set(range(len(obj_ids))) - used_rows:
                obj_id = obj_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            for col in set(range(len(input_centroids))) - used_cols:
                self.register(tuple(input_centroids[col]), input_labels[col])

        return self.objects


class LineCrossingCounter:
    """Count objects crossing a horizontal virtual line."""

    def __init__(self, line_y: int):
        self.line_y = line_y
        self.prev_positions: dict[int, int] = {}
        self.crossed_ids: set[int] = set()
        self.counts: dict[str, int] = defaultdict(int)
        self.total = 0

    def update(self, tracked: dict, tracker: CentroidTracker) -> list[str]:
        events = []
        for obj_id, centroid in tracked.items():
            cy = int(centroid[1])
            prev_cy = self.prev_positions.get(obj_id)
            if prev_cy is not None and obj_id not in self.crossed_ids:
                if prev_cy < self.line_y <= cy or prev_cy > self.line_y >= cy:
                    self.crossed_ids.add(obj_id)
                    label = tracker.labels.get(obj_id, "vehicle")
                    self.counts[label] += 1
                    self.total += 1
                    events.append(label)
            self.prev_positions[obj_id] = cy
        return events
