"""
Virtual line crossing counter using ByteTrack IDs.
"""

from collections import defaultdict


class LineCrossingCounter:
    """Count unique track IDs crossing a horizontal virtual line."""

    def __init__(self, line_y: int):
        self.line_y = line_y
        self.prev_y: dict[int, int] = {}
        self.crossed: set[int] = set()
        self.counts: dict[str, int] = defaultdict(int)
        self.total = 0

    def update(self, detections: list[dict]) -> list[str]:
        events = []
        for d in detections:
            tid = d.get("track_id")
            if tid is None:
                continue
            cx, cy = d["center"]
            prev = self.prev_y.get(tid)
            if prev is not None and tid not in self.crossed:
                if (prev < self.line_y <= cy) or (prev > self.line_y >= cy):
                    self.crossed.add(tid)
                    self.counts[d["label"]] += 1
                    self.total += 1
                    events.append(d["label"])
            self.prev_y[tid] = cy
        return events
