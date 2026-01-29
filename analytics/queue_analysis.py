"""
Queue analysis: count waiting vehicles in a ROI using tracking IDs.
Waiting = low speed (below threshold). Queue length & density computed.
Reset when signal turns GREEN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


@dataclass
class QueueMetrics:
    queue_length: int
    queue_density: float
    queue_region_area: float  # normalized or pixel area


class QueueAnalyzer:
    """
    Queue ROI is [x1, y1, x2, y2] normalized (0–1). We convert to pixel coords
    using frame size. Vehicles with center inside ROI and speed below threshold
    are "waiting". We use unique track IDs to avoid double-counting.
    """

    def __init__(
        self,
        roi_normalized: List[float] | None = None,
        speed_threshold: float = cfg.SPEED_THRESHOLD_PX_PER_FRAME,
    ):
        self.roi_n = roi_normalized or cfg.QUEUE_ROI_NORMALIZED
        self.speed_threshold = speed_threshold
        self._waiting_ids: set[int] = set()

    def set_signal_state(self, state: str) -> None:
        """Reset queue state when signal turns GREEN."""
        if state.upper() == "GREEN":
            self._waiting_ids.clear()

    def update(
        self,
        tracks: List[Tuple[int, float, float, float, float, float]],  # id, x1,y1,x2,y2, speed
        frame_shape: Tuple[int, int],
    ) -> QueueMetrics:
        """
        tracks: from tracker, with per-id speed (px/frame).
        Returns QueueMetrics (queue_length, density, area).
        """
        h, w = frame_shape[:2]
        x1 = int(self.roi_n[0] * w)
        y1 = int(self.roi_n[1] * h)
        x2 = int(self.roi_n[2] * w)
        y2 = int(self.roi_n[3] * h)
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            area = 1.0

        waiting = set()
        for (tid, x1_, y1_, x2_, y2_, speed) in tracks:
            if tid < 0:
                continue
            cx = (x1_ + x2_) / 2.0
            cy = (y1_ + y2_) / 2.0
            in_roi = x1 <= cx <= x2 and y1 <= cy <= y2
            if in_roi and speed <= self.speed_threshold:
                waiting.add(tid)
        self._waiting_ids = waiting
        length = len(waiting)
        density = length / area if area else 0.0
        return QueueMetrics(queue_length=length, queue_density=density, queue_region_area=float(area))

    def get_queue_roi_pixels(self, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) in pixel coords for drawing."""
        h, w = frame_shape[:2]
        x1 = int(self.roi_n[0] * w)
        y1 = int(self.roi_n[1] * h)
        x2 = int(self.roi_n[2] * w)
        y2 = int(self.roi_n[3] * h)
        return x1, y1, x2, y2
