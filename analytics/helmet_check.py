"""
Helmet check for bike/motorcycle riders (placeholder).
Requires person + helmet detection (e.g. custom YOLO or classifier).
COCO does not include helmet class; integrate a custom model for production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class HelmetCheckResult:
    """Result for one rider: track_id of bike, track_id of person (if any), helmet_detected."""
    bike_track_id: int
    person_track_id: Optional[int]
    helmet_detected: bool
    frame_idx: int
    note: str = ""


def check_helmets(
    frame_shape: Tuple[int, int],
    vehicle_tracks: List[Tuple[int, float, float, float, float, int]],  # tid, x1,y1,x2,y2, class_id
    person_tracks: Optional[List[Tuple[int, float, float, float, float]]] = None,
    frame_idx: int = 0,
) -> List[HelmetCheckResult]:
    """
    Placeholder: would pair person detections with motorcycle/bicycle and run helmet classifier.
    COCO has no helmet class; use a custom model (e.g. helmet detection) for real checks.
    Returns empty list until a helmet model is integrated.
    """
    # person_tracks: (tid, x1, y1, x2, y2) from person detector
    # vehicle_tracks: from our tracker with class_id (3=motorcycle, 1=bicycle)
    # TODO: overlap person bbox with bike bbox -> rider; run helmet classifier on person crop
    return []
