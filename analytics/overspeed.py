"""
Overspeed detection using pixel-to-meter calibration.
Converts per-track speed (pixels/frame) to km/h and flags vehicles exceeding limit.
"""

from __future__ import annotations

from typing import Dict, Set

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


def px_per_frame_to_kmh(
    speed_px_per_frame: float,
    process_fps: float,
    pixels_per_meter: float = cfg.PIXELS_PER_METER,
) -> float:
    """
    Convert speed from pixels/frame to km/h.
    speed_px_per_frame * process_fps = pixels per second;
    (pixels/sec) / pixels_per_meter = m/s; m/s * 3.6 = km/h.
    """
    if pixels_per_meter <= 0 or process_fps <= 0:
        return 0.0
    pixels_per_sec = speed_px_per_frame * process_fps
    meters_per_sec = pixels_per_sec / pixels_per_meter
    return meters_per_sec * 3.6


def get_overspeed_ids(
    velocities_px_per_frame: Dict[int, float],
    process_fps: float,
    pixels_per_meter: float = cfg.PIXELS_PER_METER,
    speed_limit_kmh: float = cfg.SPEED_LIMIT_KMH,
) -> Set[int]:
    """
    Return set of track IDs whose estimated speed (km/h) exceeds speed_limit_kmh.
    velocities_px_per_frame: track_id -> speed in pixels/frame.
    """
    overspeed: Set[int] = set()
    for tid, px_speed in velocities_px_per_frame.items():
        if px_speed <= 0:
            continue
        kmh = px_per_frame_to_kmh(px_speed, process_fps, pixels_per_meter)
        if kmh >= speed_limit_kmh:
            overspeed.add(tid)
    return overspeed
