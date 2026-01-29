"""
Rash driving detection via trajectory-based heuristics (rule-based, interpretable).
- Sharp direction change
- Sudden acceleration
- Abnormally high speed relative to other vehicles
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
class RashDrivingEvent:
    track_id: int
    frame_idx: int
    reason: str  # Explainability: why we flagged this


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between 2D vectors in degrees."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


class RashDrivingAnalyzer:
    """
    Uses trajectory history and per-id velocities. Rule-based checks:
    1. Sharp turn: consecutive direction vectors angle > threshold.
    2. Sudden acceleration: speed ratio between consecutive frames > threshold.
    3. High speed: speed > percentile of recent speeds and above minimum.
    """

    def __init__(
        self,
        min_traj_len: int = cfg.RASH_MIN_TRAJECTORY_LEN,
        sharp_turn_deg: float = cfg.RASH_SHARP_TURN_ANGLE_DEG,
        accel_ratio: float = cfg.RASH_ACCELERATION_RATIO,
        speed_percentile: float = cfg.RASH_SPEED_PERCENTILE,
        min_speed_px: float = cfg.RASH_MIN_SPEED_PX,
    ):
        self.min_traj_len = min_traj_len
        self.sharp_turn_deg = sharp_turn_deg
        self.accel_ratio = accel_ratio
        self.speed_percentile = speed_percentile
        self.min_speed_px = min_speed_px
        self.events: List[RashDrivingEvent] = []
        self._recent_speeds: List[float] = []
        self._max_recent = 100

    def update(
        self,
        trajectories: Dict[int, List[Tuple[int, float, float]]],
        velocities: Dict[int, float],
        frame_idx: int,
    ) -> List[RashDrivingEvent]:
        """
        trajectories: id -> [(frame, cx, cy), ...]
        velocities: id -> speed (px/frame)
        """
        new_events: List[RashDrivingEvent] = []
        speeds = [s for s in velocities.values() if s > 0]
        self._recent_speeds = (self._recent_speeds + speeds)[-self._max_recent:]
        pct = np.percentile(self._recent_speeds, self.speed_percentile) if self._recent_speeds else 0.0

        for tid, traj in trajectories.items():
            if len(traj) < self.min_traj_len:
                continue
            traj = traj[-self.min_traj_len - 2:]
            speed = velocities.get(tid, 0.0)

            # Sharp turn
            pts = np.array([(t[1], t[2]) for t in traj])
            for i in range(2, len(pts)):
                v1 = pts[i - 1] - pts[i - 2]
                v2 = pts[i] - pts[i - 1]
                if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                    continue
                ang = _angle_deg(v1, v2)
                if ang >= self.sharp_turn_deg:
                    evt = RashDrivingEvent(
                        track_id=tid,
                        frame_idx=frame_idx,
                        reason=f"Sharp direction change ({ang:.0f}°)",
                    )
                    self.events.append(evt)
                    new_events.append(evt)
                    break

            # Sudden acceleration (compare consecutive speeds from trajectory)
            if len(traj) >= 3:
                d1 = np.hypot(traj[-2][1] - traj[-3][1], traj[-2][2] - traj[-3][2])
                d2 = np.hypot(traj[-1][1] - traj[-2][1], traj[-1][2] - traj[-2][2])
                if d1 > 0.5 and d2 / d1 >= self.accel_ratio:
                    evt = RashDrivingEvent(
                        track_id=tid,
                        frame_idx=frame_idx,
                        reason=f"Sudden acceleration (ratio {d2 / d1:.1f})",
                    )
                    self.events.append(evt)
                    new_events.append(evt)

            # Abnormally high speed
            if speed >= self.min_speed_px and pct > 0 and speed >= pct:
                evt = RashDrivingEvent(
                    track_id=tid,
                    frame_idx=frame_idx,
                    reason=f"Abnormally high speed ({speed:.1f} px/frame, >{self.speed_percentile}th %ile)",
                )
                self.events.append(evt)
                new_events.append(evt)

        return new_events
