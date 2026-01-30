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
        min_traj_len: int = max(10, cfg.RASH_MIN_TRAJECTORY_LEN), # Ensure at least 10 frames
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
        self._max_recent = 200 # larger pool for more stable percentile
        self._flagged_ids_per_reason: Dict[tuple[int, str], int] = {} # (tid, reason) -> last_frame_idx

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
        
        # Only add valid speeds to the pool
        speeds = [s for s in velocities.values() if s > 0]
        if speeds:
            self._recent_speeds.extend(speeds)
            self._recent_speeds = self._recent_speeds[-self._max_recent:]
            
        pct = np.percentile(self._recent_speeds, self.speed_percentile) if len(self._recent_speeds) > 20 else 100.0 # Need some data first

        for tid, traj in trajectories.items():
            if len(traj) < self.min_traj_len:
                continue
            
            # Use a slightly longer window for turn analysis
            traj_window = traj[-self.min_traj_len:]
            speed = velocities.get(tid, 0.0)

            # 1. Sharp turn (check last few segments)
            pts = np.array([(t[1], t[2]) for t in traj_window])
            if len(pts) >= 5:
                # Average direction of last 2 segments vs previous 2
                v_recent = pts[-1] - pts[-3]
                v_prev = pts[-3] - pts[-5]
                if np.linalg.norm(v_recent) > 1.0 and np.linalg.norm(v_prev) > 1.0:
                    ang = _angle_deg(v_prev, v_recent)
                    if ang >= self.sharp_turn_deg:
                        self._trigger_event(tid, frame_idx, f"Sharp turn ({ang:.0f}°)", new_events)


            # 2. Sudden acceleration
            if len(traj_window) >= 3:
                d1 = np.hypot(traj_window[-2][1] - traj_window[-3][1], traj_window[-2][2] - traj_window[-3][2])
                d2 = np.hypot(traj_window[-1][1] - traj_window[-2][1], traj_window[-1][2] - traj_window[-2][2])
                if d1 > 1.0 and d2 / d1 >= self.accel_ratio:
                    self._trigger_event(tid, frame_idx, f"Sudden accel (x{d2/d1:.1f})", new_events)

            # 3. Abnormally high speed
            if speed >= self.min_speed_px and speed >= pct:
                self._trigger_event(tid, frame_idx, "High speed", new_events)

        return new_events

    def _trigger_event(self, tid: int, frame_idx: int, reason: str, event_list: List[RashDrivingEvent]):
        """Helper to append event and log, with cooldown to avoid duplicates."""
        last_f = self._flagged_ids_per_reason.get((tid, reason), -100)
        cooldown = 30 # frames (approx 3 seconds at 10fps)
        if frame_idx - last_f > cooldown:
            evt = RashDrivingEvent(track_id=tid, frame_idx=frame_idx, reason=reason)
            self.events.append(evt)
            event_list.append(evt)
            self._flagged_ids_per_reason[(tid, reason)] = frame_idx

