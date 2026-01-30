"""
Rash driving detection via trajectory-based heuristics (rule-based, interpretable).
- Sharp direction change (angle between consecutive direction vectors)
- Sudden acceleration (ratio of consecutive frame-to-frame displacements)
- Abnormally high speed (top percentile of recent vehicle speeds)
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
    reason: str  # Human-readable explanation


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between 2D vectors in degrees (0–180)."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


class RashDrivingAnalyzer:
    """
    Rule-based rash driving detection using trajectory and velocity.
    Only flags when thresholds are clearly exceeded to reduce false positives.
    """

    def __init__(
        self,
        min_traj_len: int = None,
        sharp_turn_deg: float = None,
        accel_ratio: float = None,
        speed_percentile: float = None,
        min_speed_px: float = None,
        min_displacement_px: float = None,
        accel_min_d1_px: float = None,
        accel_min_d2_px: float = None,
        cooldown_frames: int = None,
    ):
        self.min_traj_len = min_traj_len if min_traj_len is not None else max(5, cfg.RASH_MIN_TRAJECTORY_LEN)
        self.sharp_turn_deg = sharp_turn_deg if sharp_turn_deg is not None else cfg.RASH_SHARP_TURN_ANGLE_DEG
        self.accel_ratio = accel_ratio if accel_ratio is not None else cfg.RASH_ACCELERATION_RATIO
        self.speed_percentile = speed_percentile if speed_percentile is not None else cfg.RASH_SPEED_PERCENTILE
        self.min_speed_px = min_speed_px if min_speed_px is not None else cfg.RASH_MIN_SPEED_PX
        self.min_displacement_px = min_displacement_px if min_displacement_px is not None else getattr(
            cfg, "RASH_MIN_DISPLACEMENT_PX", 3.0
        )
        self.accel_min_d1_px = accel_min_d1_px if accel_min_d1_px is not None else getattr(
            cfg, "RASH_ACCEL_MIN_D1_PX", 2.5
        )
        self.accel_min_d2_px = accel_min_d2_px if accel_min_d2_px is not None else getattr(
            cfg, "RASH_ACCEL_MIN_D2_PX", 5.0
        )
        self.cooldown_frames = cooldown_frames if cooldown_frames is not None else getattr(
            cfg, "RASH_COOLDOWN_FRAMES", 45
        )
        self.events: List[RashDrivingEvent] = []
        self._recent_speeds: List[float] = []
        self._max_recent = 200
        # Cooldown by (tid, reason_key) so same vehicle can't spam same category
        self._last_frame_per_key: Dict[Tuple[int, str], int] = {}

    def update(
        self,
        trajectories: Dict[int, List[Tuple[int, float, float]]],
        velocities: Dict[int, float],
        frame_idx: int,
    ) -> List[RashDrivingEvent]:
        """
        trajectories: track_id -> [(frame_idx, cx, cy), ...]
        velocities: track_id -> speed (pixels per frame)
        """
        new_events: List[RashDrivingEvent] = []

        # Build speed pool for percentile (only positive speeds)
        for s in velocities.values():
            if s > 0:
                self._recent_speeds.append(s)
        self._recent_speeds = self._recent_speeds[-self._max_recent :]
        min_samples = 30
        pct = (
            np.percentile(self._recent_speeds, self.speed_percentile)
            if len(self._recent_speeds) >= min_samples
            else float("inf")
        )

        for tid, traj in trajectories.items():
            if len(traj) < self.min_traj_len:
                continue

            traj_window = traj[-self.min_traj_len :]
            speed = velocities.get(tid, 0.0)
            pts = np.array([(t[1], t[2]) for t in traj_window])

            # 1. Sharp turn: direction change between two segments
            if len(pts) >= 5:
                v_recent = pts[-1] - pts[-3]  # last 2 steps
                v_prev = pts[-3] - pts[-5]    # previous 2 steps
                n_recent = np.linalg.norm(v_recent)
                n_prev = np.linalg.norm(v_prev)
                if n_recent >= self.min_displacement_px and n_prev >= self.min_displacement_px:
                    ang = _angle_deg(v_prev, v_recent)
                    if ang >= self.sharp_turn_deg:
                        self._trigger(
                            tid, frame_idx, f"Sharp turn ({ang:.0f}°)", "sharp_turn", new_events
                        )

            # 2. Sudden acceleration: ratio of consecutive displacements
            if len(traj_window) >= 3:
                d1 = np.hypot(
                    traj_window[-2][1] - traj_window[-3][1],
                    traj_window[-2][2] - traj_window[-3][2],
                )
                d2 = np.hypot(
                    traj_window[-1][1] - traj_window[-2][1],
                    traj_window[-1][2] - traj_window[-2][2],
                )
                if (
                    d1 >= self.accel_min_d1_px
                    and d2 >= self.accel_min_d2_px
                    and d1 > 1e-6
                    and d2 / d1 >= self.accel_ratio
                ):
                    self._trigger(
                        tid, frame_idx, f"Sudden accel (x{d2/d1:.1f})", "sudden_accel", new_events
                    )

            # 3. Abnormally high speed (top percentile of recent vehicles)
            if speed >= self.min_speed_px and pct < float("inf") and speed >= pct:
                self._trigger(tid, frame_idx, "High speed", "high_speed", new_events)

        return new_events

    def _trigger(
        self,
        tid: int,
        frame_idx: int,
        reason_display: str,
        reason_key: str,
        event_list: List[RashDrivingEvent],
    ) -> None:
        """Emit one event per (tid, reason_key) per cooldown period."""
        key = (tid, reason_key)
        last = self._last_frame_per_key.get(key, -self.cooldown_frames - 1)
        if frame_idx - last <= self.cooldown_frames:
            return
        self._last_frame_per_key[key] = frame_idx
        evt = RashDrivingEvent(track_id=tid, frame_idx=frame_idx, reason=reason_display)
        self.events.append(evt)
        event_list.append(evt)
