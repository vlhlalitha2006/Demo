"""
Multi-object tracking using ByteTrack (via Ultralytics YOLOv8 .track()).
Assigns unique IDs to vehicles and maintains trajectories to avoid double-counting.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

from detection.detector import VehicleDetector


# Trajectory: list of (frame_idx, cx, cy) per track id
TrajectoryMap = Dict[int, List[Tuple[int, float, float]]]


class VehicleTracker:
    """
    Uses YOLOv8 + ByteTrack for detection and tracking.
    Exposes track IDs and stores trajectories per ID for analytics.
    """

    def __init__(
        self,
        model_size: str = cfg.YOLO_MODEL_SIZE,
        conf: float = cfg.DETECTION_CONFIDENCE,
        tracker_name: str = cfg.TRACKER_NAME,
        max_age: int = cfg.TRACK_MAX_AGE,
        min_hits: int = cfg.TRACK_MIN_HITS,
    ):
        if YOLO is None:
            raise ImportError("Install ultralytics: pip install ultralytics")
        self.detector = VehicleDetector(model_size=model_size, conf=conf)
        self.model = self.detector.model
        self.tracker_name = tracker_name
        self.max_age = max_age
        self.min_hits = min_hits
        self.trajectories: TrajectoryMap = {}
        self._last_velocities: Dict[int, float] = {}  # id -> speed (px/frame)
        self._frame_index = 0

    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Run detection + tracking on one frame.
        Returns (N, 7) array: [x1, y1, x2, y2, conf, class_id, track_id].
        Also updates trajectories and per-id velocities.
        """
        results = self.model.track(
            frame,
            conf=self.detector.conf,
            classes=cfg.VEHICLE_CLASS_IDS,
            persist=True,
            tracker=f"{self.tracker_name}.yaml",
            verbose=False,
        )
        out = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            tid = boxes.id
            ids = tid.cpu().numpy().astype(int) if tid is not None else np.full(len(boxes), -1)
            for i in range(len(boxes)):
                x1, y1, x2, y2 = xyxy[i]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                track_id = int(ids[i]) if ids[i] >= 0 else -1
                out.append([
                    float(x1), float(y1), float(x2), float(y2),
                    float(conf[i]), int(cls[i]), track_id,
                ])
                if track_id >= 0:
                    if track_id not in self.trajectories:
                        self.trajectories[track_id] = []
                    self.trajectories[track_id].append((self._frame_index, cx, cy))
                    # velocity from previous point
                    hist = self.trajectories[track_id]
                    if len(hist) >= 2:
                        _, px, py = hist[-2]
                        speed = np.hypot(cx - px, cy - py)
                        self._last_velocities[track_id] = speed
        self._frame_index += 1
        self._prune_old_tracks()
        return np.array(out) if out else np.zeros((0, 7))

    def _prune_old_tracks(self) -> None:
        """Remove tracks not seen recently to limit memory."""
        max_len = 200
        for tid in list(self.trajectories.keys()):
            traj = self.trajectories[tid]
            if traj and self._frame_index - traj[-1][0] > self.max_age:
                del self.trajectories[tid]
                self._last_velocities.pop(tid, None)
            elif len(traj) > max_len:
                self.trajectories[tid] = traj[-max_len:]

    def get_trajectory(self, track_id: int) -> List[Tuple[int, float, float]]:
        return self.trajectories.get(track_id, [])

    def get_velocity(self, track_id: int) -> float:
        return self._last_velocities.get(track_id, 0.0)

    def set_frame_index(self, idx: int) -> None:
        self._frame_index = idx

    def reset(self) -> None:
        """Call when processing a new video (tracker state reset)."""
        self.trajectories.clear()
        self._last_velocities.clear()
        self._frame_index = 0
        # Ultralytics tracker is stateful; new video = new model run, so we rely
        # on not persisting across runs. For same-video re-runs, user restarts.
