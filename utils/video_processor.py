"""
Video I/O: read pre-recorded traffic video, sample at fixed FPS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


class VideoProcessor:
    """
    Reads a pre-recorded video and yields frames at PROCESS_FPS.
    Optionally caps at MAX_FRAMES for quick testing.
    """

    def __init__(
        self,
        video_path: str | Path,
        process_fps: float = cfg.PROCESS_FPS,
        max_frames: Optional[int] = cfg.MAX_FRAMES,
    ):
        self.video_path = Path(video_path)
        self.process_fps = process_fps
        self.max_frames = max_frames
        self.cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._total_frames: int = 0
        self._frame_shape: Tuple[int, int] = (0, 0)

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            return False
        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_shape = (h, w)
        return True

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def frame_shape(self) -> Tuple[int, int]:
        return self._frame_shape

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "VideoProcessor":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def iter_frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Yield (frame_index, frame) at process_fps.
        frame_index is 0-based and increments by 1 per yielded frame.
        """
        if self.cap is None and not self.open():
            return
        step = max(1, round(self._fps / self.process_fps))
        idx = 0
        yielded = 0
        while True:
            ret, frame = self.cap.read()
            if not ret or not frame.size:
                break
            if idx % step == 0:
                yield yielded, frame
                yielded += 1
                if self.max_frames is not None and yielded >= self.max_frames:
                    break
            idx += 1
        self.close()
