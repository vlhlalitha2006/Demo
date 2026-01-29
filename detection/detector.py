"""
Vehicle detector using YOLOv8 (Ultralytics).
Detects car, bike, bus, truck. Auto-rickshaws typically map to car/bike.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore

import sys
from pathlib import Path

# Add project root for config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


class VehicleDetector:
    """
    Wraps YOLOv8 for vehicle-only detection.
    Filters detections by VEHICLE_CLASS_IDS and confidence.
    """

    def __init__(
        self,
        model_size: str = "n",
        conf: float = cfg.DETECTION_CONFIDENCE,
        vehicle_classes: Optional[List[int]] = None,
    ):
        if YOLO is None:
            raise ImportError("Install ultralytics: pip install ultralytics")
        self.conf = conf
        self.vehicle_classes = vehicle_classes or cfg.VEHICLE_CLASS_IDS
        model_name = f"yolov8{model_size}.pt"
        self.model = YOLO(model_name)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run detection on a BGR frame.
        Returns (N, 6) array: [x1, y1, x2, y2, conf, class_id].
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            classes=self.vehicle_classes,
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
            for i in range(len(boxes)):
                out.append([
                    float(xyxy[i, 0]),
                    float(xyxy[i, 1]),
                    float(xyxy[i, 2]),
                    float(xyxy[i, 3]),
                    float(conf[i]),
                    int(cls[i]),
                ])
        return np.array(out) if out else np.zeros((0, 6))

    def class_id_to_name(self, class_id: int) -> str:
        """Map COCO class id to our vehicle name."""
        idx = self.vehicle_classes.index(class_id) if class_id in self.vehicle_classes else 0
        return cfg.VEHICLE_CLASS_NAMES[min(idx, len(cfg.VEHICLE_CLASS_NAMES) - 1)]
