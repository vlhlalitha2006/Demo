"""
Overlay bounding boxes, IDs, queue ROI, stop line, queue stats, violations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

from analytics import QueueMetrics


def draw_frame_overlay(
    frame: np.ndarray,
    tracks: np.ndarray,
    *,
    detector_class_names: Optional[Dict[int, str]] = None,
    stop_line_y: float = cfg.STOP_LINE_Y,
    queue_roi: Optional[Tuple[int, int, int, int]] = None,
    queue_metrics: Optional[QueueMetrics] = None,
    signal_state: str = "RED",
    red_light_violation_ids: Optional[set] = None,
    rash_driving_ids: Optional[set] = None,
    speed_violation_ids: Optional[set] = None,
) -> np.ndarray:
    """
    tracks: (N, 7) [x1,y1,x2,y2, conf, class_id, track_id].
    Overlays vehicle IDs, queue ROI, stop line, queue/signal stats, and violation alerts.
    """
    out = frame.copy()
    red_light_violation_ids = red_light_violation_ids or set()
    rash_driving_ids = rash_driving_ids or set()
    speed_violation_ids = speed_violation_ids or set()
    if detector_class_names is None:
        detector_class_names = {}

    # Queue ROI
    if queue_roi is not None:
        x1, y1, x2, y2 = queue_roi
        cv2.rectangle(out, (x1, y1), (x2, y2), cfg.COLOR_QUEUE, 2)
        cv2.putText(
            out, "Queue ROI", (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE, cfg.COLOR_QUEUE, cfg.FONT_THICKNESS,
        )

    # Stop line (thick so visible in most videos; clamp y to frame height)
    h, w = out.shape[:2]
    y = max(0, min(int(stop_line_y), h - 1))
    line_thick = getattr(cfg, "STOP_LINE_VIS_THICKNESS", 4)
    cv2.line(out, (0, y), (w, y), cfg.COLOR_STOP_LINE, line_thick)
    cv2.putText(
        out, "Stop line", (10, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE, cfg.COLOR_STOP_LINE, cfg.FONT_THICKNESS,
    )

    # Signal state
    sig_color = cfg.COLOR_RED if signal_state.upper() == "RED" else cfg.COLOR_GREEN
    cv2.putText(
        out, f"Signal: {signal_state}", (w - 180, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, sig_color, 2,
    )

    # Queue metrics and violation counts
    if queue_metrics is not None:
        txt = (
            f"Total: {len([i for i in range(len(tracks)) if int(tracks[i, 6]) >= 0])}  "
            f"Queue: {queue_metrics.queue_length}  Density: {queue_metrics.queue_density:.4f}"
        )
        cv2.putText(
            out, txt, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, cfg.COLOR_WHITE, 2,
        )
    viol_txt = (
        f"Red-light: {len(red_light_violation_ids)}  Rash: {len(rash_driving_ids)}  "
        f"Speed: {len(speed_violation_ids)}"
    )
    cv2.putText(
        out, viol_txt, (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cfg.COLOR_WHITE, 2,
    )

    # Boxes + vehicle IDs + violation labels
    for i in range(len(tracks)):
        x1, y1, x2, y2 = int(tracks[i, 0]), int(tracks[i, 1]), int(tracks[i, 2]), int(tracks[i, 3])
        conf = float(tracks[i, 4])
        cls_id = int(tracks[i, 5])
        tid = int(tracks[i, 6])
        label = detector_class_names.get(cls_id, "vehicle")
        box_label = f"{label} #{tid}" if tid >= 0 else label
        if tid in red_light_violation_ids:
            color = cfg.COLOR_RED
            box_label += " [RED-LIGHT]"
        elif tid in speed_violation_ids:
            color = (0, 165, 255)  # Orange for overspeed
            box_label += " [SPEED]"
        elif tid in rash_driving_ids:
            color = cfg.COLOR_YELLOW
            box_label += " [RASH]"
        else:
            color = cfg.COLOR_GREEN
        cv2.rectangle(out, (x1, y1), (x2, y2), color, cfg.BOX_THICKNESS)
        cv2.putText(
            out, f"{box_label} {conf:.1f}", (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE, color, cfg.FONT_THICKNESS,
        )

    return out
