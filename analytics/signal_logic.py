"""
Traffic signal logic and red-light jump violation detection.
Assumption: Signal state (RED/GREEN) is known. Stop line defined as y = constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


@dataclass
class ViolationEvent:
    """A single red-light jump event."""
    track_id: int
    frame_idx: int
    # Human-readable explanation for explainability
    explanation: str = "Vehicle crossed stop line during RED signal"


class SignalLogic:
    """
    Rule-based red-light jump detection.
    - Stop line: y = STOP_LINE_Y (pixels from top). Vehicles below this have "crossed".
    - During RED: if a tracked vehicle crosses the stop line (center goes from above
      to below), we flag a violation. Each track_id is counted at most once per
      crossing (we track which IDs have already been flagged for this crossing).
    """

    def __init__(
        self,
        stop_line_y: float = cfg.STOP_LINE_Y,
        cross_buffer: float = cfg.STOP_LINE_CROSS_BUFFER,
    ):
        self.stop_line_y = stop_line_y
        self.cross_buffer = cross_buffer
        # Track IDs that have already triggered a violation (reset when signal turns GREEN)
        self._violated_ids: Set[int] = set()
        self._last_side: dict[int, str] = {}  # id -> "above" | "below"
        self.violations: list[ViolationEvent] = []

    def set_stop_line_y(self, y: float) -> None:
        self.stop_line_y = y

    def set_signal_state(self, state: str) -> None:
        """Call when signal turns GREEN to reset violation state and queue."""
        if state.upper() == "GREEN":
            self._violated_ids.clear()
            self._last_side.clear()

    def update(
        self,
        tracks: list[tuple[int, float, float, float, float]],  # (id, x1,y1,x2,y2)
        signal_state: str,
        frame_idx: int,
    ) -> list[ViolationEvent]:
        """
        For each track: compute center y. If signal is RED and vehicle crosses
        stop line (center moves from above to below), flag violation.
        Uses buffer to avoid duplicate triggers.
        """
        new_events: list[ViolationEvent] = []
        if signal_state.upper() != "RED":
            return new_events

        for (tid, x1, y1, x2, y2) in tracks:
            if tid < 0:
                continue
            cy = (y1 + y2) / 2.0
            side = "above" if cy < self.stop_line_y - self.cross_buffer else "below"
            prev = self._last_side.get(tid, "above")

            # Crossing: was above, now below
            if prev == "above" and side == "below" and tid not in self._violated_ids:
                self._violated_ids.add(tid)
                evt = ViolationEvent(
                    track_id=tid,
                    frame_idx=frame_idx,
                    explanation="Vehicle crossed stop line during RED signal (red-light jump)",
                )
                self.violations.append(evt)
                new_events.append(evt)

            self._last_side[tid] = side

        return new_events
