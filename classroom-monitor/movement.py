"""
Movement analysis — zone assignment, direction, entry/exit classification.
Operates on TrackedPerson objects from the registry.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


@dataclass
class MovementRecord:
    track_id:   int
    frame_id:   int
    timestamp:  str
    is_moving:  bool
    direction:  str      # compass direction or "stationary"
    zone:       str      # e.g. "Front-Center"
    is_entry:   bool
    is_exit:    bool


class MovementAnalyzer:
    """
    Computes movement state, direction, and classroom zone for each track.
    Requires the frame dimensions to build the zone grid.
    """

    ROW_LABELS = ["Front", "Mid", "Back"]
    COL_LABELS = ["Left", "Center", "Right"]

    def __init__(self, cfg: dict, frame_width: int = 640, frame_height: int = 480):
        mv_cfg = cfg["movement"]
        self._threshold_px = mv_cfg.get("displacement_threshold_px", 15)
        self._window_sec   = mv_cfg.get("analysis_window_seconds", 2.0)
        self._rows         = mv_cfg.get("zones_rows", 3)
        self._cols         = mv_cfg.get("zones_cols", 3)
        self._fw           = frame_width
        self._fh           = frame_height

        # track_id → deque of (timestamp_float, cx, cy)
        self._history: dict[int, deque] = {}

    def update_frame_size(self, width: int, height: int) -> None:
        self._fw = width
        self._fh = height

    def analyze(self, person, fps: float = 30.0) -> MovementRecord:
        """
        person: TrackedPerson dataclass from tracker.py
        fps:    camera fps used to convert window_sec → frame count
        """
        tid   = person.track_id
        cx, cy = person.centroid

        if tid not in self._history:
            self._history[tid] = deque(maxlen=int(self._window_sec * fps) + 1)

        self._history[tid].append((person.timestamp, cx, cy))

        is_moving, direction = self._compute_motion(tid)
        zone = self._assign_zone(cx, cy)

        return MovementRecord(
            track_id=tid,
            frame_id=person.frame_id,
            timestamp=person.timestamp,
            is_moving=is_moving,
            direction=direction,
            zone=zone,
            is_entry=person.is_entry,
            is_exit=person.is_exit,
        )

    def _compute_motion(self, tid: int) -> tuple[bool, str]:
        hist = self._history.get(tid, deque())
        if len(hist) < 2:
            return False, "stationary"

        _, x0, y0 = hist[0]
        _, x1, y1 = hist[-1]
        dx = x1 - x0
        dy = y1 - y0
        displacement = math.hypot(dx, dy)

        if displacement < self._threshold_px:
            return False, "stationary"

        angle_deg = math.degrees(math.atan2(-dy, dx)) % 360
        sector = int((angle_deg + 22.5) / 45) % 8
        return True, COMPASS[sector]

    def _assign_zone(self, cx: float, cy: float) -> str:
        col_idx = min(int(cx / self._fw * self._cols), self._cols - 1)
        row_idx = min(int(cy / self._fh * self._rows), self._rows - 1)

        row_labels = self.ROW_LABELS[:self._rows] if self._rows <= 3 else [
            f"Row{i}" for i in range(self._rows)
        ]
        col_labels = self.COL_LABELS[:self._cols] if self._cols <= 3 else [
            f"Col{j}" for j in range(self._cols)
        ]
        return f"{row_labels[row_idx]}-{col_labels[col_idx]}"

    def purge_track(self, track_id: int) -> None:
        self._history.pop(track_id, None)
