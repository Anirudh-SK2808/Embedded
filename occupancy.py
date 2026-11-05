"""
Occupancy analysis — current, peak, and rolling-average counts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class OccupancyRecord:
    timestamp:     str
    current:       int
    peak:          int
    average:       float


class OccupancyTracker:
    """
    Tracks the number of students present each frame.
    Rolling average uses a time-windowed deque.
    """

    def __init__(self, cfg: dict):
        mv_cfg = cfg.get("movement", {})
        # reuse the same window concept; configurable under movement.occupancy_window_seconds
        self._window_sec = cfg.get("movement", {}).get("analysis_window_seconds", 60.0)
        # override with explicit key if present
        self._window_sec = cfg.get("occupancy", {}).get("window_seconds", self._window_sec)

        self._peak = 0
        # deque of (timestamp_str, count)
        self._history: deque[tuple[str, int]] = deque()

    def update(self, current_count: int, timestamp: str) -> OccupancyRecord:
        self._peak = max(self._peak, current_count)
        self._history.append((timestamp, current_count))
        self._trim_window(timestamp)

        counts = [c for _, c in self._history]
        avg = sum(counts) / len(counts) if counts else 0.0

        return OccupancyRecord(
            timestamp=timestamp,
            current=current_count,
            peak=self._peak,
            average=round(avg, 2),
        )

    def _trim_window(self, now_str: str) -> None:
        """Remove entries older than window_sec from the history deque."""
        try:
            now = datetime.fromisoformat(now_str)
        except ValueError:
            return
        while self._history:
            ts_str, _ = self._history[0]
            try:
                ts = datetime.fromisoformat(ts_str)
                if (now - ts).total_seconds() > self._window_sec:
                    self._history.popleft()
                else:
                    break
            except ValueError:
                self._history.popleft()

    def reset(self) -> None:
        """Reset peak and history (e.g. between class periods)."""
        self._peak = 0
        self._history.clear()
