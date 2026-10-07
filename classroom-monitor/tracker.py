"""
Multi-person tracking registry.

ByteTrack is handled inside ultralytics (detector.py calls model.track()).
This module wraps the raw Detection stream into a stateful TrackRegistry
that records entry/exit events and maintains per-track position history.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TrackInfo:
    track_id:         int
    entry_time:       str
    last_seen_frame:  int
    position_history: deque = field(default_factory=lambda: deque(maxlen=500))
    behaviour_history: deque = field(default_factory=lambda: deque(maxlen=10))
    is_active:        bool = True


@dataclass
class TrackedPerson:
    track_id:      int
    frame_id:      int
    timestamp:     str
    class_id:      int
    class_name:    str
    confidence:    float
    bbox:          tuple[float, float, float, float]
    centroid:      tuple[float, float]
    smoothed_behaviour: str   # mode of last N behaviour predictions
    is_entry:      bool
    is_exit:       bool


class TrackRegistry:
    """
    Maintains state for all active tracks.
    Wraps raw Detection objects (which already have ultralytics track IDs)
    into TrackedPerson objects with entry/exit flags and position history.
    """

    def __init__(self, cfg: dict):
        self._expiry_frames  = cfg["tracking"].get("track_expiry_frames", 30)
        self._smooth_window  = cfg["behaviour"].get("smoothing_window", 5)
        self._max_ids        = cfg["tracking"].get("max_ids", 50)
        self._registry: dict[int, TrackInfo] = {}

    def update(self, detections: list, frame_id: int,
               timestamp: str) -> list[TrackedPerson]:
        """
        Feed raw detections into the registry.
        Returns one TrackedPerson per active detection, plus
        exit events for tracks that just disappeared.
        """
        from detector import Detection

        active_ids_this_frame: set[int] = set()
        results: list[TrackedPerson] = []

        for det in detections:
            tid = det.track_id
            if tid < 0:
                continue   # untracked detection (first frame edge case)

            is_entry = tid not in self._registry
            if is_entry:
                if len(self._registry) >= self._max_ids:
                    continue   # cap at max_ids
                self._registry[tid] = TrackInfo(
                    track_id=tid,
                    entry_time=timestamp,
                    last_seen_frame=frame_id,
                )

            info = self._registry[tid]
            info.last_seen_frame = frame_id
            info.is_active       = True
            info.position_history.append((frame_id, *det.centroid))
            info.behaviour_history.append(det.class_name)
            active_ids_this_frame.add(tid)

            smoothed = _mode(list(info.behaviour_history)[-self._smooth_window:])

            results.append(TrackedPerson(
                track_id=tid,
                frame_id=frame_id,
                timestamp=timestamp,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox,
                centroid=det.centroid,
                smoothed_behaviour=smoothed,
                is_entry=is_entry,
                is_exit=False,
            ))

        # Detect exits: tracks that vanished for > expiry_frames.
        # Popped from the registry (not just marked inactive) so max_ids
        # enforces a true concurrent-track cap instead of a lifetime one —
        # otherwise len(self._registry) only ever grows and, once max_ids
        # unique people have been seen, every new person is silently
        # dropped for the rest of the session even after the room empties.
        for tid, info in list(self._registry.items()):
            if tid not in active_ids_this_frame:
                frames_gone = frame_id - info.last_seen_frame
                if frames_gone > self._expiry_frames and info.is_active:
                    del self._registry[tid]
                    results.append(TrackedPerson(
                        track_id=tid,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        class_id=-1,
                        class_name="",
                        confidence=0.0,
                        bbox=(0, 0, 0, 0),
                        centroid=(0, 0),
                        smoothed_behaviour="",
                        is_entry=False,
                        is_exit=True,
                    ))

        return results

    def active_count(self) -> int:
        return sum(1 for info in self._registry.values() if info.is_active)

    def get_position_history(self, track_id: int) -> list[tuple]:
        info = self._registry.get(track_id)
        return list(info.position_history) if info else []


def _mode(values: list[str]) -> str:
    if not values:
        return ""
    return max(set(values), key=values.count)
