"""
Analytics engine — aggregates pipeline outputs and batches DB writes.
"""

from __future__ import annotations


class AnalyticsEngine:
    """
    Collects TrackedPerson, MovementRecord, and OccupancyRecord objects
    and flushes them to the database in batches.

    Session-level stats (behaviour distribution, zone counts, entry/exit
    log, per-person timelines) are read back from SQLite via database.py's
    query_* methods rather than kept in memory here — the dashboard already
    queries the DB directly, so duplicating that state in-process would just
    grow unbounded for the life of a long-running session with no reader.
    """

    def __init__(self, cfg: dict, db):
        self._write_interval = cfg["analytics"].get("write_interval_frames", 5)
        self._batch_size     = cfg["analytics"].get("batch_size", 30)
        self._db             = db

        self._frames_since_write = 0
        self._pending_tracks: list   = []
        self._pending_movements: list = []
        self._pending_occupancy: list = []

    def ingest(self, session_id: str, frame_id: int,
               tracked_persons: list,
               movement_records: list,
               occupancy_record) -> None:
        """Ingest one frame's worth of pipeline outputs."""

        for tp in tracked_persons:
            if tp.is_exit:
                continue
            self._pending_tracks.append({
                "session_id": session_id,
                "frame_id":   tp.frame_id,
                "timestamp":  tp.timestamp,
                "track_id":   tp.track_id,
                "class_id":   tp.class_id,
                "class_name": tp.class_name,
                "smoothed_behaviour": tp.smoothed_behaviour,
                "confidence": tp.confidence,
                "bbox_x1":    tp.bbox[0],
                "bbox_y1":    tp.bbox[1],
                "bbox_x2":    tp.bbox[2],
                "bbox_y2":    tp.bbox[3],
                "centroid_x": tp.centroid[0],
                "centroid_y": tp.centroid[1],
            })
        for mr in movement_records:
            self._pending_movements.append({
                "session_id": session_id,
                "frame_id":   mr.frame_id,
                "timestamp":  mr.timestamp,
                "track_id":   mr.track_id,
                "is_moving":  int(mr.is_moving),
                "direction":  mr.direction,
                "zone":       mr.zone,
                "is_entry":   int(mr.is_entry),
                "is_exit":    int(mr.is_exit),
            })

        if occupancy_record:
            self._pending_occupancy.append({
                "session_id":    session_id,
                "timestamp":     occupancy_record.timestamp,
                "current_count": occupancy_record.current,
                "peak_count":    occupancy_record.peak,
                "average_count": occupancy_record.average,
            })

        self._frames_since_write += 1
        if self._frames_since_write >= self._write_interval:
            self._flush()
            self._frames_since_write = 0

    def _flush(self) -> None:
        if self._pending_tracks:
            self._db.insert_batch("detections", self._pending_tracks)
            self._pending_tracks.clear()
        if self._pending_movements:
            self._db.insert_batch("movements", self._pending_movements)
            self._pending_movements.clear()
        if self._pending_occupancy:
            self._db.insert_batch("occupancy", self._pending_occupancy)
            self._pending_occupancy.clear()

    def flush_all(self) -> None:
        """Force-flush remaining buffered records (call on shutdown)."""
        self._flush()
