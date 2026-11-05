"""
SQLite database manager with WAL mode and batch inserts.

One shared instance is used for the whole process (both the live pipeline's
background writer thread and Flask's request-handling threads), guarded by
a lock — sqlite3 connections aren't safe for unserialized concurrent use
even with check_same_thread=False, so every method here takes the lock
rather than relying on WAL alone for thread safety.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    start_time   TEXT NOT NULL,
    end_time     TEXT,
    camera_id    TEXT
);

CREATE TABLE IF NOT EXISTS detections (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT    NOT NULL,
    frame_id         INTEGER NOT NULL,
    timestamp        TEXT    NOT NULL,
    track_id         INTEGER NOT NULL,
    class_id         INTEGER,
    class_name       TEXT,
    smoothed_behaviour TEXT,
    confidence       REAL,
    bbox_x1          REAL,
    bbox_y1          REAL,
    bbox_x2          REAL,
    bbox_y2          REAL,
    centroid_x       REAL,
    centroid_y       REAL
);

CREATE TABLE IF NOT EXISTS movements (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    frame_id    INTEGER NOT NULL,
    timestamp   TEXT    NOT NULL,
    track_id    INTEGER NOT NULL,
    is_moving   INTEGER,
    direction   TEXT,
    zone        TEXT,
    is_entry    INTEGER,
    is_exit     INTEGER
);

CREATE TABLE IF NOT EXISTS occupancy (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    current_count INTEGER,
    peak_count    INTEGER,
    average_count REAL
);

CREATE TABLE IF NOT EXISTS alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT    NOT NULL,
    track_id      INTEGER NOT NULL,
    behaviour     TEXT    NOT NULL,
    started_at    TEXT    NOT NULL,
    detected_at   TEXT    NOT NULL,
    duration_s    REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_det_session   ON detections(session_id);
CREATE INDEX IF NOT EXISTS idx_det_ts        ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_det_track     ON detections(track_id);
CREATE INDEX IF NOT EXISTS idx_mov_session   ON movements(session_id);
CREATE INDEX IF NOT EXISTS idx_mov_zone      ON movements(zone);
CREATE INDEX IF NOT EXISTS idx_occ_session   ON occupancy(session_id);
CREATE INDEX IF NOT EXISTS idx_alerts_session ON alerts(session_id);
"""


class DatabaseManager:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(DDL)
            self._conn.commit()

    # ------------------------------------------------------------------ #
    # Session management                                                   #
    # ------------------------------------------------------------------ #

    def new_session(self, camera_id: str = "") -> str:
        sid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions(session_id, start_time, camera_id) VALUES(?,?,?)",
                (sid, now, str(camera_id)),
            )
            self._conn.commit()
        return sid

    def close_session(self, session_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET end_time=? WHERE session_id=?",
                (now, session_id),
            )
            self._conn.commit()

    # ------------------------------------------------------------------ #
    # Batch inserts                                                        #
    # ------------------------------------------------------------------ #

    def insert_batch(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        cols   = list(rows[0].keys())
        ph     = ",".join("?" * len(cols))
        sql    = f"INSERT INTO {table}({','.join(cols)}) VALUES({ph})"
        values = [tuple(r[c] for c in cols) for r in rows]
        with self._lock:
            self._conn.executemany(sql, values)
            self._conn.commit()

    def insert_alert(self, session_id: str, track_id: int, behaviour: str,
                      started_at: str, detected_at: str, duration_s: float) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO alerts(session_id, track_id, behaviour, started_at, detected_at, duration_s)
                   VALUES(?,?,?,?,?,?)""",
                (session_id, track_id, behaviour, started_at, detected_at, duration_s),
            )
            self._conn.commit()

    # ------------------------------------------------------------------ #
    # Query helpers                                                         #
    # ------------------------------------------------------------------ #

    def list_sessions(self) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT session_id, start_time, end_time, camera_id FROM sessions ORDER BY start_time DESC"
            )
            return [dict(r) for r in cur.fetchall()]

    def query_detections(self, session_id: str,
                         start: str | None = None,
                         end:   str | None = None) -> list[dict]:
        sql  = "SELECT * FROM detections WHERE session_id=?"
        args: list = [session_id]
        if start:
            sql += " AND timestamp >= ?";  args.append(start)
        if end:
            sql += " AND timestamp <= ?";  args.append(end)
        sql += " ORDER BY timestamp"
        with self._lock:
            cur = self._conn.execute(sql, args)
            return [dict(r) for r in cur.fetchall()]

    def query_movements(self, session_id: str,
                        start: str | None = None,
                        end:   str | None = None) -> list[dict]:
        sql  = "SELECT * FROM movements WHERE session_id=?"
        args: list = [session_id]
        if start:
            sql += " AND timestamp >= ?";  args.append(start)
        if end:
            sql += " AND timestamp <= ?";  args.append(end)
        sql += " ORDER BY timestamp"
        with self._lock:
            cur = self._conn.execute(sql, args)
            return [dict(r) for r in cur.fetchall()]

    def query_occupancy(self, session_id: str,
                        start: str | None = None,
                        end:   str | None = None) -> list[dict]:
        sql  = "SELECT * FROM occupancy WHERE session_id=?"
        args: list = [session_id]
        if start:
            sql += " AND timestamp >= ?";  args.append(start)
        if end:
            sql += " AND timestamp <= ?";  args.append(end)
        sql += " ORDER BY timestamp"
        with self._lock:
            cur = self._conn.execute(sql, args)
            return [dict(r) for r in cur.fetchall()]

    def query_behaviour_distribution(self, session_id: str) -> list[dict]:
        sql = """
            SELECT smoothed_behaviour AS behaviour,
                   COUNT(*) AS count
            FROM detections
            WHERE session_id=? AND smoothed_behaviour IS NOT NULL AND smoothed_behaviour != ''
            GROUP BY smoothed_behaviour
            ORDER BY count DESC
        """
        with self._lock:
            cur = self._conn.execute(sql, (session_id,))
            return [dict(r) for r in cur.fetchall()]

    def query_zone_counts(self, session_id: str) -> list[dict]:
        sql = """
            SELECT zone, COUNT(*) AS count
            FROM movements
            WHERE session_id=? AND is_exit=0 AND zone IS NOT NULL
            GROUP BY zone
            ORDER BY count DESC
        """
        with self._lock:
            cur = self._conn.execute(sql, (session_id,))
            return [dict(r) for r in cur.fetchall()]

    def query_entry_exit_events(self, session_id: str) -> list[dict]:
        sql = """
            SELECT timestamp, track_id,
                   CASE WHEN is_entry=1 THEN 'entry' ELSE 'exit' END AS event,
                   zone
            FROM movements
            WHERE session_id=? AND (is_entry=1 OR is_exit=1)
            ORDER BY timestamp
        """
        with self._lock:
            cur = self._conn.execute(sql, (session_id,))
            return [dict(r) for r in cur.fetchall()]

    def query_person_timeline(self, session_id: str, track_id: int) -> list[dict]:
        sql = """
            SELECT timestamp, smoothed_behaviour AS behaviour
            FROM detections
            WHERE session_id=? AND track_id=?
            ORDER BY timestamp
        """
        with self._lock:
            cur = self._conn.execute(sql, (session_id, track_id))
            return [dict(r) for r in cur.fetchall()]

    def query_track_ids(self, session_id: str) -> list[int]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT DISTINCT track_id FROM detections WHERE session_id=? ORDER BY track_id",
                (session_id,),
            )
            return [r[0] for r in cur.fetchall()]

    def query_latest_occupancy(self, session_id: str) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM occupancy WHERE session_id=? ORDER BY timestamp DESC LIMIT 1",
                (session_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def query_alerts(self, session_id: str) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                """SELECT track_id, behaviour, started_at, detected_at, duration_s
                   FROM alerts WHERE session_id=? ORDER BY detected_at DESC""",
                (session_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def query_people_summary(self, session_id: str) -> list[dict]:
        """One row per track_id: first/last seen, dominant behaviour, alert count."""
        with self._lock:
            spans = self._conn.execute(
                """SELECT track_id, MIN(timestamp) AS first_seen, MAX(timestamp) AS last_seen
                   FROM detections WHERE session_id=? GROUP BY track_id""",
                (session_id,),
            ).fetchall()

            counts = self._conn.execute(
                """SELECT track_id, smoothed_behaviour AS behaviour, COUNT(*) AS n
                   FROM detections
                   WHERE session_id=? AND smoothed_behaviour IS NOT NULL AND smoothed_behaviour != ''
                   GROUP BY track_id, smoothed_behaviour""",
                (session_id,),
            ).fetchall()

            alert_counts = self._conn.execute(
                """SELECT track_id, COUNT(*) AS n FROM alerts
                   WHERE session_id=? GROUP BY track_id""",
                (session_id,),
            ).fetchall()

        dominant: dict[int, tuple[str, int]] = {}
        for row in counts:
            tid, behaviour, n = row["track_id"], row["behaviour"], row["n"]
            if tid not in dominant or n > dominant[tid][1]:
                dominant[tid] = (behaviour, n)

        alerts_by_track = {row["track_id"]: row["n"] for row in alert_counts}

        return [
            {
                "track_id": row["track_id"],
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "dominant_behaviour": dominant.get(row["track_id"], ("", 0))[0],
                "alert_count": alerts_by_track.get(row["track_id"], 0),
            }
            for row in spans
        ]

    # ------------------------------------------------------------------ #
    # Retention                                                            #
    # ------------------------------------------------------------------ #

    def prune_old_sessions(self, older_than_hours: float) -> int:
        """Delete closed sessions (and their rows in every table) older than
        the cutoff. Keeps the DB bounded on constrained storage. Returns the
        number of sessions removed."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=older_than_hours)).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "SELECT session_id FROM sessions WHERE end_time IS NOT NULL AND end_time < ?",
                (cutoff,),
            )
            ids = [r[0] for r in cur.fetchall()]
            for sid in ids:
                self._conn.execute("DELETE FROM detections WHERE session_id=?", (sid,))
                self._conn.execute("DELETE FROM movements WHERE session_id=?", (sid,))
                self._conn.execute("DELETE FROM occupancy WHERE session_id=?", (sid,))
                self._conn.execute("DELETE FROM alerts WHERE session_id=?", (sid,))
                self._conn.execute("DELETE FROM sessions WHERE session_id=?", (sid,))
            self._conn.commit()
            if ids:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return len(ids)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
