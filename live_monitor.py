"""
Background runner for the live camera pipeline, built for the Flask
dashboard: owns the camera + detector + tracker + analytics stack in a
daemon thread and exposes thread-safe reads of the latest annotated
frame (JPEG bytes) and current stats — including a per-person roster and
sustained-behaviour alerts. Not a GUI window — the browser is the display.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone

import cv2

from analytics import AnalyticsEngine
from capture import FrameProducer
from database import DatabaseManager
from detector import BehaviourDetector
from movement import MovementAnalyzer
from occupancy import OccupancyTracker
from tracker import TrackRegistry

ENGAGED_CLASSES = {"upright", "hand-raising", "reading", "writing", "book", "raise_head"}


class ProcessingRate:
    """Exponential moving average of frames actually processed per second.
    Movement analysis needs this, not the camera fps: on a Pi 4 the
    pipeline processes a few fps, so sizing time windows by the camera's
    30fps would stretch a "2 second" window to ~15 seconds."""

    def __init__(self, initial: float = 5.0, alpha: float = 0.1):
        self.fps = initial
        self._alpha = alpha
        self._last: float | None = None

    def tick(self) -> float:
        now = time.monotonic()
        if self._last is not None and now > self._last:
            self.fps += self._alpha * (1.0 / (now - self._last) - self.fps)
        self._last = now
        return self.fps


class RateLimiter:
    """Caps the processing loop at max_fps by sleeping off the rest of each
    period, so a Pi 4 isn't pinned at 100% CPU (and thermally throttling)
    when a couple of fps is all the analytics need. 0 / None = no cap.
    Call wait() *before* reading a frame, so the frame processed is the
    newest one rather than one that aged during the pause."""

    def __init__(self, max_fps: float | None):
        self._period = 1.0 / max_fps if max_fps else 0.0
        self._next = time.monotonic()

    def wait(self) -> None:
        if not self._period:
            return
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next = time.monotonic() + self._period


def encode_jpeg(image, width: int, quality: int = 75) -> bytes | None:
    h, w = image.shape[:2]
    if w > width:
        image = cv2.resize(image, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


class LiveMonitor:
    def __init__(self, cfg: dict, db: DatabaseManager):
        self._cfg = cfg
        self._db = db
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_stats: dict = {}
        self._running = False
        self._thread: threading.Thread | None = None

        alerts_cfg = cfg.get("alerts", {})
        self._alert_threshold_s = alerts_cfg.get("threshold_seconds", 5.0)
        self._alert_classes = set(alerts_cfg.get(
            "classes", ["sleep", "Using_phone", "turn_head"]))
        # Dashboard stream is downscaled before JPEG encoding — encoding full
        # 1280x720 every frame steals CPU the detector needs on a Pi.
        self._stream_width = cfg.get("dashboard", {}).get("stream_width", 640)
        self._proc_fps = ProcessingRate()

        # track_id -> (behaviour, start_ts float)
        self._behaviour_since: dict[int, tuple[str, float]] = {}
        # track_id -> set of start_ts already alerted, so a sustained
        # behaviour only fires one alert, not one per frame past threshold
        self._alerted: dict[int, set[float]] = {}
        self._recent_alerts: deque = deque(maxlen=100)

        self._detector: BehaviourDetector | None = None
        self._producer: FrameProducer | None = None
        self._registry: TrackRegistry | None = None
        self._analyzer: MovementAnalyzer | None = None
        self._occ_tracker: OccupancyTracker | None = None
        self._engine: AnalyticsEngine | None = None
        self.session_id: str | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> str:
        """Idempotent: returns the active session_id, starting the pipeline if needed."""
        with self._lock:
            if self._running:
                return self.session_id

            cfg = self._cfg
            self._detector    = BehaviourDetector(cfg)
            self._producer    = FrameProducer(cfg["camera"]).start()
            self._registry    = TrackRegistry(cfg)
            self._analyzer    = MovementAnalyzer(cfg,
                                                  frame_width=cfg["camera"]["width"],
                                                  frame_height=cfg["camera"]["height"])
            self._occ_tracker = OccupancyTracker(cfg)
            self._engine       = AnalyticsEngine(cfg, self._db)
            self.session_id    = self._db.new_session(str(cfg["camera"]["source"]))
            self._behaviour_since.clear()
            self._alerted.clear()
            self._recent_alerts.clear()

            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            return self.session_id

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False

        if self._thread:
            self._thread.join(timeout=3)

        if self._producer:
            self._producer.stop()
        if self._engine:
            self._engine.flush_all()
        if self.session_id:
            self._db.close_session(self.session_id)

        with self._lock:
            self._latest_jpeg = None
            self._latest_stats = {}

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._latest_stats)

    # ------------------------------------------------------------------ #
    # Background loop                                                      #
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        cfg = self._cfg
        # Capture is latest-frame-only, so the detector naturally skips
        # whatever it's too slow for; frame_skip is only for deliberately
        # throttling below that (e.g. to leave CPU headroom).
        frame_skip = cfg["analytics"].get("frame_skip", 1)
        frame_count = 0
        limiter = RateLimiter(cfg["analytics"].get("max_fps", 0))

        while self._running:
            limiter.wait()
            frame = self._producer.read(timeout=1.0)
            if frame is None:
                continue

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            proc_fps = self._proc_fps.tick()
            raw_detections = self._detector.detect(
                frame_id=frame.frame_id, timestamp=frame.timestamp, image=frame.image,
            )
            tracked = self._registry.update(raw_detections, frame.frame_id, frame.timestamp)
            active  = [t for t in tracked if not t.is_exit]
            exits   = [t for t in tracked if t.is_exit]

            movements = [self._analyzer.analyze(t, fps=proc_fps) for t in tracked]
            zone_by_track = {m.track_id: m for m in movements}

            occ = self._occ_tracker.update(self._registry.active_count(), frame.timestamp)
            self._engine.ingest(self.session_id, frame.frame_id, tracked, movements, occ)

            for t in exits:
                self._analyzer.purge_track(t.track_id)
                self._behaviour_since.pop(t.track_id, None)
                self._alerted.pop(t.track_id, None)

            roster = self._build_roster(active, zone_by_track, frame.timestamp)

            annotated = self._detector.annotate(frame.image, raw_detections)
            jpeg = encode_jpeg(annotated, self._stream_width)
            if jpeg is None:
                continue

            engaged = sum(1 for r in roster if r["behaviour"] in ENGAGED_CLASSES)
            engagement_pct = round(100 * engaged / len(roster), 1) if roster else None

            stats = {
                "session_id":     self.session_id,
                "timestamp":      frame.timestamp,
                "active_tracks":  self._registry.active_count(),
                "processing_fps": round(proc_fps, 1),
                "occupancy": {
                    "current": occ.current, "peak": occ.peak, "average": occ.average,
                } if occ else None,
                "engagement_pct": engagement_pct,
                "roster":         roster,
                "recent_alerts":  list(self._recent_alerts),
            }

            with self._lock:
                self._latest_jpeg = jpeg
                self._latest_stats = stats

    def _build_roster(self, active: list, zone_by_track: dict, timestamp: str) -> list[dict]:
        now = self._parse_ts(timestamp)
        roster = []
        for t in active:
            behaviour = t.smoothed_behaviour
            prev = self._behaviour_since.get(t.track_id)
            if prev is None or prev[0] != behaviour:
                start_ts = now
                self._behaviour_since[t.track_id] = (behaviour, start_ts)
            else:
                start_ts = prev[1]

            duration_s = max(0.0, now - start_ts)
            mv = zone_by_track.get(t.track_id)

            roster.append({
                "track_id":   t.track_id,
                "behaviour":  behaviour,
                "confidence": round(t.confidence, 2),
                "zone":       mv.zone if mv else None,
                "is_moving":  mv.is_moving if mv else False,
                "direction":  mv.direction if mv else None,
                "duration_s": round(duration_s, 1),
            })

            self._maybe_alert(t.track_id, behaviour, start_ts, duration_s, timestamp)

        return roster

    def _maybe_alert(self, track_id: int, behaviour: str, start_ts: float,
                      duration_s: float, timestamp: str) -> None:
        if behaviour not in self._alert_classes or duration_s < self._alert_threshold_s:
            return
        already = self._alerted.setdefault(track_id, set())
        if start_ts in already:
            return
        already.add(start_ts)

        started_at = datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat()
        self._db.insert_alert(self.session_id, track_id, behaviour,
                               started_at, timestamp, duration_s)
        self._recent_alerts.appendleft({
            "track_id": track_id, "behaviour": behaviour,
            "duration_s": round(duration_s, 1), "detected_at": timestamp,
        })

    @staticmethod
    def _parse_ts(timestamp: str) -> float:
        try:
            return datetime.fromisoformat(timestamp).timestamp()
        except ValueError:
            return time.time()
