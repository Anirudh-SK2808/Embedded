"""
Classroom Activity Monitoring System — main inference pipeline.

Usage:
    python main.py                     # uses config.yaml defaults
    python main.py --source 0          # webcam 0
    python main.py --source video.mp4  # video file
    python main.py --source rtsp://... # RTSP stream
    python main.py --config config.pi.yaml --no-display   # Raspberry Pi 4, headless
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

import cv2
import yaml

# ------------------------------------------------------------------ #
# Project imports                                                      #
# ------------------------------------------------------------------ #

from analytics  import AnalyticsEngine
from capture    import FrameProducer
from database   import DatabaseManager
from detector   import BehaviourDetector
from live_monitor import ProcessingRate, RateLimiter
from movement   import MovementAnalyzer
from occupancy  import OccupancyTracker
from tracker    import TrackRegistry


def load_cfg(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Classroom Activity Monitor")
    p.add_argument("--config",     default="config.yaml")
    p.add_argument("--source",     default=None,
                   help="Override camera source (index / path / RTSP URL)")
    p.add_argument("--no-display", action="store_true",
                   help="Do not open OpenCV preview window (Pi / headless)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_cfg(args.config)

    if args.source is not None:
        cfg["camera"]["source"] = args.source

    # ---------------------------------------------------------------- #
    # Initialise modules                                                 #
    # ---------------------------------------------------------------- #

    print("[INFO] Loading model …")
    detector = BehaviourDetector(cfg)

    print("[INFO] Starting camera …")
    producer = FrameProducer(cfg["camera"]).start()

    registry  = TrackRegistry(cfg)
    analyzer  = MovementAnalyzer(cfg,
                                  frame_width=cfg["camera"]["width"],
                                  frame_height=cfg["camera"]["height"])
    occ_tracker = OccupancyTracker(cfg)
    db          = DatabaseManager(cfg["database"]["path"])
    engine      = AnalyticsEngine(cfg, db)

    session_id  = db.new_session(str(cfg["camera"]["source"]))
    print(f"[INFO] Session ID: {session_id}")

    frame_skip  = cfg["analytics"].get("frame_skip", 1)
    frame_count = 0
    proc_rate   = ProcessingRate()
    limiter     = RateLimiter(cfg["analytics"].get("max_fps", 0))

    show_window = not args.no_display

    # ---------------------------------------------------------------- #
    # Graceful shutdown                                                  #
    # ---------------------------------------------------------------- #

    running = True

    def _shutdown(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("[INFO] Pipeline running. Press Ctrl+C to stop.")

    # ---------------------------------------------------------------- #
    # Inference loop                                                     #
    # ---------------------------------------------------------------- #

    t0 = time.time()
    processed = 0

    while running:
        limiter.wait()
        frame = producer.read(timeout=1.0)
        if frame is None:
            continue

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        proc_fps = proc_rate.tick()

        # 1 — Detect + behaviour classify
        raw_detections = detector.detect(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            image=frame.image,
        )

        # 2 — Track registry: entry/exit, position history, smoothing
        tracked = registry.update(raw_detections, frame.frame_id, frame.timestamp)

        active  = [t for t in tracked if not t.is_exit]
        exits   = [t for t in tracked if t.is_exit]

        # 3 — Movement analysis
        movements = [analyzer.analyze(t, fps=proc_fps) for t in tracked]

        # 4 — Occupancy
        occ = occ_tracker.update(registry.active_count(), frame.timestamp)

        # 5 — Analytics engine → batched DB write
        engine.ingest(session_id, frame.frame_id,
                      tracked, movements, occ)

        # Purge movement history for exited tracks
        for t in exits:
            analyzer.purge_track(t.track_id)

        # 6 — Annotate frame for the optional live preview
        annotated = detector.annotate(frame.image, raw_detections)

        # 7 — Optional live preview window
        if show_window:
            cv2.imshow("Classroom Monitor — press Q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        processed += 1
        if processed % 100 == 0:
            elapsed = time.time() - t0
            fps = processed / elapsed
            print(f"[INFO] Frames processed: {processed} | "
                  f"FPS: {fps:.1f} | Active tracks: {registry.active_count()}")

    # ---------------------------------------------------------------- #
    # Shutdown                                                           #
    # ---------------------------------------------------------------- #

    print("[INFO] Shutting down …")
    engine.flush_all()
    db.close_session(session_id)
    db.close()
    producer.stop()
    if show_window:
        cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
