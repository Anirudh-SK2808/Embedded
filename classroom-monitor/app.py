"""
Dashboard — Live camera monitoring + historical session analytics.
SQLite-backed via database.py; the live pipeline runs in a background
thread (live_monitor.py) so it only starts when you open the Live page.

Usage:
    python app.py                      # uses config.yaml
    python app.py --config config.pi_sim.yaml
"""

from __future__ import annotations

import argparse
import csv
import io
import time

import yaml
from flask import Flask, Response, jsonify, render_template, stream_with_context

from database import DatabaseManager
from live_monitor import LiveMonitor


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def create_app(cfg: dict) -> Flask:
    app = Flask(__name__)

    # One shared connection for the whole process — DatabaseManager guards
    # it with an internal lock, so the live pipeline's writer thread and
    # Flask's request threads can both use it safely without each request
    # opening/closing its own sqlite3 connection.
    db = DatabaseManager(cfg["database"]["path"])
    db.prune_old_sessions(cfg["database"].get("retention_hours", 72))

    monitor = LiveMonitor(cfg, db)

    # ------------------------------------------------------------------ #
    # Pages                                                                #
    # ------------------------------------------------------------------ #

    @app.route("/")
    def index():
        return render_template("live.html", active="live")

    @app.route("/live")
    def live():
        return render_template("live.html", active="live")

    @app.route("/history")
    def history():
        return render_template("history.html", active="history")

    # ------------------------------------------------------------------ #
    # Live API                                                             #
    # ------------------------------------------------------------------ #

    @app.route("/api/live/start", methods=["POST"])
    def live_start():
        session_id = monitor.start()
        return jsonify({"running": True, "session_id": session_id})

    @app.route("/api/live/stop", methods=["POST"])
    def live_stop():
        monitor.stop()
        return jsonify({"running": False})

    @app.route("/api/live/stats")
    def live_stats():
        stats = monitor.get_stats()
        stats["running"] = monitor.is_running
        return jsonify(stats)

    @app.route("/video_feed")
    def video_feed():
        def gen():
            while True:
                jpeg = monitor.get_jpeg()
                if jpeg is None:
                    time.sleep(0.1)
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(0.03)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # ------------------------------------------------------------------ #
    # History API                                                          #
    # ------------------------------------------------------------------ #

    @app.route("/api/sessions")
    def api_sessions():
        return jsonify(db.list_sessions())

    @app.route("/api/history/<session_id>/summary")
    def api_history_summary(session_id):
        return jsonify({
            "behaviour_distribution": db.query_behaviour_distribution(session_id),
            "zone_counts":            db.query_zone_counts(session_id),
            "entry_exit":             db.query_entry_exit_events(session_id),
            "latest_occupancy":       db.query_latest_occupancy(session_id),
            "track_ids":              db.query_track_ids(session_id),
        })

    @app.route("/api/history/<session_id>/occupancy")
    def api_history_occupancy(session_id):
        return jsonify(db.query_occupancy(session_id))

    @app.route("/api/history/<session_id>/track/<int:track_id>")
    def api_history_track(session_id, track_id):
        return jsonify(db.query_person_timeline(session_id, track_id))

    @app.route("/api/history/<session_id>/people")
    def api_history_people(session_id):
        return jsonify(db.query_people_summary(session_id))

    @app.route("/api/history/<session_id>/alerts")
    def api_history_alerts(session_id):
        return jsonify(db.query_alerts(session_id))

    @app.route("/api/history/<session_id>/export.csv")
    def api_history_export(session_id):
        rows = db.query_detections(session_id)

        def gen():
            buf = io.StringIO()
            writer = csv.writer(buf)
            if not rows:
                yield ""
                return
            writer.writerow(rows[0].keys())
            yield buf.getvalue()
            buf.seek(0); buf.truncate(0)
            for row in rows:
                writer.writerow(row.values())
                yield buf.getvalue()
                buf.seek(0); buf.truncate(0)

        return Response(
            stream_with_context(gen()),
            mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="session_{session_id}.csv"'},
        )

    return app


def parse_args():
    p = argparse.ArgumentParser(description="Classroom Monitor Dashboard")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5050)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_cfg(args.config)
    app = create_app(cfg)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
