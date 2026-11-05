"""
Video frame capture — background thread that always holds only the
newest frame.

Two backends:
  - picamera2: Raspberry Pi Camera Module (ribbon cable). On Raspberry Pi
    OS Bookworm the camera goes through libcamera, which cv2.VideoCapture
    can't open reliably — picamera2 is the supported path.
  - opencv: USB webcam index, video file, or RTSP URL.

Latest-frame-only (not a FIFO queue): on a Pi 4 the detector runs a few
fps, so any queue just makes the output lag behind reality. The consumer
gets whatever was captured most recently and stale frames are dropped.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Frame:
    frame_id: int
    timestamp: str          # ISO 8601
    image: np.ndarray       # BGR, shape (H, W, 3)


class FrameProducer:
    """
    Captures frames in a background thread. read() blocks until a frame
    newer than the last one returned is available (or timeout), so the
    consumer never processes the same frame twice and never falls behind.
    """

    def __init__(self, cfg: dict):
        self._source = cfg["source"]
        self._width  = cfg["width"]
        self._height = cfg["height"]
        self._fps    = cfg.get("fps", 30)
        backend = cfg.get("backend", "auto")
        if backend == "auto":
            backend = "picamera2" if str(self._source) == "picamera" else "opencv"
        self._backend = backend
        # Video files are replayed at their native rate (not as fast as
        # the CPU can decode) so timestamps/durations stay realistic.
        self._is_file = backend == "opencv" and Path(str(self._source)).is_file()

        self._cap: cv2.VideoCapture | None = None
        self._picam = None
        self._frame_id = 0
        self._latest: Frame | None = None
        self._cond = threading.Condition()
        self._last_read_id = -1
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self) -> "FrameProducer":
        if self._backend == "picamera2":
            self._open_picamera()
        else:
            self._open_opencv()
        self._thread.start()
        return self

    def _open_picamera(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise RuntimeError(
                "picamera2 not available. On the Pi: sudo apt install python3-picamera2 "
                "and create the venv with --system-site-packages (see README)."
            ) from e
        self._picam = Picamera2()
        # RGB888 in picamera2 is stored B,G,R in memory — exactly OpenCV's layout.
        config = self._picam.create_video_configuration(
            main={"size": (self._width, self._height), "format": "RGB888"},
            controls={"FrameRate": float(self._fps)},
            buffer_count=4,
        )
        self._picam.configure(config)
        self._picam.start()

    def _open_opencv(self) -> None:
        src = int(self._source) if str(self._source).isdigit() else self._source
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self._source!r}")
        if not self._is_file:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # don't let the driver queue stale frames

    def _grab(self) -> np.ndarray | None:
        if self._picam is not None:
            return self._picam.capture_array("main")
        ok, img = self._cap.read()
        if not ok:
            if self._is_file:   # loop video files so demos keep running
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return None
        return img

    def _capture_loop(self) -> None:
        file_period = 1.0 / (self._cap.get(cv2.CAP_PROP_FPS) or self._fps) if self._is_file else 0.0
        next_t = time.monotonic()
        while not self._stop_event.is_set():
            img = self._grab()
            if img is None:
                time.sleep(0.02)
                continue

            if img.shape[1] != self._width or img.shape[0] != self._height:
                img = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_AREA)

            f = Frame(frame_id=self._frame_id,
                      timestamp=datetime.now(timezone.utc).isoformat(),
                      image=img)
            self._frame_id += 1
            with self._cond:
                self._latest = f
                self._cond.notify_all()

            if file_period:
                next_t += file_period
                time.sleep(max(0.0, next_t - time.monotonic()))

    def read(self, timeout: float = 1.0) -> Frame | None:
        with self._cond:
            self._cond.wait_for(
                lambda: self._latest is not None and self._latest.frame_id != self._last_read_id,
                timeout=timeout,
            )
            f = self._latest
            if f is None or f.frame_id == self._last_read_id:
                return None
            self._last_read_id = f.frame_id
            return f

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        if self._picam:
            self._picam.stop()
            self._picam.close()
