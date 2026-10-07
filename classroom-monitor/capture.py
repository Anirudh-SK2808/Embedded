"""Video frame capture — thread-safe producer with a bounded queue."""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from queue import Empty, Full, Queue

import cv2
import numpy as np


@dataclass
class Frame:
    frame_id: int
    timestamp: str          # ISO 8601
    image: np.ndarray       # BGR, shape (H, W, 3)


class FrameProducer:
    """
    Captures frames from a camera source in a background thread and
    exposes them via read(). Drops frames when the queue is full so
    downstream modules never block the capture loop.
    """

    def __init__(self, cfg: dict):
        self._source = cfg["source"]
        self._width  = cfg["width"]
        self._height = cfg["height"]
        self._fps    = cfg.get("fps", 30)
        self._q: Queue[Frame] = Queue(maxsize=cfg.get("queue_depth", 4))

        self._cap: cv2.VideoCapture | None = None
        self._frame_id = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self) -> "FrameProducer":
        self._open_capture()
        self._thread.start()
        return self

    def _open_capture(self) -> None:
        src = int(self._source) if str(self._source).isdigit() else self._source
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self._source!r}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            ok, img = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            img = cv2.resize(img, (self._width, self._height))
            ts  = datetime.now(timezone.utc).isoformat()
            f   = Frame(frame_id=self._frame_id, timestamp=ts, image=img)
            self._frame_id += 1

            try:
                self._q.put_nowait(f)
            except Full:
                # Drop oldest, put newest
                try:
                    self._q.get_nowait()
                except Empty:
                    pass
                try:
                    self._q.put_nowait(f)
                except Full:
                    pass

    def read(self, timeout: float = 1.0) -> Frame | None:
        try:
            return self._q.get(timeout=timeout)
        except Empty:
            return None

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
