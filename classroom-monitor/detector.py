"""
YOLOv8 behaviour detector.

This model was trained on the Student Behaviour Detection dataset which
annotates students with their behaviour class directly (12 classes).
One forward pass returns both the bounding box and the behaviour label,
so no separate person-detector + classifier pipeline is needed.

Classes:
  0:Using_phone  1:bend        2:book       3:bow_head
  4:hand-raising 5:phone       6:raise_head 7:reading
  8:sleep        9:turn_head   10:upright   11:writing
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Class index → human-readable label
CLASS_NAMES: dict[int, str] = {
    0: "Using_phone",
    1: "bend",
    2: "book",
    3: "bow_head",
    4: "hand-raising",
    5: "phone",
    6: "raise_head",
    7: "reading",
    8: "sleep",
    9: "turn_head",
    10: "upright",
    11: "writing",
}

# Priority order for resolving multiple detections on one track (most to
# least severe). Used as a tie-breaker when two classes have equal confidence.
PRIORITY_ORDER: list[str] = [
    "sleep", "Using_phone", "phone", "turn_head", "bend", "bow_head",
    "hand-raising", "reading", "writing", "book", "raise_head", "upright",
]
_PRIORITY_RANK: dict[str, int] = {name: i for i, name in enumerate(PRIORITY_ORDER)}

# Colour per class for visualisation (BGR)
CLASS_COLOURS: dict[int, tuple[int, int, int]] = {
    0:  (0,   0,   255),   # Using_phone  – red
    1:  (0,   165, 255),   # bend         – orange
    2:  (0,   255, 255),   # book         – yellow
    3:  (255, 0,   0),     # bow_head     – blue
    4:  (0,   255, 0),     # hand-raising – green
    5:  (0,   0,   200),   # phone        – dark red
    6:  (255, 255, 0),     # raise_head   – cyan
    7:  (200, 0,   200),   # reading      – magenta
    8:  (128, 0,   128),   # sleep        – purple
    9:  (255, 128, 0),     # turn_head    – sky blue
    10: (200, 200, 200),   # upright      – light grey
    11: (0,   128, 255),   # writing      – orange-yellow
}


@dataclass
class Detection:
    frame_id:        int
    timestamp:       str
    class_id:        int
    class_name:      str
    confidence:      float
    bbox:            tuple[float, float, float, float]   # x1,y1,x2,y2 pixels
    centroid:        tuple[float, float]                 # cx, cy pixels
    track_id:        int = -1                            # set by ultralytics ByteTrack


def _resolve_duplicates(detections: list[Detection]) -> list[Detection]:
    """
    Collapse multiple class detections that landed on the same track_id in
    one frame (NMS doesn't suppress overlapping boxes of different classes)
    down to a single detection per person: highest confidence wins, ties
    broken by PRIORITY_ORDER. Untracked detections (track_id == -1) pass
    through unchanged since they can't be matched to a person.
    """
    best: dict[int, Detection] = {}
    passthrough: list[Detection] = []
    for d in detections:
        if d.track_id < 0:
            passthrough.append(d)
            continue
        current = best.get(d.track_id)
        if current is None or d.confidence > current.confidence:
            best[d.track_id] = d
        elif d.confidence == current.confidence:
            if _PRIORITY_RANK.get(d.class_name, len(PRIORITY_ORDER)) < \
               _PRIORITY_RANK.get(current.class_name, len(PRIORITY_ORDER)):
                best[d.track_id] = d
    return list(best.values()) + passthrough


class BehaviourDetector:
    """
    Wraps a YOLOv8 model and returns Detection objects per frame.
    Uses ultralytics' built-in ByteTrack so each detection also
    carries a persistent track_id across frames.
    """

    def __init__(self, cfg: dict):
        weights   = cfg["model"]["weights"]
        self._conf = cfg["model"]["confidence_threshold"]
        self._iou  = cfg["model"]["iou_threshold"]
        self._imgsz = cfg["model"].get("imgsz", 640)
        self._device = cfg["model"].get("device", "")
        self._tracker = cfg["tracking"].get("tracker", "bytetrack")

        # Lazy import so the module can be imported without ultralytics
        from ultralytics import YOLO
        self._model = YOLO(weights)

        # A bare name resolves against ultralytics' own bundled
        # cfg/trackers/*.yaml; tracking.tracker_config overrides it with a
        # path to a custom tracker file (e.g. trackers/botsort_reid.yaml)
        # when the shipped defaults need tuning (track_buffer, ReID, ...).
        self._tracker_cfg = cfg["tracking"].get("tracker_config") or f"{self._tracker}.yaml"

    def detect(self, frame_id: int, timestamp: str, image: np.ndarray) -> list[Detection]:
        """
        Run inference + tracking on one BGR frame.
        Returns one Detection per tracked student.
        """
        results = self._model.track(
            source=image,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=self._device,
            tracker=self._tracker_cfg,
            persist=True,       # keep tracker state between calls
            verbose=False,
        )

        detections: list[Detection] = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
            conf  = float(boxes.conf[i].cpu().numpy())
            cls   = int(boxes.cls[i].cpu().numpy())
            name  = CLASS_NAMES.get(cls, str(cls))
            cx    = (x1 + x2) / 2
            cy    = (y1 + y2) / 2

            # Extract ByteTrack ID if available
            tid = -1
            if boxes.id is not None:
                tid = int(boxes.id[i].cpu().numpy())

            detections.append(Detection(
                frame_id=frame_id,
                timestamp=timestamp,
                class_id=cls,
                class_name=name,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                centroid=(cx, cy),
                track_id=tid,
            ))

        return _resolve_duplicates(detections)

    def annotate(self, image: np.ndarray, detections: list[Detection],
                 track_ids: dict[int, int] | None = None) -> np.ndarray:
        """Draw bounding boxes and labels onto a copy of the image."""
        import cv2
        out = image.copy()
        for d in detections:
            x1, y1, x2, y2 = map(int, d.bbox)
            colour = CLASS_COLOURS.get(d.class_id, (200, 200, 200))
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            label = f"{d.class_name} {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    def export_onnx(self, output_dir: str = "models") -> str:
        """Export the loaded model to ONNX format (for Pi 5 deployment)."""
        path = self._model.export(format="onnx", opset=17, dynamic=False, imgsz=self._imgsz)
        return str(path)
