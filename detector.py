"""
YOLO26 behaviour detector.

The model is trained on the Student Behaviour Detection dataset with the
"phone" object class removed (see training/prepare_dataset.py), so one
forward pass returns a box + behaviour label per student — no separate
person-detector + classifier stage.

Class names are read from the loaded model itself (.pt or NCNN export
metadata), never hard-coded here, so the app always matches the weights.

Works with either weights format:
  - models/behaviour_yolo26n.pt                       (Mac / training machine)
  - models/pi/behaviour_yolo26n_ncnn_model/           (Raspberry Pi, NCNN FP16)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Priority order for resolving multiple detections on one person (most to
# least severe). Used so a confident "upright" can't bury a less-confident
# but significant "Using_phone"/"sleep" on the same student.
PRIORITY_ORDER: list[str] = [
    "sleep", "Using_phone", "turn_head", "bend", "bow_head",
    "hand-raising", "reading", "writing", "book", "raise_head", "upright",
]
_PRIORITY_RANK: dict[str, int] = {name: i for i, name in enumerate(PRIORITY_ORDER)}

# "Posture" classes: the box describes the person's own physical extent
# (head/torso). "Activity" classes: a small box near the hands/desk that
# belongs to whichever person it's next to. Used by _resolve_duplicates to
# decide what can anchor a person-cluster vs what should attach to one.
POSTURE_CLASSES: set[str] = {
    "upright", "bow_head", "raise_head", "turn_head", "bend", "sleep", "hand-raising",
}
ACTIVITY_CLASSES: set[str] = {"book", "Using_phone", "reading", "writing"}

# Containment ratio (intersection / area of the smaller box) needed to treat
# two boxes as the same physical person. Containment, not IoU: a tight
# head-box nested inside a wide torso-box has low IoU but high containment.
_MERGE_CONTAINMENT = 0.5

# Colour per class for visualisation (BGR), keyed by name.
CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "Using_phone":  (0,   0,   255),   # red
    "bend":         (0,   165, 255),   # orange
    "book":         (0,   255, 255),   # yellow
    "bow_head":     (255, 0,   0),     # blue
    "hand-raising": (0,   255, 0),     # green
    "raise_head":   (255, 255, 0),     # cyan
    "reading":      (200, 0,   200),   # magenta
    "sleep":        (128, 0,   128),   # purple
    "turn_head":    (255, 128, 0),     # sky blue
    "upright":      (200, 200, 200),   # light grey
    "writing":      (0,   128, 255),   # orange-yellow
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


def _containment(a: tuple[float, float, float, float],
                  b: tuple[float, float, float, float]) -> float:
    """Intersection area / area of the smaller of the two boxes (0 if disjoint)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return inter / min(area_a, area_b)


def _best_of(candidates: list[Detection]) -> Detection:
    """Highest-PRIORITY class wins, not highest confidence. Confidence only
    breaks ties within the same priority tier."""
    return min(
        candidates,
        key=lambda d: (_PRIORITY_RANK.get(d.class_name, len(PRIORITY_ORDER)), -d.confidence),
    )


def _resolve_duplicates(detections: list[Detection]) -> list[Detection]:
    """
    Collapse all boxes landing on one physical person, in one frame, down to
    a single detection. NMS only suppresses overlapping boxes of the SAME
    class, so one student commonly produces several simultaneous,
    differently-shaped boxes (torso "upright", head "turn_head", hand/desk
    "Using_phone"). Grouping by spatial *containment*:

      1. Cluster POSTURE_CLASSES boxes with each other by containment.
      2. Attach each activity box to the posture cluster it overlaps most.
      3. Within each cluster, the label is the highest-PRIORITY class present.

    The merged detection keeps the anchor's (largest posture box) bbox and
    track_id — the whole-body box is the stable thing to track — but the
    winning class's name/confidence. Activity boxes with no overlapping
    posture box pass through unchanged.
    """
    posture = [d for d in detections if d.class_name in POSTURE_CLASSES]
    activity = [d for d in detections if d.class_name not in POSTURE_CLASSES]

    parent = list(range(len(posture)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(posture)):
        for j in range(i + 1, len(posture)):
            if _containment(posture[i].bbox, posture[j].bbox) >= _MERGE_CONTAINMENT:
                union(i, j)

    clusters: dict[int, list[Detection]] = {}
    for i, d in enumerate(posture):
        clusters.setdefault(find(i), []).append(d)

    anchors: dict[int, Detection] = {
        root: max(members, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
        for root, members in clusters.items()
    }
    # Snapshot before attachment so matching is order-independent.
    posture_members = {root: list(members) for root, members in clusters.items()}

    unattached: list[Detection] = []
    for d in activity:
        best_root, best_ratio = None, 0.0
        for root, members in posture_members.items():
            ratio = max(_containment(d.bbox, m.bbox) for m in members)
            if ratio > best_ratio:
                best_root, best_ratio = root, ratio
        if best_root is not None and best_ratio >= _MERGE_CONTAINMENT:
            clusters[best_root].append(d)
        else:
            unattached.append(d)

    resolved: list[Detection] = []
    for root, members in clusters.items():
        winner = _best_of(members)
        anchor = anchors[root]
        resolved.append(Detection(
            frame_id=winner.frame_id,
            timestamp=winner.timestamp,
            class_id=winner.class_id,
            class_name=winner.class_name,
            confidence=winner.confidence,
            bbox=anchor.bbox,
            centroid=anchor.centroid,
            track_id=anchor.track_id,
        ))

    return resolved + unattached


class BehaviourDetector:
    """
    Wraps a YOLO26 model (.pt or NCNN export) and returns Detection objects
    per frame. Uses ultralytics' built-in ByteTrack/BoT-SORT so each
    detection also carries a persistent track_id across frames.
    """

    def __init__(self, cfg: dict):
        m = cfg["model"]
        self._conf   = m["confidence_threshold"]
        self._iou    = m["iou_threshold"]
        # [h, w] — the model is trained at 832; 16:9 camera frames letterbox
        # to 832x468, so a 480x832 input carries ~40% less compute than
        # 832x832 with no accuracy loss. Must match the NCNN export size.
        self._imgsz  = m.get("imgsz", [480, 832])
        self._device = m.get("device", "")

        # Lazy import so the module can be imported without ultralytics
        from ultralytics import YOLO
        self._model = YOLO(m["weights"], task="detect")
        self.class_names: dict[int, str] = dict(self._model.names)

        # .pt weights remember their training size; inferring at a different
        # scale silently costs accuracy on these small objects.
        train_imgsz = self._model.overrides.get("imgsz") if str(m["weights"]).endswith(".pt") else None
        long_side = max(self._imgsz) if isinstance(self._imgsz, (list, tuple)) else self._imgsz
        if isinstance(train_imgsz, int) and train_imgsz != long_side:
            h = math.ceil(train_imgsz * 9 / 16 / 32) * 32
            print(f"[WARN] model.imgsz {self._imgsz} doesn't match the weights' training "
                  f"imgsz {train_imgsz} — set it to [{h}, {train_imgsz}]")

        # A bare name resolves against ultralytics' bundled cfg/trackers/*.yaml;
        # tracking.tracker_config overrides it with a custom file.
        tracker = cfg["tracking"].get("tracker", "bytetrack")
        self._tracker_cfg = cfg["tracking"].get("tracker_config") or f"{tracker}.yaml"

    def raw_detections(self, frame_id: int, timestamp: str,
                       image: np.ndarray) -> list[Detection]:
        """Inference + tracking on one BGR frame, before per-person merging."""
        results = self._model.track(
            source=image,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=self._device or None,
            tracker=self._tracker_cfg,
            persist=True,       # keep tracker state between calls
            verbose=False,
        )

        detections: list[Detection] = []
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return detections

        boxes = results[0].boxes
        xyxy  = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss  = boxes.cls.cpu().numpy().astype(int)
        ids   = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            cls = int(clss[i])
            detections.append(Detection(
                frame_id=frame_id,
                timestamp=timestamp,
                class_id=cls,
                class_name=self.class_names.get(cls, str(cls)),
                confidence=float(confs[i]),
                bbox=(x1, y1, x2, y2),
                centroid=((x1 + x2) / 2, (y1 + y2) / 2),
                track_id=int(ids[i]) if ids is not None else -1,
            ))
        return detections

    def detect(self, frame_id: int, timestamp: str, image: np.ndarray) -> list[Detection]:
        """Run inference + tracking on one BGR frame. One Detection per student."""
        return _resolve_duplicates(self.raw_detections(frame_id, timestamp, image))

    @staticmethod
    def annotate(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels onto a copy of the image."""
        import cv2
        out = image.copy()
        for d in detections:
            x1, y1, x2, y2 = map(int, d.bbox)
            colour = CLASS_COLOURS.get(d.class_name, (200, 200, 200))
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            label = f"{d.class_name} {d.confidence:.2f}"
            if d.track_id >= 0:
                label = f"#{d.track_id} {label}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return out
