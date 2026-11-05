#!/usr/bin/env python3
"""
Run the detector (same model/thresholds/merging as the app) over a folder
of images and save annotated copies — one box per student, labelled with
the winning behaviour.

Usage:
    cd "Embedded 2"
    python tools/visualize.py                                   # samples/ -> outputs/visualize/
    python tools/visualize.py --images dataset/test/images --raw # every box, before merging
    python tools/visualize.py --config config.pi.yaml            # check the NCNN export
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector import BehaviourDetector, _resolve_duplicates  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Annotate images with detector output")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--images", default=str(ROOT / "samples"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "visualize"))
    p.add_argument("--raw", action="store_true", help="Draw every box, skip per-person merging")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))
    det = BehaviourDetector(cfg)
    w, h = cfg["camera"]["width"], cfg["camera"]["height"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in Path(args.images).iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    counts: Counter = Counter()
    ts = datetime.now(timezone.utc).isoformat()

    for i, path in enumerate(images):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        # Same resolution the camera delivers, so results match the live app.
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        dets = det.raw_detections(i, ts, frame)
        if not args.raw:
            dets = _resolve_duplicates(dets)
        counts.update(d.class_name for d in dets)
        cv2.imwrite(str(out_dir / path.name), det.annotate(frame, dets))

    print(f"[DONE] {len(images)} images -> {out_dir}")
    for name, n in counts.most_common():
        print(f"  {name:<13} {n}")


if __name__ == "__main__":
    main()
