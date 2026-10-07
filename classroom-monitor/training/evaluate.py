"""
Evaluate a trained behaviour detector on the test split.

Usage:
    cd classroom-monitor
    python training/evaluate.py
    python training/evaluate.py --weights models/behaviour_detector.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATASET_ROOT = (
    Path(__file__).parent.parent.parent
    / "Student Behaviour Detection.v1i.yolov8 2"
).resolve()

CLASS_NAMES = [
    "Using_phone", "bend", "book", "bow_head",
    "hand-raising", "phone", "raise_head", "reading",
    "sleep", "turn_head", "upright", "writing",
]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLOv8 behaviour detector")
    p.add_argument("--weights", default="models/behaviour_detector.pt")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--iou",     type=float, default=0.45)
    p.add_argument("--device",  default="")
    p.add_argument("--split",   default="test", choices=["train", "val", "test"])
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.")
        sys.exit(1)

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights}")
        print("        Run training/train.py first.")
        sys.exit(1)

    # Build data.yaml for validation
    import yaml
    data_yaml = Path(__file__).parent / "data.yaml"
    if not data_yaml.exists():
        d = {
            "train": str(DATASET_ROOT / "train" / "images"),
            "val":   str(DATASET_ROOT / "valid" / "images"),
            "test":  str(DATASET_ROOT / "test"  / "images"),
            "nc":    12,
            "names": CLASS_NAMES,
        }
        with open(data_yaml, "w") as f:
            yaml.dump(d, f)

    print(f"[INFO] Evaluating: {weights}")
    print(f"[INFO] Split:      {args.split}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device or None,
        verbose=True,
        plots=True,
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"mAP@0.50:      {metrics.box.map50:.4f}")
    print(f"mAP@0.50-0.95: {metrics.box.map:.4f}")
    print(f"Precision:     {metrics.box.mp:.4f}")
    print(f"Recall:        {metrics.box.mr:.4f}")
    print("=" * 60)

    if hasattr(metrics.box, "ap_class_index"):
        print("\nPer-class AP@0.50:")
        for idx, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
            name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
            bar  = "█" * int(ap * 30)
            print(f"  {name:<15} {ap:.4f}  {bar}")


if __name__ == "__main__":
    main()
