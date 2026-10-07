"""
Train YOLO11s (default) on the merged-class behaviour dataset.

Run prepare_dataset.py first to generate ./dataset/data.yaml.

Usage:
    cd pi-deploy
    python train.py                              # YOLO11s, defaults below
    python train.py --model yolo11n.pt            # lighter, if YOLO11s is too slow to train
    python train.py --epochs 150 --patience 30
    python train.py --device mps                  # Apple Silicon (training machine)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA_YAML = HERE / "dataset" / "data.yaml"


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO11 behaviour detector")
    p.add_argument("--model", default="yolo11s.pt",
                    help="Base weights: yolo11n.pt / yolo11s.pt / yolo11m.pt")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="",
                    help="'' = auto, 'cpu', '0' = GPU 0, 'mps' = Apple Silicon")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="behaviour_detector_v2")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    if not DATA_YAML.exists():
        print(f"[ERROR] {DATA_YAML} not found. Run prepare_dataset.py first.")
        sys.exit(1)

    model = YOLO(args.model)

    print(f"[INFO] Model:    {args.model}")
    print(f"[INFO] Epochs:   {args.epochs} (patience={args.patience})")
    print(f"[INFO] Batch:    {args.batch}  Imgsz: {args.imgsz}")
    print(f"[INFO] Device:   {args.device or 'auto'}")
    print(f"[INFO] Data:     {DATA_YAML}")

    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device or None,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=args.patience,
        # Augmentations (same recipe as the v1 yolov8n run)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        plots=True,
        save=True,
        save_period=10,
        val=True,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    dest_pt = HERE / "models" / "behaviour_detector_v2.pt"
    dest_pt.parent.mkdir(exist_ok=True)

    if best_pt.exists():
        shutil.copy(best_pt, dest_pt)
        print(f"\n[INFO] Best weights copied to: {dest_pt}")
        print("[INFO] Next: python export_pi.py --weights "
              f"{dest_pt}")
    else:
        print("[WARN] best.pt not found — check training output directory.")


if __name__ == "__main__":
    main()
