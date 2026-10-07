"""
Train YOLOv8 on the Student Behaviour Detection dataset.

Usage:
    cd classroom-monitor
    python training/train.py                          # default (batch=128, early stopping)
    python training/train.py --model yolov8s.pt       # larger model
    python training/train.py --epochs 500 --patience 25  # custom early stopping
    python training/train.py --resume                 # resume last run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

DATASET_ROOT = (
    Path(__file__).parent.parent.parent
    / "Student Behaviour Detection.v1i.yolov8 2"
).resolve()


def build_data_yaml(output_path: Path) -> Path:
    """
    Write a data.yaml with absolute paths so it works regardless of cwd.
    """
    data = {
        "train": str(DATASET_ROOT / "train" / "images"),
        "val":   str(DATASET_ROOT / "valid" / "images"),
        "test":  str(DATASET_ROOT / "test"  / "images"),
        "nc":    12,
        "names": [
            "Using_phone", "bend", "book", "bow_head",
            "hand-raising", "phone", "raise_head", "reading",
            "sleep", "turn_head", "upright", "writing",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"[INFO] data.yaml written to: {output_path}")
    return output_path


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 behaviour detector")
    p.add_argument("--model",   default="yolov8n.pt",
                   help="Base weights: yolov8n.pt / yolov8s.pt / yolov8m.pt")
    p.add_argument("--epochs",  type=int, default=100,
                   help="Max epochs (early stopping may end sooner)")
    p.add_argument("--batch",   type=int, default=64)
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience (stop if no improvement for N epochs)")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--device",  default="",
                   help="'' = auto, 'cpu', '0' = GPU 0, 'mps' = Apple Silicon")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", default="training/runs",
                   help="Directory for training outputs")
    p.add_argument("--name",    default="behaviour_detector",
                   help="Run name")
    p.add_argument("--resume",  action="store_true",
                   help="Resume last interrupted training run")
    p.add_argument("--export-onnx", action="store_true",
                   help="After training, export best.pt to ONNX")
    p.add_argument("--verbose", action="store_true",
                   help="Verbose output (default: one line per epoch)")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Validate dataset path
    if not DATASET_ROOT.exists():
        print(f"[ERROR] Dataset not found at: {DATASET_ROOT}")
        print("        Update DATASET_ROOT in training/train.py to match your path.")
        sys.exit(1)

    data_yaml = build_data_yaml(
        Path(__file__).parent / "data.yaml"
    )

    model = YOLO(args.model)

    print(f"[INFO] Starting training —")
    print(f"       Model:    {args.model}")
    print(f"       Epochs:   {args.epochs} (max, early stopping enabled)")
    print(f"       Batch:    {args.batch}")
    print(f"       Patience: {args.patience} epochs")
    print(f"       Imgsz:    {args.imgsz}")
    print(f"       Device:   {args.device or 'auto'}")
    print(f"       Dataset:  {DATASET_ROOT}")

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device or None,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # Early stopping
        patience=args.patience,
        # Augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        # Optimiser
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        # Logging
        plots=True,
        save=True,
        save_period=10,
        val=True,
        verbose=args.verbose,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    dest_pt = Path(__file__).parent.parent / "models" / "behaviour_detector.pt"
    dest_pt.parent.mkdir(exist_ok=True)

    if best_pt.exists():
        shutil.copy(best_pt, dest_pt)
        print(f"\n[INFO] Best weights copied to: {dest_pt}")

        if args.export_onnx:
            export_model = YOLO(str(dest_pt))
            onnx_path = export_model.export(
                format="onnx", opset=17, dynamic=False, imgsz=args.imgsz
            )
            dest_onnx = dest_pt.with_suffix(".onnx")
            shutil.copy(onnx_path, dest_onnx)
            print(f"[INFO] ONNX model exported to: {dest_onnx}")
    else:
        print("[WARN] best.pt not found — check training output directory.")

    print("\n[INFO] Training complete.")
    print(f"       Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
