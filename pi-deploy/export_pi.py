"""
Export a trained .pt model to NCNN (INT8-quantized) for CPU inference on
a Raspberry Pi 5. NCNN is the fastest supported runtime for YOLO on Pi
CPUs (no GPU/NPU required).

INT8 export needs a small calibration set — point it at a folder of
representative frames (a few hundred images sampled from the training
set is enough).

Usage:
    cd pi-deploy
    python export_pi.py --weights models/behaviour_detector_v2.pt
    python export_pi.py --weights models/behaviour_detector_v2.pt --no-int8   # fp32, if accuracy drop is too large
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Export YOLO model to NCNN for Raspberry Pi")
    p.add_argument("--weights", required=True, help="Path to trained .pt weights")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--no-int8", action="store_true",
                    help="Skip INT8 quantization (larger, slightly more accurate, slower)")
    p.add_argument("--calib-data", default="dataset/data.yaml",
                    help="data.yaml used to sample calibration images for INT8")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed. Run: pip install -r requirements.txt")

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    model = YOLO(str(weights))

    export_kwargs = dict(format="ncnn", imgsz=args.imgsz)
    if not args.no_int8:
        export_kwargs.update(int8=True, data=args.calib_data)

    print(f"[INFO] Exporting {weights} -> NCNN "
          f"({'int8' if not args.no_int8 else 'fp32'}, imgsz={args.imgsz})")
    exported_dir = model.export(**export_kwargs)

    dest = Path(__file__).parent / "models" / "pi" / weights.stem
    dest.parent.mkdir(parents=True, exist_ok=True)
    if Path(exported_dir).is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(exported_dir, dest)
    else:
        dest.parent.mkdir(exist_ok=True)
        shutil.copy(exported_dir, dest)

    print(f"\n[INFO] NCNN model ready at: {dest}")
    print("[INFO] Copy this folder to the Pi and point config.yaml's")
    print(f"       model.weights at it, e.g.: model: pi/{weights.stem}")


if __name__ == "__main__":
    main()
