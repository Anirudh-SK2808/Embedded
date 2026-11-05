#!/usr/bin/env python3
"""
Measure real detector speed on this machine, using exactly the same model,
imgsz and tracker as the app (it loads them from the config).

Two modes:
  - images (default): sample images resized to the camera resolution, so
    you can benchmark without a camera attached.
  - --camera: frames straight from the configured camera, end to end
    (capture -> detect -> track -> merge), which is what the app sees.

On a Raspberry Pi it also reports CPU temperature and whether the Pi
throttled during the run — a Pi 4 without a fan slows itself down at
~80 °C and the fps drops with it.

Usage:
    cd "Embedded 2"
    python tools/benchmark.py --config config.pi.yaml               # on the Pi
    python tools/benchmark.py --config config.pi.yaml --camera
    python tools/benchmark.py --config config.yaml --frames 100      # on the Mac
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector import BehaviourDetector  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark the behaviour detector")
    p.add_argument("--config", default="config.pi.yaml")
    p.add_argument("--frames", type=int, default=200, help="Timed frames (after warm-up)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--images", default=str(ROOT / "samples"),
                   help="Folder of images to cycle through (images mode)")
    p.add_argument("--camera", action="store_true", help="Benchmark from the live camera instead")
    return p.parse_args()


def cpu_temp_c() -> float | None:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        return None


def throttled_flags() -> str | None:
    """vcgencmd get_throttled bits: 0x1/0x10000 = under-voltage now/since boot,
    0x4/0x40000 = throttled now/since boot."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                             text=True, timeout=2).stdout.strip()
        return out.split("=")[-1] if out else None
    except (OSError, subprocess.SubprocessError):
        return None


def frame_source(args, cfg):
    """Yields BGR frames at the configured camera resolution, forever."""
    w, h = cfg["camera"]["width"], cfg["camera"]["height"]
    if args.camera:
        from capture import FrameProducer
        producer = FrameProducer(cfg["camera"]).start()
        try:
            while True:
                f = producer.read(timeout=2.0)
                if f is not None:
                    yield f.image
        finally:
            producer.stop()
    else:
        paths = sorted(p for p in Path(args.images).iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not paths:
            raise SystemExit(f"No images found in {args.images}")
        frames = [cv2.resize(cv2.imread(str(p)), (w, h), interpolation=cv2.INTER_AREA)
                  for p in paths]
        while True:
            yield from frames


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))
    m = cfg["model"]

    print(f"[INFO] Model:  {m['weights']}")
    print(f"[INFO] imgsz:  {m.get('imgsz')}   device: {m.get('device') or 'auto'}")
    print(f"[INFO] Input:  {'camera ' + str(cfg['camera']['source']) if args.camera else args.images} "
          f"@ {cfg['camera']['width']}x{cfg['camera']['height']}")

    t_load = time.perf_counter()
    det = BehaviourDetector(cfg)
    print(f"[INFO] Model loaded in {time.perf_counter() - t_load:.1f} s "
          f"({len(det.class_names)} classes)")

    source = frame_source(args, cfg)
    ts = datetime.now(timezone.utc).isoformat()
    for i in range(args.warmup):
        det.detect(i, ts, next(source))

    temp_start = cpu_temp_c()
    latencies = []
    n_people = []
    t0 = time.perf_counter()
    for i in range(args.frames):
        frame = next(source)
        t = time.perf_counter()
        dets = det.detect(args.warmup + i, ts, frame)
        latencies.append((time.perf_counter() - t) * 1000)
        n_people.append(len(dets))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.frames}  last {latencies[-1]:.0f} ms")
    wall = time.perf_counter() - t0
    temp_end = cpu_temp_c()

    lat = sorted(latencies)
    p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]  # noqa: E731
    print("\n" + "=" * 52)
    print(f"Frames:            {args.frames}")
    print(f"Latency  mean:     {statistics.mean(lat):7.1f} ms")
    print(f"         p50:      {p(0.50):7.1f} ms")
    print(f"         p90:      {p(0.90):7.1f} ms")
    print(f"         max:      {lat[-1]:7.1f} ms")
    print(f"Throughput:        {args.frames / wall:7.2f} fps  (detect + track + merge)")
    print(f"People / frame:    {statistics.mean(n_people):7.1f}")
    if temp_start is not None:
        print(f"CPU temp:          {temp_start:.0f} °C -> {temp_end:.0f} °C")
    flags = throttled_flags()
    if flags is not None:
        bits = int(flags, 16)
        state = "THROTTLING NOW" if bits & 0x4 else ("throttled earlier since boot" if bits & 0x40000 else "never throttled")
        print(f"Throttling:        {flags} ({state})")
        if bits & 0x40004:
            print("                   -> add a heatsink + fan; throttling cuts fps by up to ~1/3")
        if bits & 0x10001:
            print("                   -> UNDER-VOLTAGE detected: use the official 5V/3A USB-C supply")
    print("=" * 52)


if __name__ == "__main__":
    main()
