# pi-deploy

Retrain the classroom-monitor behaviour detector as YOLO11s instead of
YOLOv8n, with `Using_phone`/`phone` merged into one class, then export it
to NCNN (INT8) for real-time-enough inference on a Raspberry Pi 5 (8GB,
CPU-only).

## Why

The original `yolov8n` model scored weakest on exactly the classes
`live_monitor.py` cares about for alerts (`phone` 0.42 mAP, `turn_head`
0.35 mAP), and `Using_phone`/`phone` were two separate classes for what
is really one behaviour, splitting the training signal. See the training
run at `../classroom-monitor/runs/detect/training/runs/behaviour_detector-4/`
for the original metrics.

## Pipeline

```
# 1. Merge Using_phone + phone, write a new dataset copy (original untouched)
python prepare_dataset.py

# 2. Train YOLO11s on the merged dataset (run on a real GPU/MPS machine, not the Pi)
pip install -r requirements.txt
python train.py --device mps      # or --device 0 for CUDA, omit for auto

# 3. Export best.pt to NCNN INT8 for Pi CPU inference
python export_pi.py --weights models/behaviour_detector_v2.pt
```

This produces `models/pi/behaviour_detector_v2/` — copy that folder to
the Raspberry Pi.

## Deploying to the Pi

1. Copy `models/pi/behaviour_detector_v2/` onto the Pi, into
   `classroom-monitor/models/pi/behaviour_detector_v2/`.
2. Copy `pi_config.yaml` over `classroom-monitor/config.yaml` on the Pi.
3. Run the app as usual (`python main.py` / `python app.py`).

`pi_config.yaml` differs from the original config in three ways:
- `model.weights` points at the exported NCNN directory, `device: cpu`.
- `analytics.frame_skip: 4` — the detector doesn't need to run every
  frame; `bytetrack` interpolates motion between detections, which is
  enough for smooth-looking tracking without needing Pi-CPU inference to
  hit 30fps.
- `tracking.tracker: bytetrack` instead of `botsort` (no ReID) — cheaper
  on CPU, since botsort+reid's extra accuracy isn't worth the CPU cost
  on Pi.
- `dataset.names` / `alerts.classes` updated to the merged 11-class list
  (`Using_phone` removed, folded into `phone`).

## If this still isn't fast/accurate enough

The CPU-only ceiling on Pi 5 is roughly YOLO11s at a few fps with
frame-skip. To go further (e.g. `yolo11m` at real-time), add a
**Raspberry Pi AI HAT+ (Hailo-8L, ~13 TOPS)** and export with
`format="hailo"` instead of `format="ncnn"` — inference then runs on the
HAT, off the CPU entirely.
