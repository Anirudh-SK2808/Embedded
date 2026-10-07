# Classroom Monitor

A YOLOv8-based system that watches a classroom camera feed, classifies each
student's behaviour per frame (phone use, sleeping, hand-raising, etc.),
tracks individuals across frames, and surfaces live + historical analytics
through a Flask dashboard. Built with a Raspberry Pi 5 deployment target
in mind.

## Repo layout

```
classroom-monitor/   Flask app — live monitoring, tracking, analytics, dashboard
pi-deploy/            Retraining pipeline for a leaner/stronger model, tuned for Pi 5 CPU inference
```

`pi-deploy/` is self-contained and doesn't import from `classroom-monitor/`
— it only produces a model file that gets copied back into
`classroom-monitor/models/` once trained. See
[`pi-deploy/README.md`](pi-deploy/README.md) for that pipeline specifically.

## How it works

One YOLOv8 model does both detection and behaviour classification in a
single forward pass — there's no separate person-detector + classifier
stage. Each detected student is tracked across frames (ByteTrack/BoT-SORT),
and `live_monitor.py` smooths the per-track behaviour prediction over a
rolling window, computes movement/zone occupancy, and raises alerts when a
flagged behaviour (configurable, e.g. `sleep`, `phone`, `turn_head`)
persists past a duration threshold. Everything is written to SQLite and
served through a Flask dashboard.

### Detected classes (current model)

```
Using_phone  bend        book       bow_head
hand-raising phone       raise_head reading
sleep        turn_head   upright    writing
```

## Running it

```bash
cd classroom-monitor
pip install -r requirements.txt
python app.py                      # uses config.yaml
python app.py --config config.pi_sim.yaml   # Pi-throttled local sim
```

Then open the dashboard (Flask dev server, default `http://127.0.0.1:5000`):

- **Live** — starts the camera pipeline, shows the annotated feed + running stats
- **History** — past sessions: behaviour distribution, zone occupancy, per-person timelines, alerts

### Pi 5 local simulation (no real Pi needed)

```bash
./run_pi_demo.sh
```

Throttles your dev machine (CPU-only, 4 threads, memory-capped, frame-skip)
to approximate a Raspberry Pi 5's constraints, using your own webcam. Run
it directly in a terminal — it needs camera + display access.

## Configuration

All runtime behaviour is driven by `classroom-monitor/config.yaml`
(see `config.pi_sim.yaml` for the Pi-simulation variant):

| Section | Key knobs |
|---|---|
| `camera` | source, resolution, fps |
| `model` | weights path, confidence/IoU thresholds, device (`cpu`/`mps`/`cuda`) |
| `tracking` | tracker (`bytetrack`/`botsort`), track expiry |
| `analytics.frame_skip` | run the detector on 1-in-N frames; tracker interpolates the rest — the main lever for CPU-constrained deployment |
| `alerts` | which behaviour classes trigger alerts, and how long they must persist |

## The model

Current model: `classroom-monitor/models/behaviour_detector.pt`
(YOLOv8n, trained on the
[Student Behaviour Detection](https://universe.roboflow.com/anirudh-suchindrum-kasi-23bce1593/student-behaviour-detection-neazg-wcvjm-8rmuw)
Roboflow dataset — 1,598 train / 581 val images, 12 classes).

Final metrics (epoch 90/100, see
`classroom-monitor/runs/detect/training/runs/behaviour_detector-4/results.csv`):

| Metric | Value |
|---|---|
| Precision | 0.741 |
| Recall | 0.676 |
| mAP50 | 0.701 |
| mAP50-95 | 0.457 |

Per-class accuracy is uneven — strong on posture classes (`upright` 0.95,
`sleep` 0.86), weak on the classes the alert system actually relies on most
(`phone` 0.42, `turn_head` 0.35), partly because `Using_phone`/`phone` were
annotated as two separate classes for one real behaviour. `pi-deploy/`
retrains a YOLO11s model with those merged, aimed at CPU-only Pi 5
inference via NCNN export — see its README for the full pipeline and
rationale.

### Retraining the original model

```bash
cd classroom-monitor
python training/train.py                      # yolov8n, defaults
python training/train.py --model yolov8s.pt    # larger variant
```

## Requirements

- Python 3.10+
- `ultralytics`, `opencv-python`, `torch`/`torchvision`, `boxmot`, `Flask`, `scipy` — see `classroom-monitor/requirements.txt`
- A camera (webcam index or RTSP URL)

## Notes

- Training run artifacts (`runs/`), local databases (`*.db`), `__pycache__`,
  and the raw dataset directory are gitignored — they're either generated
  or too large to version. The deployed model weights
  (`models/behaviour_detector.pt`) are tracked.
- `pi-deploy/dataset/` and `pi-deploy/models/` are similarly gitignored —
  regenerate them with `prepare_dataset.py` / `train.py` / `export_pi.py`.
