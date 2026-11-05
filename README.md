# Classroom Monitor 2 — YOLO26n on Raspberry Pi 4

Watches a classroom camera, labels each student's behaviour (using a phone,
sleeping, turning their head, writing, …), tracks each student across frames,
and shows live and historical analytics on a web dashboard. It's a rebuild of
`../classroom-monitor`, tuned for a **Raspberry Pi 4 (8 GB)** with a
**Pi Camera Module**.

## What changed from `classroom-monitor`

| | `classroom-monitor` | `Embedded 2` |
|---|---|---|
| Model | YOLOv8n | **YOLO26n**: no DFL layer, faster on CPU, better small-object accuracy. It also has an NMS-free head, but NCNN can't run it, so the Mac and the Pi both use the standard head + NMS and give the same results |
| Classes | 12: `Using_phone` and the `phone` object both present, plus a relabel workaround | **11**: the `phone` object class is removed, so only `Using_phone` (the behaviour) remains |
| Camera | `cv2.VideoCapture` at 640×480 (4:3) | **picamera2** (Pi Camera Module) or OpenCV, at **1280×720 (16:9)**, the same shape as the training images |
| Training size | 640 | **832**: the objects are tiny (median `Using_phone` box ≈ 14×10 px at 640), so every box is ~1.3× larger per side |
| Pi model format | NCNN "INT8" (actually FP32, because Ultralytics has no INT8 NCNN export) | **NCNN FP16 at a 480×832 input**, about 40% less compute than 832×832 |
| Frame handling | 4-frame queue plus a fixed `frame_skip` | **Latest frame only**: no lag build-up, and the detector skips frames on its own when it's busy |
| Movement window | Sized by the camera's fps (wrong when processing runs at 4 fps) | Sized by the **measured processing fps** |
| Dashboard stream | JPEG encoded at full frame size | Downscaled to 640 px wide before encoding, saving Pi CPU |
| Class names | Hard-coded in five places | **Read from the model** |

The old one-off scripts (`track_and_store.py`, `run_tracking.py`,
`process_image*.py`) weren't carried over. They duplicated the main pipeline
using hard-coded absolute paths.

## Layout

```
app.py              Flask dashboard (live + history)
main.py             Headless or OpenCV-window pipeline
capture.py          picamera2 / OpenCV capture, latest-frame-only
detector.py         YOLO26 + ByteTrack, merges each student's boxes into one per frame
tracker.py          Per-track registry: entry/exit, smoothing
movement.py, occupancy.py, analytics.py, database.py
config.yaml         Mac: .pt weights, webcam
config.pi.yaml      Pi 4: NCNN weights, Pi Camera
trackers/           ByteTrack settings (desktop and Pi)
training/
  prepare_dataset.py   Builds dataset/ with the phone object class removed
  train.py             Trains YOLO26n at 832
  evaluate.py          Test-split mAP, optionally side by side with the old YOLOv8n
  export_pi.py         NCNN FP16 480×832 export + accuracy check against the .pt
tools/
  benchmark.py         fps/latency on this machine (run it on the Pi)
  visualize.py         Draws detections on images
samples/            24 test images, used for benchmarking on the Pi
setup_pi.sh         One-time Pi setup
train_gpu.sh        Setup + training on an NVIDIA GPU server (L4)
```

---

## Step 1: Train on your Mac

```bash
cd "Embedded 2"
python3 -m venv .venv && source .venv/bin/activate     # or reuse ../classroom-monitor/.venv
pip install -r requirements.txt

python training/prepare_dataset.py        # builds dataset/ (11 classes)
python training/train.py                  # YOLO26n @ 832, 200 epochs: ~6–7 h on Apple GPU (MPS)
python training/train.py --device 0       # or on an NVIDIA GPU (e.g. L4): ~1.5–2 h
```

**On an NVIDIA GPU server (e.g. L4):** `train_gpu.sh` does the whole job.
It checks the driver, installs what's missing, confirms PyTorch can see the
GPU, rebuilds `dataset/`, and trains on GPU 0 with a log in `logs/`. The
copy commands to and from the server are at the top of the script. Run it
inside `tmux` so an SSH drop doesn't kill training.

Training runs the full 200 epochs. Early stopping is off by default because
the cosine learning-rate decay and the final 15 epochs without mosaic both
happen at the end, and that's where most of the last accuracy gain comes
from. The best epoch (by validation mAP50-95) is copied to
`models/behaviour_yolo26n.pt`. If training is interrupted, resume it with
`python training/train.py --resume`.

## Step 2: Check it beats the old model

```bash
python training/evaluate.py --baseline
```

This prints per-class AP50 and AP50-95 for the old YOLOv8n and the new
YOLO26n on the same 292 test images. To look at the results:

```bash
python tools/visualize.py                 # samples/ -> outputs/visualize/
```

Then choose the runtime confidence threshold on the validation split:

```bash
python training/evaluate.py --split val   # prints "Best mean F1 … at conf X"
```

Put X in `model.confidence_threshold` in both configs. Both are set to
0.39, from the current `models/behaviour_yolo26n.pt`; re-run this after
retraining.

## Step 3: Export for the Pi

```bash
python training/export_pi.py
```

The export size (480×832) comes from the size the weights were trained
at, matching `model.imgsz` in both configs. It writes
`models/pi/behaviour_yolo26n_ncnn_model/`, then runs the export and
the `.pt` model on 30 test images and compares their boxes. **"Export OK"**
means they agree on at least 90% of boxes.

## Step 4: Set up the Raspberry Pi 4

Requirements: **64-bit** Raspberry Pi OS (Bookworm or newer), a Pi Camera
Module on the ribbon port, a **heatsink and fan**, and the official 5V/3A
power supply.

1. Copy the project to the Pi. You don't need the dataset, `runs/` or `.venv/`:
   ```bash
   rsync -av --exclude dataset --exclude runs --exclude .venv --exclude '*.pt' \
     "Embedded 2/" pi@<pi-ip>:~/embedded2/
   ```
2. On the Pi:
   ```bash
   cd ~/embedded2
   bash setup_pi.sh                       # apt picamera2 + venv + pip packages, then a check
   source .venv/bin/activate
   ```
3. Measure speed:
   ```bash
   python tools/benchmark.py --config config.pi.yaml             # samples/, no camera needed
   python tools/benchmark.py --config config.pi.yaml --camera    # live camera, end to end
   ```
4. Run it:
   ```bash
   python app.py --config config.pi.yaml          # dashboard: http://<pi-ip>:5050 → Start
   # or headless, logging to the database only:
   python main.py --config config.pi.yaml --no-display
   ```

**USB webcam instead of the Pi Camera:** in `config.pi.yaml`, set
`camera.source: 0`.

---

## Performance notes (Pi 4)

- **Expected speed: about 2–3 fps.** That's an estimate; `tools/benchmark.py`
  gives the real number. Behaviours last seconds and alerts need 5 s, so a
  few fps is enough. ByteTrack keeps each student's ID between processed
  frames.
- **FP16 on a Pi 4** halves the model size but isn't much faster. The
  Cortex-A72 can store FP16 but can't calculate in it, so NCNN computes in
  FP32. On a Pi 5 it would be faster.
- **Heat:** without a fan, a Pi 4 slows itself down at about 80 °C and
  loses up to a third of its fps. `benchmark.py` reports this.
- **More speed:** INT8 is roughly 1.3–1.8× faster for a small accuracy loss.
  Ultralytics can't produce it for NCNN; build NCNN's tools on the Pi and use
  them on the FP16 export (`ncnn2table` to calibrate on `samples/`, then
  `ncnn2int8`). See the NCNN wiki "quantized int8 inference". This step is
  untested here.
- **Why 832 and not 640:** at 640 the median `Using_phone` box is only
  about 14×10 px. 832 costs about 1.7× the Pi inference time of 640, which
  is why the speed is 2–3 fps rather than 3–5. The detector prints a
  warning if `model.imgsz` doesn't match the weights.

## Troubleshooting

- **`picamera2 not available`**: the venv must be created with
  `--system-site-packages`, which `setup_pi.sh` does. Also run
  `sudo apt install python3-picamera2`.
- **`numpy.dtype size changed` / simplejpeg errors on Bookworm**: pip
  installed a NumPy that conflicts with the system one that picamera2 was
  built against. Fix: `pip install "numpy<2"`.
- **No camera found**: run `rpicam-hello --list-cameras`. Check that the
  ribbon cable is seated with its contacts facing the right way.
- **Boxes in the wrong place, or a `shape` error, on the Pi**: `model.imgsz`
  in `config.pi.yaml` must match the size you exported with (`[480, 832]`
  by default).

## Dataset

Student Behaviour Detection v1 (Roboflow, CC BY 4.0): 1,598 train,
581 valid and 292 test images, mostly 1920×1080. It's read from
`../Student Behaviour Detection.v1i.yolov8 2`, and the original is never
modified.
