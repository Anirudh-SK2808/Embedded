#!/usr/bin/env bash
# One-time setup on the Raspberry Pi 4 (64-bit Raspberry Pi OS Bookworm or newer).
#
#   cd ~/embedded2
#   bash setup_pi.sh
#
# The venv is created with --system-site-packages so it can see the
# apt-installed picamera2 (it isn't pip-installable on the Pi).

set -euo pipefail
cd "$(dirname "$0")"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "[ERROR] Needs 64-bit Raspberry Pi OS (uname -m = $(uname -m), expected aarch64)."
  exit 1
fi

# Without the +cpu pin, ultralytics pulls PyPI's CUDA torch (several GB of
# nvidia-* wheels) — an old requirements-pi.txt did exactly that.
if ! grep -q '^torch==.*+cpu' requirements-pi.txt; then
  echo "[ERROR] requirements-pi.txt doesn't pin torch==...+cpu — this is an old copy."
  echo "        Copy embedded2_pi_cpu.zip to the Pi and unzip it over this folder."
  exit 1
fi

sudo apt update
sudo apt install -y python3-venv python3-picamera2

python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip

# Bookworm's apt picamera2 (simplejpeg) is built against NumPy 1.x; a pip
# NumPy 2.x in the venv shadows it and picamera2 fails with "numpy.dtype
# size changed". Trixie and newer ship NumPy 2.x, so only pin on 1.x systems.
numpy_pin=""
if python3 -c 'import numpy, sys; sys.exit(int(numpy.__version__.split(".")[0]) >= 2)' 2>/dev/null; then
  numpy_pin="numpy<2"
  echo "[INFO] System NumPy is 1.x (Bookworm) — pinning numpy<2 for picamera2."
fi
.venv/bin/pip install -r requirements-pi.txt $numpy_pin

echo
echo "[CHECK] Importing runtime packages ..."
.venv/bin/python - <<'PY'
import importlib
for name in ("numpy", "torch", "ncnn", "ultralytics", "cv2", "lap", "flask", "picamera2"):
    try:
        mod = importlib.import_module(name)
        print(f"  ok   {name} {getattr(mod, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {name}: {e}")
try:
    import torch
    print(f"  torch build: {'CUDA ' + torch.version.cuda if torch.version.cuda else 'CPU-only'}")
except Exception:  # noqa: BLE001
    pass
PY
# nvidia-ml-py is a tiny GPU-query lib ultralytics always installs — harmless.
if .venv/bin/pip list 2>/dev/null | grep -i '^nvidia' | grep -qvi '^nvidia-ml-py'; then
  echo "[ERROR] nvidia-* packages are installed — the CUDA torch slipped in. Clean up and rerun:"
  echo "        rm -rf .venv ~/.cache/pip && bash setup_pi.sh"
  exit 1
fi

echo
echo "[CHECK] Cameras seen by libcamera:"
rpicam-hello --list-cameras || libcamera-hello --list-cameras || echo "  (no camera tool found)"

echo
echo "[DONE] Next:"
echo "  .venv/bin/python tools/benchmark.py --config config.pi.yaml"
echo "  .venv/bin/python app.py --config config.pi.yaml      # http://$(hostname -I | awk '{print $1}'):5050"
