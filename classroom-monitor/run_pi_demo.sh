#!/usr/bin/env bash
# Live demo on the MacBook's camera, throttled to approximate a
# Raspberry Pi 5 (4GB RAM, 4-core Cortex-A76, no GPU/accelerator).
#
#   - OMP/MKL thread count capped at 4 (Pi 5's core count)
#   - torch forced onto CPU (config.pi_sim.yaml: device "cpu")
#   - virtual memory capped at 4GB via ulimit
#   - frame_skip 3 + 15fps capture, matching the Pi-tuned settings
#     already called out in config.yaml's comments
#
# Run this yourself in a terminal — it needs camera permission and a
# window to show the live preview, neither of which an agent should
# trigger on your behalf.

set -euo pipefail
cd "$(dirname "$0")"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Cap virtual memory at 4GB to mirror the Pi 5's total RAM.
# (macOS enforcement of -v is best-effort, not a hard guarantee like cgroups on Linux.)
ulimit -v 4194304 || echo "[warn] ulimit -v not enforced on this shell; continuing without a hard cap"

exec .venv/bin/python main.py --config config.pi_sim.yaml --source 0
