#!/usr/bin/env bash
# One-shot reproduction: install deps + download data/ckpt + run PETR eval.
# Safe to re-run (idempotent). See REPRODUCE.md for details.
set -euo pipefail

# Resolve PETR repo root = parent of this script's directory (petr_port/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"
echo ">> PETR repo root: $REPO_ROOT"

echo ">> [1/5] Installing Python packages (pure-Python, no CUDA build)..."
python3 -m pip install "mmengine==0.10.7" "mmcv-lite==2.1.0"
python3 -m pip install --no-deps "mmdet==3.2.0"
python3 -m pip install pycocotools terminaltables scipy shapely
python3 -m pip install nuscenes-devkit gdown

echo ">> [2/5] Downloading nuScenes v1.0-mini (~4GB, resumable)..."
mkdir -p data ckpts data/nuscenes
if [ ! -f data/v1.0-mini.tgz ]; then
  curl -L -C - "https://www.nuscenes.org/data/v1.0-mini.tgz" -o data/v1.0-mini.tgz
fi
if [ ! -d data/nuscenes/v1.0-mini ]; then
  tar xzf data/v1.0-mini.tgz -C data/nuscenes
fi

echo ">> [3/5] Downloading PETR VoVNet-p4-800x320 checkpoint (~986MB)..."
if [ ! -f ckpts/petr_vovnet_p4_800x320.pth ]; then
  python3 -m gdown "1-afU8MhAf92dneOIbhoVxl_b72IAWOEJ" \
    -O ckpts/petr_vovnet_p4_800x320.pth
fi

echo ">> [4/5] Generating nuScenes mini info pkls..."
cd "$SCRIPT_DIR"
python3 gen_infos.py

echo ">> [5/5] Running inference + nuScenes evaluation on mini_val..."
python3 petr_infer.py

echo ">> Done. Outputs in $REPO_ROOT/work_dirs/petr_vov_mini/"
