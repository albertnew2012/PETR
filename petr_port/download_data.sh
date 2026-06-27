#!/usr/bin/env bash
# Download everything PETR needs to run, and nothing else:
#   1) nuScenes v1.0-mini dataset  (~4 GB, public, no login)  -> data/nuscenes/
#   2) PETR VoVNet-p4-800x320 checkpoint (~986 MB, Google Drive) -> ckpts/
#   3) the nuScenes mini info .pkl files (generated locally)
#
# Safe to re-run: each step is skipped if its output already exists, and the
# dataset download resumes if interrupted.
#
# Usage:  cd petr_port && ./download_data.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"
echo ">> PETR repo root: $REPO_ROOT"

MINI_URL="https://www.nuscenes.org/data/v1.0-mini.tgz"
MINI_TGZ="data/v1.0-mini.tgz"
MINI_SIZE=4167696325                              # expected byte size (~3.9 GiB)
CKPT_ID="1-afU8MhAf92dneOIbhoVxl_b72IAWOEJ"       # Google Drive file id (PETR README)
CKPT="ckpts/petr_vovnet_p4_800x320.pth"

mkdir -p data/nuscenes ckpts

# --------------------------------------------------------------------------- #
# 1) nuScenes v1.0-mini
# --------------------------------------------------------------------------- #
echo ">> [1/4] nuScenes v1.0-mini dataset (public, no login)"
if [ -d data/nuscenes/v1.0-mini ]; then
  echo "    already extracted at data/nuscenes/ (skip)"
else
  cur=$(stat -c%s "$MINI_TGZ" 2>/dev/null || echo 0)
  if [ ! -f "$MINI_TGZ" ] || [ "$cur" -lt "$MINI_SIZE" ]; then
    echo "    downloading (resumable, -C -)..."
    curl -L -C - "$MINI_URL" -o "$MINI_TGZ"
  else
    echo "    archive already complete (skip download)"
  fi
  echo "    verifying tarball integrity..."
  tar tzf "$MINI_TGZ" >/dev/null
  echo "    extracting -> data/nuscenes/ ..."
  tar xzf "$MINI_TGZ" -C data/nuscenes
fi

# --------------------------------------------------------------------------- #
# 2) Pretrained checkpoint (Google Drive)
# --------------------------------------------------------------------------- #
echo ">> [2/4] PETR VoVNet-p4-800x320 checkpoint (~986 MB)"
if [ -f "$CKPT" ]; then
  echo "    already present at $CKPT (skip)"
else
  if ! python3 -c "import gdown" 2>/dev/null; then
    echo "    installing gdown (needed for Google Drive large files)..."
    python3 -m pip install --quiet gdown
  fi
  python3 -m gdown "$CKPT_ID" -O "$CKPT"
  echo "    verifying it is a torch checkpoint..."
  python3 - "$CKPT" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd = ck.get("state_dict", ck)
assert len(sd) > 100, "checkpoint looks wrong (got an HTML error page?)"
print(f"    OK: {len(sd)} params, epoch={ck.get('meta', {}).get('epoch')}")
PY
fi

# --------------------------------------------------------------------------- #
# 3) Info .pkl files (generated locally from the dataset)
# --------------------------------------------------------------------------- #
echo ">> [3/4] nuScenes mini info .pkl files"
if [ -f data/nuscenes/nuscenes_infos_val.pkl ]; then
  echo "    already present (skip)"
elif python3 -c "import mmengine, nuscenes" 2>/dev/null; then
  ( cd "$SCRIPT_DIR" && python3 gen_infos.py )
else
  echo "    SKIPPED: python deps (mmengine / nuscenes-devkit) not installed yet."
  echo "    after installing them, run:  cd petr_port && python3 gen_infos.py"
fi

# --------------------------------------------------------------------------- #
# 4) Summary
# --------------------------------------------------------------------------- #
echo ">> [4/4] Summary"
[ -d data/nuscenes ] && du -sh data/nuscenes 2>/dev/null | sed 's/^/    dataset: /'
[ -f "$CKPT" ] && ls -lh "$CKPT" | awk '{print "    ckpt:    "$5"  "$NF}'
if ls data/nuscenes/*.pkl >/dev/null 2>&1; then
  ls data/nuscenes/*.pkl | sed 's/^/    info:    /'
else
  echo "    info:    (not generated yet -- run gen_infos.py)"
fi
echo ">> Done."
