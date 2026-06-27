# Reproducing PETR eval on Python 3.12 / PyTorch 2.9 (no venv, no old stack)

This document records **everything** that was added/changed to make the original
[megvii PETR](https://github.com/megvii-research/PETR) repo run a real nuScenes
detection **evaluation** on a modern runtime, so you can redo it on another PC
that uses the same Docker container.

The original PETR targets a 2021 stack (Python 3.8, `mmcv-full 1.4`,
`mmdet 2.24`, `mmdet3d 0.17`) which **cannot** be installed on Python 3.12 /
torch 2.9. Instead of creating a separate Python 3.8 environment, we keep the
container's interpreter and add a thin **compatibility shim** so the unmodified
PETR source builds and runs on `mmengine + mmcv 2.x + mmdet 3.x`.

> **Nothing in the original repo was modified.** All new code lives in
> `petr_port/`. The only side effect of installation is that `numpy` is pinned
> to `1.26.x` (the old ecosystem requires numpy < 2).

---

## 0. Target environment (the container we ran on)

| Component | Value |
|---|---|
| OS | Ubuntu 24.04 (inside a Docker container, GPU passthrough) |
| Python | 3.12.3 (system interpreter, no venv/conda) |
| PyTorch | 2.9.1+cu129 (CUDA build 12.9) |
| GPU | NVIDIA RTX 3090 24 GB (driver 580), `sm_86` |
| sudo | passwordless (not actually needed below) |

GPU check (should print `True` and the device name):
```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 1. Python packages to install

All packages below are **pure-Python wheels** — there is **no CUDA compilation**,
so this works on any torch/CUDA version the container ships.

```bash
# 1) OpenMMLab core (pure python; mmcv-LITE has no compiled ops on purpose)
python3 -m pip install "mmengine==0.10.7" "mmcv-lite==2.1.0"

# 2) mmdet WITHOUT deps -- critical, so pip does NOT pull `mmcv` (full),
#    which would try to compile against torch 2.9 and fail.
python3 -m pip install --no-deps "mmdet==3.2.0"
python3 -m pip install pycocotools terminaltables scipy shapely

# 3) nuScenes devkit + Google-Drive downloader.
#    nuscenes-devkit pins numpy<2 (downgrades numpy 2.x -> 1.26.x), which is
#    exactly what the old-style code needs.
python3 -m pip install nuscenes-devkit gdown
```

Resulting versions that were verified working:

```
mmengine==0.10.7   mmcv-lite==2.1.0   mmdet==3.2.0   (mmcv full: NOT installed)
numpy==1.26.4      nuscenes-devkit==1.2.0  gdown==6.1.0
pyquaternion==0.9.9  shapely==2.0.7
```

Why these exact pins:
- `mmengine==0.10.7` — earlier 0.10.4 crashes on torch 2.9 with
  `KeyError: 'Adafactor is already registered'` (torch 2.5+ added
  `torch.optim.Adafactor`, colliding with mmengine's own). 0.10.7 fixes it.
- `mmcv-lite==2.1.0` — `lite` = no compiled CUDA ops (we stub the one import
  point). mmdet3d 1.x wants `mmcv<2.2`, so 2.1.0 is in range.
- `mmdet==3.2.0` installed `--no-deps` — prevents pip from installing full
  `mmcv` (which compiles) and from changing torch/numpy.

> Troubleshooting: if `import cv2` complains about numpy, force a numpy<2 build:
> `python3 -m pip install "opencv-python-headless<4.12"`.

---

## 2. Dataset (nuScenes v1.0-mini, ~4 GB, no login required)

> Shortcut: `cd petr_port && ./download_data.sh` does this section **and** §3
> (checkpoint) **and** generates the info pkls, idempotently/resumably.

```bash
cd <PETR_repo_root>
mkdir -p data ckpts
curl -L -C - "https://www.nuscenes.org/data/v1.0-mini.tgz" -o data/v1.0-mini.tgz
mkdir -p data/nuscenes
tar xzf data/v1.0-mini.tgz -C data/nuscenes
# -> data/nuscenes/{maps,samples,sweeps,v1.0-mini}
```

## 3. Pretrained checkpoint (PETR VoVNet-p4-800x320, ~986 MB)

```bash
python3 -m gdown "1-afU8MhAf92dneOIbhoVxl_b72IAWOEJ" -O ckpts/petr_vovnet_p4_800x320.pth
```
(Google-Drive file id from the PETR README "PETR-vov-p4-800x320 / gdrive" link.
The checkpoint is `meta.epoch=24`, 854 params.)

---

## 4. Copy the port over (the only new code)

Copy the whole `petr_port/` directory into the PETR repo root. It contains
4 files and **no external data**:

| File | Purpose |
|---|---|
| `petr_port/compat.py` | The compatibility shim. **Must be imported first.** Recreates the legacy module paths PETR imports and maps them to the modern stack. |
| `petr_port/gen_infos.py` | Generates `nuscenes_infos_{train,val}.pkl` for the mini split using PETR's *own* converter (identical calibration / `lidar2img` math). |
| `petr_port/build_test.py` | Sanity check: builds the model + loads the checkpoint, reports key-match (expect `missing=0 unexpected=0`). |
| `petr_port/petr_infer.py` | Inference + official nuScenes evaluation on `mini_val`. |

---

## 5. Generate info files, then run

```bash
cd <PETR_repo_root>/petr_port

# (a) build the per-sample camera-calibration info pkls for v1.0-mini
python3 gen_infos.py
#   -> data/nuscenes/nuscenes_infos_train.pkl (323 samples)
#   -> data/nuscenes/nuscenes_infos_val.pkl   (81 samples, mini_val)

# (b) OPTIONAL sanity check (model build + checkpoint match)
python3 build_test.py
#   expect: "state_dict: model=854 ckpt=854 missing=0 unexpected=0"

# (c) inference + nuScenes detection evaluation on mini_val
python3 petr_infer.py
```

Outputs land in `work_dirs/petr_vov_mini/`:
`results_nusc.json` (predicted boxes), `metrics_summary.json`, `metrics_details.json`.

---

## 6. Expected result

```
mAP : 0.3406
NDS : 0.3633
car 0.609 | bus 0.549 | traffic_cone 0.598 | pedestrian 0.506 | truck 0.496 | motorcycle 0.486 | bicycle 0.163
construction_vehicle 0.000 | trailer 0.000 | barrier 0.000   (<- 0 GT instances in mini_val)
```

Interpretation: the published **full-val** numbers are mAP 0.378 / NDS 0.426.
mini_val is only 2 scenes / 81 samples, and 3 of the 10 classes have **zero**
ground-truth instances there, so they score 0 AP by definition and pull the
10-class mean down. Over the 7 classes that appear, mean AP is **0.487**, and
the per-class APs (car 0.61, etc.) confirm the port is faithful.

To compare strictly against 0.378 / 0.426 you must run on the **full** nuScenes
`val` set (download `v1.0-trainval*`, ~300+ GB, then
`python3 gen_infos.py --version v1.0-trainval` and point `petr_infer.py` at it).

---

## 7. What the shim actually does (`compat.py`)

For anyone studying the port, the shim performs these mappings (no PETR file is
edited):

1. **Stub `mmcv._ext`** with a `MagicMock` — PETR's VoVNet path is NMS-free and
   uses no DCN, so the compiled mmcv ops are never *called*; this lets
   `mmdet`'s layers import without building CUDA ops.
2. **Recreate deleted legacy modules** and point them at modern equivalents:
   - `mmcv.runner` → `BaseModule` (mmengine), `force_fp32`/`auto_fp16` (no-ops),
     `load_checkpoint`.
   - `mmcv.parallel.DataContainer` (minimal), `mmcv.cnn.bricks.registry`
     (`ATTENTION`/`TRANSFORMER_LAYER`/… → mmdet `MODELS`).
   - `mmcv.cnn` weight-init helpers (`xavier_init`, …) re-exported from
     `mmengine.model`; `mmcv.*` IO helpers (`dump`, `load`,
     `track_iter_progress`, …) re-exported from `mmengine`.
   - `mmdet.core`, `mmdet.models.builder`, `mmdet.models.utils.builder`,
     `mmdet.models.utils.transformer` → modern `mmdet 3.x` locations.
3. **Vendor a minimal `mmdet3d`** (not installed): `LiDARInstance3DBoxes` with
   0.17 semantics, `bbox3d2result`, `build_bbox_coder`, and a minimal
   `MVXTwoStageDetector` base — so PETR's detector/coder/box conventions match
   the pretrained weights exactly.
4. **Register all PETR components** into mmdet's `MODELS`/`TASK_UTILS` and build
   under `init_default_scope('mmdet')`. A registry override is enabled so PETR's
   type names win over identically-named mmdet built-ins.
5. **Namespace-package trick** so `projects.mmdet3d_plugin.*` submodules import
   without executing the heavy package `__init__` (which would drag in
   dataset code that needs real mmdet3d).

mmcv 2.1 still ships the transformer "bricks" (`BaseTransformerLayer`, `FFN`,
`MultiheadAttention`, …) on a unified `MODELS` registry, so those did **not**
need vendoring — only the legacy registry *names*.

---

## 8. One-shot script (optional)

```bash
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"                      # run from PETR repo root
python3 -m pip install "mmengine==0.10.7" "mmcv-lite==2.1.0"
python3 -m pip install --no-deps "mmdet==3.2.0"
python3 -m pip install pycocotools terminaltables scipy shapely nuscenes-devkit gdown
mkdir -p data ckpts data/nuscenes
[ -f data/v1.0-mini.tgz ] || curl -L -C - "https://www.nuscenes.org/data/v1.0-mini.tgz" -o data/v1.0-mini.tgz
tar xzf data/v1.0-mini.tgz -C data/nuscenes
[ -f ckpts/petr_vovnet_p4_800x320.pth ] || python3 -m gdown "1-afU8MhAf92dneOIbhoVxl_b72IAWOEJ" -O ckpts/petr_vovnet_p4_800x320.pth
cd petr_port
python3 gen_infos.py
python3 petr_infer.py
```
