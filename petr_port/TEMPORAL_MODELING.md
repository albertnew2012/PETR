# Temporal modeling in PETR (this repo) — how the t-1 frame is used

You saw `figs/overall.png` (the **PETRv2** figure) and asked how temporal info is
used. Short answer: **the model you've been studying/evaluating
(`petr_vovnet_p4_800x320`) is single-frame PETR v1 — no temporal.** Temporal
modeling is the **PETRv2** variant, which *is* in this repo:

- configs: [configs/petrv2/](../projects/configs/petrv2) (e.g.
  `petrv2_vovnet_gridmask_p4_800x320.py`)
- head: [dense_heads/petrv2_head.py](../projects/mmdet3d_plugin/models/dense_heads/petrv2_head.py)
  (`PETRv2Head`, with `with_time=True`, `with_fpe=True`)
- multi-frame loader: `LoadMultiViewImageFromMultiSweepsFiles`
  ([pipelines/loading.py](../projects/mmdet3d_plugin/datasets/pipelines/loading.py))
- sweep info generator: [tools/generate_sweep_pkl.py](../tools/generate_sweep_pkl.py)

The detector class is still `Petr3D`; only the **head** changes (v1 `PETRHead`
→ v2 `PETRv2Head`).

---

## 1. Reading `figs/overall.png`

```
 t   frame: multi-view imgs ─► 2D features ┐                      ┌► (C) concat ─► VALUE
                                           ├─(C) concat 2D feats ─┤
 t-1 frame: multi-view imgs ─► 2D features ┘                      └► FPE ─► KEY (3D PE, feature-guided)
 t   frame: 3D coordinates ───────────────► (C) concat 3D coords ─┘
 t-1 frame: 3D coordinates ──(A) align──────┘
                                  │
                            Transformer Decoder  ◄── det queries (+ seg queries)
                                  │
                       Det head → 3D boxes (BEV)   │   Seg head → BEV map
```
Two operators matter: **(C) concatenate** the current and previous frame, and
**(A) align** the previous frame's 3D coordinates into the current frame.

---

## 2. The mechanism, end to end

**Step 1 — load two frames.** The pipeline adds one previous sweep:
```python
dict(type='LoadMultiViewImageFromFiles', ...)                          # 6 current images
dict(type='LoadMultiViewImageFromMultiSweepsFiles', sweeps_num=1,
     sweep_range=[3,27], ...)                                          # +6 previous images
```
So `img` becomes **12 images** = 2 frames × 6 cameras, and the loader appends the
previous frame's `lidar2img`, `intrinsics`, `extrinsics`, `timestamp`.

**Step 2 — alignment "A" happens in the DATA, not the network.** The previous
sweep's `lidar2img` is built so its rays land in the **current** frame's LiDAR
coordinate system (using the ego-pose change between t-1 and t). The repo README
says it directly: *"the info files contain 30 previous frames, whose
transformation matrix is aligned with the current frame."* So once loaded, **both
frames' 3D coordinates live in one common (current) frame**.

**Step 3 — backbone.** `Petr3D.extract_img_feat` runs VoVNet+CPFPN on all 12
images → features `[B, 12, C, H, W]`.

**Step 4 — 3D PE for both frames (the "C" concat).**
`PETRv2Head.position_embeding` is the *same* back-projection as v1, but now over
`N=12`: it builds the 3D coordinates for every image (current + aligned previous)
and encodes them → a 3D PE for **12 × H × W** key tokens. Because of Step 2 they
are all in the current frame, so a static object's tokens from t and t-1 get
*similar* 3D PE, and a moving object's tokens get *offset* 3D PE.

**Step 5 — Feature-guided Position Encoder (FPE), `with_fpe=True`.**
```python
self.fpe = SELayer(self.embed_dims)                                   # squeeze-excite
coords_position_embeding = self.fpe(coords_position_embeding, x)      # gate PE by image features
```
The image features `x` produce per-channel gates that **modulate** the geometric
3D PE — so the position encoding becomes *content-aware* (helps where pure
geometry is ambiguous). This is the FPE box in the figure.

**Step 6 — decoder over both frames.** The 900 queries cross-attend over **all
12 × H × W** keys. A query for a moving object now sees its evidence at **two time
positions** → it can read motion; a static object gets **2× observations** →
better localization.

**Step 7 — velocity from time, `with_time=True`** (`PETRv2Head.forward`, ~line 488):
```python
time_stamp = ...view(B, -1, 6)                                  # (B, 2 frames, 6 cams)
mean_time_stamp = (time_stamp[:,1,:] - time_stamp[:,0,:]).mean(-1)   # Δt between frames
...
tmp[..., 8:] = tmp[..., 8:] / mean_time_stamp[:, None, None]    # velocity = displacement / Δt
```
The reg head predicts a **displacement** for the velocity channels (dims 8–9 =
vx, vy); dividing by the real **time gap Δt** turns it into a metric velocity.
That's only possible because the network saw the object at two times.

---

## 3. Why this is the win (and the `code_weights` tell)

- **Velocity:** v1 can barely estimate velocity from one frame, so its config
  **down-weights** velocity in the loss: `code_weights = [...,1,1,1,1,1,1,1,1, 0.2, 0.2]`.
  v2, which *can* measure it from two frames, sets **all `code_weights = 1.0`**.
  Empirically PETRv2's `mAVE` drops sharply and **NDS jumps** (NDS heavily weights
  velocity/attribute errors). (Recall our v1 mini_val run had `mAVE ≈ 0.89`.)
- **Detection:** two aligned observations of static structure improve recall/
  localization.

---

## 4. PETRv2 also adds (same figure)

- **BEV segmentation** — extra *segmentation queries* predict BEV map patches
  (`with_multi`, [petr_head_seg.py](../projects/mmdet3d_plugin/models/dense_heads/petr_head_seg.py),
  config `petrv2_BEVseg.py`). That's the "Seg head → BEV map" branch.
- **Query denoising** (in the `denoise/` configs and `petrv2_dnhead.py`) — a
  DN-DETR-style training aid, orthogonal to temporal.

---

## 5. Running PETRv2 here — DONE ✅

PETRv2 (temporal) now runs end-to-end on the same modern stack via
[petrv2_infer.py](petrv2_infer.py) (debug config **14**). What it took:

1. **PETRv2 checkpoint** — `petrv2_vovnet_p4_800x320.pth` (repo README's PETRv2
   row), in `ckpts/`.
2. **Register `PETRv2Head`** (+ its internal `SELayer`) into the same shim;
   the detector is still `Petr3D`, only the head and `with_time/with_fpe` flags
   change.
3. **Temporal sweep alignment, computed on-the-fly** — instead of pre-generating
   the multi-sweep pkls with `tools/generate_sweep_pkl.py`, `petrv2_infer.py`
   replicates its `add_frame` math with the nuScenes devkit: for each sample it
   walks the camera `prev` chains, picks the test-mode sweep (index 14 of the
   30-deep list, `sweep_range=[3,27]`), and aligns the previous frame's
   `lidar2img` into the **current** LiDAR frame via the ego-pose chain.
4. **12-image input** — current 6 cams + previous 6 cams, with 12 `lidar2img`
   and 12 timestamps laid out `[current×6, prev×6]` so `with_time` recovers
   `Δt ≈ 1.2 s` and divides displacement → velocity.

### Reproduced results (nuScenes **mini_val**, 81 samples)

| metric | PETR v1 | **PETRv2** | Δ |
|---|---:|---:|---:|
| mAP | 0.3406 | **0.3907** | **+0.050** |
| **NDS** | 0.3633 | **0.4250** | **+0.062** |
| mAVE (vel_err) | 0.8889 | **0.5701** | **−0.319** ⬇ |
| mAOE (orient_err) | 0.7101 | 0.6729 | −0.037 |
| mATE (trans_err) | 0.7124 | 0.6914 | −0.021 |

The NDS jump is driven mostly by the **36 % drop in velocity error** — exactly
the temporal contribution this doc describes. (mini has too few
trailer/construction_vehicle/barrier instances to score those classes, same
caveat as v1.)

Run it yourself:
```bash
cd petr_port
python3 petrv2_infer.py                      # full eval (prints the table above)
python3 petrv2_infer.py --limit 5 --no-eval  # quick smoke test (5 samples)
```

---

## 6. One-line summary

> Temporal = **load a previous frame, align its 3D coordinates into the current
> frame (in the data), concatenate both frames' position-aware tokens, let the
> decoder attend over both, and divide the predicted displacement by Δt to get
> velocity.** Geometry alignment is data-side; the network just attends over 2×
> the tokens.

## Code map

| piece | location |
|---|---|
| 2-frame loader | `LoadMultiViewImageFromMultiSweepsFiles` — loading.py |
| sweep info (alignment) | tools/generate_sweep_pkl.py |
| v2 head | petrv2_head.py — `PETRv2Head` |
| 3D PE over 12 imgs | `position_embeding` ~line 341 |
| FPE (feature-guided) | `self.fpe = SELayer(...)` ~line 328; applied ~line 454 |
| velocity / Δt | `with_time` block ~line 488 |
| config (flags) | configs/petrv2/petrv2_vovnet_gridmask_p4_800x320.py (`with_time/with_fpe`, `code_weights=1`) |

See also: `3D_PE_DEEPDIVE.md` (the single-frame 3D PE this builds on).
