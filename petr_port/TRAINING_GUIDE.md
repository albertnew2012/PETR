# Studying how to TRAIN PETR — losses, the prediction head, and every moving part

Companion to [STUDY_GUIDE.md](STUDY_GUIDE.md). That one covers the architecture/
inference; this one covers **training**: what the losses are, how the head is
supervised, the bipartite matching, the data/augmentation pipeline, the
optimiser/schedule, and the practical knobs. Everything is anchored in this
repo's code, and there are **two runnable demos** that compute the real loss and
even run a tiny training loop on your stack.

> Run these first; the numbers below come from them:
> ```bash
> cd petr_port
> python3 study/loss_demo.py    --index 0          # loss anatomy on real GT
> python3 study/train_overfit.py --steps 40        # watch the loss drop
> ```

---

## 1. Where training is wired

It's DETR's training loop lifted to 3D. The data flow:

```
Petr3D.forward(return_loss=True)              # detectors/petr3d.py
  └─ forward_train(img, img_metas, gt_bboxes_3d, gt_labels_3d, ...)
       ├─ img_feats = extract_feat(img)        # VoVNet + CPFPN (+ GridMask in train)
       └─ forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d, img_metas)
            ├─ outs = pts_bbox_head(img_feats, img_metas)   # 6-layer decoder
            └─ losses = pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs)
```

`loss()` lives in
[petr_head.py](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py)
(`loss`, `loss_single`, `get_targets`, `_get_target_single`, ~lines 460–700).

---

## 2. The losses (what they are)

PETR uses **DETR-style set prediction** — exactly two loss types, applied to
**every** decoder layer (deep supervision):

| Loss | Module | Applies to | Weight |
|---|---|---|---|
| **Classification** | `FocalLoss(use_sigmoid, γ=2, α=0.25)` | all 900 queries | `2.0` |
| **Box regression** | `L1Loss` on the normalised 10-d box | matched queries only | `0.25` |
| (iou) | `GIoULoss` | — | `0.0` (disabled) |

With 6 decoder layers you therefore get **12 loss terms**: `loss_cls`,
`loss_bbox` (final layer) + `d0..d4.loss_cls/loss_bbox` (auxiliary). They are all
summed for backprop. Live output of `study/loss_demo.py` on sample 0 (20 GT):

```
Hungarian matching (final layer): 20 positive queries (= #GT), 880 background.
  loss_cls   : 0.4332   loss_bbox : 0.8067      # final layer
  d0.loss_cls: 0.5273   d0.loss_bbox: 0.7795    # earlier layers: higher cls loss
  ... d1..d4 ...
  TOTAL      : 7.5424   (sum of 12 terms)
```

Key facts to internalise:
- **Classification loss is over ALL queries** (focal down-weights the 880 easy
  negatives). `avg_factor = #positives` (≈ #GT), matching DETR's normalisation.
  See `loss_single` (~line 600).
- **Box loss is L1 on NORMALISED targets.** Targets are encoded by
  [`normalize_bbox`](../projects/mmdet3d_plugin/core/bbox/util.py):
  `(cx,cy,w,l,cz,h, sinθ,cosθ, vx,vy)` with `w,l,h` in **log** space. The L1 is
  weighted by `code_weights = [1]*8 + [0.2,0.2]` (velocity down-weighted), and
  normalised by `#positives`. See `loss_single` (~line 620).
- **NMS-free** falls out of one-to-one matching (next section).

> Correctness note for *this* port: mmdet's `FocalLoss` would normally call a
> compiled mmcv op; since we run with stubbed ops, [compat.py](compat.py)
> swaps in the pure-python focal loss (with index→one-hot handling). Without
> that fix `loss_cls` silently reads 0. Good thing to be aware of if you hack on
> the loss.

---

## 3. The prediction head

Built in
[`PETRHead._init_layers`](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py)
(~line 213):

```python
# classification branch (per the FocalLoss): 2x [Linear+LN+ReLU] -> Linear(256, 10)
# regression branch:                          2x [Linear+ReLU]    -> Linear(256, code_size=10)
self.cls_branches = ModuleList([fc_cls]  * num_pred)   # num_pred = 6
self.reg_branches = ModuleList([reg_branch]* num_pred) #  ^ SAME module -> SHARED weights
```

- Outputs per decoder layer: `cls (B,900,10)`, `reg (B,900,10)`. (`trace_shapes.py`
  shows both, x6.)
- **Anchor refinement:** the reg head predicts an *offset*; the query's reference
  point is added before sigmoid (tail of `forward`, ~line 430):
  ```python
  tmp[...,0:2] += reference[...,0:2]; sigmoid    # x,y
  tmp[...,4:5] += reference[...,2:3]; sigmoid    # z
  # then scale by pc_range
  ```
- **Box code (10-d):** `(cx,cy,w,l,cz,h,sinθ,cosθ,vx,vy)`. At test,
  [`NMSFreeCoder`](../projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py)
  +`denormalize_bbox` turn this into `(x,y,z,w,l,h,yaw,vx,vy)`.
- **Heads are shared across the 6 layers** — fewer params, and every layer is
  trained to decode boxes (deep supervision). Contrast DETR, where this is also
  common but not mandatory.

---

## 4. Bipartite matching (why no NMS)

[`HungarianAssigner3D`](../projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_3d.py):

```python
cls_cost = FocalLossCost(weight=2.0)(cls_pred, gt_labels)          # focal-style cls cost
reg_cost = BBox3DL1Cost(weight=0.25)(bbox_pred[:, :8],             # L1 on first 8 dims
                                     normalize_bbox(gt)[:, :8])    # (no velocity in matching)
cost = cls_cost + reg_cost                                        # iou_cost weight 0 -> unused
row, col = scipy.linear_sum_assignment(cost)                     # one-to-one
```

- Costs match the loss weights (the head even `assert`s this).
- The match cost is computed on the **first 8 dims only** (centre+size+yaw,
  *excluding* velocity) — a detail worth noting.
- One prediction ↔ one GT ⇒ exactly `#GT` positives, rest are background ⇒
  **no duplicate boxes ⇒ no NMS** (the demo prints `20 positives = #GT`).
- `PseudoSampler` just splits pos/neg (no sampling) — DETR convention.

The full per-image target assembly is `_get_target_single` (~line 460): positives
get their GT label + GT box target; negatives get the background class (`10`) and
zero box weight.

---

## 5. The training data pipeline & augmentations

`train_pipeline` in the
[config](../projects/configs/petr/petr_vovnet_gridmask_p4_800x320.py):

| Stage | What it does |
|---|---|
| `LoadMultiViewImageFromFiles` | 6 camera images |
| `LoadAnnotations3D(with_bbox_3d, with_label_3d)` | GT 3D boxes + labels |
| `ObjectRangeFilter(point_cloud_range)` | drop GT outside `±51.2 m` |
| `ObjectNameFilter(class_names)` | keep the 10 detection classes |
| `ResizeCropFlipImage(ida_aug_conf, training=True)` | **random** resize∈(0.47,0.625), crop, random horizontal flip; updates each camera's intrinsics ⇒ `lidar2img` |
| `GlobalRotScaleTransImage(...)` | **3D scene aug** (next paragraph) |
| `NormalizeMultiviewImage` / `PadMultiViewImage(32)` | normalise + pad |
| `DefaultFormatBundle3D` / `Collect3D(['gt_bboxes_3d','gt_labels_3d','img'])` | to tensors |

**The PETR-specific bit:**
[`GlobalRotScaleTransImage`](../projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py)
(~line 611). PETR has *no point cloud*, so a global BEV **rotation** (±0.3925 rad
≈ ±22.5°) and **scale** (0.95–1.05) are applied by transforming the projection
matrices and the GT boxes *together*:
```python
lidar2img[v] = lidar2img[v] @ rot_mat_inv     # rotate/scale the LiDAR frame
extrinsics[v] = rot_mat_inv.T @ extrinsics[v] # (images are untouched)
gt_bboxes_3d.rotate(angle); gt_bboxes_3d.scale(ratio)
```
Because the 3D PE is built from `lidar2img` (Phase 3 of the architecture guide),
this rotates the *coordinate field* the network sees while keeping pixels fixed —
a geometry-consistent 3D augmentation. **This is a great thing to study**: it's
how a camera-only method gets LiDAR-style global augmentation.

**GridMask** (`use_grid_mask=True`) is an image-space regulariser applied inside
`Petr3D.extract_img_feat` *only in training* (no-op under `eval()`).

---

## 6. Optimiser, schedule, and the practical knobs

All from the config tail:

```python
optimizer = AdamW(lr=2e-4, weight_decay=0.01,
                  paramwise_cfg=custom_keys{'img_backbone': lr_mult=0.1})  # backbone learns 10x slower
optimizer_config = Fp16OptimizerHook(loss_scale=512.,                      # mixed precision
                                     grad_clip=dict(max_norm=35, norm_type=2))
lr_config = CosineAnnealing(warmup='linear', warmup_iters=500,
                            warmup_ratio=1/3, min_lr_ratio=1e-3)
total_epochs = 24
data = dict(samples_per_gpu=1, workers_per_gpu=4)   # effective batch = 1 x #GPUs (paper: 8)
```

Things to know before launching a real run:
- **Backbone init matters a lot.** `img_backbone` is initialised from the
  **VoVNet V2-99** weights pretrained with FCOS3D/DD3D (put in `ckpts/`, see the
  repo README). Training from ImageNet/scratch is much weaker.
- **`lr_mult=0.1` on the backbone** stabilises fine-tuning the big pretrained net.
- **FP16 + loss_scale 512** keeps the 6-view forward in memory; **grad-clip 35**
  tames the DETR-style early instability.
- **Effective batch size** in the paper is 8 (8×2080Ti, 1 sample each). With 900
  queries + 6 cameras, memory is the constraint.
- **CBGS** (the `*_cbgs.py` configs) wraps the dataset in `CBGSDataset` to
  **class-balance** sampling (helps rare classes: trailer, construction_vehicle).
  Compare a `cbgs` vs non-`cbgs` config to see the only difference.

---

## 7. Two runnable training demos (on your stack)

**`study/loss_demo.py`** — builds the head *with* `train_cfg` (assigner active),
loads real GT for one sample, runs the forward pass, performs the Hungarian
match, and prints all 12 loss terms. Use it to connect every number above to
code.

**`study/train_overfit.py`** — proves the *whole* train path works here: it
freezes the backbone+neck, caches their features, and optimises the head with
AdamW. The loss collapses, confirming forward → match → loss → backward → step:
```
 step     total  loss_cls loss_bbox
    0    7.6710    0.4231    0.8518
   20    2.3780    0.0754    0.2757
   39    1.3741    0.0153    0.1926
```
(Try `--num-samples 4 --steps 80`; or unfreeze the backbone to feel the memory
cost — that's why the paper uses FP16 + small batch.)

**`study/train_tiny.py`** — a tiny **end-to-end** job: trains the *whole* model
(VoVNet-99 backbone + neck + head) for a few steps, with the backbone at 0.1x LR
and grad-clip 35, then saves a checkpoint to `work_dirs/tiny_train/`. It runs in
FP32 (the real FP16 path needs the mmcv `Fp16OptimizerHook` + `force_fp32`
upcasting, which isn't ported). The loss drops e.g. `9.08 -> 5.17` in 10 steps,
confirming end-to-end training works on this stack:
```bash
python3 study/train_tiny.py --steps 20 --num-samples 3
```

> **VS Code debugging:** all of these (plus eval/build/gen-infos) are wired as
> launch configs in [`.vscode/launch.json`](../.vscode/launch.json), each with
> `stopOnEntry` + `justMyCode` so you step only through *your* code (the PETR
> source + `petr_port/`), skipping torch/mmengine internals. Pick a config from
> the Run-and-Debug panel and press F5.


> Honesty about this port: `loss_demo`/`train_overfit` exercise the **real**
> PETR head, assigner, costs and losses on the modern stack. What is *not* ported
> is the full mmengine training **Runner** (multi-GPU dataloader with the
> augmentation pipeline of §5, EMA/hooks, the LR schedule of §6). For a real,
> full training run, use the original stack via `tools/dist_train.sh` (the
> [REPRODUCE.md](REPRODUCE.md) `--no-deps` install gets you the inference stack;
> full training wants the original `mmdet3d 0.17` environment). The demos here
> are for *learning the mechanics*, and they use the genuine loss code.

---

## 8. Exercises

1. **Read the matching cost vs the loss** and confirm the weights line up
   (`FocalLossCost 2.0` ↔ `loss_cls 2.0`; `BBox3DL1Cost 0.25` ↔ `loss_bbox 0.25`).
   Why does the *cost* use 8 dims but the *loss* use 10? (velocity.)
2. **`code_weights` sweep**: in `loss_demo.py`, set the head's `code_weights`
   velocity entries to `1.0` and re-run; see how `loss_bbox` changes.
3. **Deep supervision**: in `train_overfit.py`, drop the `d0..d4` aux terms (only
   optimise final-layer loss) and compare convergence speed.
4. **Augmentation geometry**: apply a 20° `GlobalRotScaleTransImage` rotation to
   one sample, then run `study/demo_visualize.py`-style projection and verify the
   GT boxes still land on the objects (the matrices were rotated consistently).
5. **Matching stability**: print the matched query indices across the 6 decoder
   layers for one sample — do later layers match more confidently? (Connects to
   why aux losses help.)
6. **Imbalance**: count per-class GT in `nuscenes_infos_train.pkl`; relate the
   long tail to why CBGS exists.

---

## 9. One-page training recap

```
inputs: 6 imgs + lidar2img         (+ GT boxes/labels)
   │  augment: ResizeCropFlip(img+intrinsics) ; GlobalRotScaleTrans(lidar2img+GT) ; GridMask
   ▼
VoVNet-V99(pretrained) → CPFPN → 3D-PE → 6x decoder → cls(900,10), reg(900,10) x6 layers
   │
   ▼  per layer:
Hungarian match (FocalCost + L1Cost, one-to-one)  →  FocalLoss(all) + L1(matched, code_weighted)
   │  sum 12 terms (6 layers x 2)
   ▼
AdamW(lr 2e-4, backbone x0.1) + FP16(512) + clip(35) + Cosine(24 ep)   →  no NMS at test
```

Start with `study/loss_demo.py`, read `PETRHead.loss_single` beside it, then run
`study/train_overfit.py` and watch the loss fall. That closes the loop from the
architecture guide to a trainable model.
