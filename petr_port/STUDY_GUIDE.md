# Studying PETR — a guided plan for someone who already knows DETR

You know DETR. This guide builds **PETR** on top of that knowledge, anchored in
the *actual code in this repo*, with runnable examples. Work through the phases
in order; each has a concrete "do this" and pointers to the exact files/lines.

> All file links are relative to this `petr_port/` folder. Code lives under
> `../projects/mmdet3d_plugin/`. Runnable scripts live in `petr_port/`.

---

## 0. The one-paragraph mental model (DETR → DETR3D → PETR)

- **DETR (2D):** object *queries* cross-attend to flattened CNN features; each
  key gets a **2D sine positional encoding**. A query decodes a 2D box via an
  MLP. One-to-one Hungarian matching ⇒ no NMS.
- **DETR3D (multi-view 3D):** each query owns a **3D reference point**; you
  *project* that point into every camera and **bilinearly sample** local
  features (sparse, deformable-style). Geometry is applied *explicitly, every
  layer*, via projection.
- **PETR (this repo):** **don't sample.** Instead, bake 3D geometry **into the
  keys**. For every image-feature pixel, back-project a camera **frustum** into
  the 3D (LiDAR) frame, and MLP-encode those 3D coordinates into a **3D Position
  Embedding (3D PE)**. Add 3D PE to the image features → *“3D position-aware
  features.”* Queries (built from learnable 3D anchor points) then do **vanilla
  global cross-attention** over all multi-view 3D-aware features. The network
  learns the query↔region association implicitly; the geometry lives in the key
  positional encoding.

**The single most important file to understand is the 3D PE builder:**
[`PETRHead.position_embeding()`](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py)
(~line 282). Everything else is DETR you already know, lifted to 3D.

---

## Phase 0 — Run it and look (30 min)

Everything is already set up (see [REPRODUCE.md](REPRODUCE.md)). Do these three:

```bash
cd petr_port

# (a) full nuScenes mini_val evaluation  ->  mAP 0.3406 / NDS 0.3633
python3 petr_infer.py

# (b) trace tensor shapes through the whole forward pass
python3 study/trace_shapes.py

# (c) render predicted 3D boxes onto the 6 cameras (saves a montage)
python3 study/demo_visualize.py --index 0 --score-thr 0.3
#   -> work_dirs/study/demo_boxes_0.jpg   (open it!)
```

`trace_shapes.py` prints exactly this (memorise these shapes — the rest of the
guide refers to them), for input `img = (1, 6, 3, 320, 800)`:

```
img_backbone        (6,3,320,800)         -> [(6,768,20,50), (6,1024,10,25)]   # VoVNet stage4/5
img_neck (CPFPN)    ^                      -> [(6,256,20,50), (6,256,10,25)]    # 2 FPN levels, 256-d
head.input_proj     (6,256,20,50)         -> (6,256,20,50)                     # 1x1 conv
head.position_encoder (6,192,20,50)       -> (6,256,20,50)                     # 3D PE MLP  (192 = 3 coords x 64 depth bins)
head.positional_encoding (1,6,20,50)      -> (1,6,384,20,50)                   # sine 3D (multiview)
head.adapt_pos3d    (6,384,20,50)         -> (6,256,20,50)
head.query_embedding (900,384)            -> (900,256)                         # 900 queries
head.transformer    (1,6,256,20,50)       -> [(6,1,900,256), memory]           # 6 decoder layers
head.cls/reg_branch[0] (1,900,256)        -> (1,900,10)  x6 layers
DECODED: boxes_3d (300,9)=(x,y,z,w,l,h,yaw,vx,vy), 48 boxes score>0.3
```

The head uses only **FPN level 0** (the `20×50` map): `position_level = 0`.

---

## Phase 1 — The papers & the gap (1 h)

Read with your DETR hat on, looking only for *what changes*:
- PETR (ECCV'22) §3.2–3.3: the **3D Coordinates Generator** and **3D Position
  Encoder**. That's the whole novelty.
- Skim DETR3D's "feature sampling" to appreciate what PETR *removes*.

Checkpoint question you should be able to answer after this phase: *“Why does
PETR not need deformable attention or per-layer projection?”* (Because geometry
is precomputed once into the key PE; cross-attention is then plain MHA.)

---

## Phase 2 — Geometry & coordinate systems (1–2 h)

This is the part that's new vs 2D DETR. Three frames matter: **LiDAR** (the
canonical ego frame PETR predicts in), **camera**, **image (pixels)**.

**`lidar2img` (4×4 per camera)** is the workhorse. Its derivation is in
[`CustomNuScenesDataset.get_data_info()`](../projects/mmdet3d_plugin/datasets/nuscenes_dataset.py)
(~line 56) and mirrored in
[`petr_infer.build_cam_matrices()`](petr_infer.py):

```python
lidar2cam_r  = inv(sensor2lidar_rotation)
lidar2cam_t  = sensor2lidar_translation @ lidar2cam_r.T
lidar2cam_rt = eye(4); lidar2cam_rt[:3,:3] = lidar2cam_r.T; lidar2cam_rt[3,:3] = -lidar2cam_t
viewpad      = eye(4); viewpad[:3,:3] = cam_intrinsic          # K
lidar2img    = viewpad @ lidar2cam_rt.T                        # 4x4: LiDAR point -> image homogeneous
```

> Gotcha: the image preprocessing (`ResizeCropFlipImage`, test path in
> [`transform_3d.py`](../projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py)
> ~line 363) **modifies the intrinsics** (resize 0.5 → crop 130px) and therefore
> recomputes `lidar2img`. If you ever change input resolution, `lidar2img`
> changes too. See `petr_infer._img_transform()` for the exact 3×3 `ida_mat`.

**Exercise 2.1** — back-projection by hand. For `index 0`, take `lidar2img` of
`CAM_FRONT`, pick a pixel `(u,v)=(800,160)` and depth `d=15`, and compute the
LiDAR point `inv(lidar2img) @ (u*d, v*d, d, 1)`. Confirm it lands ~15 m in front
of the ego. (This is literally what the 3D PE does for *every* pixel × depth.)

---

## Phase 3 — The 3D Position Embedding (the heart) (2 h)

> **Dedicated deep dive + paper Figure 4 reproduction:**
> **[3D_PE_DEEPDIVE.md](3D_PE_DEEPDIVE.md)** with
> `study/visualize_3d_pe.py` → `work_dirs/study/fig4_3d_pe_0.jpg`.

Read [`PETRHead.position_embeding()`](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py)
(~line 282) line by line. The algorithm:

1. **Feature→pixel grid.** `H,W = 20,50`. Map feature cells to padded-image
   pixels: `coords_w = arange(W)*pad_w/W`, `coords_h = arange(H)*pad_h/H`.
2. **Depth bins (D=64)** with **LID** (linear-increasing discretization — bins
   widen with range, like CaDDN/DD3D):
   ```python
   bin = (position_range[3] - depth_start) / (depth_num*(1+depth_num))
   coords_d = depth_start + bin * index * (index+1)        # i = 0..63
   ```
3. **Frustum** `coords[W,H,D,3] = (u,v,d)` → homogeneous, then scale pixel coords
   by depth: `(u·d, v·d, d, 1)`. This is the image-ray point at depth `d`.
4. **Back-project** to LiDAR: `coords3d = inv(lidar2img) @ (u·d, v·d, d, 1)`
   → a real 3D point per `(cam, pixel, depth)`.
5. **Normalise** by `position_range` to `[0,1]`; **mask** points that fall
   outside the volume (if >½ the depth bins are out).
6. `inverse_sigmoid(coords3d)` → **`position_encoder`** (two 1×1 convs,
   `192→1024→256`, where `192 = 3 coords × 64 depth bins`) → **3D PE**
   `[B,N,256,H,W]`.

In [`PETRHead.forward()`](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py)
(~line 359) the 3D PE is *added* to a sine 3D PE (`positional_encoding` →
`adapt_pos3d`) to form the final **key positional encoding** `pos_embed`.

**Why it works:** two pixels from *different cameras* that correspond to the same
physical 3D location get **similar** 3D PE, so a query can attend to all views of
an object through ordinary attention — no explicit cross-view wiring.

**Exercise 3.1** — visualise the geometry. Run
`python3 study/demo_visualize.py` and confirm boxes wrap real objects (the green
cars in `CAM_BACK`, the pedestrian in `CAM_BACK_RIGHT`). That projection uses the
*same* `lidar2img` the 3D PE inverts.

**Exercise 3.2** — ablate the 3D PE. In `petr_infer.import_and_build()` set
`with_position=False` (and/or `with_multiview=False`) in the head dict, re-run
`petr_infer.py`, and watch mAP collapse. This isolates the contribution of the
3D PE. (Revert afterwards.)

---

## Phase 4 — Queries & the decoder (1–2 h)

**Queries are 3D anchors.** In
[`_init_layers()`](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py)
(~line 265):
```python
self.reference_points = nn.Embedding(num_query=900, 3)      # learnable 3D anchors in [0,1]^3
self.query_embedding  = Sequential(Linear(384,256), ReLU, Linear(256,256))
# forward: query_embeds = query_embedding(pos2posemb3d(reference_points))
```
`pos2posemb3d` (top of `petr_head.py`) is DETR's sine PE generalised to a 3D
point `(x,y,z) → 384-d`. So a query "is" a sine-encoded 3D location — directly
comparable to the key 3D PE. Compare this to DETR's learned query embeddings:
PETR's are *geometrically meaningful*.

**Decoder** in
[`petr_transformer.py`](../projects/mmdet3d_plugin/models/utils/petr_transformer.py):
- `PETRTransformer.forward` flattens features `[B,N,C,H,W] → memory [N·H·W, B, C]`,
  sets `key_pos = 3D PE`, `query_pos = query_embeds`, `target = 0`.
- 6× `PETRTransformerDecoderLayer` with
  `operation_order = (self_attn, norm, cross_attn, norm, ffn, norm)`:
  - **self_attn**: standard MHA among the 900 queries.
  - **cross_attn**: `PETRMultiheadAttention` — `query = q+query_pos`,
    `key = memory + key_pos(3D PE)`, `value = memory`. **Plain global attention**
    (the key difference from DETR3D's sampling).
- `return_intermediate=True` ⇒ outputs from all 6 layers `[6,B,900,256]`.

**Exercise 4.1** — attention rollout. Add a hook on the cross-attention of layer
5, take one high-score query, reshape its attention over keys back to
`N×H×W`, and overlay on the 6 images. You'll see the query light up the region
of its object across views.

---

## Phase 5 — Boxes, losses, matching, NMS-free decode (1 h)

> Deep dive + runnable loss/training demos: see
> **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** (`study/loss_demo.py`,
> `study/train_overfit.py`).

**Box parameterisation** (tail of `PETRHead.forward`, ~line 430). Per decoder
layer: `cls = cls_branches[l](dec)`, `reg = reg_branches[l](dec)`. The
**reference point is added** to the predicted center before sigmoid (residual /
anchor refinement), then scaled by `pc_range`:
```python
tmp[...,0:2] += reference[...,0:2]; tmp[...,0:2] = sigmoid(...)     # x,y
tmp[...,4:5] += reference[...,2:3]; tmp[...,4:5] = sigmoid(...)     # z
# then * (pc_range span) + pc_range min
```
Box code (10-d): `(cx,cy,w,l,cz,h,sinθ,cosθ,vx,vy)` (normalised). See
[`util.normalize_bbox/denormalize_bbox`](../projects/mmdet3d_plugin/core/bbox/util.py).

> Note: `cls_branches`/`reg_branches` are built as `[module]*num_pred` — the
> **same** module repeated, i.e. **weights are shared across all 6 layers**.

**Training = DETR set prediction, in 3D:**
- Matching:
  [`HungarianAssigner3D`](../projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_3d.py)
  — cost = `FocalLossCost(cls) + BBox3DL1Cost(normalised 10-d box) + IoUCost(0,
  fake)`, solved with `scipy.linear_sum_assignment`. One-to-one ⇒ **no NMS**.
- Losses: `FocalLoss(cls) + L1(box, code_weights)`, applied to **every** decoder
  layer (deep supervision). See `PETRHead.loss()` / `loss_single()`.

**Inference decode:**
[`NMSFreeCoder.decode_single()`](../projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py)
— `sigmoid(cls)`, `topk(300)` over `query×class`, `label = idx % num_classes`,
`denormalize_bbox`, range-filter by `post_center_range`. No NMS anywhere.

**Exercise 5.1** — change `num_query` 900→300 in the head dict, re-run eval,
compare mAP/latency. **Exercise 5.2** — read `loss_single`; map every term back
to DETR's loss and identify the only genuinely new piece (the 3D L1 on the
9/10-dim box with `code_weights`).

---

## Phase 6 — Backbone, neck, configs (1 h)

- Backbone [`VoVNetCP`](../projects/mmdet3d_plugin/models/backbones/vovnetcp.py)
  (`V-99-eSE`) → outputs `stage4 (768)`, `stage5 (1024)`.
- Neck [`CPFPN`](../projects/mmdet3d_plugin/models/necks/cp_fpn.py) → 2 levels,
  256-d. (A lighter FPN than mmdet's.)
- Config you ran:
  [`petr_vovnet_gridmask_p4_800x320.py`](../projects/configs/petr/petr_vovnet_gridmask_p4_800x320.py).
  Read it top-to-bottom; you now recognise every block (model, `ida_aug_conf`,
  `train/test_pipeline`, optimizer). Compare with the ResNet config
  [`petr_r50dcn_gridmask_c5.py`](../projects/configs/petr/petr_r50dcn_gridmask_c5.py)
  to see backbone/resolution trade-offs.

**Exercise 6.1** — diff `800x320` vs `1600x640` configs; predict the effect on
`H,W` of the feature map (hence number of keys) and on accuracy/speed.

---

## Phase 7 — PETRv2 (temporal, FPE, seg, denoise) (2 h)

All present in the repo:
- **Temporal (2 frames):** previous-frame features are aligned to the current
  frame using ego motion; the **3D PE is pose-aligned** so a moving object's
  past and present 3D coords are consistent. Config:
  [`petrv2_vovnet_gridmask_p4_800x320.py`](../projects/configs/petrv2/petrv2_vovnet_gridmask_p4_800x320.py).
- **Feature-guided Position Encoder (FPE):** the 3D PE is *modulated by the image
  features* (a small gating network) instead of being purely geometric — see
  [`petrv2_head.py`](../projects/mmdet3d_plugin/models/dense_heads/petrv2_head.py).
- **Query denoising (DN-DETR style):** noised GT boxes are fed as auxiliary
  queries with a reconstruction loss to stabilise/ speed up matching — see
  [`petrv2_dnhead.py`](../projects/mmdet3d_plugin/models/dense_heads/petrv2_dnhead.py)
  and the [`denoise/`](../projects/configs/denoise) configs.
- **BEV segmentation:** extra segmentation queries each predict a BEV patch —
  [`petr_head_seg.py`](../projects/mmdet3d_plugin/models/dense_heads/petr_head_seg.py)
  and [`petrv2_BEVseg.py`](../projects/configs/petrv2/petrv2_BEVseg.py).

**Exercise 7.1** — in `petrv2_head.py`, find where the previous-frame 3D coords
are transformed into the current frame and connect it to the temporal-alignment
claim in the PETRv2 paper.

---

## Phase 8 — Consolidation exercises

1. **One-pager**: redraw the PETR diagram from memory, labelling tensor shapes
   from Phase 0.
2. **Ablations table**: run eval for `{with_position, with_multiview} ∈
   {on,off}²` (4 runs) and tabulate mAP. Explain the ranking.
3. **Geometry probe**: extend `study/trace_shapes.py` to also dump the
   normalised `coords3d` range and the `coords_mask` coverage per camera.
4. **DETR3D vs PETR**: write 5 bullet points contrasting feature *sampling* vs
   feature *position-encoding*, including compute/memory implications (PETR's
   cross-attention is over `N·H·W` keys — quadratic-ish; DETR3D samples a few
   points).

---

## File map (concept → code)

| Concept | File | Symbol |
|---|---|---|
| Detector glue (extract feats, test) | [detectors/petr3d.py](../projects/mmdet3d_plugin/models/detectors/petr3d.py) | `Petr3D` |
| **3D Position Embedding** | [dense_heads/petr_head.py](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py) | `position_embeding` |
| Head forward / box param | [dense_heads/petr_head.py](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py) | `forward`, `_init_layers` |
| Query sine PE | [dense_heads/petr_head.py](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py) | `pos2posemb3d` |
| Transformer / cross-attn | [models/utils/petr_transformer.py](../projects/mmdet3d_plugin/models/utils/petr_transformer.py) | `PETRTransformer`, `PETRMultiheadAttention` |
| Sine 3D PE (multiview) | [models/utils/positional_encoding.py](../projects/mmdet3d_plugin/models/utils/positional_encoding.py) | `SinePositionalEncoding3D` |
| Box (de)normalisation | [core/bbox/util.py](../projects/mmdet3d_plugin/core/bbox/util.py) | `normalize_bbox`, `denormalize_bbox` |
| NMS-free decode | [core/bbox/coders/nms_free_coder.py](../projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py) | `NMSFreeCoder` |
| Hungarian matching | [core/bbox/assigners/hungarian_assigner_3d.py](../projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_3d.py) | `HungarianAssigner3D` |
| `lidar2img` build | [datasets/nuscenes_dataset.py](../projects/mmdet3d_plugin/datasets/nuscenes_dataset.py) | `get_data_info` |
| Test image transform | [datasets/pipelines/transform_3d.py](../projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py) | `ResizeCropFlipImage` |
| Backbone / neck | [backbones/vovnetcp.py](../projects/mmdet3d_plugin/models/backbones/vovnetcp.py) · [necks/cp_fpn.py](../projects/mmdet3d_plugin/models/necks/cp_fpn.py) | `VoVNetCP`, `CPFPN` |

---

## Conventions & gotchas (keep this handy)

- **Box tensor (LiDAR):** `[x, y, z_bottom, w, l, h, yaw, vx, vy]`. `z` stored as
  **bottom** center; `gravity_center.z = z + h/2`.
- **yaw → nuScenes:** `box_yaw = -yaw - π/2` (see `petr_infer.output_to_nusc_box`).
- **Two ranges:** `point_cloud_range` (box decode, `±51.2`) vs `position_range`
  (3D PE volume, `±61.2`) — they are **different** on purpose.
- **Class order = config order** (`car,truck,construction_vehicle,bus,trailer,
  barrier,motorcycle,bicycle,pedestrian,traffic_cone`), *not* the devkit order.
- **GridMask** is training-only (no-op under `model.eval()`).
- **Shared heads:** all 6 decoder layers share one `cls`/`reg` branch.
- **Modern-stack note:** this repo runs via the shim in
  [compat.py](compat.py); the original files are unmodified. If something about
  imports/registries confuses you, that's the shim, not PETR — see
  [REPRODUCE.md](REPRODUCE.md) §7.

---

## Suggested schedule

| When | Phases |
|---|---|
| Morning | 0 (run everything) → 1 → 2 |
| Midday | 3 (the 3D PE — the crux) |
| Afternoon | 4 → 5 |
| Next day | 6 → 7 → 8 (exercises) |

Start by running the three commands in Phase 0 and opening
`work_dirs/study/demo_boxes_0.jpg`. Then read `position_embeding()` with this
guide side-by-side. That function *is* PETR.
