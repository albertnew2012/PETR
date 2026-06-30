# PETR's 3D Position Embedding — a deep dive (for a DETR person)

You know DETR. This note is **only** about the one idea that makes PETR work:
the **3D Position Embedding (3D PE)**. It explains the mechanism, the exact math
and code, and a faithful reproduction of the paper's **Figure 4** on your own
runtime.

```bash
cd petr_port
python3 study/visualize_3d_pe.py --index 0      # -> work_dirs/study/fig4_3d_pe_0.jpg
```

---

## 1. The gap PETR fills (vs DETR / DETR3D)

| | keys the decoder attends to | positional encoding of those keys |
|---|---|---|
| **DETR** | 2D image tokens | **2D** sine PE (where in the image) |
| **DETR3D** | features *sampled* at projected 3D ref-points | geometry applied by projection, per layer |
| **PETR** | the **same** 2D image tokens (all 6 views) | **3D** PE (where in 3D space this token's ray lives) |

DETR's PE answers *“where in the image is this key?”*. PETR's 3D PE answers
*“where in **3D space** does this key look?”*. Once keys carry 3D position, an
object query (which is itself a 3D point) can find its object with **ordinary
cross-attention across all views** — no deformable sampling, no per-layer
projection. That's the whole trick.

Code entry point — one function:
[`PETRHead.position_embeding()`](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py)
(~line 282).

---

## 2. Step 1 — the 3D Coordinates Generator (a camera frustum in 3D)

> **See it first:** `python3 study/visualize_frustum.py --index 0`
> → `work_dirs/study/frustum_bev_0.jpg`.

**The intuition (this is the whole confusing part):**
- A 2D pixel is **not** a 3D point — it's a **ray** (a line of sight). You know
  the *direction* but not *how far*.
- PETR doesn't *predict* the distance; it **samples** it — it picks `D=64`
  candidate depths along the ray. So **one pixel → 64 candidate 3D points** (a
  "ray of points").
- Do that for every pixel of a camera and you get a **frustum** — a fan/cone of
  3D points, narrow near the lens, spreading out far away (the *right* panel of
  the figure shows one camera's fan, coloured by depth).
- Convert each `(pixel, depth)` into `(x,y,z)` in the shared **LiDAR** frame, and
  now **all 6 cameras' frustums live in one common 3D space** and tile 360°
  around the car (the *left* panel). That shared frame is what lets a query
  (a 3D point) talk to keys from any camera.

Concretely:

For every camera and every feature-map cell `(H=20, W=50)`, PETR builds a
**frustum of 3D points** by sampling `D=64` depths along the pixel ray and
back-projecting them into the **LiDAR** frame.

**2a. Pixel grid → input-image pixels** (feature cells map back to the 800×320
input the model saw):
```python
coords_h = arange(H) * pad_h / H          # row -> pixel-v
coords_w = arange(W) * pad_w / W          # col -> pixel-u
```

**2b. Depth bins via LID** (Linear-Increasing Discretization — bins widen with
range, like CaDDN/DD3D, so near depths are sampled finely):

$$ d_i = d_{\text{start}} + \frac{d_{\max}-d_{\text{start}}}{D\,(1+D)}\; i\,(i+1),\quad i=0\dots 63 $$

```python
bin = (position_range[3] - depth_start) / (depth_num * (1 + depth_num))
coords_d = depth_start + bin * index * (index + 1)
```

**2c. Frustum point, homogeneous, scaled by depth** → `(u·d, v·d, d, 1)` (a
camera-ray point at depth `d`).

**2d. Back-project to LiDAR** with `img2lidar = inv(lidar2img)`:

$$ \mathbf{p}^{\text{lidar}} = \text{lidar2img}^{-1}\,(u\,d,\; v\,d,\; d,\; 1)^\top $$

So a single feature cell becomes a **64-point 3D ray**. The script prints this
for the centre-front pixel (real numbers from `--index 0`):
```
depth   3.0 m  ->  LiDAR xyz = ( -0.1,   3.5,  -0.5)
depth  15.0 m  ->  LiDAR xyz = ( -0.3,  15.5,  -1.4)
depth  45.0 m  ->  LiDAR xyz = ( -1.0,  45.7,  -3.6)
```
i.e. the centre pixel looks straight ahead (`x≈0`), marching forward in `y`,
dipping toward the road in `z`. **That 3D ray is the geometric identity of the
token.**

Finally the coords are normalised to `[0,1]` by `position_range` and points
outside the volume are masked (`coords_mask`).

---

## 3. Step 2 — the 3D Position Encoder (an MLP over the ray)

The 64 ray points (×3 xyz = **192 channels**) are fed through a tiny 2-layer
1×1-conv MLP to produce a **256-d** embedding per cell
([`_init_layers`](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py)
~line 261):

```python
self.position_encoder = Sequential(
    Conv2d(3*depth_num=192, 1024, 1), ReLU,
    Conv2d(1024, embed_dims=256, 1))
# 3D PE  = position_encoder(inverse_sigmoid(coords3d))   # [B, N, 256, H, W]
```

Shapes (from `study/trace_shapes.py`):
`position_encoder: (6,192,20,50) -> (6,256,20,50)`.

This **3D PE** is added to the image features (after a sine 3D PE via
`adapt_pos3d`) to form the decoder's **key positional encoding**
(`PETRHead.forward`, ~line 395). The value/key content is the image feature; the
*position* is now 3D.

**Query side (the symmetric half):** queries are learnable **3D anchor points**
`reference_points ∈ [0,1]³`, sine-encoded by `pos2posemb3d` and MLP'd to 256-d
(`query_embedding`). Query PE and key PE live in the *same* 3D-encoded space —
that's why plain dot-product attention can match a query to its object's tokens.

---

## 3b. Common confusion: "image space" vs "3D space"

A natural question while reading this code:

> *"Sine PE happens in image space, and the 3D PE is the 3D discretized points
> projected to image space, then embedded, then added to the 2D sine PE — and
> the sum is the key position?"*

Almost — **one arrow is backwards**. What's right and what to fix:

- ✅ **Sine PE is image space** — `SinePositionalEncoding3D` encodes *(which
  camera, row, column)* on the feature grid.
- ✅ **The sum is the key position** — `pos_embed = 3D PE + sine PE` is the
  `key_pos` fed to cross-attention.
- ❌ **Direction:** the geometry runs **image → 3D (back-projection)**, *not*
  3D → image. Pixels are **un-projected** into 3D, not 3D points projected onto
  the image.

**What actually happens**, per feature cell `(u, v)` on the `20×50` grid:
1. sample **64 depths** `d` (LID),
2. form homogeneous image points `(u·d, v·d, d, 1)`,
3. multiply by **`img2lidar = inv(lidar2img)`** → 64 real **3D points in the
   LiDAR frame** (the ray / frustum),
4. normalize those `(x, y, z)` and run the `position_encoder` MLP → a 256-d
   vector.

So it's *"image pixel + depths → back-projected to 3D coordinates → embed those
3D coordinates,"* not *"3D points projected onto the image."*

**The reconciliation (the real source of confusion):** both PEs are stored on
the **same image grid** `[B, N, 256, H, W]` — that's *why* they can be summed —
but they **encode different things**:

| | layout (where it is stored) | value (what it encodes) |
|---|---|---|
| **sine PE** | per image cell `[H, W]` | "I am pixel (row, col) of camera n" — image/view identity |
| **3D PE** | per image cell `[H, W]` | "my pixel's ray passes through these 3D coordinates" — LiDAR-space geometry |

The 3D PE is therefore **indexed by image location but valued by 3D position** —
it attaches a *3D meaning* to each *image token*.

```
image cell (u,v) --+64 depths--> back-project (inv lidar2img) --> 3D coords --MLP--> 3D PE  ┐
                                                                                           ├ add -> key_pos (1,6,256,20,50)
camera n, row, col -----------------------------------------> sine PE --adapt_pos3d-->      ┘
```

To *see* the back-projection literally, run `study/visualize_frustum.py` — those
fanned-out 3D points are exactly the `coords3d` that get embedded into the 3D PE.

---

## 4. Reproducing Figure 4 — "3D PE similarity"

Paper (Sec. 3.3): *“we randomly select the PE at three points in the front view
and compute the PE similarity between these three points and all multi-view PEs
… when we select the left point in the front view, the right region of the
front-left view will have relatively higher response. It indicates that 3D PE
implicitly establishes the position correlation of different views in 3D space.”*

[`study/visualize_3d_pe.py`](study/visualize_3d_pe.py) does exactly this on the
real model:
1. hook `position_encoder` to grab the 3D PE `[6, 256, 20, 50]`;
2. pick 3 points (left/centre/right) in **FRONT**;
3. cosine-similarity of each point's 256-d PE vs **every cell in all 6 views**;
4. overlay the 6 heatmaps (columns laid out FL · F · FR · BL · B · BR).

**What you see in `work_dirs/study/fig4_3d_pe_0.jpg`** (reproduced faithfully):
- A **localised warm blob** around the red point in FRONT (a BEV-like arc — the
  back-projected ray's footprint).
- **LEFT point →** the blob hugs the left of FRONT *and bleeds into the **right
  edge of FRONT_LEFT*** — the overlapping field of view. (The paper's exact
  example.)
- **CENTER point →** symmetric; spills into the inner edges of both FRONT_LEFT
  and FRONT_RIGHT.
- **RIGHT point →** mirror image; bleeds into the **left edge of FRONT_RIGHT**.
- **All BACK cameras stay cold** (deep blue) — front 3D points are far from what
  the rear cameras see.

That cross-view bleed is the proof: two pixels in *different* cameras that look
at the *same 3D region* get *similar* 3D PE. Attention can therefore associate
them for free.

### Why the heatmap lights up other views — the geometry

The cosine heatmap is the *measurement*; the tail of
[`study/visualize_3d_pe.py`](study/visualize_3d_pe.py) prints the *cause*. It
back-projects the CENTER front point and shows that the 3D PE of one pixel is
really the embedding of a **ray**, not a single point:

```python
l2i       = img_metas[0]["lidar2img"][FRONT]     # 3D -> image (4x4)
img2lidar = np.linalg.inv(l2i)                   # image -> 3D
u, v      = cols[1]*16 + 8, r*16 + 8             # feature cell -> 800x320 pixel
for d in (3.0, 15.0, 45.0):                      # sweep depth along the ray
    p = img2lidar @ np.array([u*d, v*d, d, 1.0])
    p = p[:3] / p[3]
```

**The math.** Forward, `lidar2img` sends a LiDAR point to *homogeneous* image
coordinates $(u z,\, v z,\, z,\, 1)$ — the pixel scaled by its depth $z$. A
single pixel can't be inverted (the depth was multiplied away), so you **pick** a
depth $d$, rebuild the homogeneous vector, invert, and divide by the homogeneous
coordinate $w$:

$$ \mathbf{p}^{\text{lidar}}(d) = \text{lidar2img}^{-1}\,(u\,d,\; v\,d,\; d,\; 1)^\top \;\;\xrightarrow{\;/\,w\;}\;\; (x, y, z) $$

Sweeping $d = 3 \to 15 \to 45\,\text{m}$ marches the point **outward along one
straight line of sight** — the pixel's ray. (`*16+8` first turns the feature
cell back into an 800×320 input pixel: the feature map has stride 16, and `+8`
hits the cell centre.)

**Why other views light up.** PETR's 3D PE for a pixel encodes that *whole ray*
(Section 2: 64 depths → 64 points → MLP). Take any other camera with an
overlapping field of view: some of its pixels look at the **same stretch of 3D
space** the front ray passes through, so *their* rays intersect the front ray's
points. Same 3D position → same 3D-coordinate input to the encoder → **similar
256-d PE** → high cosine → a **warm blob in that neighbour view**. Cameras that
never see that region (the rear three) get near-orthogonal PE and stay cold.

**Why the side matches.** The lateral sign of $\mathbf{p}^{\text{lidar}}$ follows
the pixel's column — the LEFT front pixel back-projects to one side, the RIGHT to
the other (the script's printed `xyz` confirms the sign). The overlapping camera
on that side is the one whose FOV contains those 3D points, which is exactly why
**LEFT point → FRONT_LEFT bleed** and **RIGHT point → FRONT_RIGHT bleed**, never
the reverse.

In one line: *back-projection shows each pixel's PE is a 3D ray; wherever another
camera's pixels sit on that ray their PE matches — and that match **is** the
cross-view heatmap.*

> Knobs: `--index N` (different scene), `--row-frac` (move the 3 points up/down).
> Try a back-row scene to see BACK cameras light up instead.

---

## 5. Why this is the elegant part

- **Geometry is precomputed, once, into the keys.** The transformer is vanilla.
- **Multi-view fusion is implicit.** No explicit cross-camera wiring; the shared
  3D coordinate frame does it (Fig 4 is the evidence).
- **Depth is handled by sampling, not predicting.** The 64-depth ray means a
  pixel's PE is "the set of 3D points it could be" — the network learns to weight
  them. (Contrast monocular methods that regress a single depth.)
- **`position_range` (±61.2) ⊃ `point_cloud_range` (±51.2)** on purpose: the PE
  volume is padded beyond the detection volume so boundary objects still get
  well-defined 3D PE.

---

## 6. Exercises (all runnable)

1. **Move the point** (`--row-frac 0.75`) — lower points are nearer road; watch
   the blob shrink toward the ego and the cross-view bleed change.
2. **Ablate depth** — set `depth_num=1` in the model dict (one depth) and
   regenerate; the PE loses its ray structure and the blobs blur. (Then restore.)
3. **LID vs uniform** — flip `LID=False` and compare bin spacing / blob shape.
4. **3D PE off** — set `with_position=False` and re-run `petr_infer.py`: mAP
   collapses, quantifying the figure.
5. **Query↔key space** — confirm `pos2posemb3d` (query) and the key PE are the
   *same* dimensionality (256) and meaning; that's why attention works.
6. **Numbers** — change the printed back-projection depths in the script; verify
   a left/right pixel back-projects to `x<0` / `x>0` in LiDAR (matches the bleed
   direction in Fig 4).

---

## 7. The 30-second recap

```
each feature cell (per camera)
   └─ 64 depths (LID)  ─►  64 ray points  ─►  back-project via inv(lidar2img)
        ─►  3D coords (lidar frame, 192 ch)  ─►  MLP  ─►  256-d 3D PE
key = image_feature + 3D PE        query = MLP(sine(3D anchor point))
   └────────────── plain cross-attention (no sampling) ──────────────┘
Fig 4: same-3D-region pixels in different cameras  ⇒  similar 3D PE  ⇒  fusion
```

Read `position_embeding()` with this page open, then run
`study/visualize_3d_pe.py` and stare at `fig4_3d_pe_0.jpg`. That function + that
figure *are* PETR's contribution.
