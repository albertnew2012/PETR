# PETR queries & reference points — how object queries are made

A focused note on **how PETR generates its object queries** and what the
`reference_points` `(900, 3)` actually are. All code refs are
[dense_heads/petr_head.py](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py);
the numbers below come from the trained `petr_vovnet_p4_800x320` checkpoint.

---

## TL;DR

```
reference_points : nn.Embedding(900, 3)   ← the ONLY learnable query parameter (≈ nn.Parameter(900,3))
        │  init uniform[0,1]; values are NORMALISED coords (0↔-51.2 m, 1↔+51.2 m … = pc_range)
        ├─ pos2posemb3d(·)   sine-encode each (x,y,z)        (900,3) → (900,384)
        │  query_embedding   MLP 384→256→256                  (900,384) → (900,256)  = the queries
        └─ inverse_sigmoid(·) → added to the predicted box centre  (anchor / reference)
After training these 900 anchors are FROZEN and SHARED across every scene.
```

So: **`query = MLP(sine(3D anchor point))`**, and the *only* learned query
parameter is the `(900, 3)` table of 3D anchor points.

---

## 1. The learnable thing: `reference_points` `(900, 3)`

`_init_layers` (~line 265):
```python
self.reference_points = nn.Embedding(self.num_query, 3)        # 900 × 3
self.query_embedding  = nn.Sequential(
    nn.Linear(self.embed_dims*3//2, self.embed_dims),          # 384 → 256
    nn.ReLU(),
    nn.Linear(self.embed_dims, self.embed_dims))               # 256 → 256
```
- `nn.Embedding(900, 3)` is just a container; its **`.weight` is an `nn.Parameter`
  of shape `(900, 3)`** — i.e. effectively `nn.Parameter(torch.empty(900, 3))`.
  PETR only uses `.weight` directly (no lookup).
- These are **900 learnable 3D anchor points**. `900 × 3 = 2700` learned numbers —
  that's the entire "query bank".

## 2. Initialization

`init_weights` (~line 276):
```python
nn.init.uniform_(self.reference_points.weight.data, 0, 1)
```
→ a **random-uniform scatter** in the unit cube `[0,1]³`. **Not** a regular grid,
**not** evenly *spaced* — just random, then learned.

## 3. What the numbers mean: normalised coords ↔ `pc_range`

The values are **normalised**, not metres. `[0,1]` maps **linearly** onto
`pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]`:

| normalised | x, y | z |
|---|---|---|
| 0.0 | −51.2 m | −5.0 m |
| 0.5 | 0 m (ego) | −1.0 m |
| 1.0 | +51.2 m | +3.0 m |

The mapping is applied to the **final** box prediction (`forward`, ~line 440):
```python
all_bbox_preds[...,0:1] = pred*(pc_range[3]-pc_range[0]) + pc_range[0]   # [0,1] → metres
```
So a reference point at normalised `(0.5, 0.5, 0.4)` ≈ "right at the ego, near
ground height".

> Nuance: they're only *nominally* `[0,1]`; training lets a few drift slightly
> outside (see §7). The box path runs them through `inverse_sigmoid`, which
> **clamps to `[0,1]`** first, so effectively `[0,1] ↔ pc_range` holds.

## 4. How a query is generated — `query = MLP(sine(anchor))`

`forward` (~line 415):
```python
reference_points = self.reference_points.weight                 # (900, 3)  ← the Parameter
query_embeds     = self.query_embedding(pos2posemb3d(reference_points))   # (900, 256)
```

| step | op | shape |
|---|---|---|
| `pos2posemb3d` | **sine** encode each axis (128 feats × 3) | `(900,3) → (900,384)` |
| `query_embedding` | **MLP** `384→256→256` | `(900,384) → (900,256)` |

`pos2posemb3d` (top of file, ~line 28) is DETR's sine PE generalised to a 3D
point. The resulting `(900, 256)` `query_embeds` are the actual object queries
(used as the decoder's `query_pos`). The query is **derived**, not stored.

## 5. The anchor is also the box *reference* (used twice)

`forward` (~line 424), per decoder layer:
```python
reference = inverse_sigmoid(reference_points.clone())
tmp[..., 0:2] += reference[..., 0:2]   # predicted offset + anchor xy
tmp[..., 4:5] += reference[..., 2:3]   # + anchor z
tmp = tmp.sigmoid()
```
So each box centre is predicted as an **offset from its anchor** (Deformable-DETR
style). One `(900,3)` parameter therefore serves **both** as the query seed
**and** the box reference.

## 6. Fixed after training, and input-independent

- `reference_points` is a normal `nn.Parameter`: updated by gradients **during
  training**, then **frozen** at inference.
- It is **input-independent** — the **same 900 anchors are used for every scene**.
  They don't depend on the cameras or image content.

Then why do predictions differ per scene? The **decoder** does the per-scene work:
```
fixed anchors (same every scene) ─► queries ─► cross-attn over THIS scene's features+3D PE ─► scene boxes
```
The anchors are fixed priors of "where objects tend to be"; the transformer
adapts them to each scene via cross-attention, and the box is an offset from the
anchor. (PETRv2 / StreamPETR later make queries partly *input-dependent* — a
deliberate improvement.)

## 7. The actual trained distribution (real numbers)

From the trained checkpoint's `pts_bbox_head.reference_points.weight`:

| axis | normalised min / mean / max | metres min / mean / max | std (uniform ≈ 0.289) |
|---|---|---|---|
| x | −0.14 / 0.48 / 1.06 | −65 / −2 / 58 | 0.268 |
| y | −0.19 / 0.47 / 1.17 | −70 / −3 / 69 | 0.272 |
| z | −0.15 / 0.44 / 1.08 | −6 / −1.5 / 4 | **0.211** |

Reading it:
- **x, y stay broadly spread** (std ≈ 0.27, close to uniform's 0.29) → they tile
  the BEV plane around the ego.
- **z is more concentrated** (std 0.21, mean ≈ −1.5 m) → objects sit at similar
  heights near the ground, so anchors cluster vertically there.
- A few points drifted slightly **outside `[0,1]`** — training nudged them past
  the nominal range.

So: random-uniform at init → **learned into a data-driven spread**, not a grid.

## 8. Contrast with DETR / DETR3D

| | object query | reference point |
|---|---|---|
| **DETR** | directly learned `Embedding(N, 256)` (no geometry) | none |
| **PETR** | **computed** `MLP(sine(3D point))` | **the learned `(900,3)` point** |

PETR's win: because the query is `MLP(sine(3D point))` and the key 3D PE is
`MLP(3D coords)`, **query and key live in the same 3D-encoded space**, so plain
dot-product attention can match a query to the image tokens whose rays pass near
its anchor.

## 9. Verify it yourself (debugger / REPL)

```python
h = model.pts_bbox_head
h.reference_points.weight.shape            # torch.Size([900, 3])
type(h.reference_points.weight)            # torch.nn.parameter.Parameter
h.reference_points.weight.requires_grad    # True (trainable)

from projects.mmdet3d_plugin.models.dense_heads.petr_head import pos2posemb3d
rp = h.reference_points.weight
pos2posemb3d(rp).shape                      # torch.Size([900, 384])
h.query_embedding(pos2posemb3d(rp)).shape   # torch.Size([900, 256])  ← the queries
```

Reproduce the §7 table:
`python3 -c "..."` (the one-liner used to measure the trained distribution — see
this session, or load `ckpts/petr_vovnet_p4_800x320.pth` and read
`state_dict['pts_bbox_head.reference_points.weight']`).

---

## Code map

| concept | location |
|---|---|
| `pos2posemb3d` (sine of a 3D point) | petr_head.py ~line 28 |
| `reference_points` + `query_embedding` defined | `_init_layers` ~line 265 |
| init `uniform_(…, 0, 1)` | `init_weights` ~line 276 |
| query generation `MLP(sine(anchor))` | `forward` ~line 415 |
| anchor used as box reference (offset) | `forward` ~line 424 |
| `[0,1] → metres` mapping | `forward` ~line 440 |

See also: [3D_PE_DEEPDIVE.md](3D_PE_DEEPDIVE.md) (the key side), `MODEL_STRUCTURE.md`.
