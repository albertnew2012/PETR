# PETR model structure

This document describes the complete inference path for the checked-in
PETR VoVNet-99, p4, 800×320 configuration.

## Shape notation

- `B`: batch size
- `N=6`: number of cameras
- `C=256`: transformer feature dimension
- `H=20`, `W=50`: feature-map height and width
- `Q=900`: number of object queries
- `D=64`: number of candidate depths per feature cell

## Complete model in one diagram

```text
INPUT
Six camera images
[B, 6, 3, 320, 800]

        │ merge batch and camera dimensions
        ▼

[B×6, 3, 320, 800]

        │ GridMask (training augmentation; model module retained at inference)
        ▼

SHARED VoVNet-99-eSE BACKBONE
The same weights process every camera.

stage4: [B×6,  768, 20, 50]
stage5: [B×6, 1024, 10, 25]

        │ CPFPN: convert both levels to 256 channels
        ▼

level 0: [B×6, 256, 20, 50]  ← PETR head uses this level
level 1: [B×6, 256, 10, 25]

        │ restore camera dimension
        ▼

IMAGE FEATURES
[B, 6, 256, 20, 50]

        ├──────────────── IMAGE CONTENT PATH ────────────────┐
        │                                                    │
        │ input_proj: Conv2d(256 → 256)                      │
        │ flatten 6×20×50                                    │
        ▼                                                    │
image memory / key / value                                   │
[6000, B, 256]                                               │
                                                             │
        ┌──────────── IMAGE POSITION PATH ───────────────────┘
        │
        ├─ METRIC 3D POSITION
        │
        │  feature cells [6,20,50] × 64 candidate depths
        │       ↓ inverse(lidar2img)
        │  LiDAR-frame XYZ [B,6,50,20,64,3]
        │       ↓ concatenate 64×XYZ
        │  [B×6,192,20,50]
        │       ↓ position_encoder: 192 → 1024 → 256
        │  metric 3D PE [B,6,256,20,50]
        │
        └─ CAMERA/GRID POSITION

           mask [B,6,20,50]
                ↓ SinePositionalEncoding3D
           camera + row + column PE [B,6,384,20,50]
                ↓ adapt_pos3d: 384 → 1024 → 256
           camera/grid PE [B,6,256,20,50]

metric 3D PE + camera/grid PE
[B,6,256,20,50]

        │ flatten
        ▼

image key_pos
[6000,B,256]

===============================================================================

QUERY SIDE

trained normalized 3D reference points
reference_points.weight: [900,3]

        │ pos2posemb3d
        │ encode x, y, z with multiple sine/cosine frequencies
        ▼

[900,384] = 128_y + 128_x + 128_z

        │ query_embedding MLP: 384 → 256 → 256
        ▼

query_pos
[900,256]

        │ repeat across batch
        ▼

query_pos
[900,B,256]

initial query content = zeros_like(query_pos)
[900,B,256]

===============================================================================

TRANSFORMER DECODER × 6

QUERY SIDE                              IMAGE SIDE

query content [900,B,256]               memory [6000,B,256]
       +                                       +
query_pos [900,B,256]                   key_pos [6000,B,256]

        └──────────────────┬───────────────────┘
                           ▼

For each decoder layer:

1. query self-attention: 900 queries ↔ 900 queries
2. cross-attention: 900 queries → 6000 image tokens
3. FFN: 256 → 2048 → 256

                           ▼

decoder outputs
[6,B,900,256]

===============================================================================

PREDICTION HEADS

final query features [B,900,256]

        ├─ classification MLP: 256 → 256 → 256 → 10
        │  class logits [B,900,10]
        │
        └─ regression MLP: 256 → 256 → 256 → 10
           box code [B,900,10]

The regression center is predicted relative to the query's original
reference point.

===============================================================================

NMS-FREE DECODING

class logits [B,900,10]
        ↓ sigmoid
class probabilities [B,900,10]
        ↓ flatten query and class dimensions
900×10 = 9000 candidates
        ↓ top 300
        ↓ retrieve corresponding box predictions
        ↓ denormalize

FINAL DETECTIONS

- class
- confidence
- center: x, y, z
- dimensions: width, length, height
- yaw
- velocity: vx, vy
```

## 1. Input and shared image backbone

PETR receives the cameras in a fixed order:

```text
CAM_FRONT
CAM_FRONT_RIGHT
CAM_FRONT_LEFT
CAM_BACK
CAM_BACK_LEFT
CAM_BACK_RIGHT
```

The detector reshapes `[B,6,3,320,800]` into `[B×6,3,320,800]`, so all six
images pass through one shared VoVNet. There are not six backbone copies.
After VoVNet and CPFPN, the camera dimension is restored.

Source:

- [`Petr3D.extract_img_feat()`](../projects/mmdet3d_plugin/models/detectors/petr3d.py#L53-L84)
- [PETR model configuration](petr_infer.py#L74-L112)

## 2. Image content tokens

The PETR head uses the `20×50` FPN level. Every camera therefore contributes
`1000` image tokens:

```text
6 cameras × 20 rows × 50 columns = 6000 tokens
```

Each token contains a 256-dimensional visual feature. Flattening gives
`memory [6000,B,256]`, which supplies both transformer keys and values.

Source:

- [`PETRHead.forward()` image preparation](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L365-L388)
- [`PETRTransformer.forward()` flattening](../projects/mmdet3d_plugin/models/utils/petr_transformer.py#L84-L94)

## 3. Metric 3D image position

For each camera feature cell, PETR creates 64 candidate points along the
corresponding camera ray. A point begins in image/depth coordinates:

$$
p_{img}=(u d,\;v d,\;d,\;1)^T
$$

PETR applies the inverse camera projection:

$$
p_{lidar}=\operatorname{inverse}(\mathrm{lidar2img})\,p_{img}
$$

This expresses every camera's candidate points in the same LiDAR coordinate
system. The 64 XYZ samples become `64×3=192` channels, and a 1×1 convolutional
MLP converts them to one 256-dimensional 3D positional embedding per image
token.

Source:

- [`PETRHead.position_embeding()`](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L282-L327)

## 4. Camera/grid image position

`SinePositionalEncoding3D` encodes:

- camera index
- feature-map row
- feature-map column

Each component receives 128 sine/cosine values, producing 384 channels. The
`adapt_pos3d` MLP reduces this to 256 channels. PETR adds it to the metric 3D
position embedding:

$$
\mathrm{key\_pos}
=
\mathrm{metric\ 3D\ PE}
+
\mathrm{camera/grid\ PE}
$$

Source:

- [`SinePositionalEncoding3D`](../projects/mmdet3d_plugin/models/utils/positional_encoding.py#L15-L91)
- [Position addition in `PETRHead.forward()`](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L390-L402)

## 5. How the 900 queries are created

### 5.1 Reference points

PETR defines a trainable table:

```python
self.reference_points = nn.Embedding(900, 3)
```

The shape is `[900,3]`. Row `q` stores the normalized 3D anchor associated
with query `q`:

$$
r_q=(x_q,y_q,z_q),\qquad x_q,y_q,z_q\in[0,1]
$$

The intended training initialization samples every coordinate independently:

$$
r_{q,d}\sim U(0,1)
$$

This is random uniform initialization, not an evenly spaced grid. Training
updates the points through backpropagation. During inference, the checkpoint
contains fixed trained points; they are not resampled for each image.

Source:

- [Reference-point parameter](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L265)
- [Uniform initialization](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L276)
- [Checkpoint loading](petr_infer.py#L55-L59)

### 5.2 Sine/cosine encoding of one 3D point

The phrase "3D sine/cosine encoding" means that PETR encodes each scalar
coordinate independently at multiple frequencies; it does not apply a literal
3D wave.

For a coordinate $p$, the encoding is conceptually:

$$
E(p)=\left[
\sin\left(\frac{2\pi p}{\lambda_0}\right),
\cos\left(\frac{2\pi p}{\lambda_0}\right),
\ldots,
\sin\left(\frac{2\pi p}{\lambda_{63}}\right),
\cos\left(\frac{2\pi p}{\lambda_{63}}\right)
\right]
$$

There are 128 values per coordinate: 64 sine values and 64 cosine values.
PETR concatenates them in the implementation's `y,x,z` order:

```text
E(y) [128] + E(x) [128] + E(z) [128] = [384]
```

Multiple frequencies give the transformer a richer and less ambiguous
representation than raw `[x,y,z]` values.

#### Why not pass `[x,y,z]` directly?

PETR could theoretically use an MLP such as `Linear(3→256)` directly. The
sine/cosine step is therefore not required for tensor shapes, and expanding
three values to 384 values does not create new spatial information. Instead,
it changes the coordinates into a representation in which spatial patterns
are easier for the learned MLP and dot-product attention to use.

Raw coordinates are a very low-dimensional, nearly linear description. If
only `[x,y,z]` is supplied, the learned layers must discover useful nonlinear
functions of position from data. Neural networks also tend to learn smooth,
low-frequency functions before fine spatial variations. This is often called
**spectral bias**.

`pos2posemb3d()` first expands every coordinate with 64 sine/cosine frequency
pairs. This provides both coarse and fine descriptions of position:

- low-frequency components vary slowly and represent broad spatial regions;
- high-frequency components vary more quickly and distinguish nearby points;
- nearby coordinates produce related encodings;
- the fixed frequency basis means the learned MLP does not have to discover
  all of these spatial scales from scratch;
- the 384-dimensional representation can be projected naturally into the
  transformer's 256-dimensional query space.

Sine and cosine are used together because their dot product exposes relative
displacement. For one frequency:

$$
\sin(a)\sin(b)+\cos(a)\cos(b)=\cos(a-b)
$$

Consequently, comparing the encodings of two positions can reveal how far
apart they are, not merely their absolute coordinate values. Using many
frequencies makes this comparison available at multiple spatial scales.

The ordering is therefore:

```text
raw normalized location [x,y,z]
        ↓ fixed multi-frequency sine/cosine basis
explicit coarse-to-fine spatial features [384]
        ↓ learned query_embedding MLP
task-specific transformer query position [256]
```

The fixed encoding answers **how to describe location at many scales**. The
learned embedding then answers **which combinations of those scales are
useful for 3D object detection**. Putting the encoding first separates these
two jobs and makes learning easier than asking one MLP to invent both the
spatial basis and the task-specific query representation.

The sinusoidal conversion does not learn parameters or change the reference
point. It only represents the same `(x,y,z)` location in a richer form. The
following `query_embedding` MLP is learned and decides how to use that form.

Source:

- [`pos2posemb3d()`](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L29-L41)

### 5.3 Is `pos2posemb3d()` similar to `self.positional_encoding`?

Yes. Both use fixed, multi-frequency sine/cosine functions to turn a small set
of coordinates into a richer positional representation. Neither operation
has learned parameters. Their inputs and roles are different:

| Property | `pos2posemb3d()` | `self.positional_encoding` |
|---|---|---|
| Describes | Object-query reference points | Image-feature locations |
| Input | Learned normalized `(x,y,z)` points | Padding mask `[B,N,H,W]` |
| Coordinates encoded | `y`, `x`, `z` | camera index, row, column |
| Input shape here | `[900,3]` | `[B,6,20,50]` |
| Direct output | `[900,384]` | `[B,6,384,20,50]` |
| Learned adapter | Linear MLP `384→256→256` | 1×1-convolution MLP `384→1024→256` |
| Transformer role | Query position (`query_pos`) | Part of image-token position (`key_pos`) |

Thus, they use the same general encoding idea on opposite sides of
cross-attention:

```text
query XYZ ── pos2posemb3d ──► query_pos

camera/row/column ── SinePositionalEncoding3D ──► key_pos
```

This helps attention compare **where a query is searching in 3D** with
**where each image token came from**.

### 5.4 Query position and query content

The 384-dimensional coordinate encoding passes through a learned MLP:

```text
reference_points [900,3]
        ↓ pos2posemb3d
[900,384]
        ↓ Linear(384→256), ReLU, Linear(256→256)
query_pos [900,256]
```

Exact query-position creation:

```python
reference_points = self.reference_points.weight
query_embeds = self.query_embedding(pos2posemb3d(reference_points))
```

The transformer repeats `query_embeds` across the batch and independently
creates zero-valued initial query content:

```python
query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
target = torch.zeros_like(query_embed)
```

Therefore, before decoder layer 1:

```text
query content:  [900,B,256] = zeros
query position: [900,B,256] = encoded trained 3D anchors
```

The query position says where a query begins searching. Cross-attention fills
its initially empty content with image evidence.

Source:

- [Query-position creation](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L415-L417)
- [Zero query-content creation](../projects/mmdet3d_plugin/models/utils/petr_transformer.py#L89-L93)
- [Query and key positions passed to the decoder](../projects/mmdet3d_plugin/models/utils/petr_transformer.py#L95-L104)

## 6. Transformer decoder

PETR has no transformer encoder in this configuration. Its six decoder layers
operate directly on flattened image memory.

Each layer performs:

```text
query self-attention
    ↓
LayerNorm
    ↓
image cross-attention
    ↓
LayerNorm
    ↓
FFN 256→2048→256
    ↓
LayerNorm
```

Self-attention allows queries to coordinate and avoid duplicating objects.
Cross-attention lets each query collect evidence from all 6000 image tokens.

## 7. Classification and regression

Each decoder output enters two branches:

```text
query feature [256]
        ├─ classification branch → 10 class logits
        └─ regression branch     → 10-value box code
```

The regression code represents center, dimensions, orientation, and planar
velocity. PETR adds the predicted center offsets to the query reference point,
then rescales the normalized center into the configured LiDAR range.

### 7.1 Box parameterization design

The ten regression channels deliberately use different mathematical spaces:

| Index | Predicted quantity | Representation |
|---:|---|---|
| `0` | center $x$ | normalized coordinate refined in logit space |
| `1` | center $y$ | normalized coordinate refined in logit space |
| `2` | width $w$ | $\log(w)$ |
| `3` | length $l$ | $\log(l)$ |
| `4` | center $z$ | normalized coordinate refined in logit space |
| `5` | height $h$ | $\log(h)$ |
| `6` | yaw | $\sin(\mathrm{yaw})$ |
| `7` | yaw | $\cos(\mathrm{yaw})$ |
| `8` | velocity $v_x$ | direct value |
| `9` | velocity $v_y$ | direct value |

These choices encode three physical constraints: centers remain inside the
normalized spatial range, dimensions remain positive, and orientation remains
continuous across the angle wraparound.

#### Center coordinates: inverse-sigmoid reference refinement

Reference points are normalized coordinates in $[0,1]$, while a linear
regression head produces unrestricted real-valued corrections. PETR therefore
converts each reference coordinate into unrestricted **logit space**:

$$
\operatorname{logit}(r)
=
\operatorname{inverse\_sigmoid}(r)
=
\log\left(\frac{r}{1-r}\right)
$$

It adds the decoder's predicted correction $\Delta$ there and returns to
normalized coordinate space with sigmoid:

$$
\hat r
=
\sigma\left(\operatorname{logit}(r)+\Delta\right)
$$

This has the useful identity:

$$
\Delta=0
\quad\Longrightarrow\quad
\hat r=\sigma(\operatorname{logit}(r))=r
$$

Thus, zero correction means "keep the query's learned reference location."
Positive and negative corrections move it while the final sigmoid guarantees
that the result remains in $[0,1]$.

The implementation maps reference `(x,y,z)` into box-code channels `(0,1,4)`:

```python
reference = inverse_sigmoid(reference_points.clone())
tmp[..., 0:2] += reference[..., 0:2]  # x and y
tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
tmp[..., 4:5] += reference[..., 2:3]  # z
tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
```

`clone()` gives the transformation separate tensor storage so the original
reference-point tensor cannot be accidentally overwritten. It does not
detach the computation graph, so gradients still reach the learned reference
points. With the current non-in-place inverse-sigmoid operation, the clone is
primarily defensive.

After refinement, PETR linearly maps normalized $x$, $y$, and $z$ into the
configured metric LiDAR range.

#### Dimensions: why predict `log(w)`, `log(l)`, and `log(h)`?

Width, length, and height must be strictly positive, but an ordinary linear
layer can output any real number. PETR predicts their logarithms and restores
the dimensions with exponentiation:

$$
s_w=\log(w),\qquad w=e^{s_w}>0
$$

The same applies to $l$ and $h$. This prevents physically invalid negative
dimensions without imposing a fixed upper bound.

Log space also changes additive prediction errors into multiplicative size
changes:

$$
\log(w)+\Delta
\quad\Longrightarrow\quad
e^{\log(w)+\Delta}=w e^\Delta
$$

This makes the regression more sensitive to relative scale. A fixed error is
more important for a small object than for a very large object, and log-space
regression reflects that distinction better than raw-meter regression.

#### Orientation: why predict both `sin(yaw)` and `cos(yaw)`?

Yaw is periodic. For example, $179^\circ$ and $-179^\circ$ are only
$2^\circ$ apart physically, but direct scalar subtraction reports
$358^\circ$. This discontinuity makes raw-angle regression difficult.

PETR instead represents orientation as a point on a circle:

$$
	heta\longmapsto(\sin\theta,\cos\theta)
$$

Nearby physical orientations then have nearby representations even across
the $-\pi/\pi$ boundary. Both values are required: sine alone cannot
distinguish $\theta$ from $\pi-\theta$, and cosine alone cannot distinguish
$\theta$ from $-\theta$. The pair identifies orientation uniquely modulo
$2\pi$.

During decoding, PETR reconstructs yaw using:

$$
	heta=\operatorname{atan2}(\sin\theta,\cos\theta)
$$

The predicted pair is not explicitly forced to have unit length. `atan2()`
mainly uses its direction, so a common positive scaling of the two values does
not change the reconstructed angle.

The complete parameterization flow is:

```text
center:    reference [0,1] → logit + correction → sigmoid → metric range
dimensions: unrestricted log values → exp → positive w, l, h
orientation: unrestricted sin/cos pair → atan2 → periodic yaw
velocity:   unrestricted direct values → vx, vy
```

Source:

- [Prediction branches](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L222-L270)
- [Reference-point regression offsets](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L424-L443)
- [Box target parameterization](../projects/mmdet3d_plugin/core/bbox/util.py#L46-L65)
- [Box decoding with `exp()` and `atan2()`](../projects/mmdet3d_plugin/core/bbox/util.py#L67-L95)

## 8. NMS-free output

The final decoder layer produces `900×10=9000` query/class probabilities.
PETR applies sigmoid, selects the top 300 candidates, retrieves the associated
query boxes, and removes centers outside the configured range. It does not use
conventional NMS.

Source:

- [`NMSFreeCoder.decode_single()`](../projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py#L47-L86)

## Core mental model

```text
IMAGE TOKENS
"What visual evidence exists, and where could it lie in shared 3D space?"

OBJECT QUERIES
"Search for an object near this learned 3D reference point."

CROSS-ATTENTION
"Collect supporting image evidence from all six cameras."

OUTPUT HEADS
"Classify the object and adjust the reference point into its final 3D box."
```
