# PETR classification, box regression, and confidence

After every PETR decoder layer, each of the 900 object queries has one learned
feature vector:

$$
q \in \mathbb{R}^{256}
$$

PETR sends the **same query feature** through two separate prediction heads:

```text
decoder query feature [B, 900, 256]
              ├── cls_branch → class logits [B, 900, 10]
              └── reg_branch → box code    [B, 900, 10]
```

The head definitions are in
[`petr_head.py`](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L222-L243),
and they are applied after each decoder layer in
[`petr_head.py`](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L419-L439).

## 1. Classification branch: “what is this?”

For the default configuration, `num_reg_fcs = 2`, `embed_dims = 256`, and
there are 10 nuScenes object categories. Thus `cls_branch` is:

```text
Linear(256 → 256)
LayerNorm(256)
ReLU
Linear(256 → 256)
LayerNorm(256)
ReLU
Linear(256 → 10)
```

Its 10 outputs are **raw logits**, not confidences yet. For query $q$ and class
$c$:

$$
\ell_{q,c} = \texttt{cls\_branch}(q)_c
$$

The ten categories are:

```text
car, truck, construction_vehicle, bus, trailer,
barrier, motorcycle, bicycle, pedestrian, traffic_cone
```

### Why `Linear → LayerNorm → ReLU`?

`LayerNorm` normalizes the intermediate 256-D feature of **each query** before
the nonlinearity. It gives the classification MLP a more consistent feature
scale, which is helpful when learning class scores among 900 mostly-empty
queries. This is an empirical architecture choice, not a required property of
classification.

The classification head is supervised by sigmoid Focal Loss. The configuration
is in
[`petr_vovnet_gridmask_p4_800x320.py`](../projects/configs/petr/petr_vovnet_gridmask_p4_800x320.py#L74-L78).

## 2. Regression branch: “where is the 3D box?”

`reg_branch` has the same input and hidden width, but no `LayerNorm`:

```text
Linear(256 → 256)
ReLU
Linear(256 → 256)
ReLU
Linear(256 → 10)
```

It outputs a continuous 10-value 3D box code:

$$
(x, y, w, l, z, h, \sin\theta, \cos\theta, v_x, v_y)
$$

where $(x,y,z)$ is the box center in ego/LiDAR space after decoding;
$(w,l,h)$ is the box size; $(\sin\theta,\cos\theta)$ represents yaw without
angle wrap-around; and $(v_x,v_y)$ is planar velocity.

The first two center coordinates and the height coordinate are predicted as
offsets relative to the query's learned 3D reference point:

```python
tmp[..., 0:2] += reference[..., 0:2]  # x, y
tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
tmp[..., 4:5] += reference[..., 2:3]  # z
tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
```

Then PETR rescales the normalized centers into the configured physical point
cloud range. This happens in
[`petr_head.py`](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L425-L443).

### Why only `Linear → ReLU` here?

The regression task predicts precise continuous offsets and dimensions. PETR
uses a simpler MLP without `LayerNorm`, following the original architecture
choice. This does not mean normalization would be mathematically invalid; it
would be an architectural change that must be retrained and evaluated.

During training, PETR applies a weighted L1 regression loss only to queries
matched to ground-truth boxes. The loss code is in
[`petr_head.py`](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L573-L615).

## 3. Where confidence comes from

There is **no separate objectness/confidence head** in this PETR configuration.
The confidence of a candidate detection is its class probability:

$$
p_{q,c} = \sigma(\ell_{q,c}) = \frac{1}{1 + e^{-\ell_{q,c}}}
$$

At inference, the NMS-free decoder applies `sigmoid()` to all $900\times10$
class logits:

```python
cls_scores = cls_scores.sigmoid()
```

This occurs in
[`nms_free_coder.py`](../projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py#L53-L60).

For example, if the `car` logit for query 17 is $3.2$:

$$
\sigma(3.2) \approx 0.961
$$

Then `0.961` is the confidence for the candidate “query 17 predicts a car.”

## 4. Selecting final detections

PETR flattens the $900\times10=9000$ class probabilities and takes the top 300:

```python
scores, indexs = cls_scores.view(-1).topk(max_num)
labels = indexs % self.num_classes
bbox_index = indexs // self.num_classes
bbox_preds = bbox_preds[bbox_index]
```

So each final output joins:

```text
class label   = which class channel had the high score
confidence    = sigmoid(class logit)
3D box        = reg_branch output from the corresponding query
```

The decoder performs no NMS; it is an NMS-free DETR-style detector. It also
filters predictions whose centers fall outside `post_center_range`.

## One-line summary

```text
cls_branch says “what class, and how confident?”;
reg_branch says “what 3D box belongs to that query?”
```