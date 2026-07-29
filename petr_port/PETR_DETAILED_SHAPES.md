# PETR detailed data-flow diagram (with tensor shapes)

This diagram follows the checked-in `petr_vovnet_gridmask_p4_800x320` model.
It uses `B` for batch size. The example shapes use `B=1`, six nuScenes cameras,
and an image size of `320 x 800`.

## End-to-end inference

```mermaid
flowchart TB
    IN["Six camera images\nimg: [B, 6, 3, 320, 800]\nexample: [1, 6, 3, 320, 800]"]

    subgraph VIS["Image feature extraction — independently per camera"]
        FLAT["merge B and cameras\n[B*6, 3, 320, 800]\n[6, 3, 320, 800]"]
        BACKBONE["VoVNet-99 backbone\nstage4: [B*6, 768, 20, 50]\nstage5: [B*6, 1024, 10, 25]"]
        FPN["CPFPN\nlevel 0: [B*6, 256, 20, 50]\nlevel 1: [B*6, 256, 10, 25] — unused by this head"]
        CONTENT["input_proj, 1x1 conv\nimage content x: [B, 6, 256, 20, 50]\n1,000 tokens/camera; 6,000 total"]
    end

    subgraph KPE["Key positional encoding — one 256-D position per image token"]
        GRID["Feature-grid locations\n(u, v): [W=50, H=20]\ninput-pixel coordinates"]
        DEPTH["64 LID depth samples\nd: [D=64]"]
        RAY["For every (u,v,d):\n(u*d, v*d, d, 1)\nraw frustum: [W, H, D, 4]\n[50, 20, 64, 4]"]
        BACKPROJECT["Per-camera img2lidar = inverse(lidar2img)\nback-project into the shared LiDAR/ego frame\nxyz: [B, 6, W, H, D, 3]\n[1, 6, 50, 20, 64, 3]"]
        NORM["Normalize xyz by position_range\nmask invalid rays\nreorder/flatten 64 xyz samples\n[B*6, 3*64, H, W]\n[6, 192, 20, 50]"]
        PE3D["position_encoder: 1x1 Conv MLP\n192 → 1024 → 256\n3D PE: [B, 6, 256, 20, 50]"]
        SINE["SinePositionalEncoding3D\nencodes camera + row + column\n[B, 6, 384, 20, 50]"]
        ADAPT["adapt_pos3d: 1x1 Conv MLP\n384 → 1024 → 256\n[B, 6, 256, 20, 50]"]
        KEYPOS["key_pos = 3D PE + adapted sine PE\n[B, 6, 256, 20, 50]"]
    end

    subgraph QPE["Object queries — 900 learned 3D anchors"]
        REF["reference_points embedding\n[900, 3] in normalized xyz"]
        QPOS["pos2posemb3d + query_embedding MLP\n[900, 384] → [900, 256]\nrepeat batch: [B, 900, 256]"]
    end

    subgraph DEC["PETR Transformer decoder — 6 layers"]
        TOKENS["Flatten image grid\ncontent/value: [B, 6, 256, 20, 50]\nkeys: 6 × 20 × 50 = 6,000"]
        LAYER["Each layer:\n1. query self-attention: 900 queries\n2. cross-attention: 900 queries → 6,000 image keys\n3. FFN: 256 → 2048 → 256\nlayer output: [B, 900, 256]"]
        DECOUT["all decoder outputs\n[6 layers, B, 900, 256]\n[6, 1, 900, 256]"]
    end

    subgraph PRED["Detection prediction"]
        CLS["classification branch per decoder layer\n[B, 900, 10] class logits"]
        REG["box regression branch per decoder layer\n[B, 900, 10] normalized box code"]
        DECODE["NMSFreeCoder\nuse final layer; select top 300\nboxes: [300, 9] = x,y,z,w,l,h,yaw,vx,vy\nscores: [300], labels: [300]"]
    end

    IN --> FLAT --> BACKBONE --> FPN --> CONTENT
    FPN --> GRID
    GRID --> RAY
    DEPTH --> RAY --> BACKPROJECT --> NORM --> PE3D --> KEYPOS
    SINE --> ADAPT --> KEYPOS
    REF --> QPOS
    CONTENT --> TOKENS --> LAYER
    KEYPOS -->|"position supplied with image keys"| LAYER
    QPOS -->|"position supplied with queries"| LAYER
    LAYER -->|"repeat ×6"| DECOUT --> CLS --> DECODE
    DECOUT --> REG --> DECODE

    classDef input fill:#e3f2fd,stroke:#1565c0,color:#000;
    classDef geometry fill:#e8f5e9,stroke:#2e7d32,color:#000;
    classDef query fill:#fff3e0,stroke:#ef6c00,color:#000;
    classDef output fill:#fce4ec,stroke:#ad1457,color:#000;
    class IN,FLAT,CONTENT,TOKENS input;
    class GRID,DEPTH,RAY,BACKPROJECT,NORM,PE3D,SINE,ADAPT,KEYPOS geometry;
    class REF,QPOS query;
    class CLS,REG,DECODE output;
```

## Zoom-in: exactly what happens to one image token

```mermaid
flowchart LR
    PIXEL["One level-0 feature cell\n(u,v) on [H=20, W=50]\nIts visual feature: 256 numbers"]
    DS["D=64 candidate depths\nd₁ ... d₆₄"]
    CAM["Build camera-ray points\n(u*dᵢ, v*dᵢ, dᵢ, 1)\n64 × 4"]
    LIDAR["Apply this camera's inverse projection\nimg2lidar @ point\n64 × xyz in shared LiDAR frame\n64 × 3"]
    VECTOR["Normalize and concatenate\n[x₁,y₁,z₁, ..., x₆₄,y₆₄,z₆₄]\n192-vector"]
    MLP["position_encoder\n1×1 MLP: 192 → 1024 → 256"]
    ONEPE["One 3D PE: 256-vector\nThis describes the complete possible 3D ray,\nnot one chosen depth."]
    ATTN["Cross-attention\nAn object query learns to favor this token\nwhen its 3D anchor is compatible with the ray\nand the visual feature supports an object."]

    PIXEL --> DS --> CAM --> LIDAR --> VECTOR --> MLP --> ONEPE --> ATTN
```

## Shape ledger

| Quantity | Shape | Meaning |
|---|---:|---|
| Input images | `[B, 6, 3, 320, 800]` | batch, cameras, BGR/RGB channels, height, width |
| Head feature level | `[B, 6, 256, 20, 50]` | stride-16 image features |
| Geometry before flattening | `[B, 6, 50, 20, 64, 3]` | one LiDAR-frame 3D point per camera, cell, and sampled depth |
| Geometry fed to `position_encoder` | `[B*6, 192, 20, 50]` | $192=64\times3$ coordinate channels for each cell |
| 3D position embedding | `[B, 6, 256, 20, 50]` | one learned 3D position vector per image token |
| Sine/view position embedding | `[B, 6, 384, 20, 50]` | fixed camera/row/column encoding before adaptation |
| Final key position | `[B, 6, 256, 20, 50]` | $\mathrm{3D\ PE}+\mathrm{adapted\ sine\ PE}$ |
| Image keys/values | `[B, 6000, 256]` conceptually | $6\times20\times50=6000$ image tokens |
| Query positions | `[B, 900, 256]` | 900 learned normalized 3D reference points encoded into transformer space |
| Decoder output | `[6, B, 900, 256]` | six decoder layers, retained for deep supervision |
| Final predictions | `[B, 900, 10]` | 10 class logits and 10 regression values before decoding |

## The key interpretation

PETR does **not** turn a pixel into 64 separate transformer tokens, and it does
not select one of the 64 depths before attention. It uses the 64 back-projected
points to construct a 192-number geometry descriptor, maps that descriptor to
one 256-D positional embedding, and attaches it to the corresponding **single**
image token. Thus the transformer has 6,000 image tokens, not
$6\times20\times50\times64=384,000$ tokens.

The relevant implementation is `PETRHead.position_embeding()` in
[`petr_head.py`](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L282-L327),
and the addition of the two key-position components is in
[`petr_head.py`](../projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L390-L405).