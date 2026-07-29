# How Temporal Information Is Used in PETRv2 (and Why It Works)

This note explains the temporal mechanism in PETRv2 in plain terms, then ties each part to the code in this repo.

## 1) The core idea in one sentence

PETRv2 gives the detector two moments in time (current cameras + previous cameras), expresses both moments in one common 3D coordinate frame, and lets the transformer attend over both so it can infer motion.

## 2) What temporal input actually is

For each sample, PETRv2 uses:
- 6 current camera images
- 6 previous camera images (a selected sweep)

So the head receives 12 camera views total.

In this repo, this is built in:
- petr_port/petrv2_infer.py
  - preprocess_sample_v2
  - build_sweep_lists
  - pick_sweep

Original pipeline behavior is in:
- projects/mmdet3d_plugin/datasets/pipelines/loading.py
  - LoadMultiViewImageFromMultiSweepsFiles

Important detail:
- At test time, one sweep is chosen at index
  int((3 + 27)/2) - 1 = 14
  (roughly about 1.2 seconds before the current frame).

## 3) The hard part: alignment across time

If you simply feed an older image, geometry would be inconsistent because ego vehicle pose changed.
PETRv2 fixes this in data processing:

- Previous-frame camera extrinsics are transformed so they are expressed in the current LiDAR frame.
- After this, both current and previous views can be back-projected into a shared 3D frame.

That is why temporal fusion is possible without explicit optical flow.

In this repo:
- petr_port/petrv2_infer.py
  - add_frame

This function mirrors the official sweep preprocessing math from:
- tools/generate_sweep_pkl.py

Conceptually, the transform chain is:
- previous camera -> previous ego -> global -> current ego -> current LiDAR

Then intrinsics are combined to form aligned lidar2img for each previous camera.

## 4) Where temporal information enters the network

Inside PETRv2Head forward:
- It receives features with shape [B, Ncams, C, H, W], where Ncams is 12 for temporal mode.
- It builds 3D position embedding from lidar2img for all 12 camera views.
- Transformer attention can now attend across space and time jointly.

Code location:
- projects/mmdet3d_plugin/models/dense_heads/petrv2_head.py
  - position_embeding
  - forward

## 5) Timestamp usage and velocity recovery

PETRv2 does not directly regress velocity in meters/second first.
Instead:
- It predicts displacement-like terms in the regression head.
- Then divides by measured delta time to convert to velocity.

In code:
- It reads img_meta[timestamp]
- Reshapes to [B, 2, 6]
  - row 0 = current 6 cameras
  - row 1 = previous 6 cameras
- Computes mean_time_stamp = mean(row1 - row0)
- Divides velocity channels by mean_time_stamp

Code location:
- projects/mmdet3d_plugin/models/dense_heads/petrv2_head.py
  - with_time block in forward

Why this matters:
- If the actual gap between frames varies slightly, normalization by measured delta time keeps velocity physically meaningful.

## 6) Why this is possible at all

Temporal fusion works because all three conditions are satisfied:

1. Shared 3D reference frame
- Previous-frame geometry is re-expressed in current-frame coordinates.

2. Consistent 3D positional encoding
- Both times are encoded into comparable 3D-aware tokens.

3. Attention over combined tokens
- The decoder can match object evidence across time and infer motion/state change.

So PETRv2 is not doing naive frame stacking. It is frame stacking with geometric alignment plus time-aware regression.

## 7) Role of FPE and multi-view fusion

PETRv2 adds:
- with_fpe: Feature-guided position enhancement (SE-style gating on 3D positional embedding)
- with_multi: multi-frame multi-view usage

Practical effect:
- Better position cues under ambiguity
- Better temporal consistency
- Better velocity estimates

Code location:
- projects/mmdet3d_plugin/models/dense_heads/petrv2_head.py
  - self.fpe initialization
  - application of fpe on coords_position_embeding

## 8) Minimal mental model

Think of PETRv2 as:
- Build one 3D world coordinate system anchored at current time
- Project both current and previous camera evidence into that world
- Let queries attend to both moments
- Convert temporal displacement to velocity using true delta time

That is the temporal mechanism.

## 9) In this workspace, where to inspect quickly

- End-to-end temporal input construction:
  - petr_port/petrv2_infer.py
    - preprocess_sample_v2
    - add_frame
    - build_sweep_lists
    - pick_sweep

- Original temporal loader behavior:
  - projects/mmdet3d_plugin/datasets/pipelines/loading.py
    - LoadMultiViewImageFromMultiSweepsFiles

- Temporal usage inside the head:
  - projects/mmdet3d_plugin/models/dense_heads/petrv2_head.py
    - forward
    - with_time branch
    - position_embeding

## 10) Quick self-checks you can run

1. Confirm Ncams is 12 in PETRv2 forward.
2. Print img_meta timestamp length and ordering (should be 12, current first then previous).
3. Print computed mean_time_stamp and verify about 1.2s on mini.
4. Compare velocity error versus PETR v1 to see temporal gain.

---

## 11) FAQ: where does ego pose come from if no IMU is input?

This repo does not feed raw IMU measurements into PETRv2.

What it does use is nuScenes pose metadata for each timestamp:
- ego_pose: vehicle pose in global frame (rotation + translation)
- calibrated_sensor: camera extrinsic to ego frame

In code, pose is fetched at:
- petr_port/petrv2_infer.py (add_frame)
  - pose = nusc.get("ego_pose", sd["ego_pose_token"])
  - calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

So think of it as having known SE(3) transforms from the dataset annotations.
No IMU packet stream is consumed by the model; only the resulting poses are used.

The alignment chain is:
- previous camera -> previous ego -> global -> current ego -> current lidar

That chain produces previous-frame sensor2lidar in the current lidar frame.
Then the repo inverts/combines with intrinsics to form aligned lidar2img.

Because both times are represented in one common current-lidar frame,
cross-time attention is geometrically meaningful.

If ego_pose were removed (or wrong), temporal fusion would degrade badly:
- previous frame points/tokens would be placed in the wrong 3D locations
- velocity and orientation estimates would become inconsistent

Bottom line:
- No raw IMU input to PETRv2
- Yes, pose alignment from nuScenes metadata
- Temporal gain comes from geometry-aligned multi-time camera evidence

---

If you want, I can also add a second note with a small figure (diagram) showing the exact coordinate transform chain step-by-step.
