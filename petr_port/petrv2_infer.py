"""Run PETRv2 (temporal, VoVNet p4 800x320) inference + nuScenes eval on mini_val.

PETRv2 = PETR + temporal modelling.  It feeds TWO frames (12 images: the current
6 cameras + 6 cameras from a previous "sweep" ~1.2 s earlier) into the same
`Petr3D` detector, but swaps the head for `PETRv2Head` which adds:

  * FPE  (feature-guided position encoder, an SE layer on the 3D PE),
  * with_time (the regression head predicts a *displacement* over the two
    frames, then divides by the measured Δt to recover velocity),
  * with_multi (multi-frame 3D position embedding over all 12 images).

The crucial trick lives in the DATA: the previous frame's `lidar2img` matrices
are pre-aligned into the *current* frame's LiDAR coordinate system via the ego
pose chain (exactly what `tools/generate_sweep_pkl.py` does with `add_frame`).
We replicate that alignment on-the-fly with the nuScenes devkit, then mirror the
`LoadMultiViewImageFromMultiSweepsFiles` test-mode sweep selection (index 14 of
the 30-deep sweep list, sweep_range=[3,27]) and the `with_time` timestamp layout.

Everything else (ResizeCropFlip / Normalize / Pad, box decoding, the official
nuScenes detection eval) is shared with `petr_infer.py`.
"""
import argparse
import os

import compat  # noqa: F401  (installs shims first)

import numpy as np
import torch
import mmcv
import mmengine
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.eval.detection.config import config_factory

from mmdet.registry import MODELS

# reuse the v1 plumbing verbatim
from petr_infer import (
    CLASS_NAMES, CAMERAS, IMG_NORM, MODALITY,
    _sample_augmentation, _img_transform, build_cam_matrices,
    output_to_nusc_box, lidar_nusc_box_to_global, make_anno,
)

POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
VOXEL_SIZE = [0.2, 0.2, 8]
SWEEP_RANGE = (3, 27)            # config: sweep_range=[3,27]
SWEEPS_NUM = 1                   # config: sweeps_num=1
NUM_PREV = 5                     # generate_sweep_pkl: num previous key frames
NUM_SWEEP = 5                    # generate_sweep_pkl: sweeps between key frames
MEAN_TIME = (SWEEP_RANGE[0] + SWEEP_RANGE[1]) / 2.0 * 0.083  # pad Δt ~= 1.245 s


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
def register_petrv2_modules():
    import projects.mmdet3d_plugin.models.backbones.vovnetcp  # noqa: F401
    import projects.mmdet3d_plugin.models.necks.cp_fpn  # noqa: F401
    import projects.mmdet3d_plugin.models.utils.positional_encoding  # noqa
    import projects.mmdet3d_plugin.models.utils.petr_transformer  # noqa
    import projects.mmdet3d_plugin.core.bbox.coders.nms_free_coder  # noqa
    import projects.mmdet3d_plugin.models.dense_heads.petrv2_head as _ph  # noqa
    import projects.mmdet3d_plugin.models.detectors.petr3d  # noqa
    _ph.PETRv2Head.__abstractmethods__ = frozenset()


def build_v2_model_cfg():
    return dict(
        type="Petr3D", use_grid_mask=True,
        img_backbone=dict(type="VoVNetCP", spec_name="V-99-eSE",
                          norm_eval=True, frozen_stages=-1, input_ch=3,
                          out_features=("stage4", "stage5")),
        img_neck=dict(type="CPFPN", in_channels=[768, 1024], out_channels=256,
                      num_outs=2),
        pts_bbox_head=dict(
            type="PETRv2Head", num_classes=10, in_channels=256, num_query=900,
            LID=True, with_position=True, with_multiview=True,
            with_fpe=True, with_time=True, with_multi=True,
            position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            code_weights=[1.0] * 10,
            transformer=dict(type="PETRTransformer", decoder=dict(
                type="PETRTransformerDecoder", return_intermediate=True,
                num_layers=6, transformerlayers=dict(
                    type="PETRTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(type="MultiheadAttention", embed_dims=256,
                             num_heads=8, dropout=0.1),
                        dict(type="PETRMultiheadAttention", embed_dims=256,
                             num_heads=8, dropout=0.1)],
                    feedforward_channels=2048, ffn_dropout=0.1,
                    operation_order=("self_attn", "norm", "cross_attn", "norm",
                                     "ffn", "norm")))),
            bbox_coder=dict(type="NMSFreeCoder",
                            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2,
                                               10.0],
                            pc_range=POINT_CLOUD_RANGE, max_num=300,
                            voxel_size=VOXEL_SIZE, num_classes=10),
            positional_encoding=dict(type="SinePositionalEncoding3D",
                                     num_feats=128, normalize=True),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0,
                          alpha=0.25, loss_weight=2.0),
            loss_bbox=dict(type="L1Loss", loss_weight=0.25),
            loss_iou=dict(type="GIoULoss", loss_weight=0.0)),
        train_cfg=None, test_cfg=None, pretrained=None)


def import_and_build_v2(ckpt):
    register_petrv2_modules()
    model = MODELS.build(build_v2_model_cfg())
    from mmengine.runner import load_checkpoint
    load_checkpoint(model, ckpt, map_location="cpu", strict=True)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
#  Temporal sweep alignment (mirror tools/generate_sweep_pkl.py :: add_frame)
# --------------------------------------------------------------------------- #
def add_frame(nusc, sd, e2g_t, l2e_t, l2e_r_mat, e2g_r_mat, data_root):
    """Align one previous camera sample_data into the *current* LiDAR frame."""
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    l2e_r_s_mat = Quaternion(calib["rotation"]).rotation_matrix
    e2g_r_s_mat = Quaternion(pose["rotation"]).rotation_matrix
    l2e_t_s = np.array(calib["translation"])
    e2g_t_s = np.array(pose["translation"])

    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sensor2lidar_rotation = R.T
    sensor2lidar_translation = T

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    viewpad = np.eye(4)
    viewpad[:3, :3] = np.array(calib["camera_intrinsic"])
    lidar2img = viewpad @ lidar2cam_rt.T
    return dict(
        data_path=os.path.join(data_root, sd["filename"]),
        timestamp=sd["timestamp"],
        intrinsics=viewpad.astype(np.float32),
        extrinsics=lidar2cam_rt.astype(np.float32),
        lidar2img=lidar2img.astype(np.float32),
    )


def build_sweep_lists(nusc, token, info, data_root):
    """Replicate generate_sweep_pkl.py: list of previous (aligned) cam frames."""
    e2g_t = np.array(info["ego2global_translation"])
    l2e_t = np.array(info["lidar2ego_translation"])
    l2e_r_mat = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    e2g_r_mat = Quaternion(info["ego2global_rotation"]).rotation_matrix

    sample = nusc.get("sample", token)
    current_cams = {cam: nusc.get("sample_data", sample["data"][cam])
                    for cam in CAMERAS}
    sweep_lists = []
    for _ in range(NUM_PREV):
        if sample["prev"] == "":
            break
        for _ in range(NUM_SWEEP):                       # sweeps between keys
            sweep_cams = dict()
            for cam in CAMERAS:
                if current_cams[cam]["prev"] == "":
                    sweep_cams = sweep_lists[-1]
                    break
                sd = nusc.get("sample_data", current_cams[cam]["prev"])
                sweep_cams[cam] = add_frame(nusc, sd, e2g_t, l2e_t,
                                            l2e_r_mat, e2g_r_mat, data_root)
                current_cams[cam] = sd
            sweep_lists.append(sweep_cams)
        sample = nusc.get("sample", sample["prev"])      # previous key frame
        sweep_cams = dict()
        for cam in CAMERAS:
            sd = nusc.get("sample_data", sample["data"][cam])
            sweep_cams[cam] = add_frame(nusc, sd, e2g_t, l2e_t,
                                        l2e_r_mat, e2g_r_mat, data_root)
            current_cams[cam] = sd
        sweep_lists.append(sweep_cams)
    return sweep_lists


def pick_sweep(sweep_lists):
    """Mirror LoadMultiViewImageFromMultiSweepsFiles test-mode selection."""
    n = len(sweep_lists)
    if n == 0:
        return None
    idx = 0 if n <= SWEEPS_NUM else int(sum(SWEEP_RANGE) / 2) - 1   # 14
    sweep_idx = min(idx, n - 1)
    sweep = sweep_lists[sweep_idx]
    if len(sweep.keys()) < len(CAMERAS):
        sweep = sweep_lists[sweep_idx - 1]
    return sweep


# --------------------------------------------------------------------------- #
#  Data: build the 12-image temporal input
# --------------------------------------------------------------------------- #
def _imread(path, data_root):
    if not os.path.isabs(path):
        path = os.path.join(data_root, path)
    return mmcv.imread(path, "unchanged").astype(np.float32)


def preprocess_sample_v2(info, data_root, nusc):
    # ---- current frame (6 cams) ----
    paths, _, intr_cur, extr_cur = build_cam_matrices(info)
    lidar_ts = info["timestamp"] / 1e6
    cur_ts = [lidar_ts - info["cams"][cam]["timestamp"] / 1e6 for cam in CAMERAS]
    imgs = [_imread(p, data_root) for p in paths]
    intr = [m.copy() for m in intr_cur]
    extr = [m.copy() for m in extr_cur]
    ts = list(cur_ts)

    # ---- previous frame (6 cams, aligned to current LiDAR) ----
    sweep = pick_sweep(build_sweep_lists(nusc, info["token"], info, data_root))
    if sweep is None:                          # first frame of a scene -> pad
        for i, cam in enumerate(CAMERAS):
            imgs.append(imgs[i].copy())
            intr.append(intr_cur[i].copy())
            extr.append(extr_cur[i].copy())
            ts.append(cur_ts[i] + MEAN_TIME)
    else:
        for cam in CAMERAS:
            sc = sweep[cam]
            imgs.append(_imread(sc["data_path"], data_root))
            intr.append(sc["intrinsics"].astype(np.float64).copy())
            extr.append(sc["extrinsics"].astype(np.float64).copy())
            ts.append(lidar_ts - sc["timestamp"] / 1e6)

    # ---- ResizeCropFlip (test) on all 12, update intrinsics + lidar2img ----
    resize, resize_dims, crop, flip, rotate = _sample_augmentation()
    new_imgs = []
    for i in range(len(imgs)):
        pil = Image.fromarray(np.uint8(imgs[i]))
        pil, ida_mat = _img_transform(pil, resize, resize_dims, crop, flip,
                                      rotate)
        new_imgs.append(np.array(pil).astype(np.float32))
        intr[i][:3, :3] = ida_mat @ intr[i][:3, :3]
    lidar2img = [intr[i] @ extr[i].T for i in range(len(extr))]

    # ---- Normalize + Pad/32 + stack -> [1, 12, 3, H, W] ----
    new_imgs = [mmcv.imnormalize(im, IMG_NORM["mean"], IMG_NORM["std"],
                                 IMG_NORM["to_rgb"]) for im in new_imgs]
    new_imgs = [mmcv.impad_to_multiple(im, 32, pad_val=0) for im in new_imgs]
    img_shape = [im.shape for im in new_imgs]
    img_tensor = np.ascontiguousarray(np.stack(new_imgs).transpose(0, 3, 1, 2))
    img_tensor = torch.from_numpy(img_tensor).float().unsqueeze(0)
    img_meta = dict(
        img_shape=img_shape, pad_shape=img_shape, lidar2img=lidar2img,
        timestamp=ts, box_type_3d=compat.LiDARInstance3DBoxes,
        sample_idx=info["token"], scale_factor=1.0, flip=False,
    )
    return img_tensor, [img_meta]


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        compat.REPO_ROOT, "ckpts/petrv2_vovnet_p4_800x320.pth"))
    ap.add_argument("--data-root", default=os.path.join(
        compat.REPO_ROOT, "data/nuscenes"))
    ap.add_argument("--out-dir", default=os.path.join(
        compat.REPO_ROOT, "work_dirs/petrv2_vov_mini"))
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the first N samples (0 = all). A partial "
                         "run auto-skips the official eval.")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    print("Building PETRv2 model + loading checkpoint ...")
    model = import_and_build_v2(args.ckpt).to(device)

    print("Loading nuScenes v1.0-mini (for temporal sweep alignment) ...")
    from nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-mini", dataroot=args.data_root, verbose=False)

    infos = mmengine.load(os.path.join(args.data_root,
                                       "nuscenes_infos_val.pkl"))["infos"]
    full_count = len(infos)
    if args.limit:
        infos = infos[:args.limit]
    partial = len(infos) < full_count
    print(f"Running PETRv2 inference on {len(infos)} samples ...")

    eval_cfg = config_factory("detection_cvpr_2019")
    nusc_annos = {}
    for info in mmcv.track_iter_progress(infos):
        img, img_metas = preprocess_sample_v2(info, args.data_root, nusc)
        img = img.to(device)
        with torch.no_grad():
            result = model.simple_test(img_metas, img)
        det = result[0]["pts_bbox"]
        boxes = output_to_nusc_box(det)
        boxes = lidar_nusc_box_to_global(info, boxes, CLASS_NAMES, eval_cfg)
        annos = []
        for b in boxes:
            anno = make_anno(b, CLASS_NAMES[b.label])
            anno["sample_token"] = info["token"]
            annos.append(anno)
        nusc_annos[info["token"]] = annos

    submission = dict(meta=MODALITY, results=nusc_annos)
    res_path = os.path.join(args.out_dir, "results_nusc.json")
    mmengine.dump(submission, res_path)
    print("Wrote", res_path)

    if args.no_eval or partial:
        reason = "--no-eval" if args.no_eval else \
            f"partial run ({len(infos)}/{full_count} samples)"
        print(f"\nSkipping official nuScenes evaluation ({reason}).")
        return

    from nuscenes.eval.detection.evaluate import NuScenesEval
    nusc_eval = NuScenesEval(nusc, config=eval_cfg, result_path=res_path,
                             eval_set="mini_val", output_dir=args.out_dir,
                             verbose=True)
    nusc_eval.main(render_curves=False)
    metrics = mmengine.load(os.path.join(args.out_dir, "metrics_summary.json"))
    print("\n========= PETRv2 VoVNet-p4-800x320 (temporal) on nuScenes "
          "mini_val =========")
    print(f"mAP : {metrics['mean_ap']:.4f}")
    print(f"NDS : {metrics['nd_score']:.4f}")
    for k, v in metrics["tp_errors"].items():
        print(f"{k:>12}: {v:.4f}")
    print("per-class AP:")
    for name, ap in metrics["mean_dist_aps"].items():
        print(f"  {name:>22}: {ap:.4f}")


if __name__ == "__main__":
    main()
