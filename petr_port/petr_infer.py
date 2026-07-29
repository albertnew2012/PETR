"""Run PETR (VoVNet p4 800x320) inference + nuScenes detection eval on mini_val.

Reproduces the original PETR test pipeline (LoadMultiViewImageFromFiles ->
ResizeCropFlipImage(test) -> NormalizeMultiviewImage -> PadMultiViewImage) and
the original mmdet3d 0.17 nuScenes formatting / evaluation, on the modern stack.
"""
import argparse
import os
import subprocess
import sys

import compat  # noqa: F401  (installs shims first)

import numpy as np
import torch
import mmcv
import mmengine
from PIL import Image
import pyquaternion
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.eval.detection.config import config_factory

from mmdet.registry import MODELS

# nuScenes detection class order used by the *model* (the config's class_names).
CLASS_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
VOXEL_SIZE = [0.2, 0.2, 8]
IMG_NORM = dict(mean=np.array([103.530, 116.280, 123.675], dtype=np.float32),
                std=np.array([57.375, 57.120, 58.395], dtype=np.float32),
                to_rgb=False)
IDA_AUG_CONF = {"resize_lim": (0.47, 0.625), "final_dim": (320, 800),
                "bot_pct_lim": (0.0, 0.0), "rot_lim": (0.0, 0.0),
                "H": 900, "W": 1600, "rand_flip": True}
MODALITY = dict(use_lidar=False, use_camera=True, use_radar=False,
                use_map=False, use_external=True)
DefaultAttribute = {
    "car": "vehicle.parked", "pedestrian": "pedestrian.moving",
    "trailer": "vehicle.parked", "truck": "vehicle.parked",
    "bus": "vehicle.moving", "motorcycle": "cycle.without_rider",
    "construction_vehicle": "vehicle.parked", "bicycle": "cycle.without_rider",
    "barrier": "", "traffic_cone": "",
}


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
def import_and_build(ckpt, train_cfg=None, test_cfg=None):
    register_petr_modules()
    model = MODELS.build(build_model_cfg(train_cfg, test_cfg))
    from mmengine.runner import load_checkpoint
    load_checkpoint(model, ckpt, map_location="cpu", strict=True)
    model.eval()
    return model


def register_petr_modules():
    import projects.mmdet3d_plugin.models.backbones.vovnetcp  # noqa: F401
    import projects.mmdet3d_plugin.models.necks.cp_fpn  # noqa: F401
    import projects.mmdet3d_plugin.models.utils.positional_encoding  # noqa
    import projects.mmdet3d_plugin.models.utils.petr_transformer  # noqa
    import projects.mmdet3d_plugin.core.bbox.coders.nms_free_coder  # noqa
    import projects.mmdet3d_plugin.models.dense_heads.petr_head as _ph  # noqa
    import projects.mmdet3d_plugin.models.detectors.petr3d  # noqa
    _ph.PETRHead.__abstractmethods__ = frozenset()


def build_model_cfg(train_cfg=None, test_cfg=None):
    return dict(
        type="Petr3D", use_grid_mask=True,
        img_backbone=dict(type="VoVNetCP", spec_name="V-99-eSE",
                          norm_eval=True, frozen_stages=-1, input_ch=3,
                          out_features=("stage4", "stage5")),
        img_neck=dict(type="CPFPN", in_channels=[768, 1024], out_channels=256,
                      num_outs=2),
        pts_bbox_head=dict(
            type="PETRHead", num_classes=10, in_channels=256, num_query=900,
            LID=True, with_position=True, with_multiview=True,
            position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            normedlinear=False,
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
        train_cfg=train_cfg, test_cfg=test_cfg, pretrained=None)



# --------------------------------------------------------------------------- #
#  Data: build per-sample camera matrices (mirror CustomNuScenesDataset)
# --------------------------------------------------------------------------- #
def build_cam_matrices(info):
    lidar2img_rts, intrinsics, extrinsics, paths = [], [], [], []
    for cam in CAMERAS:
        cam_info = info["cams"][cam]
        lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
        lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t
        intrinsic = cam_info["cam_intrinsic"]
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        lidar2img_rt = viewpad @ lidar2cam_rt.T
        intrinsics.append(viewpad)
        extrinsics.append(lidar2cam_rt)
        lidar2img_rts.append(lidar2img_rt)
        paths.append(cam_info["data_path"])
    return paths, lidar2img_rts, intrinsics, extrinsics


def _get_rot(h):
    return np.array([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]],
                    dtype=np.float32)


def _img_transform(img, resize, resize_dims, crop, flip, rotate):
    """Deterministic test-time transform, identical to ResizeCropFlipImage."""
    ida_rot = np.eye(2, dtype=np.float32)
    ida_tran = np.zeros(2, dtype=np.float32)
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)
    ida_rot *= resize
    ida_tran -= np.array(crop[:2], dtype=np.float32)
    if flip:
        A = np.array([[-1, 0], [0, 1]], dtype=np.float32)
        b = np.array([crop[2] - crop[0], 0], dtype=np.float32)
        ida_rot = A @ ida_rot
        ida_tran = A @ ida_tran + b
    A = _get_rot(rotate / 180 * np.pi)
    b = np.array([crop[2] - crop[0], crop[3] - crop[1]], dtype=np.float32) / 2
    b = A @ (-b) + b
    ida_rot = A @ ida_rot
    ida_tran = A @ ida_tran + b
    ida_mat = np.eye(3, dtype=np.float32)
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 2] = ida_tran
    return img, ida_mat


def _sample_augmentation():
    H, W = IDA_AUG_CONF["H"], IDA_AUG_CONF["W"]
    fH, fW = IDA_AUG_CONF["final_dim"]
    resize = max(fH / H, fW / W)
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = int((1 - np.mean(IDA_AUG_CONF["bot_pct_lim"])) * newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    return resize, resize_dims, crop, False, 0


def preprocess_sample(info, data_root):
    paths, lidar2img, intrinsics, extrinsics = build_cam_matrices(info)
    # LoadMultiViewImageFromFiles (BGR, float32 list)
    imgs = []
    for p in paths:
        if not os.path.isabs(p):
            p = os.path.join(data_root, p)
        img = mmcv.imread(p, "unchanged").astype(np.float32)
        imgs.append(img)
    # ResizeCropFlipImage (test): resize/crop + update intrinsics + lidar2img
    resize, resize_dims, crop, flip, rotate = _sample_augmentation()
    new_imgs = []
    intr = [m.copy() for m in intrinsics]
    for i in range(len(imgs)):
        pil = Image.fromarray(np.uint8(imgs[i]))
        pil, ida_mat = _img_transform(pil, resize, resize_dims, crop, flip,
                                      rotate)
        new_imgs.append(np.array(pil).astype(np.float32))
        intr[i][:3, :3] = ida_mat @ intr[i][:3, :3]
    lidar2img = [intr[i] @ extrinsics[i].T for i in range(len(extrinsics))]
    # NormalizeMultiviewImage
    new_imgs = [mmcv.imnormalize(im, IMG_NORM["mean"], IMG_NORM["std"],
                                 IMG_NORM["to_rgb"]) for im in new_imgs]
    # PadMultiViewImage size_divisor=32
    new_imgs = [mmcv.impad_to_multiple(im, 32, pad_val=0) for im in new_imgs]
    img_shape = [im.shape for im in new_imgs]
    # stack -> [N, 3, H, W] -> [1, N, 3, H, W]
    img_tensor = np.ascontiguousarray(np.stack(new_imgs).transpose(0, 3, 1, 2))
    img_tensor = torch.from_numpy(img_tensor).float().unsqueeze(0)
    img_meta = dict(
        img_shape=img_shape, pad_shape=img_shape,
        lidar2img=lidar2img, box_type_3d=compat.LiDARInstance3DBoxes,
        sample_idx=info["token"], scale_factor=1.0, flip=False,
    )
    return img_tensor, [img_meta]


# --------------------------------------------------------------------------- #
#  Results formatting (vendored from mmdet3d 0.17 NuScenesDataset)
# --------------------------------------------------------------------------- #
def output_to_nusc_box(detection):
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()
    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    box_yaw = -box_yaw - np.pi / 2
    box_list = []
    for i in range(len(box3d)):
        quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9].tolist(), 0.0)
        box = NuScenesBox(box_gravity_center[i], box_dims[i], quat,
                          label=labels[i], score=scores[i], velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info, boxes, classes, eval_configs):
    box_list = []
    for box in boxes:
        box.rotate(Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = eval_configs.class_range[classes[box.label]]
        if radius > det_range:
            continue
        box.rotate(Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list


def make_anno(box, name):
    if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
        if name in ("car", "construction_vehicle", "bus", "truck", "trailer"):
            attr = "vehicle.moving"
        elif name in ("bicycle", "motorcycle"):
            attr = "cycle.with_rider"
        else:
            attr = DefaultAttribute[name]
    else:
        if name == "pedestrian":
            attr = "pedestrian.standing"
        elif name == "bus":
            attr = "vehicle.stopped"
        else:
            attr = DefaultAttribute[name]
    return dict(translation=box.center.tolist(), size=box.wlh.tolist(),
                rotation=box.orientation.elements.tolist(),
                velocity=box.velocity[:2].tolist(), detection_name=name,
                detection_score=float(box.score), attribute_name=attr)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        compat.REPO_ROOT, "ckpts/petr_vovnet_p4_800x320.pth"))
    ap.add_argument("--data-root", default=os.path.join(
        compat.REPO_ROOT, "data/nuscenes"))
    ap.add_argument("--out-dir", default=os.path.join(
        compat.REPO_ROOT, "work_dirs/petr_vov_mini"))
    ap.add_argument("--limit", type=int, default=0,
                    help="run inference on only the first N samples (0 = all). "
                         "A partial run auto-skips the official eval, which "
                         "needs the whole mini_val split.")
    ap.add_argument("--no-eval", action="store_true",
                    help="write detections but skip the official nuScenes eval")
    ap.add_argument(
        "--blackout-cams", nargs="*", choices=CAMERAS, default=[],
        help="camera slots to zero after preprocessing (keeps geometry/shape)")
    ap.add_argument("--visualize", action="store_true",
                    help="render sample 0 immediately after inference")
    ap.add_argument("--visual-name", default="petr",
                    help="unique visualization filename/model label")
    ap.add_argument("--score-thr", type=float, default=0.5,
                    help="visualization confidence threshold")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    print("Building model + loading checkpoint ...")
    model = import_and_build(args.ckpt).to(device)

    infos = mmengine.load(os.path.join(args.data_root,
                                       "nuscenes_infos_val.pkl"))["infos"]
    full_count = len(infos)
    if args.limit:
        infos = infos[:args.limit]
    partial = len(infos) < full_count
    print(f"Running inference on {len(infos)} samples ...")
    blackout_indices = [CAMERAS.index(camera)
                        for camera in args.blackout_cams]
    if args.blackout_cams:
        print("Zeroing camera tensors after preprocessing:",
              ", ".join(args.blackout_cams))

    eval_cfg = config_factory("detection_cvpr_2019")
    nusc_annos = {}
    for idx, info in enumerate(mmcv.track_iter_progress(infos)):
        img, img_metas = preprocess_sample(info, args.data_root)
        if blackout_indices:
            img[:, blackout_indices] = 0.0
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

    if args.visualize:
        visualizer = os.path.join(os.path.dirname(__file__),
                                  "petrv2_visualize.py")
        command = [
            sys.executable, visualizer,
            "--index", "0",
            "--score-thr", str(args.score_thr),
            "--gt",
            "--model-name", args.visual_name,
            "--output-name", f"{args.visual_name}_sample_0.jpg",
            "--result-dir", args.out_dir,
            "--data-root", args.data_root,
        ]
        if args.blackout_cams:
            command.extend(["--blackout-cams", *args.blackout_cams])
        print("Creating visualization ...")
        subprocess.run(command, check=True)

    # The official nuScenes eval requires predictions for EVERY sample in the
    # mini_val split, so skip it for partial (--limit) or explicit --no-eval.
    if args.no_eval or partial:
        reason = "--no-eval" if args.no_eval else \
            f"partial run ({len(infos)}/{full_count} samples)"
        print(f"\nSkipping official nuScenes evaluation ({reason}).")
        print(f"Detections were written to {res_path}.")
        print(f"For the full mAP / NDS, run all {full_count} samples: "
              f"python3 petr_infer.py   (no --limit / --no-eval).")
        return

    # ---- official nuScenes evaluation ----
    from nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import NuScenesEval
    nusc = NuScenes(version="v1.0-mini", dataroot=args.data_root, verbose=False)
    nusc_eval = NuScenesEval(nusc, config=eval_cfg, result_path=res_path,
                             eval_set="mini_val", output_dir=args.out_dir,
                             verbose=True)
    nusc_eval.main(render_curves=False)
    metrics = mmengine.load(os.path.join(args.out_dir, "metrics_summary.json"))
    print("\n================ PETR VoVNet-p4-800x320 on nuScenes mini_val "
          "================")
    print(f"mAP : {metrics['mean_ap']:.4f}")
    print(f"NDS : {metrics['nd_score']:.4f}")
    for k, v in metrics["tp_errors"].items():
        print(f"{k:>12}: {v:.4f}")
    print("per-class AP:")
    for name, ap in metrics["mean_dist_aps"].items():
        print(f"  {name:>22}: {ap:.4f}")


if __name__ == "__main__":
    main()
