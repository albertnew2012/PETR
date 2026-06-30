"""Combined GIF: 6 camera views (with projected 3D boxes) + BEV, per frame.

For every keyframe of a scene it renders the 6 cameras with the predicted 3D
boxes drawn on them (left, 2x3) next to the top-down BEV panel (right), then
stitches the frames into a GIF.

Run:  python3 study/make_combined_gif.py --scene 0 --gt --fps 4
Out:  work_dirs/study/combined_<scene>.gif
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import compat  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import cv2  # noqa: E402
import mmengine  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
import petr_infer as P  # noqa: E402
from study.demo_visualize import draw_box_on_image, COLORS as CAM_COLORS, MONTAGE  # noqa: E402
from study.visualize_bev import render_bev_ax, load_gt  # noqa: E402


def camera_montage(info, det, scores, data_root, score_thr, pw=420, ph=236):
    boxes = P.output_to_nusc_box(det)
    keep = [i for i in range(len(boxes)) if scores[i] >= score_thr]
    paths, lidar2img, _, _ = P.build_cam_matrices(info)
    cam_idx = {c: i for i, c in enumerate(P.CAMERAS)}
    panels = {}
    for cam, ci in cam_idx.items():
        p = paths[ci] if os.path.isabs(paths[ci]) else os.path.join(data_root, paths[ci])
        im = cv2.imread(p)
        for i in keep:
            b = boxes[i]
            name = P.CLASS_NAMES[b.label]
            draw_box_on_image(im, b.corners(), lidar2img[ci], CAM_COLORS[name],
                              f"{name}:{scores[i]:.2f}")
        cv2.putText(im, cam, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)
        panels[cam] = cv2.resize(im, (pw, ph))
    rows = [np.concatenate([panels[c] for c in row], axis=1) for row in MONTAGE]
    return np.concatenate(rows, axis=0)            # BGR, (2*ph, 3*pw, 3)


def bev_bgr(det, scores, info, data_root, score_thr, nusc, size_px):
    boxes = P.output_to_nusc_box(det)
    keep = [(boxes[i], P.CLASS_NAMES[boxes[i].label])
            for i in range(len(boxes)) if scores[i] >= score_thr]
    gt = load_gt(info, data_root, nusc=nusc) if nusc is not None else None
    dpi = 100
    fig, ax = plt.subplots(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    render_bev_ax(ax, keep, gt, 55.0, 10.0, f"BEV (score \u2265 {score_thr})")
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--gt", action="store_true")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--width", type=int, default=1100, help="output GIF width")
    ap.add_argument("--out", default=os.path.join(compat.REPO_ROOT,
                                                  "work_dirs/study"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data_root = os.path.join(compat.REPO_ROOT, "data/nuscenes")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = P.import_and_build(os.path.join(
        compat.REPO_ROOT, "ckpts/petr_vovnet_p4_800x320.pth")).to(device)
    infos = mmengine.load(os.path.join(data_root,
                                       "nuscenes_infos_val.pkl"))["infos"]
    token2info = {it["token"]: it for it in infos}

    from nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-mini", dataroot=data_root, verbose=False)
    val_scenes = [sc for sc in nusc.scene
                  if sc["first_sample_token"] in token2info]
    sc = val_scenes[args.scene]
    ordered, tok = [], sc["first_sample_token"]
    while tok:
        if tok in token2info:
            ordered.append(token2info[tok])
        tok = nusc.get("sample", tok)["next"]
    ordered = ordered[::args.stride]
    print(f"scene {sc['name']}: {len(ordered)} frames")

    frames = []
    for k, info in enumerate(ordered):
        img, img_metas = P.preprocess_sample(info, data_root)
        with torch.no_grad():
            det = model.simple_test(img_metas, img.to(device))[0]["pts_bbox"]
        scores = det["scores_3d"].numpy()
        mont = camera_montage(info, det, scores, data_root, args.score_thr)
        bev = bev_bgr(det, scores, info, data_root, args.score_thr,
                      nusc if args.gt else None, size_px=mont.shape[0])
        frame = np.concatenate([mont, bev], axis=1)          # BGR
        # downscale to target width
        h, w = frame.shape[:2]
        nw = args.width
        nh = int(h * nw / w)
        frame = cv2.resize(frame, (nw, nh))
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        print(f"  frame {k+1}/{len(ordered)}", end="\r")

    gif = os.path.join(args.out, f"combined_{sc['name']}.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    sz = os.path.getsize(gif) / 1e6
    print(f"\nwrote {gif}  ({len(frames)} frames, {sz:.1f} MB)")


if __name__ == "__main__":
    main()
