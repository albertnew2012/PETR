"""Make an animated BEV GIF of PETR detections over one nuScenes scene.

Renders the top-down detection view (ego at centre, range rings) for every
keyframe of a scene in time order, then stitches the frames into a GIF with
Pillow. The view is ego-centric: the ego stays fixed at the centre and the
detected objects move relative to it as the car drives.

Run:  python3 study/make_bev_gif.py --scene 0 --gt --fps 4
Out:  work_dirs/study/bev_<scene-name>.gif
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import compat  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import mmengine  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
import petr_infer as P  # noqa: E402
from study.visualize_bev import render_bev_ax, load_gt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=0,
                    help="which val scene (0 or 1 for mini_val)")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--max-range", type=float, default=55.0)
    ap.add_argument("--ring-step", type=float, default=10.0)
    ap.add_argument("--gt", action="store_true")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--out", default=os.path.join(
        compat.REPO_ROOT, "work_dirs/study"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data_root = os.path.join(compat.REPO_ROOT, "data/nuscenes")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = P.import_and_build(os.path.join(
        compat.REPO_ROOT, "ckpts/petr_vovnet_p4_800x320.pth")).to(device)
    infos = mmengine.load(
        os.path.join(data_root, "nuscenes_infos_val.pkl"))["infos"]
    token2info = {it["token"]: it for it in infos}

    # Order the chosen scene's keyframes in time, restricted to our val infos.
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
    print(f"scene {sc['name']}: {len(ordered)} frames")

    frames = []
    for k, info in enumerate(ordered):
        img, img_metas = P.preprocess_sample(info, data_root)
        with torch.no_grad():
            det = model.simple_test(img_metas, img.to(device))[0]["pts_bbox"]
        scores = det["scores_3d"].numpy()
        pb = P.output_to_nusc_box(det)
        keep = [(pb[i], P.CLASS_NAMES[pb[i].label])
                for i in range(len(pb)) if scores[i] >= args.score_thr]
        gt = load_gt(info, data_root, nusc=nusc) if args.gt else None

        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)   # fixed 800x800 frames
        render_bev_ax(ax, keep, gt, args.max_range, args.ring_step,
                      f"{sc['name']}   frame {k+1}/{len(ordered)}   "
                      f"(score \u2265 {args.score_thr})")
        fig.canvas.draw()
        frames.append(Image.fromarray(
            np.asarray(fig.canvas.buffer_rgba())).convert("RGB"))
        plt.close(fig)
        print(f"  frame {k+1}/{len(ordered)}: {len(keep)} boxes", end="\r")

    gif = os.path.join(args.out, f"bev_{sc['name']}.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"\nwrote {gif}  ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
