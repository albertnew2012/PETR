"""BEV (bird's-eye view) visualization of PETR detections.

Top-down view of the predicted 3D boxes (all 6 cameras fused into the single
LiDAR/ego frame), with concentric range rings showing distance from the ego.
Optionally overlays ground-truth boxes.

Run:  python3 study/visualize_bev.py --index 0 --score-thr 0.3
      python3 study/visualize_bev.py --index 0 --gt          # add GT (needs devkit)
Out:  work_dirs/study/bev_<index>.jpg
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
from matplotlib.patches import Circle, Polygon  # noqa: E402
import petr_infer as P  # noqa: E402

# class -> matplotlib RGB
COLORS = {
    "car": "#2ca02c", "truck": "#17becf", "bus": "#ff7f0e",
    "trailer": "#8c564b", "construction_vehicle": "#7f4f24",
    "pedestrian": "#1f77b4", "motorcycle": "#e377c2",
    "bicycle": "#ff9896", "traffic_cone": "#bcbd22", "barrier": "#9467bd",
}


def box_bev_polygon(box):
    """nuScenes Box (LiDAR frame) -> (4,2) bottom-face corners in BEV (x,y)."""
    corners = box.bottom_corners()          # (3, 4) in the box's frame (LiDAR)
    return corners[:2].T                     # (4, 2) -> x, y


def heading_segment(box):
    """A short line from the box centre along its forward (length) axis."""
    c = box.center[:2]
    fwd = box.orientation.rotation_matrix[:, 0][:2]   # local x-axis in LiDAR
    tip = c + fwd * (box.wlh[1] / 2.0)                # length is wlh[1]
    return np.array([c, tip])


def draw_boxes(ax, boxes, color=None, lw=1.6, label_set=None):
    for box, name in boxes:
        col = color or COLORS.get(name, "#333333")
        poly = box_bev_polygon(box)
        lbl = None
        if label_set is not None and name not in label_set:
            lbl = name
            label_set.add(name)
        ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor=col,
                             lw=lw, label=lbl))
        seg = heading_segment(box)
        ax.plot(seg[:, 0], seg[:, 1], color=col, lw=lw)


def load_gt(info, data_root, nusc=None):
    """Ground-truth boxes in the LiDAR frame, as [(Box, class_name), ...]."""
    if nusc is None:
        from nuscenes import NuScenes
        nusc = NuScenes(version="v1.0-mini", dataroot=data_root, verbose=False)
    _, gt_boxes, _ = nusc.get_sample_data(info["lidar_token"])
    nm = compat._NuScenesDatasetStub.NameMapping
    return [(b, nm.get(b.name, b.name)) for b in gt_boxes
            if nm.get(b.name, b.name) in COLORS]


def render_bev_ax(ax, pred, gt, max_range, ring_step, title):
    """Draw range rings, ego, GT (optional) and predicted boxes onto `ax`."""
    R = max_range
    for r in np.arange(ring_step, R + 1e-3, ring_step):
        ax.add_patch(Circle((0, 0), r, fill=False, ls="--", lw=0.8,
                            edgecolor="0.6"))
        ax.text(0, r, f"{int(r)} m", ha="center", va="bottom", fontsize=8,
                color="0.45")
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.axvline(0, color="0.85", lw=0.6, zorder=0)
    # ego (triangle pointing +y / forward)
    ax.add_patch(Polygon([[-1.0, -1.8], [1.0, -1.8], [0.0, 2.2]], closed=True,
                         facecolor="black", edgecolor="black", zorder=5))
    if gt:
        draw_boxes(ax, gt, color="0.55", lw=1.2)
        ax.plot([], [], color="0.55", lw=1.2, label="ground truth")
    draw_boxes(ax, pred, label_set=set())
    ax.set_aspect("equal")
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_xlabel("x  (m, lateral)")
    ax.set_ylabel("y  (m, forward →)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--max-range", type=float, default=55.0)
    ap.add_argument("--ring-step", type=float, default=10.0)
    ap.add_argument("--gt", action="store_true", help="overlay ground truth")
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
    info = infos[args.index]

    img, img_metas = P.preprocess_sample(info, data_root)
    with torch.no_grad():
        det = model.simple_test(img_metas, img.to(device))[0]["pts_bbox"]
    scores = det["scores_3d"].numpy()
    pred_boxes = P.output_to_nusc_box(det)        # NuScenesBox in LiDAR frame
    keep = [(pred_boxes[i], P.CLASS_NAMES[pred_boxes[i].label])
            for i in range(len(pred_boxes)) if scores[i] >= args.score_thr]
    print(f"sample {args.index}: {len(keep)} boxes >= {args.score_thr}")

    gt = None
    if args.gt:
        try:
            gt = load_gt(info, data_root)
        except Exception as e:
            print("GT overlay skipped:", e)

    fig, ax = plt.subplots(figsize=(10, 10))
    render_bev_ax(ax, keep, gt, args.max_range, args.ring_step,
                  f"PETR detections in BEV — sample {args.index} "
                  f"(score ≥ {args.score_thr}); ego ▲ at centre")
    out = os.path.join(args.out, f"bev_{args.index}.jpg")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
