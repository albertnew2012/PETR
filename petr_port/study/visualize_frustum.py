"""Visualize the '3D Coordinates Generator' = camera frustums in 3D.

For each camera it samples a grid of pixels x the 64 LID depth bins and
back-projects them into the LiDAR frame (exactly what PETRHead.position_embeding
does), then draws:
  (left)  a bird's-eye (top-down) view of all 6 camera frustums around the ego,
  (right) the FRONT camera's frustum coloured by depth, showing the LID spacing.

Run:  python3 study/visualize_frustum.py --index 0
Out:  work_dirs/study/frustum_bev.jpg
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import compat  # noqa: E402
import numpy as np  # noqa: E402
import mmengine  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import petr_infer as P  # noqa: E402

IMG_W, IMG_H = 1600, 900          # original nuScenes image size
DEPTH_START, DEPTH_MAX, DEPTH_NUM = 1.0, 61.2, 64
CAM_COLORS = {
    "CAM_FRONT": "#e6194B", "CAM_FRONT_RIGHT": "#f58231",
    "CAM_FRONT_LEFT": "#ffe119", "CAM_BACK": "#3cb44b",
    "CAM_BACK_LEFT": "#4363d8", "CAM_BACK_RIGHT": "#911eb4",
}


def lid_depth_bins():
    """PETR's Linear-Increasing Discretization depth bins."""
    i = np.arange(DEPTH_NUM, dtype=np.float64)
    bin_size = (DEPTH_MAX - DEPTH_START) / (DEPTH_NUM * (1 + DEPTH_NUM))
    return DEPTH_START + bin_size * i * (i + 1)


def backproject(lidar2img, us, vs, ds):
    """Return [M,3] LiDAR points for the grid (us x vs x ds)."""
    img2lidar = np.linalg.inv(np.asarray(lidar2img, dtype=np.float64))
    uu, vv, dd = np.meshgrid(us, vs, ds, indexing="ij")
    u, v, d = uu.ravel(), vv.ravel(), dd.ravel()
    homo = np.stack([u * d, v * d, d, np.ones_like(d)], axis=0)  # (4, M)
    pts = img2lidar @ homo                                       # (4, M)
    pts = (pts[:3] / pts[3]).T                                   # (M, 3)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(
        compat.REPO_ROOT, "work_dirs/study"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data_root = os.path.join(compat.REPO_ROOT, "data/nuscenes")

    info = mmengine.load(
        os.path.join(data_root, "nuscenes_infos_val.pkl"))["infos"][args.index]
    _, lidar2img, _, _ = P.build_cam_matrices(info)
    depths = lid_depth_bins()

    fig, (axb, axd) = plt.subplots(1, 2, figsize=(15, 7))

    # ---- (left) BEV of all six camera frustums ----
    us = np.linspace(0, IMG_W, 13)
    vs = np.linspace(150, 800, 4)           # skip extreme sky/hood rows
    for ci, cam in enumerate(P.CAMERAS):
        pts = backproject(lidar2img[ci], us, vs, depths)
        m = (np.abs(pts[:, 0]) < 80) & (np.abs(pts[:, 1]) < 80)
        axb.scatter(pts[m, 0], pts[m, 1], s=3, color=CAM_COLORS[cam],
                    label=cam, alpha=0.5)
    for r, c, lbl in [(51.2, "k", "point_cloud_range"),
                      (61.2, "gray", "position_range")]:
        axb.add_patch(Rectangle((-r, -r), 2 * r, 2 * r, fill=False,
                                edgecolor=c, ls="--", lw=1.2))
        axb.text(0, r + 1, lbl, ha="center", color=c, fontsize=8)
    axb.plot(0, 0, "k*", ms=16)
    axb.annotate("ego", (0, 0), (3, -6), fontsize=9)
    axb.annotate("FRONT (+y)", (0, 70), ha="center", fontsize=9)
    axb.set_xlabel("x  (left/right, m)")
    axb.set_ylabel("y  (forward, m)")
    axb.set_title("Bird's-eye view: 6 camera frustums in the LiDAR frame")
    axb.set_aspect("equal")
    axb.set_xlim(-75, 75)
    axb.set_ylim(-75, 75)
    axb.legend(loc="upper right", fontsize=7, markerscale=2)
    axb.grid(alpha=0.2)

    # ---- (right) FRONT frustum coloured by depth (shows LID spacing) ----
    front = P.CAMERAS.index("CAM_FRONT")
    us2 = np.linspace(0, IMG_W, 9)
    vs2 = np.array([450.0])                  # one horizontal scanline
    img2lidar = np.linalg.inv(np.asarray(lidar2img[front], dtype=np.float64))
    for u in us2:
        homo = np.stack([u * depths, vs2[0] * depths, depths,
                         np.ones_like(depths)], axis=0)
        p = img2lidar @ homo
        p = (p[:3] / p[3]).T
        axd.scatter(p[:, 0], p[:, 1], c=depths, cmap="viridis", s=10)
    sc = axd.scatter([], [], c=[], cmap="viridis")
    sc.set_clim(depths.min(), depths.max())
    cb = plt.colorbar(sc, ax=axd)
    cb.set_label("depth bin value (m)  — note widening LID spacing")
    axd.plot(0, 0, "k*", ms=16)
    axd.set_xlabel("x  (left/right, m)")
    axd.set_ylabel("y  (forward, m)")
    axd.set_title("FRONT camera: one scanline x 64 LID depths\n"
                  "(each fan-line = one pixel's ray of 3D points)")
    axd.set_aspect("equal")
    axd.grid(alpha=0.2)

    fig.suptitle("PETR 3D Coordinates Generator — a pixel becomes a ray of 3D "
                 "points; a camera becomes a frustum", fontsize=12)
    fig.tight_layout()
    out = os.path.join(args.out, f"frustum_bev_{args.index}.jpg")
    fig.savefig(out, dpi=110)
    print("wrote", out)
    print(f"LID depth bins: first few = {depths[:5].round(2)}, "
          f"last few = {depths[-3:].round(1)} (m)")


if __name__ == "__main__":
    main()
