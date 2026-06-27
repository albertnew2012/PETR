"""Reproduce PETR paper Figure 4 -- "3D position embedding similarity".

Picks 3 points in the FRONT camera, takes their learned 3D position embedding
(the output of PETRHead.position_encoder), and computes the cosine similarity
against the 3D PE of every location in all 6 camera views. Nearby-in-3D regions
light up -- e.g. a left point in the front view raises the response on the right
part of the FRONT_LEFT view -- showing the 3D PE encodes cross-view 3D position.

Run:  python3 study/visualize_3d_pe.py --index 0
Out:  work_dirs/study/fig4_3d_pe.jpg
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import compat  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import cv2  # noqa: E402
import mmengine  # noqa: E402
import petr_infer as P  # noqa: E402

# physical layout for the montage columns -> indices into P.CAMERAS
#   CAMERAS = [FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT]
COL_ORDER = [("FRONT_LEFT", 2), ("FRONT", 0), ("FRONT_RIGHT", 1),
             ("BACK_LEFT", 4), ("BACK", 3), ("BACK_RIGHT", 5)]
FRONT = 0
PANEL_W, PANEL_H = 480, 270


def denormalize(img_tensor_v):
    """[3,H,W] normalised BGR -> uint8 BGR image for display."""
    img = img_tensor_v.permute(1, 2, 0).cpu().numpy()
    img = img * P.IMG_NORM["std"] + P.IMG_NORM["mean"]
    return np.clip(img, 0, 255).astype(np.uint8)


def overlay_heat(img, heat01, alpha=0.5):
    heat = (np.clip(heat01, 0, 1) * 255).astype(np.uint8)
    heat = cv2.resize(heat, (img.shape[1], img.shape[0]),
                      interpolation=cv2.INTER_CUBIC)
    color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    # blend over a desaturated base so the heatmap structure pops
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                        cv2.COLOR_GRAY2BGR)
    base = cv2.addWeighted(img, 0.5, gray, 0.5, 0)
    return cv2.addWeighted(base, 1 - alpha, color, alpha, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--row-frac", type=float, default=0.55,
                    help="vertical position of the 3 points (0=top,1=bottom)")
    ap.add_argument("--out", default=os.path.join(
        compat.REPO_ROOT, "work_dirs/study"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = os.path.join(compat.REPO_ROOT, "data/nuscenes")

    model = P.import_and_build(os.path.join(
        compat.REPO_ROOT, "ckpts/petr_vovnet_p4_800x320.pth")).to(device)
    head = model.pts_bbox_head

    infos = mmengine.load(
        os.path.join(data_root, "nuscenes_infos_val.pkl"))["infos"]
    img_tensor, img_metas = P.preprocess_sample(infos[args.index], data_root)
    disp = [denormalize(img_tensor[0, v]) for v in range(6)]  # BGR uint8

    # --- capture the 3D PE (output of position_encoder) via a forward hook ---
    cap = {}
    h = head.position_encoder.register_forward_hook(
        lambda m, i, o: cap.__setitem__("pe", o.detach()))
    with torch.no_grad():
        feats = model.extract_feat(img=img_tensor.to(device),
                                   img_metas=img_metas)
        _ = head(feats, img_metas)
    h.remove()
    pe = cap["pe"].float()                       # [6, 256, H, W]
    N, C, H, W = pe.shape
    pe_norm = F.normalize(pe, dim=1)             # cosine-ready
    print(f"3D PE tensor: {tuple(pe.shape)}  (views, dim, Hf, Wf)")

    # --- 3 points in the FRONT view: left / centre / right at a fixed row ---
    r = int(args.row_frac * H)
    cols = [int(0.18 * W), int(0.5 * W), int(0.82 * W)]
    sx, sy = PANEL_W / W, PANEL_H / H            # feature->panel scale

    rows_img = []
    for ci, c in enumerate(cols):
        vec = pe_norm[FRONT, :, r, c]                        # [256]
        sims = (pe_norm * vec[None, :, None, None]).sum(1)   # [6, H, W]
        # robust contrast: clip to a [floor, high] percentile window across all
        # 6 views (so the broad "everything in front is similar" background dims
        # and the genuinely-high regions pop), then gamma for the mid-tones.
        s = sims.cpu().numpy()
        lo = np.percentile(s, 70.0)
        hi = np.percentile(s, 99.5)
        sims01 = np.clip((s - lo) / (hi - lo + 1e-9), 0, 1) ** 0.7

        panels = []
        for name, vi in COL_ORDER:
            base = cv2.resize(disp[vi], (PANEL_W, PANEL_H))
            panel = overlay_heat(base, sims01[vi], alpha=0.5)
            if vi == FRONT:  # mark the selected red point
                px, py = int(c * sx), int(r * sy)
                cv2.circle(panel, (px, py), 7, (0, 0, 255), -1)
                cv2.circle(panel, (px, py), 8, (255, 255, 255), 2)
            cv2.putText(panel, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
        row = np.concatenate(panels, axis=1)
        label = ["LEFT point", "CENTER point", "RIGHT point"][ci]
        cv2.putText(row, label, (6, PANEL_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
        rows_img.append(row)

    fig = np.concatenate(rows_img, axis=0)
    out_path = os.path.join(args.out, f"fig4_3d_pe_{args.index}.jpg")
    cv2.imwrite(out_path, fig)
    print("wrote", out_path)

    # --- numeric geometry: back-project the CENTER front anchor's ray ---
    l2i = np.asarray(img_metas[0]["lidar2img"][FRONT], dtype=np.float64)
    img2lidar = np.linalg.inv(l2i)
    u, v = cols[1] * 16 + 8, r * 16 + 8     # feature cell -> 800x320 input pixel
    print(f"\nBack-projection of the CENTER front pixel (u={u}, v={v}) — this is "
          f"what the 3D PE encodes for that location:")
    for d in (3.0, 15.0, 45.0):
        p = img2lidar @ np.array([u * d, v * d, d, 1.0])
        p = p[:3] / p[3]
        print(f"   depth {d:5.1f} m  ->  LiDAR xyz = "
              f"({p[0]:6.1f}, {p[1]:6.1f}, {p[2]:6.1f})")
    print("\nEach row = one red point in FRONT; warm = high 3D-PE cosine "
          "similarity. Note the cross-view response in adjacent cameras.")


if __name__ == "__main__":
    main()
