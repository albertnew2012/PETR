"""Study demo: render predicted 3D boxes onto the 6 camera images.

Runs PETR on one nuScenes mini sample, projects the predicted 3D boxes (in the
LiDAR frame) into every camera using the same `lidar2img` matrices the model
consumes, and saves a 2x3 montage. This makes the geometry concrete.

Run:  python3 study/demo_visualize.py --index 0 --score-thr 0.3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import compat  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402
import mmengine  # noqa: E402
import petr_infer as P  # noqa: E402

# 12 edges of an 8-corner nuScenes box (corners(): first 4 front, last 4 back).
EDGES = [(i, (i + 1) % 4) for i in range(4)] + \
        [(i + 4, (i + 1) % 4 + 4) for i in range(4)] + \
        [(i, i + 4) for i in range(4)]
# class -> BGR colour
COLORS = {
    "car": (0, 255, 0), "truck": (0, 200, 200), "bus": (0, 128, 255),
    "trailer": (0, 100, 200), "construction_vehicle": (0, 60, 160),
    "pedestrian": (255, 0, 0), "motorcycle": (255, 0, 255),
    "bicycle": (255, 128, 0), "traffic_cone": (180, 180, 0),
    "barrier": (128, 128, 255),
}
# montage layout (nuScenes physical camera arrangement)
MONTAGE = [["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
           ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]]


def draw_box_on_image(img, corners3d, lidar2img, color, label):
    """corners3d: (3,8) in LiDAR frame. Returns True if drawn."""
    pts = np.concatenate([corners3d, np.ones((1, 8))], axis=0)  # (4,8)
    cam = lidar2img @ pts  # (4,8)
    depths = cam[2, :]
    if np.any(depths < 0.1):  # box (partly) behind the camera
        return False
    uv = cam[:2, :] / depths[None, :]
    if uv[0].max() < 0 or uv[0].min() > img.shape[1] or \
       uv[1].max() < 0 or uv[1].min() > img.shape[0]:
        return False
    uv = uv.astype(np.int32)
    for a, b in EDGES:
        cv2.line(img, tuple(uv[:, a]), tuple(uv[:, b]), color, 2, cv2.LINE_AA)
    c = uv[:, :4].mean(axis=1).astype(np.int32)
    cv2.putText(img, label, (c[0], c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2, cv2.LINE_AA)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--out", default=os.path.join(
        compat.REPO_ROOT, "work_dirs/study"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data_root = os.path.join(compat.REPO_ROOT, "data/nuscenes")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = P.import_and_build(
        os.path.join(compat.REPO_ROOT,
                     "ckpts/petr_vovnet_p4_800x320.pth")).to(device)
    infos = mmengine.load(
        os.path.join(data_root, "nuscenes_infos_val.pkl"))["infos"]
    info = infos[args.index]

    img, img_metas = P.preprocess_sample(info, data_root)
    with torch.no_grad():
        det = model.simple_test(img_metas, img.to(device))[0]["pts_bbox"]

    # NuScenesBox list in the LiDAR frame (no global transform here).
    boxes = P.output_to_nusc_box(det)
    scores = det["scores_3d"].numpy()
    keep = [i for i in range(len(boxes)) if scores[i] >= args.score_thr]
    print(f"sample {args.index}: {len(keep)} boxes with score>={args.score_thr}")

    paths, lidar2img, _, _ = P.build_cam_matrices(info)
    name_by_cam = {c: i for i, c in enumerate(P.CAMERAS)}

    panels = {}
    for cam, ci in name_by_cam.items():
        im = cv2.imread(paths[ci] if os.path.isabs(paths[ci])
                        else os.path.join(data_root, paths[ci]))
        drawn = 0
        for i in keep:
            box = boxes[i]
            name = P.CLASS_NAMES[box.label]
            ok = draw_box_on_image(im, box.corners(), lidar2img[ci],
                                   COLORS[name], f"{name}:{scores[i]:.2f}")
            drawn += int(ok)
        cv2.putText(im, f"{cam}  ({drawn} boxes)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        panels[cam] = im

    rows = []
    for row in MONTAGE:
        ims = [cv2.resize(panels[c], (800, 450)) for c in row]
        rows.append(np.concatenate(ims, axis=1))
    montage = np.concatenate(rows, axis=0)
    out_path = os.path.join(args.out, f"demo_boxes_{args.index}.jpg")
    cv2.imwrite(out_path, montage)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
