"""Study tool: compute and explain the PETR *training* loss on one real sample.

Builds the head WITH train_cfg (so the Hungarian assigner is active), loads the
ground-truth boxes for one nuScenes mini sample, runs a forward pass, performs
the bipartite matching, and prints every loss term (final layer + 5 aux layers).

Run:  python3 study/loss_demo.py --index 0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import compat  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import mmengine  # noqa: E402
from mmdet.registry import MODELS  # noqa: E402
import petr_infer as P  # noqa: E402

TRAIN_CFG = dict(pts=dict(
    grid_size=[512, 512, 1], voxel_size=P.VOXEL_SIZE,
    point_cloud_range=P.POINT_CLOUD_RANGE, out_size_factor=4,
    assigner=dict(
        type="HungarianAssigner3D",
        cls_cost=dict(type="FocalLossCost", weight=2.0),
        reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
        iou_cost=dict(type="IoUCost", weight=0.0),
        pc_range=P.POINT_CLOUD_RANGE)))


def build_gt(info, device):
    """Convert info pkl GT to a LiDARInstance3DBoxes (gravity-center -> bottom)."""
    nm = compat._NuScenesDatasetStub.NameMapping  # raw nuScenes -> det name
    gt = info["gt_boxes"].astype(np.float32)             # [N,7] x,y,zc,w,l,h,yaw
    vel = info["gt_velocity"].astype(np.float32)         # [N,2]
    vel = np.nan_to_num(vel)
    names = info["gt_names"]
    valid = info.get("valid_flag", np.ones(len(gt), bool))

    boxes9, labels = [], []
    for i in range(len(gt)):
        name = nm.get(names[i], names[i])
        if name not in P.CLASS_NAMES or not valid[i]:
            continue
        x, y, zc = gt[i, 0], gt[i, 1], gt[i, 2]
        if not (P.POINT_CLOUD_RANGE[0] <= x <= P.POINT_CLOUD_RANGE[3] and
                P.POINT_CLOUD_RANGE[1] <= y <= P.POINT_CLOUD_RANGE[4]):
            continue
        boxes9.append([*gt[i, :6], gt[i, 6], vel[i, 0], vel[i, 1]])
        labels.append(P.CLASS_NAMES.index(name))
    boxes9 = torch.tensor(boxes9, dtype=torch.float32)
    # gravity-center z -> bottom-center z, so .gravity_center gives the centre back
    boxes9[:, 2] = boxes9[:, 2] - boxes9[:, 5] * 0.5
    gt_box = compat.LiDARInstance3DBoxes(boxes9, box_dim=9).to(device)
    gt_labels = torch.tensor(labels, dtype=torch.long, device=device)
    return gt_box, gt_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = os.path.join(compat.REPO_ROOT, "data/nuscenes")

    # Build the model WITH train_cfg so head.assigner / head.sampler exist.
    P.register_petr_modules()
    import projects.mmdet3d_plugin.core.bbox.match_costs.match_cost  # noqa
    import projects.mmdet3d_plugin.core.bbox.assigners.hungarian_assigner_3d  # noqa
    model = MODELS.build(P.build_model_cfg(train_cfg=TRAIN_CFG)).to(device)
    from mmengine.runner import load_checkpoint
    load_checkpoint(model, os.path.join(
        compat.REPO_ROOT, "ckpts/petr_vovnet_p4_800x320.pth"),
        map_location="cpu", strict=True)
    model.eval()
    head = model.pts_bbox_head

    infos = mmengine.load(
        os.path.join(data_root, "nuscenes_infos_val.pkl"))["infos"]
    info = infos[args.index]
    img, img_metas = P.preprocess_sample(info, data_root)
    gt_box, gt_labels = build_gt(info, device)

    print("=" * 70)
    print(f"Sample {args.index}: {len(gt_box)} GT boxes, {head.num_query} queries")
    print("GT class histogram:",
          {P.CLASS_NAMES[c]: int((gt_labels == c).sum()) for c in
           sorted(set(gt_labels.tolist()))})

    # Forward pass -> raw head outputs (all 6 decoder layers).
    with torch.no_grad():
        feats = model.extract_feat(img=img.to(device), img_metas=img_metas)
        outs = head(feats, img_metas)
    print(f"\nHead outputs: all_cls_scores {tuple(outs['all_cls_scores'].shape)} "
          f"(layers,B,Q,classes)")
    print(f"             all_bbox_preds {tuple(outs['all_bbox_preds'].shape)} "
          f"(layers,B,Q,code=10)")

    # Explicit bipartite matching on the FINAL layer (what the loss does inside).
    gt_for_match = torch.cat(
        [gt_box.gravity_center, gt_box.tensor[:, 3:]], dim=1).to(device)
    assign = head.assigner.assign(outs["all_bbox_preds"][-1][0],
                                  outs["all_cls_scores"][-1][0],
                                  gt_for_match, gt_labels)
    n_pos = int((assign.gt_inds > 0).sum())
    print(f"\nHungarian matching (final layer): {n_pos} positive queries "
          f"(= #GT), {head.num_query - n_pos} background.")

    # Full deep-supervised loss over all 6 decoder layers.
    with torch.no_grad():
        loss_dict = head.loss([gt_box], [gt_labels], outs)

    print("\n--- LOSS TERMS (weights already applied: cls x2.0, bbox x0.25) ---")
    total = 0.0
    # final-layer terms first, then aux layers d0..d4
    order = ["loss_cls", "loss_bbox"] + \
            [f"d{i}.loss_{t}" for i in range(5) for t in ("cls", "bbox")]
    for k in order:
        if k in loss_dict:
            v = float(loss_dict[k])
            total += v
            tag = "  (final layer)" if "." not in k else ""
            print(f"  {k:<16}: {v:8.4f}{tag}")
    print(f"  {'TOTAL':<16}: {total:8.4f}  (sum of all 12 terms -> backprop)")

    print("\nWhat to notice:")
    print("  * one-to-one matching => #positives == #GT (no NMS needed)")
    print("  * the SAME loss is applied to every decoder layer (deep supervision)")
    print("  * loss_bbox is L1 on NORMALISED 10-d boxes, weighted by code_weights")
    print("    (velocity dims x0.2); see PETRHead.loss_single + util.normalize_bbox")


if __name__ == "__main__":
    main()
