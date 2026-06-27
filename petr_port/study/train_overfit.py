"""Study demo: prove the TRAINING path works on the modern stack by overfitting
the PETR head to a couple of nuScenes mini samples and watching the loss drop.

For speed/memory this freezes the backbone+neck and caches their features, then
optimises only the detection head (the part that learns query->object). Real
training also fine-tunes the backbone (lr_mult 0.1) -- see TRAINING_GUIDE.md.

Run:  python3 study/train_overfit.py --steps 40 --num-samples 2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import compat  # noqa: E402
import torch  # noqa: E402
import mmengine  # noqa: E402
from mmdet.registry import MODELS  # noqa: E402
import petr_infer as P  # noqa: E402
from study.loss_demo import TRAIN_CFG, build_gt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--num-samples", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = os.path.join(compat.REPO_ROOT, "data/nuscenes")

    P.register_petr_modules()
    import projects.mmdet3d_plugin.core.bbox.match_costs.match_cost  # noqa
    import projects.mmdet3d_plugin.core.bbox.assigners.hungarian_assigner_3d  # noqa
    model = MODELS.build(P.build_model_cfg(train_cfg=TRAIN_CFG)).to(device)
    from mmengine.runner import load_checkpoint
    load_checkpoint(model, os.path.join(
        compat.REPO_ROOT, "ckpts/petr_vovnet_p4_800x320.pth"),
        map_location="cpu", strict=True)
    head = model.pts_bbox_head

    # Freeze backbone+neck; cache their features once (head-only overfit).
    model.eval()
    for p in model.img_backbone.parameters():
        p.requires_grad_(False)
    for p in model.img_neck.parameters():
        p.requires_grad_(False)

    infos = mmengine.load(
        os.path.join(data_root, "nuscenes_infos_val.pkl"))["infos"]
    cache = []
    for i in range(args.num_samples):
        info = infos[i]
        img, img_metas = P.preprocess_sample(info, data_root)
        with torch.no_grad():
            feats = model.extract_feat(img=img.to(device), img_metas=img_metas)
            feats = [f.detach() for f in feats]
        gt_box, gt_labels = build_gt(info, device)
        cache.append((feats, img_metas, gt_box, gt_labels))
        print(f"sample {i}: {len(gt_box)} GT boxes")

    opt = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad], lr=args.lr,
        weight_decay=0.01)
    head.train()

    print(f"\noverfitting head on {args.num_samples} sample(s) for "
          f"{args.steps} steps (lr={args.lr})...\n"
          f"{'step':>5} {'total':>9} {'loss_cls':>9} {'loss_bbox':>9}")
    for step in range(args.steps):
        tot = clsv = boxv = 0.0
        for feats, metas, gt_box, gt_labels in cache:
            outs = head(feats, metas)
            losses = head.loss([gt_box], [gt_labels], outs)
            loss = sum(losses.values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 35)
            opt.step()
            tot += float(loss)
            clsv += float(losses["loss_cls"])
            boxv += float(losses["loss_bbox"])
        if step % 5 == 0 or step == args.steps - 1:
            n = len(cache)
            print(f"{step:5d} {tot/n:9.4f} {clsv/n:9.4f} {boxv/n:9.4f}")

    print("\nIf 'total' drops markedly, the full train path "
          "(forward -> match -> loss -> backward -> step) works on this stack.")


if __name__ == "__main__":
    main()
