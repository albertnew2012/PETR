"""A *tiny* end-to-end PETR training job (smoke / learning scale).

Unlike study/train_overfit.py (head-only, frozen backbone), this trains the
WHOLE model (backbone + neck + head) for a few steps on a few nuScenes mini
samples, mirroring the real training recipe on a tiny scale:

  * AdamW with the backbone at 0.1x LR (config's `lr_mult`)
  * gradient clipping at max_norm=35
  * GridMask + transformer gradient-checkpointing active in train mode

Runs in FP32 (the original recipe uses an mmcv Fp16OptimizerHook + `force_fp32`
upcasting in the loss; that hook isn't ported here, so we keep it simple).

It saves a checkpoint at the end. This is a SMOKE/LEARNING job to see real
training step on this stack -- not a full training run (no multi-GPU dataloader,
no full augmentation pipeline / LR schedule; see TRAINING_GUIDE.md).

Run:  python3 study/train_tiny.py --steps 20 --num-samples 3
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
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default=os.path.join(
        compat.REPO_ROOT, "work_dirs/tiny_train"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    data_root = os.path.join(compat.REPO_ROOT, "data/nuscenes")

    # --- build model WITH train_cfg (assigner) and load pretrained weights ---
    P.register_petr_modules()
    import projects.mmdet3d_plugin.core.bbox.match_costs.match_cost  # noqa
    import projects.mmdet3d_plugin.core.bbox.assigners.hungarian_assigner_3d  # noqa
    model = MODELS.build(P.build_model_cfg(train_cfg=TRAIN_CFG)).to(device)
    from mmengine.runner import load_checkpoint
    load_checkpoint(model, os.path.join(
        compat.REPO_ROOT, "ckpts/petr_vovnet_p4_800x320.pth"),
        map_location="cpu", strict=True)
    model.train()

    # --- optimiser: backbone at 0.1x LR (config's paramwise_cfg) ---
    backbone_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith("img_backbone") else
         other_params).append(p)
    optimizer = torch.optim.AdamW(
        [dict(params=backbone_params, lr=args.lr * 0.1),
         dict(params=other_params, lr=args.lr)], weight_decay=0.01)

    # --- a few samples (test-time preprocessing; GT from info pkl) ---
    infos = mmengine.load(
        os.path.join(data_root, "nuscenes_infos_val.pkl"))["infos"]
    samples = []
    for i in range(args.num_samples):
        img, metas = P.preprocess_sample(infos[i], data_root)
        gt_box, gt_labels = build_gt(infos[i], device)
        samples.append((img, metas, gt_box, gt_labels))
    print(f"tiny training: {args.num_samples} samples, {args.steps} steps, "
          f"FP32, lr={args.lr} (backbone {args.lr*0.1})")
    print(f"{'step':>5} {'total':>9} {'loss_cls':>9} {'loss_bbox':>9} {'gnorm':>8}")

    for step in range(args.steps):
        img, metas, gt_box, gt_labels = samples[step % len(samples)]
        losses = model.forward_train(
            img=img.to(device), img_metas=metas,
            gt_bboxes_3d=[gt_box], gt_labels_3d=[gt_labels])
        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 35)
        optimizer.step()
        print(f"{step:5d} {float(loss):9.4f} {float(losses['loss_cls']):9.4f} "
              f"{float(losses['loss_bbox']):9.4f} {float(gnorm):8.2f}")

    ckpt_path = os.path.join(args.out, "tiny_petr.pth")
    torch.save({"state_dict": model.state_dict(),
                "meta": dict(note="tiny smoke-training job",
                             steps=args.steps)}, ckpt_path)
    print(f"\nSaved checkpoint -> {ckpt_path}")
    print("Done. (Full training: use the original mmdet3d 0.17 stack + "
          "tools/dist_train.sh -- see TRAINING_GUIDE.md.)")


if __name__ == "__main__":
    main()
