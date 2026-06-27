"""Study tool: trace tensor shapes through the PETR forward pass.

Registers forward hooks on the key modules and runs one nuScenes mini sample so
you can *see* how data flows: images -> backbone -> neck -> 3D position encoder
-> transformer decoder -> classification / regression branches.

Run:  python3 study/trace_shapes.py
"""
import os
import sys

# make petr_port/ importable (compat.py, petr_infer.py live there)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import compat  # noqa: E402  (installs shims first)
import torch  # noqa: E402
import mmengine  # noqa: E402
import petr_infer as P  # noqa: E402


def shape(x):
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)):
        return [shape(i) for i in x]
    if isinstance(x, dict):
        return {k: shape(v) for k, v in x.items()}
    return type(x).__name__


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = os.path.join(compat.REPO_ROOT, "ckpts/petr_vovnet_p4_800x320.pth")
    model = P.import_and_build(ckpt).to(device)
    head = model.pts_bbox_head

    logs = []

    def hook(name):
        def fn(module, inp, out):
            logs.append((name, shape(inp[0]) if inp else None, shape(out)))
        return fn

    handles = [
        model.img_backbone.register_forward_hook(hook("img_backbone")),
        model.img_neck.register_forward_hook(hook("img_neck (CPFPN)")),
        head.input_proj.register_forward_hook(hook("head.input_proj 1x1")),
        head.position_encoder.register_forward_hook(
            hook("head.position_encoder (3D PE MLP)")),
        head.positional_encoding.register_forward_hook(
            hook("head.positional_encoding (sine 3D)")),
        head.adapt_pos3d.register_forward_hook(hook("head.adapt_pos3d")),
        head.query_embedding.register_forward_hook(
            hook("head.query_embedding")),
        head.transformer.register_forward_hook(hook("head.transformer")),
        head.cls_branches[0].register_forward_hook(hook("head.cls_branch[0]")),
        head.reg_branches[0].register_forward_hook(hook("head.reg_branch[0]")),
    ]

    infos = mmengine.load(os.path.join(
        compat.REPO_ROOT, "data/nuscenes/nuscenes_infos_val.pkl"))["infos"]
    img, img_metas = P.preprocess_sample(infos[0], os.path.join(
        compat.REPO_ROOT, "data/nuscenes"))
    print(f"\nINPUT  img tensor          : {tuple(img.shape)}  "
          f"(B, N_cams, C, H, W)")
    print(f"INPUT  lidar2img (per cam) : {len(img_metas[0]['lidar2img'])} x "
          f"{img_metas[0]['lidar2img'][0].shape}\n")

    with torch.no_grad():
        result = model.simple_test(img_metas, img.to(device))

    print(f"{'module':<34}{'input':<26}output")
    print("-" * 90)
    for name, i, o in logs:
        print(f"{name:<34}{str(i):<26}{o}")

    det = result[0]["pts_bbox"]
    print("\nDECODED OUTPUT (after NMSFreeCoder):")
    print(f"  boxes_3d : {tuple(det['boxes_3d'].tensor.shape)}  "
          f"(x,y,z,w,l,h,yaw,vx,vy)")
    print(f"  scores_3d: {tuple(det['scores_3d'].shape)}")
    print(f"  labels_3d: {tuple(det['labels_3d'].shape)}")
    print(f"  #boxes with score>0.3: "
          f"{int((det['scores_3d'] > 0.3).sum())}")

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
