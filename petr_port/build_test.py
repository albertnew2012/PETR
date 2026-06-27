"""Build the PETR VoVNet model on the modern stack and load the checkpoint."""
import argparse
import os
import compat  # noqa: F401  (must be first: installs shims)
import torch
from mmdet.registry import MODELS

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]


def import_petr_modules():
    # Importing these registers the components into the MODELS / TASK_UTILS
    # registries (decorators run at import time).
    import projects.mmdet3d_plugin.models.backbones.vovnetcp  # noqa: F401
    import projects.mmdet3d_plugin.models.necks.cp_fpn  # noqa: F401
    import projects.mmdet3d_plugin.models.utils.positional_encoding  # noqa: F401
    import projects.mmdet3d_plugin.models.utils.petr_transformer  # noqa: F401
    import projects.mmdet3d_plugin.core.bbox.coders.nms_free_coder  # noqa: F401
    import projects.mmdet3d_plugin.models.dense_heads.petr_head as _ph  # noqa
    import projects.mmdet3d_plugin.models.detectors.petr3d  # noqa: F401

    # PETR heads were written for mmdet 2.x and don't implement the abstract
    # `loss_by_feat`/`predict_by_feat` of mmdet 3.x BaseDenseHead. We only need
    # inference (forward + get_bboxes), so make the class concrete.
    _ph.PETRHead.__abstractmethods__ = frozenset()


def build_model():
    model = dict(
        type="Petr3D",
        use_grid_mask=True,
        img_backbone=dict(
            type="VoVNetCP", spec_name="V-99-eSE", norm_eval=True,
            frozen_stages=-1, input_ch=3, out_features=("stage4", "stage5")),
        img_neck=dict(
            type="CPFPN", in_channels=[768, 1024], out_channels=256, num_outs=2),
        pts_bbox_head=dict(
            type="PETRHead", num_classes=10, in_channels=256, num_query=900,
            LID=True, with_position=True, with_multiview=True,
            position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            normedlinear=False,
            transformer=dict(
                type="PETRTransformer",
                decoder=dict(
                    type="PETRTransformerDecoder",
                    return_intermediate=True, num_layers=6,
                    transformerlayers=dict(
                        type="PETRTransformerDecoderLayer",
                        attn_cfgs=[
                            dict(type="MultiheadAttention", embed_dims=256,
                                 num_heads=8, dropout=0.1),
                            dict(type="PETRMultiheadAttention", embed_dims=256,
                                 num_heads=8, dropout=0.1),
                        ],
                        feedforward_channels=2048, ffn_dropout=0.1,
                        operation_order=("self_attn", "norm", "cross_attn",
                                         "norm", "ffn", "norm")))),
            bbox_coder=dict(
                type="NMSFreeCoder",
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                pc_range=point_cloud_range, max_num=300, voxel_size=voxel_size,
                num_classes=10),
            positional_encoding=dict(
                type="SinePositionalEncoding3D", num_feats=128, normalize=True),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0,
                          alpha=0.25, loss_weight=2.0),
            loss_bbox=dict(type="L1Loss", loss_weight=0.25),
            loss_iou=dict(type="GIoULoss", loss_weight=0.0)),
        train_cfg=None, test_cfg=None, pretrained=None)
    return MODELS.build(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default=os.path.join(compat.REPO_ROOT,
                                         "ckpts/petr_vovnet_p4_800x320.pth"))
    args = ap.parse_args()

    import_petr_modules()
    print("[1] modules imported & registered")
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2] model built: {type(model).__name__}, "
          f"{n_params/1e6:.1f}M params")

    from mmengine.runner import load_checkpoint
    ck = load_checkpoint(model, args.ckpt, map_location="cpu", strict=False)
    print("[3] checkpoint loaded")

    # Manual diff of keys
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    model_keys = set(model.state_dict().keys())
    ck_keys = set(sd.keys())
    missing = model_keys - ck_keys
    unexpected = ck_keys - model_keys
    print(f"[4] state_dict: model={len(model_keys)} ckpt={len(ck_keys)} "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("    MISSING (first 15):")
        for k in list(sorted(missing))[:15]:
            print("      ", k)
    if unexpected:
        print("    UNEXPECTED (first 15):")
        for k in list(sorted(unexpected))[:15]:
            print("      ", k)
    print("OK build_test done")


if __name__ == "__main__":
    main()
