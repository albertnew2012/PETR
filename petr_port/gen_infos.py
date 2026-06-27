"""Generate nuScenes info .pkl files for the v1.0-mini split.

Reuses PETR's exact converter (tools/data_converter/nuscenes_converter.py) so
the per-camera calibration / lidar2img math is identical to training. The
converter's only mmdet3d dependencies (points_cam2img, NuScenesDataset.
NameMapping) are provided by compat.py.
"""
import argparse
import importlib.util
import os

import compat  # noqa: F401  (installs shims first)


def load_converter():
    path = os.path.join(compat.REPO_ROOT, "tools", "data_converter",
                        "nuscenes_converter.py")
    spec = importlib.util.spec_from_file_location("nuscenes_converter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(compat.REPO_ROOT,
                                                   "data", "nuscenes"))
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--max-sweeps", type=int, default=10)
    args = ap.parse_args()

    conv = load_converter()
    print(f"Generating infos: root={args.root} version={args.version}")
    conv.create_nuscenes_infos(args.root, "nuscenes", version=args.version,
                               max_sweeps=args.max_sweeps)
    for split in ("train", "val"):
        p = os.path.join(args.root, f"nuscenes_infos_{split}.pkl")
        print(f"  {split}: {'OK ' + p if os.path.exists(p) else 'MISSING'}")


if __name__ == "__main__":
    main()
