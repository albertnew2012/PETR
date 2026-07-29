"""Render saved PETRv2 detections on the six current camera images.

``results_nusc.json`` stores boxes in the nuScenes global frame. This script
transforms them back through global -> ego -> LiDAR before applying the
current-frame ``lidar2img`` matrices.
"""
import argparse
import json
import os

import cv2
import mmengine
import numpy as np
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion

import compat
import petr_infer as P


EDGES = [(i, (i + 1) % 4) for i in range(4)] + \
        [(i + 4, (i + 1) % 4 + 4) for i in range(4)] + \
        [(i, i + 4) for i in range(4)]
PRED_COLOR = (255, 0, 0)  # Blue in OpenCV BGR.
GT_COLOR = (0, 255, 0)    # Green in OpenCV BGR.
MONTAGE = [["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
           ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]]


def global_anno_to_lidar_box(anno, info):
    """Convert one nuScenes result annotation from global to LiDAR frame."""
    box = NuScenesBox(
        center=anno["translation"],
        size=anno["size"],
        orientation=Quaternion(anno["rotation"]),
        score=anno["detection_score"],
        velocity=(*anno.get("velocity", [0.0, 0.0]), 0.0),
    )

    # Inverse of PETR's LiDAR -> ego -> global conversion.
    box.translate(-np.asarray(info["ego2global_translation"]))
    box.rotate(Quaternion(info["ego2global_rotation"]).inverse)
    box.translate(-np.asarray(info["lidar2ego_translation"]))
    box.rotate(Quaternion(info["lidar2ego_rotation"]).inverse)
    return box


def draw_box_on_image(image, corners, lidar2img, color, label):
    """Project a LiDAR-frame box and return whether it was visible/drawn."""
    points = np.concatenate([corners, np.ones((1, 8))], axis=0)
    camera_points = lidar2img @ points
    depths = camera_points[2]
    if np.any(depths <= 0.1):
        return False

    pixels = camera_points[:2] / depths[None, :]
    height, width = image.shape[:2]
    if (pixels[0].max() < 0 or pixels[0].min() >= width or
            pixels[1].max() < 0 or pixels[1].min() >= height):
        return False

    pixels = pixels.astype(np.int32)
    for start, end in EDGES:
        cv2.line(image, tuple(pixels[:, start]), tuple(pixels[:, end]),
                 color, 2, cv2.LINE_AA)
    center = pixels.mean(axis=1).astype(np.int32)
    cv2.putText(image, label, tuple(center), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)
    return True


def corners_at_camera_time(box, time_delta, compensate_motion):
    """Return corners at camera time using constant-velocity compensation."""
    if not compensate_motion:
        return box.corners()
    velocity = np.nan_to_num(np.asarray(box.velocity), nan=0.0)
    center_at_camera_time = box.center + velocity * time_delta
    shifted = NuScenesBox(center_at_camera_time, box.wlh, box.orientation)
    return shifted.corners()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--data-root", default=os.path.join(
        compat.REPO_ROOT, "data/nuscenes"))
    parser.add_argument("--result-dir", default=os.path.join(
        compat.REPO_ROOT, "work_dirs/petrv2_vov_mini"))
    parser.add_argument("--model-name", default="PETRv2",
                        help="model label and output filename prefix")
    parser.add_argument("--output-name", default=None,
                        help="explicit unique output image filename")
    parser.add_argument("--blackout-cams", nargs="*", choices=P.CAMERAS,
                        default=[], help="render these camera panels black")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--gt", action="store_true",
                        help="overlay official GT boxes in green")
    parser.add_argument(
        "--no-motion-compensation", action="store_true",
        help="show boxes at LiDAR time instead of each camera capture time")
    parser.add_argument("--out-dir", default=os.path.join(
        compat.REPO_ROOT, "work_dirs/study"))
    args = parser.parse_args()

    results_path = os.path.join(args.result_dir, "results_nusc.json")
    if not os.path.isfile(results_path):
        raise FileNotFoundError(
            f"{results_path} does not exist; run inference first")

    infos = mmengine.load(
        os.path.join(args.data_root, "nuscenes_infos_val.pkl"))["infos"]
    if not 0 <= args.index < len(infos):
        raise IndexError(f"--index must be in [0, {len(infos) - 1}]")
    info = infos[args.index]

    with open(results_path, encoding="utf-8") as result_file:
        result_by_token = json.load(result_file)["results"]
    annos = [anno for anno in result_by_token.get(info["token"], [])
             if anno["detection_score"] >= args.score_thr]
    boxes = [(global_anno_to_lidar_box(anno, info), anno) for anno in annos]
    print(f"Sample {args.index} ({info['token']}): {len(boxes)} detections "
          f"(score >= {args.score_thr})")

    gt_boxes = []
    if args.gt:
        from nuscenes import NuScenes
        nusc = NuScenes(version="v1.0-mini", dataroot=args.data_root,
                        verbose=False)
        _, gt_boxes, _ = nusc.get_sample_data(info["lidar_token"])
        print(f"Official ground truth: {len(gt_boxes)} boxes (green)")

    paths, lidar2img, _, _ = P.build_cam_matrices(info)
    camera_index = {camera: index for index, camera in enumerate(P.CAMERAS)}
    panels = {}
    for camera, index in camera_index.items():
        path = paths[index] if os.path.isabs(paths[index]) else os.path.join(
            args.data_root, paths[index])
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Could not read camera image: {path}")
        if camera in args.blackout_cams:
            image[:] = 0

        # nuScenes cameras are captured asynchronously around the LiDAR
        # keyframe. Move dynamic boxes to this camera's timestamp so their
        # projections line up with the photographed objects.
        time_delta = ((info["cams"][camera]["timestamp"] -
                       info["timestamp"]) / 1e6)
        compensate_motion = not args.no_motion_compensation

        gt_drawn = 0
        for gt_box in gt_boxes:
            gt_drawn += int(draw_box_on_image(
                image, corners_at_camera_time(
                    gt_box, time_delta, compensate_motion), lidar2img[index],
                GT_COLOR, "GT"))

        drawn = 0
        for box, anno in boxes:
            name = anno["detection_name"]
            label = f"{name}:{anno['detection_score']:.2f}"
            drawn += int(draw_box_on_image(
                image, corners_at_camera_time(
                    box, time_delta, compensate_motion), lidar2img[index],
                PRED_COLOR, label))
        count_text = f"{args.model_name} {camera} (pred {drawn}"
        if args.gt:
            count_text += f", GT {gt_drawn}"
        count_text += ")"
        cv2.putText(image, count_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3,
                    cv2.LINE_AA)
        if args.gt:
            cv2.putText(image,
                        "blue=detection  green=GT",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)
        panels[camera] = image

    rows = [np.concatenate([cv2.resize(panels[camera], (800, 450))
                            for camera in row], axis=1)
            for row in MONTAGE]
    montage = np.concatenate(rows, axis=0)
    os.makedirs(args.out_dir, exist_ok=True)
    output_prefix = args.model_name.lower().replace(" ", "")
    output_name = args.output_name or \
        f"{output_prefix}_boxes_{args.index}.jpg"
    output_path = os.path.join(args.out_dir, output_name)
    if not cv2.imwrite(output_path, montage):
        raise OSError(f"Failed to write {output_path}")
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
