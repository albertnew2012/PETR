"""
compat.py - Compatibility shim to run the original PETR codebase
(written for mmcv 1.4 / mmdet 2.24 / mmdet3d 0.17 era) on a modern stack:
    Python 3.12 + torch 2.9 + mmengine 0.10 + mmcv 2.1 (lite) + mmdet 3.2.

Import this module FIRST (before importing any PETR module). It injects
legacy module paths into sys.modules and patches modern packages so that the
unmodified PETR source files import and build correctly.

Strategy:
  * Register every PETR component into mmdet's MODELS / TASK_UTILS registries
    and build under default scope 'mmdet'.
  * mmcv 2.1 still ships the transformer "bricks" (BaseTransformerLayer, FFN,
    MultiheadAttention, ...) on a unified MODELS registry, so we only recreate
    the legacy registry *names*.
  * mmdet3d is NOT installed; we vendor a minimal LiDARInstance3DBoxes (0.17
    semantics) and a minimal MVXTwoStageDetector base so PETR's detector works.
"""
import os
import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 0. Stub the compiled mmcv ops BEFORE importing any mmdet model layer.
#    PETR's VoVNet path never *calls* these ops (no DCN, NMS-free), so a stub
#    is runtime-safe and avoids compiling mmcv against bleeding-edge torch.
# ---------------------------------------------------------------------------
sys.modules.setdefault("mmcv._ext", MagicMock())

import torch  # noqa: E402
import numpy as np  # noqa: E402
import mmcv  # noqa: E402
import mmengine  # noqa: E402
from mmengine.model import BaseModule, ModuleList  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from mmengine.registry import Registry as _Registry  # noqa: E402
from mmdet.registry import MODELS, TASK_UTILS  # noqa: E402

# PETR re-uses some type names that already exist in mmdet/mmcv (e.g.
# SinePositionalEncoding3D). Allow later registrations to override so the
# PETR definitions win instead of raising "already registered".
_orig_register_module = _Registry._register_module


def _register_module_force(self, module, module_name=None, force=False):
    return _orig_register_module(self, module, module_name=module_name,
                                 force=True)


_Registry._register_module = _register_module_force


def _mod(name):
    """Get or create a module object registered in sys.modules."""
    m = sys.modules.get(name)
    if m is None:
        m = types.ModuleType(name)
        sys.modules[name] = m
    return m


# ---------------------------------------------------------------------------
# 1. no-op fp16 decorators (force_fp32 / auto_fp16)
# ---------------------------------------------------------------------------
def _noop_decorator(*dargs, **dkwargs):
    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        return dargs[0]

    def deco(fn):
        return fn

    return deco


force_fp32 = _noop_decorator
auto_fp16 = _noop_decorator

# ---------------------------------------------------------------------------
# 2. mmcv.runner  (BaseModule, fp16 decorators, load_checkpoint)
# ---------------------------------------------------------------------------
from mmengine.runner import load_checkpoint as _load_checkpoint  # noqa: E402

_runner = _mod("mmcv.runner")
_runner.BaseModule = BaseModule
_runner.ModuleList = ModuleList
_runner.force_fp32 = force_fp32
_runner.auto_fp16 = auto_fp16
_runner.load_checkpoint = _load_checkpoint

_base_module = _mod("mmcv.runner.base_module")
_base_module.BaseModule = BaseModule
_base_module.ModuleList = ModuleList
_runner.base_module = _base_module

# ---------------------------------------------------------------------------
# 3. mmcv.parallel.DataContainer  (minimal)
# ---------------------------------------------------------------------------
_parallel = _mod("mmcv.parallel")


class DataContainer:
    def __init__(self, data, stack=False, padding_value=0, cpu_only=False,
                 pad_dims=2):
        self._data = data
        self.stack = stack
        self.padding_value = padding_value
        self.cpu_only = cpu_only
        self.pad_dims = pad_dims

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        return type(self._data)

    def __repr__(self):
        return f"DataContainer({self._data!r})"


_parallel.DataContainer = DataContainer

# ---------------------------------------------------------------------------
# 4. mmcv.utils additions (ConfigDict, build_from_cfg, to_2tuple, ...)
# ---------------------------------------------------------------------------
import mmcv.utils as _mmcv_utils  # noqa: E402
from mmengine.config import ConfigDict as _ConfigDict  # noqa: E402
from mmengine.registry import build_from_cfg as _build_from_cfg  # noqa: E402

if not hasattr(_mmcv_utils, "ConfigDict"):
    _mmcv_utils.ConfigDict = _ConfigDict
if not hasattr(_mmcv_utils, "build_from_cfg"):
    _mmcv_utils.build_from_cfg = _build_from_cfg
if not hasattr(_mmcv_utils, "to_2tuple"):
    try:
        from mmengine.utils import to_2tuple as _to_2tuple
    except Exception:
        import collections.abc as _cabc
        from itertools import repeat as _repeat

        def _to_2tuple(x):
            if isinstance(x, _cabc.Iterable):
                return tuple(x)
            return tuple(_repeat(x, 2))

    _mmcv_utils.to_2tuple = _to_2tuple
if not hasattr(_mmcv_utils, "deprecated_api_warning"):
    try:
        from mmengine.utils import deprecated_api_warning as _daw
    except Exception:
        def _daw(name_dict, cls_name=None):
            def deco(fn):
                return fn
            return deco
    _mmcv_utils.deprecated_api_warning = _daw

# mmcv 2.x moved IO / progress / path helpers to mmengine. Re-expose the few
# the original PETR data-converter and dataset code call as `mmcv.*`.
import mmengine as _mmengine_io  # noqa: E402


def _alias_mmcv(attr, value):
    if not hasattr(mmcv, attr):
        setattr(mmcv, attr, value)


try:
    from mmengine.fileio import dump as _dump, load as _load, \
        list_from_file as _list_from_file
    _alias_mmcv("dump", _dump)
    _alias_mmcv("load", _load)
    _alias_mmcv("list_from_file", _list_from_file)
except Exception:
    pass
try:
    from mmengine.utils import (mkdir_or_exist as _mkdir, scandir as _scandir,
                                track_iter_progress as _track,
                                check_file_exist as _cfe, is_filepath as _isfp)
    _alias_mmcv("mkdir_or_exist", _mkdir)
    _alias_mmcv("scandir", _scandir)
    _alias_mmcv("track_iter_progress", _track)
    _alias_mmcv("check_file_exist", _cfe)
    _alias_mmcv("is_filepath", _isfp)
except Exception:
    pass

# ---------------------------------------------------------------------------
# 5. mmcv.cnn.bricks.registry  +  transformer additions
# ---------------------------------------------------------------------------
_reg = _mod("mmcv.cnn.bricks.registry")
_reg.ATTENTION = MODELS
_reg.TRANSFORMER_LAYER = MODELS
_reg.TRANSFORMER_LAYER_SEQUENCE = MODELS
_reg.POSITIONAL_ENCODING = MODELS
_reg.FEEDFORWARD_NETWORK = MODELS

import mmcv.cnn.bricks.transformer as _tf  # noqa: E402

if not hasattr(_tf, "POSITIONAL_ENCODING"):
    _tf.POSITIONAL_ENCODING = MODELS
if not hasattr(_tf, "build_positional_encoding"):
    def build_positional_encoding(cfg, default_args=None):
        return MODELS.build(cfg, default_args=default_args)

    _tf.build_positional_encoding = build_positional_encoding

# mmcv 2.x moved the weight-init helpers to mmengine.model; re-export them on
# mmcv.cnn so legacy `from mmcv.cnn import xavier_init, ...` keeps working.
import mmcv.cnn as _mmcv_cnn  # noqa: E402
import mmengine.model as _mm_model  # noqa: E402

for _nm in ("xavier_init", "constant_init", "kaiming_init", "normal_init",
            "uniform_init", "trunc_normal_init", "bias_init_with_prob",
            "caffe2_xavier_init"):
    if not hasattr(_mmcv_cnn, _nm) and hasattr(_mm_model, _nm):
        setattr(_mmcv_cnn, _nm, getattr(_mm_model, _nm))

# ---------------------------------------------------------------------------
# 6. mmdet.core  (and a couple of submodules)
# ---------------------------------------------------------------------------
from mmdet.models.utils import multi_apply  # noqa: E402
from mmdet.utils import reduce_mean  # noqa: E402
from mmdet.structures.bbox import (  # noqa: E402
    bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh)
from mmdet.models.task_modules.builder import (  # noqa: E402
    build_assigner, build_sampler)
from mmdet.models.task_modules.coders import BaseBBoxCoder  # noqa: E402
from mmdet.models.layers import inverse_sigmoid, NormedLinear  # noqa: E402

_core = _mod("mmdet.core")
_core.multi_apply = multi_apply
_core.reduce_mean = reduce_mean
_core.bbox_cxcywh_to_xyxy = bbox_cxcywh_to_xyxy
_core.bbox_xyxy_to_cxcywh = bbox_xyxy_to_cxcywh
_core.build_assigner = build_assigner
_core.build_sampler = build_sampler

_core_bbox = _mod("mmdet.core.bbox")
_core_bbox.BaseBBoxCoder = BaseBBoxCoder
_core_bbox.build_assigner = build_assigner
_core_bbox.build_sampler = build_sampler
_core_bbox.bbox_cxcywh_to_xyxy = bbox_cxcywh_to_xyxy
_core_bbox.bbox_xyxy_to_cxcywh = bbox_xyxy_to_cxcywh
_core.bbox = _core_bbox

_core_bbox_builder = _mod("mmdet.core.bbox.builder")
_core_bbox_builder.BBOX_CODERS = TASK_UTILS
_core_bbox_builder.build_assigner = build_assigner
_core_bbox_builder.build_sampler = build_sampler
_core_bbox.builder = _core_bbox_builder

# --- training extras: assigners / samplers / match costs / iou calculators ---
# (needed only when building the head with train_cfg, e.g. the loss demo)
from mmdet.models.task_modules.assigners import (  # noqa: E402
    AssignResult, BaseAssigner)
from mmdet.structures.bbox import bbox_overlaps  # noqa: E402

_core_bbox_builder.BBOX_ASSIGNERS = TASK_UTILS
_core_bbox_builder.BBOX_SAMPLERS = TASK_UTILS
_core_bbox.BBOX_ASSIGNERS = TASK_UTILS
_core_bbox.BBOX_SAMPLERS = TASK_UTILS

_assigners = _mod("mmdet.core.bbox.assigners")
_assigners.AssignResult = AssignResult
_assigners.BaseAssigner = BaseAssigner
_core_bbox.assigners = _assigners

_match_costs = _mod("mmdet.core.bbox.match_costs")
_match_costs.build_match_cost = (
    lambda cfg, default_args=None: TASK_UTILS.build(cfg, default_args=default_args))
_match_costs_builder = _mod("mmdet.core.bbox.match_costs.builder")
_match_costs_builder.MATCH_COST = TASK_UTILS
_match_costs.builder = _match_costs_builder
_core_bbox.match_costs = _match_costs

_iou_calc = _mod("mmdet.core.bbox.iou_calculators")
_iou_calc.bbox_overlaps = bbox_overlaps
_core_bbox.iou_calculators = _iou_calc


# mmdet 3.x rewrote match costs to the InstanceData API (cost(pred_instances,
# gt_instances)). PETR's HungarianAssigner3D calls them the old way
# cost(cls_pred_tensor, gt_labels). Register old-signature versions so the
# unmodified PETR assigner works. (BBox3DL1Cost is PETR's own and already OK.)
class _FocalLossCostOld:
    def __init__(self, weight=1., alpha=0.25, gamma=2, eps=1e-12,
                 binary_input=False):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels):
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight


class _IoUCostOld:
    def __init__(self, iou_mode="giou", weight=1.):
        self.iou_mode = iou_mode
        self.weight = weight

    def __call__(self, bboxes, gt_bboxes):
        overlaps = bbox_overlaps(
            bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)
        return -overlaps * self.weight


TASK_UTILS.register_module(name="FocalLossCost", module=_FocalLossCostOld,
                           force=True)
TASK_UTILS.register_module(name="IoUCost", module=_IoUCostOld, force=True)


# mmdet 3.x PseudoSampler also uses the InstanceData API. PETR calls
# sampler.sample(assign_result, bbox_pred_tensor, gt_bboxes_tensor). Provide the
# old-style sampler + a minimal SamplingResult with the attributes PETR reads.
class _SamplingResult:
    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]
        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = (assign_result.gt_inds[pos_inds] - 1).long()
        if gt_bboxes.numel() == 0:
            self.pos_gt_bboxes = gt_bboxes.new_zeros((0, gt_bboxes.size(-1)))
        else:
            gtb = gt_bboxes if gt_bboxes.dim() >= 2 else gt_bboxes.view(-1, 4)
            self.pos_gt_bboxes = gtb[self.pos_assigned_gt_inds, :]


class _PseudoSampler:
    def __init__(self, **kwargs):
        pass

    def sample(self, assign_result, bboxes, gt_bboxes, *args, **kwargs):
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        return _SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                               assign_result, gt_flags)


TASK_UTILS.register_module(name="PseudoSampler", module=_PseudoSampler,
                           force=True)

# ---------------------------------------------------------------------------
# 7. mmdet.models legacy registry names + build_loss + utils helpers
# ---------------------------------------------------------------------------
import mmdet.models as _mmdet_models  # noqa: E402

_mmdet_models.DETECTORS = MODELS
_mmdet_models.HEADS = MODELS
_mmdet_models.BACKBONES = MODELS
_mmdet_models.NECKS = MODELS
_mmdet_models.build_loss = lambda cfg, default_args=None: MODELS.build(
    cfg, default_args=default_args)

# FocalLoss(use_sigmoid=True) dispatches to the compiled mmcv op
# `sigmoid_focal_loss` on CUDA tensors -- but we stubbed mmcv._ext, so that
# would silently yield NaN (-> 0 after nan_to_num). Replace it with a wrapper
# around the pure-python focal loss that also accepts class-index targets
# (the compiled op accepted indices; the python one needs one-hot).
try:
    import torch.nn.functional as _F  # noqa: E402
    import mmdet.models.losses.focal_loss as _focal_loss_mod  # noqa: E402

    def _sigmoid_focal_loss_compat(pred, target, weight=None, gamma=2.0,
                                   alpha=0.25, reduction="mean",
                                   avg_factor=None):
        if target.dim() != pred.dim():  # class indices -> one-hot
            num_classes = pred.size(1)
            target = _F.one_hot(target.long(), num_classes=num_classes + 1)
            target = target[:, :num_classes].type_as(pred)
        return _focal_loss_mod.py_sigmoid_focal_loss(
            pred, target, weight=weight, gamma=gamma, alpha=alpha,
            reduction=reduction, avg_factor=avg_factor)

    _focal_loss_mod.sigmoid_focal_loss = _sigmoid_focal_loss_compat
except Exception:
    pass

# mmdet 3.x removed mmdet.models.builder -> recreate it with legacy names.
_mmdet_builder = _mod("mmdet.models.builder")
for _nm in ("BACKBONES", "NECKS", "HEADS", "DETECTORS", "ROI_EXTRACTORS",
            "SHARED_HEADS", "LOSSES"):
    setattr(_mmdet_builder, _nm, MODELS)
_mmdet_builder.build_loss = _mmdet_models.build_loss
_mmdet_builder.build_backbone = (
    lambda cfg: MODELS.build(cfg))
_mmdet_builder.build_neck = (lambda cfg: MODELS.build(cfg))
_mmdet_builder.build_head = (lambda cfg: MODELS.build(cfg))
_mmdet_models.builder = _mmdet_builder

import mmdet.models.utils as _mmdet_utils  # noqa: E402

if not hasattr(_mmdet_utils, "build_transformer"):
    _mmdet_utils.build_transformer = (
        lambda cfg, default_args=None: MODELS.build(cfg, default_args=default_args))
if not hasattr(_mmdet_utils, "NormedLinear"):
    _mmdet_utils.NormedLinear = NormedLinear

_mu_builder = _mod("mmdet.models.utils.builder")
_mu_builder.TRANSFORMER = MODELS
_mu_builder.build_transformer = _mmdet_utils.build_transformer
_mmdet_utils.builder = _mu_builder

_mu_transformer = _mod("mmdet.models.utils.transformer")
_mu_transformer.inverse_sigmoid = inverse_sigmoid
_mmdet_utils.transformer = _mu_transformer

# ---------------------------------------------------------------------------
# 8. mmdet3d shims (NOT installed): minimal box structures + detector base
# ---------------------------------------------------------------------------
class LiDARInstance3DBoxes:
    """Minimal re-implementation matching mmdet3d 0.17 LiDAR box semantics.

    tensor layout: [x, y, z(bottom), w, l, h, yaw, (vx, vy)]
    """

    def __init__(self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0)):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32)
        tensor = tensor.float()
        if tensor.numel() == 0:
            tensor = tensor.reshape(0, box_dim)
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.shape
        self.tensor = tensor.clone()
        self.box_dim = box_dim
        self.with_yaw = with_yaw

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, item):
        return self.tensor[item]

    @property
    def dims(self):
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        return self.tensor[:, 6]

    @property
    def bottom_center(self):
        return self.tensor[:, :3]

    @property
    def gravity_center(self):
        bottom = self.tensor[:, :3]
        g = bottom.clone()
        g[:, 2] = bottom[:, 2] + self.tensor[:, 5] * 0.5
        return g

    def to(self, device):
        self.tensor = self.tensor.to(device)
        return self

    def cpu(self):
        return self.to("cpu")

    def numpy(self):
        return self.tensor.numpy()


class CameraInstance3DBoxes(LiDARInstance3DBoxes):
    pass


def bbox3d2result(bboxes, scores, labels, attrs=None):
    result = dict(
        boxes_3d=bboxes.to("cpu") if hasattr(bboxes, "to") else bboxes,
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu(),
    )
    if attrs is not None:
        result["attrs_3d"] = attrs.cpu()
    return result


def show_multi_modality_result(*args, **kwargs):
    pass


_m3d = _mod("mmdet3d")
_m3d.__version__ = "0.17.1-compat"

_m3d_core = _mod("mmdet3d.core")
_m3d_core.bbox3d2result = bbox3d2result
_m3d_core.LiDARInstance3DBoxes = LiDARInstance3DBoxes
_m3d_core.CameraInstance3DBoxes = CameraInstance3DBoxes
_m3d_core.show_multi_modality_result = show_multi_modality_result
_m3d.core = _m3d_core

_m3d_core_bbox = _mod("mmdet3d.core.bbox")
_m3d_core_bbox.LiDARInstance3DBoxes = LiDARInstance3DBoxes
_m3d_core_bbox.CameraInstance3DBoxes = CameraInstance3DBoxes
_m3d_core.bbox = _m3d_core_bbox

_m3d_coders = _mod("mmdet3d.core.bbox.coders")
_m3d_coders.build_bbox_coder = (
    lambda cfg, default_args=None: TASK_UTILS.build(cfg, default_args=default_args))
_m3d_core_bbox.coders = _m3d_coders

_m3d_points = _mod("mmdet3d.core.points")


class BasePoints:
    pass


def get_points_type(points_type):
    return BasePoints


_m3d_points.BasePoints = BasePoints
_m3d_points.get_points_type = get_points_type
_m3d_core.points = _m3d_points


class MVXTwoStageDetector(BaseModule):
    """Minimal base providing just what PETR's Petr3D needs."""

    def __init__(self, pts_voxel_layer=None, pts_voxel_encoder=None,
                 pts_middle_encoder=None, pts_fusion_layer=None,
                 img_backbone=None, pts_backbone=None, img_neck=None,
                 pts_neck=None, pts_bbox_head=None, img_roi_head=None,
                 img_rpn_head=None, train_cfg=None, test_cfg=None,
                 pretrained=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if img_backbone:
            self.img_backbone = MODELS.build(img_backbone)
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
        if pts_bbox_head:
            def _get(cfg, key):
                if cfg is None:
                    return None
                if isinstance(cfg, dict):
                    return cfg.get(key)
                return getattr(cfg, key, None)

            head_cfg = dict(pts_bbox_head)
            head_cfg.update(train_cfg=_get(train_cfg, "pts"))
            head_cfg.update(test_cfg=_get(test_cfg, "pts"))
            self.pts_bbox_head = MODELS.build(head_cfg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_img_backbone(self):
        return hasattr(self, "img_backbone") and self.img_backbone is not None

    @property
    def with_img_neck(self):
        return hasattr(self, "img_neck") and self.img_neck is not None

    @property
    def with_pts_bbox(self):
        return hasattr(self, "pts_bbox_head") and self.pts_bbox_head is not None


_m3d_models = _mod("mmdet3d.models")
_m3d_detectors = _mod("mmdet3d.models.detectors")
_m3d_mvx = _mod("mmdet3d.models.detectors.mvx_two_stage")
_m3d_mvx.MVXTwoStageDetector = MVXTwoStageDetector
_m3d_detectors.mvx_two_stage = _m3d_mvx
_m3d_detectors.MVXTwoStageDetector = MVXTwoStageDetector
_m3d_models.detectors = _m3d_detectors
_m3d.models = _m3d_models

# minimal mmdet3d.datasets stub (NameMapping used by the data converter)
_m3d_datasets = _mod("mmdet3d.datasets")


class _NuScenesDatasetStub:
    NameMapping = {
        "movable_object.barrier": "barrier",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.car": "car",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }


_m3d_datasets.NuScenesDataset = _NuScenesDatasetStub
_m3d.datasets = _m3d_datasets

# mmdet3d.core.bbox.box_np_ops.points_cam2img (only used by 2D-anno path)
_m3d_box_np_ops = _mod("mmdet3d.core.bbox.box_np_ops")


def points_cam2img(points_3d, proj_mat):  # pragma: no cover - unused path
    raise NotImplementedError("points_cam2img stub (unused in PETR eval)")


_m3d_box_np_ops.points_cam2img = points_cam2img
_m3d_core_bbox.box_np_ops = _m3d_box_np_ops

# ---------------------------------------------------------------------------
# 9. Make `projects.mmdet3d_plugin.*` importable WITHOUT executing the heavy
#    package __init__ files (which pull in datasets needing real mmdet3d).
#    We register namespace packages pointing at the real directories.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _namespace_pkg(name, path):
    m = sys.modules.get(name)
    if m is None:
        m = types.ModuleType(name)
        m.__path__ = [path]
        m.__package__ = name
        sys.modules[name] = m
    return m


_namespace_pkg("projects", os.path.join(REPO_ROOT, "projects"))
_namespace_pkg("projects.mmdet3d_plugin",
               os.path.join(REPO_ROOT, "projects", "mmdet3d_plugin"))
for _sub in (
    "core", "core.bbox", "core.bbox.coders", "core.bbox.assigners",
    "core.bbox.match_costs", "core.bbox.iou_calculators",
    "models", "models.detectors", "models.utils", "models.dense_heads",
    "models.backbones", "models.necks", "models.losses",
    "datasets", "datasets.pipelines",
):
    _namespace_pkg(
        "projects.mmdet3d_plugin." + _sub,
        os.path.join(REPO_ROOT, "projects", "mmdet3d_plugin", *_sub.split(".")),
    )

# Build everything under the mmdet scope so registry lookups resolve PETR +
# mmdet + mmcv components consistently.
init_default_scope("mmdet")

# Expose handy names
__all__ = ["MODELS", "TASK_UTILS", "LiDARInstance3DBoxes",
           "CameraInstance3DBoxes", "bbox3d2result", "DataContainer",
           "REPO_ROOT"]
