#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1-TaylorLike structured iterative pruning for RTDETRMM RGB/IR fusion models.

This script supports the two RTDETRMM families used in this project:
- PIAFusionBlock-based mid-fusion models
- CMXFusion-based RTMM variants such as A-DWConv-C3k2_PConv

Key design points:
- Reuses the active Taylor pruning pipeline structure for a fair comparison experiment.
- Uses torch-pruning MetaPruner/DepGraph with L1 magnitude importance for root pruning.
- Preserves fusion branch alignment by skipping the fusion input roots and pruning structurally safe
  backbone/neck outputs, including Conv/DWConv layers and CSP/C3k2-style block output convolutions.
- Ignores RTDETRDecoder and all DAttention-related modules during tracing/pruning.
- Keeps FLOPs bias and slimming interfaces available, but defaults them to neutral/off for fairness.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

SUPPORTED_FUSION_TYPES = {"PIAFusionBlock", "CMXFusion"}
CONV_LIKE_TYPES = {"Conv", "DWConv"}


def log(msg: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def parse_args() -> argparse.Namespace:
    pia_default = Path("ReTest") / "A-DWConv-C3k2_PConv" / "weights" / "best.pt"
    default_weights = pia_default if pia_default.exists() else Path("MODEL") / "pruned_finetune_best.pt"
    parser = argparse.ArgumentParser(description="L1-TaylorLike iterative pruning for RTDETRMM VIF models")
    parser.add_argument("--weights", type=str, default=str(default_weights), help="Input .pt checkpoint")
    parser.add_argument("--output", type=str, default="prune_outputs", help="Output .pt path or directory")
    parser.add_argument("--data", type=str, default=r"D:\BaiduNetdiskDownload\M3FD\M3FD_split\data.yaml", help="Dataset yaml for the printed resume command")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0, 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size")
    parser.add_argument("--prune-ratio", type=float, default=0.7, help="Target total pruning ratio")
    parser.add_argument("--iterations", type=int, default=8, help="Number of iterative pruning rounds")
    parser.add_argument("--grad-samples", type=int, default=2, help="Dummy backward passes per round")
    parser.add_argument("--min-channels", type=int, default=8, help="Minimum channels to keep for any root module")
    parser.add_argument("--round-to", type=int, default=8, help="Round pruned channels to multiples of N")
    parser.add_argument(
        "--include-neck",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include a curated set of safe neck layers in addition to the backbone roots.",
    )
    parser.add_argument(
        "--flops-bias",
        type=float,
        default=0.0,
        help="Bias pruning selection toward channels that save more FLOPs. Default 0.0 keeps the fair baseline score-only.",
    )
    parser.add_argument(
        "--layer-end",
        "--backbone-end",
        dest="layer_end",
        type=int,
        default=16,
        help="Exclusive layer end index for pruning roots. Default=16 keeps pruning focused on the backbone.",
    )
    parser.add_argument("--fps-warmup", type=int, default=10, help="Warmup iterations for FPS")
    parser.add_argument("--fps-iters", type=int, default=50, help="Benchmark iterations for FPS")
    parser.add_argument("--resume-epochs", type=int, default=30, help="Suggested epochs for the printed resume command")
    parser.add_argument("--resume-batch", type=int, default=1, help="Suggested batch size for the printed resume command")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--enable-internal-slimming",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Slim large module internals such as CSP_MutilScaleEdgeInformationEnhance, C3k2_PConv and DAttention FFN.",
    )
    parser.add_argument(
        "--decoder-keep-ratio",
        type=float,
        default=1.0,
        help="Keep ratio for RTDETRDecoder hidden width. Default 1.0 keeps decoder width unchanged for fairness.",
    )
    return parser.parse_args()


def ensure_torch_pruning():
    try:
        import torch_pruning as tp
    except ImportError as exc:  # pragma: no cover - direct runtime guidance
        raise SystemExit("torch-pruning is required. Install it in the MM environment first.") from exc
    return tp


def resolve_device(device: str) -> torch.device:
    ds = str(device).strip().lower()
    if ds == "cpu":
        return torch.device("cpu")
    if ds.isdigit():
        ds = f"cuda:{ds}"
    if ds.startswith("cuda") and torch.cuda.is_available():
        return torch.device(ds)
    if torch.cuda.is_available():
        log(f"设备 {device} 不可用，自动回退到 cuda:0")
        return torch.device("cuda:0")
    log("CUDA 不可用，自动回退到 CPU。FPS 将不代表 GPU 性能。")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_example_inputs(batch_size: int, imgsz: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    vis = torch.randn(batch_size, 3, imgsz, imgsz, device=device)
    ir = torch.randn(batch_size, 3, imgsz, imgsz, device=device)
    return vis, ir


def make_output_path(output_arg: str, ratio: float) -> Path:
    out = Path(output_arg)
    if out.suffix.lower() == ".pt":
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    ratio_str = f"{ratio:.2f}".rstrip("0").rstrip(".")
    timestamp = datetime.now().strftime("%m%d_%H%M")
    return out / f"pruned_l1_taylorlike_r{ratio_str}_{timestamp}.pt"


def remove_args_recursive(obj: object) -> None:
    if obj is None:
        return
    for module in obj.modules() if isinstance(obj, nn.Module) else []:
        if hasattr(module, "args"):
            with contextlib.suppress(Exception):
                delattr(module, "args")
    if hasattr(obj, "args"):
        with contextlib.suppress(Exception):
            delattr(obj, "args")


def sanitize_checkpoint_for_save(orig_ckpt: dict | None, model: nn.Module) -> dict:
    ckpt = copy.copy(orig_ckpt) if isinstance(orig_ckpt, dict) else {}
    ckpt.pop("train_args", None)
    for key in ("optimizer", "ema", "updates", "best_fitness"):
        ckpt.pop(key, None)
    save_model = copy.deepcopy(model).eval().half()
    if hasattr(save_model, "criterion"):
        save_model.criterion = None
    remove_args_recursive(save_model)
    for p in save_model.parameters():
        p.requires_grad_(False)
    ckpt["epoch"] = -1
    ckpt["model"] = save_model
    ckpt["pruned"] = True
    ckpt["date"] = datetime.now().isoformat()
    ckpt["prune_method"] = "l1_taylorlike"
    return ckpt


def load_model(weights: Path, device: torch.device):
    from ultralytics import RTDETRMM
    from ultralytics.nn.tasks import torch_safe_load

    if not weights.exists():
        raise FileNotFoundError(f"权重不存在: {weights}")

    log(f"加载模型: {weights}")
    wrapper = RTDETRMM(str(weights))
    model = wrapper.model.to(device).float().eval()
    for param in model.parameters():
        param.requires_grad_(True)
    ckpt = None
    if weights.suffix.lower() == ".pt":
        ckpt, _ = torch_safe_load(str(weights))
    return model, ckpt


def validate_expected_architecture(model: nn.Module) -> None:
    if len(model.model) <= 11:
        raise RuntimeError("当前模型层数不足，无法匹配 RT-DETR-r18 PIAFusionBlock 双模态结构。")
    fusion = model.model[10]
    fusion_type = type(fusion).__name__
    fusion_from = list(getattr(fusion, "f", [])) if isinstance(getattr(fusion, "f", None), (list, tuple)) else []
    compress_type = type(model.model[11]).__name__
    if fusion_type != "PIAFusionBlock" or fusion_from != [4, 9] or compress_type != "Conv":
        raise RuntimeError(
            f"当前模型不符合要求的 PIAFusionBlock 架构，检测到 layer10={fusion_type}, from={fusion_from}, layer11={compress_type}。"
        )


class ModelWrapper(nn.Module):
    """Expose the multimodal model as forward(vis, ir) for TP tracing."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.head = model.model[-1]
        self.old_export = getattr(self.head, "export", False)
        self.head.export = True

    def forward(self, vis: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        out = self.model(torch.cat([vis, ir], dim=1))
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

    def restore(self) -> None:
        self.head.export = self.old_export


@dataclass
class RootSpec:
    name: str
    layer_idx: int
    module: nn.Module
    pruning_fn: Callable
    mode: str = "out"


def find_ignored_layers(model: nn.Module) -> list[nn.Module]:
    ignored = []
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name == "RTDETRDecoder" or "DAttention" in cls_name:
            ignored.append(module)
    return ignored


def find_fusion_info(model: nn.Module) -> tuple[int | None, set[int]]:
    fusion_idx = None
    branch_tail_idxs: set[int] = set()
    for idx, layer in enumerate(model.model):
        if type(layer).__name__ == "PIAFusionBlock":
            fusion_idx = idx
            if isinstance(getattr(layer, "f", None), (list, tuple)):
                branch_tail_idxs.update(int(x) for x in layer.f if isinstance(x, int) and x >= 0)
            break
    if fusion_idx == 10 and not branch_tail_idxs:
        branch_tail_idxs.update({4, 9})
    return fusion_idx, branch_tail_idxs


def get_input_layer_indices(layer: nn.Module) -> list[int]:
    source = getattr(layer, "f", None)
    if isinstance(source, int):
        return [source] if source >= 0 else []
    if isinstance(source, (list, tuple)):
        return [int(x) for x in source if isinstance(x, int) and x >= 0]
    return []


def find_protected_layer_indices(model: nn.Module) -> set[int]:
    protected: set[int] = set()
    fusion_idx, branch_tail_idxs = find_fusion_info(model)
    if fusion_idx is not None:
        protected.add(fusion_idx)
        protected.update(branch_tail_idxs)
        next_idx = fusion_idx + 1
        if 0 <= next_idx < len(model.model) and type(model.model[next_idx]).__name__ == "Conv":
            protected.add(next_idx)
        for idx, layer in enumerate(model.model):
            if fusion_idx in get_input_layer_indices(layer) and type(layer).__name__ == "Conv":
                protected.add(idx)

    for idx, layer in enumerate(model.model):
        if type(layer).__name__ != "RTDETRDecoder":
            continue
        for src_idx in get_input_layer_indices(layer):
            if 0 <= src_idx < len(model.model):
                protected.add(src_idx)
    return protected


def get_candidate_root_names_for_layer(idx: int, layer: nn.Module, include_neck: bool) -> set[str]:
    cls_name = type(layer).__name__
    backbone_conv_layers = {0, 1, 3, 5, 6, 8, 12, 14}
    backbone_csp_layers = {2, 4, 7, 9, 13, 15}
    neck_conv_layers = {18, 20, 23, 25, 28, 31, 37}
    neck_rep_layers = {22, 27, 39}

    if idx in backbone_conv_layers and cls_name == "Conv":
        return {"conv"}
    if idx in backbone_csp_layers and hasattr(layer, "cv2"):
        return {"cv2.conv"}
    if include_neck and idx in neck_conv_layers and cls_name == "Conv":
        return {"conv"}
    if include_neck and idx in neck_rep_layers and cls_name == "RepC3":
        return {"cv3.conv"}
    return set()


def build_meta_pruner(wrapper: ModelWrapper, tp, args, vis, ir):
    ignored_layers = find_ignored_layers(wrapper.model)
    importance = tp.importance.MagnitudeImportance(p=1, group_reduction="mean", normalizer="mean")
    pruner = tp.pruner.MetaPruner(
        wrapper,
        example_inputs=(vis, ir),
        importance=importance,
        pruning_ratio=args.prune_ratio,
        iterative_steps=args.iterations,
        ignored_layers=ignored_layers,
        root_module_types=[nn.Conv2d],
        round_to=args.round_to,
    )
    return pruner, importance


def collect_root_specs(model: nn.Module, pruner, layer_end: int, include_neck: bool) -> list[RootSpec]:
    specs: list[RootSpec] = []
    name_map = {module: name for name, module in model.named_modules()}
    protected_idxs = find_protected_layer_indices(model)
    seen_modules: set[int] = set()
    for idx in range(min(layer_end, len(model.model))):
        if idx in protected_idxs:
            continue
        layer = model.model[idx]
        allowed_sub_names = get_candidate_root_names_for_layer(idx, layer, include_neck)
        if not allowed_sub_names:
            continue
        for sub_name, module in layer.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            if sub_name not in allowed_sub_names:
                continue
            if id(module) in seen_modules or module not in pruner.DG.module2node:
                continue
            if module.groups != 1:
                continue
            seen_modules.add(id(module))
            specs.append(
                RootSpec(
                    name=name_map.get(module, f"model.{idx}.{sub_name or 'conv'}"),
                    layer_idx=idx,
                    module=module,
                    pruning_fn=tp_global().pruner.function.prune_conv_out_channels,
                    mode="out",
                )
            )
    return specs


# Override the architecture helpers below so this script can also prune
# CMXFusion-based RTMM variants such as A-DWConv-C3k2_PConv.
def validate_expected_architecture(model: nn.Module) -> None:
    if len(model.model) <= 11:
        raise RuntimeError("Unexpected model depth for the current RTDETRMM architecture.")
    fusion_idx, branch_tail_idxs = find_fusion_info(model)
    if fusion_idx is None:
        raise RuntimeError(
            f"Unsupported architecture: no supported fusion block found. Expected one of {sorted(SUPPORTED_FUSION_TYPES)}."
        )
    fusion = model.model[fusion_idx]
    fusion_type = type(fusion).__name__
    fusion_from = list(getattr(fusion, "f", [])) if isinstance(getattr(fusion, "f", None), (list, tuple)) else []
    if fusion_from != [4, 9]:
        raise RuntimeError(
            f"Unexpected fusion wiring: layer{fusion_idx}={fusion_type}, from={fusion_from}, branch_tails={sorted(branch_tail_idxs)}"
        )
    if type(model.model[-1]).__name__ != "RTDETRDecoder":
        raise RuntimeError(f"Unexpected decoder type: {type(model.model[-1]).__name__}")


def find_fusion_info(model: nn.Module) -> tuple[int | None, set[int]]:
    fusion_idx = None
    branch_tail_idxs: set[int] = set()
    for idx, layer in enumerate(model.model):
        if type(layer).__name__ in SUPPORTED_FUSION_TYPES:
            fusion_idx = idx
            if isinstance(getattr(layer, "f", None), (list, tuple)):
                branch_tail_idxs.update(int(x) for x in layer.f if isinstance(x, int) and x >= 0)
            break
    if fusion_idx == 10 and not branch_tail_idxs:
        branch_tail_idxs.update({4, 9})
    return fusion_idx, branch_tail_idxs


def get_arch_family(model: nn.Module) -> str:
    fusion_idx, _ = find_fusion_info(model)
    if fusion_idx is None:
        return "unknown"
    fusion_type = type(model.model[fusion_idx]).__name__
    if fusion_type == "PIAFusionBlock":
        return "pia"
    if fusion_type == "CMXFusion":
        return "cmx"
    return "unknown"


def has_dwconv_consumer(model: nn.Module, producer_idx: int) -> bool:
    for idx, layer in enumerate(model.model):
        if type(layer).__name__ != "DWConv":
            continue
        source = getattr(layer, "f", None)
        if isinstance(source, int) and source == -1 and idx - 1 == producer_idx:
            return True
        if isinstance(source, (list, tuple)) and -1 in source and idx - 1 == producer_idx:
            return True
        if producer_idx in get_input_layer_indices(layer):
            return True
    return False


def find_protected_layer_indices(model: nn.Module) -> set[int]:
    protected: set[int] = set()
    fusion_idx, branch_tail_idxs = find_fusion_info(model)
    if fusion_idx is not None:
        protected.add(fusion_idx)
        protected.update(branch_tail_idxs)
        next_idx = fusion_idx + 1
        if 0 <= next_idx < len(model.model) and type(model.model[next_idx]).__name__ in CONV_LIKE_TYPES:
            protected.add(next_idx)
        for idx, layer in enumerate(model.model):
            if fusion_idx in get_input_layer_indices(layer) and type(layer).__name__ in CONV_LIKE_TYPES:
                protected.add(idx)

    for idx, layer in enumerate(model.model):
        if type(layer).__name__ != "RTDETRDecoder":
            continue
        for src_idx in get_input_layer_indices(layer):
            if 0 <= src_idx < len(model.model):
                protected.add(src_idx)
    return protected


def get_candidate_root_names_for_layer_v2(
    model: nn.Module, idx: int, layer: nn.Module, include_neck: bool
) -> set[str]:
    cls_name = type(layer).__name__
    arch_family = get_arch_family(model)

    if arch_family == "pia":
        backbone_conv_layers = {0, 1, 3, 5, 6, 8, 12, 14}
        backbone_csp_layers = {2, 4, 7, 9, 13, 15}
        neck_conv_layers = {18, 20, 23, 25, 28, 31, 37}
        neck_rep_layers = {22, 27, 39}

        if idx in backbone_conv_layers and cls_name == "Conv":
            return {"conv"}
        if idx in backbone_csp_layers and hasattr(layer, "cv2"):
            return {"cv2.conv"}
        if include_neck and idx in neck_conv_layers and cls_name == "Conv":
            return {"conv"}
        if include_neck and idx in neck_rep_layers and cls_name == "RepC3":
            return {"cv3.conv"}
        return set()

    if arch_family == "cmx":
        backbone_conv_layers = {0, 1, 3, 5, 6, 8, 11, 13}
        backbone_csp_layers = {2, 4, 7, 9, 12, 14}
        neck_conv_layers = {22, 24, 27, 30, 36}
        neck_c3k2_layers = {21, 26, 29, 32, 38}

        # torch-pruning dependency propagation is not stable here when the
        # pruned producer is consumed directly by a DWConv wrapper.
        if has_dwconv_consumer(model, idx):
            return set()

        if idx in backbone_conv_layers and cls_name in CONV_LIKE_TYPES:
            return {"conv"}
        if idx in backbone_csp_layers and hasattr(layer, "cv2"):
            return {"cv2.conv"}
        if include_neck and idx in neck_conv_layers and cls_name in CONV_LIKE_TYPES:
            return {"conv"}
        if include_neck and idx in neck_c3k2_layers and hasattr(layer, "cv2"):
            return {"cv2.conv"}
    return set()


def collect_root_specs(model: nn.Module, pruner, layer_end: int, include_neck: bool) -> list[RootSpec]:
    specs: list[RootSpec] = []
    name_map = {module: name for name, module in model.named_modules()}
    protected_idxs = find_protected_layer_indices(model)
    seen_modules: set[int] = set()
    for idx in range(min(layer_end, len(model.model))):
        if idx in protected_idxs:
            continue
        layer = model.model[idx]
        allowed_sub_names = get_candidate_root_names_for_layer_v2(model, idx, layer, include_neck)
        if not allowed_sub_names:
            continue
        for sub_name, module in layer.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            if sub_name not in allowed_sub_names:
                continue
            if id(module) in seen_modules or module not in pruner.DG.module2node:
                continue
            if module.groups != 1:
                continue
            seen_modules.add(id(module))
            specs.append(
                RootSpec(
                    name=name_map.get(module, f"model.{idx}.{sub_name or 'conv'}"),
                    layer_idx=idx,
                    module=module,
                    pruning_fn=tp_global().pruner.function.prune_conv_out_channels,
                    mode="out",
                )
            )
    return specs


_TP_CACHE = None


def tp_global():
    global _TP_CACHE
    if _TP_CACHE is None:
        _TP_CACHE = ensure_torch_pruning()
    return _TP_CACHE


def collect_taylor_gradients(wrapper: ModelWrapper, vis: torch.Tensor, ir: torch.Tensor, grad_samples: int) -> None:
    wrapper.eval()
    wrapper.zero_grad(set_to_none=True)
    for _ in range(grad_samples):
        out = wrapper(vis, ir)
        dummy_loss = out.float().mean()
        dummy_loss.backward()


def round_prune_count(channels: int, local_ratio: float, round_to: int, min_channels: int) -> int:
    max_prunable = max(channels - min_channels, 0)
    if max_prunable == 0:
        return 0
    n_prune = int(channels * local_ratio)
    if round_to > 1:
        n_prune = (n_prune // round_to) * round_to
        max_prunable = (max_prunable // round_to) * round_to
    return max(0, min(n_prune, max_prunable))


def ceil_round_prune_count(channels: int, local_ratio: float, round_to: int) -> int:
    if channels <= 0:
        return 0
    n_prune = int(torch.ceil(torch.tensor(channels * local_ratio)).item())
    if round_to > 1:
        n_prune = ((n_prune + round_to - 1) // round_to) * round_to
    return max(0, min(n_prune, channels))


def build_cumulative_schedule(total_ratio: float, steps: int) -> list[float]:
    return [total_ratio * step / steps for step in range(1, steps + 1)]


def pick_pruning_indices(scores: torch.Tensor, n_prune: int) -> list[int]:
    if scores is None or n_prune <= 0:
        return []
    finite = torch.isfinite(scores)
    if not finite.all():
        safe_scores = scores.clone()
        safe_scores[~finite] = torch.finfo(scores.dtype).max
        scores = safe_scores
    return torch.argsort(scores)[:n_prune].tolist()


def pair_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores is None or scores.numel() == 0 or scores.numel() % 2 != 0:
        raise ValueError("PIAFusion paired pruning requires an even number of scores.")
    half = scores.numel() // 2
    return 0.5 * (scores[:half] + scores[half:])


def expand_paired_indices(shared_idxs: list[int], half: int) -> list[int]:
    return sorted(shared_idxs + [i + half for i in shared_idxs])


def taylor_scores_for_conv_in_channels(conv: nn.Conv2d) -> torch.Tensor:
    if conv.weight.grad is None:
        raise RuntimeError("Conv input-channel Taylor scoring requires weight gradients.")
    w = conv.weight.detach().transpose(0, 1).flatten(1)
    dw = conv.weight.grad.detach().transpose(0, 1).flatten(1)
    return (w * dw).abs().sum(1)


def l1_scores_for_conv_out_channels(conv: nn.Conv2d) -> torch.Tensor:
    w = conv.weight.detach().flatten(1)
    return w.abs().sum(1)


def conv_out_scores_or_magnitude(conv: nn.Conv2d) -> torch.Tensor:
    w = conv.weight.detach().flatten(1)
    if conv.weight.grad is None:
        return w.abs().sum(1)
    dw = conv.weight.grad.detach().flatten(1)
    return (w * dw).abs().sum(1)


def collect_output_spatial_map(
    model: nn.Module, specs: list[RootSpec], vis: torch.Tensor, ir: torch.Tensor
) -> dict[str, tuple[int, int]]:
    spatial_map: dict[str, tuple[int, int]] = {}
    hooks = []
    module_to_name = {id(spec.module): spec.name for spec in specs}

    def make_hook(name: str):
        def _hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor) and output.ndim >= 4:
                spatial_map[name] = (int(output.shape[-2]), int(output.shape[-1]))

        return _hook

    for spec in specs:
        hooks.append(spec.module.register_forward_hook(make_hook(module_to_name[id(spec.module)])))

    wrapper = ModelWrapper(model)
    try:
        wrapper.eval()
        with torch.no_grad():
            wrapper(vis, ir)
    finally:
        wrapper.restore()
        for handle in hooks:
            handle.remove()

    return spatial_map


def estimate_conv_pruning_gain(conv: nn.Conv2d, spatial_hw: tuple[int, int], n_prune: int) -> float:
    h, w = spatial_hw
    if h <= 0 or w <= 0 or n_prune <= 0:
        return 0.0
    kernel_h, kernel_w = conv.kernel_size
    cin = conv.in_channels
    groups = max(conv.groups, 1)
    return float(n_prune) * float(h * w * cin * kernel_h * kernel_w / groups)


def collect_current_root_channel_map(model: nn.Module, args, device: torch.device) -> dict[str, int]:
    vis, ir = create_example_inputs(args.batch_size, args.imgsz, device)
    wrapper = ModelWrapper(model)
    pruner = None
    try:
        pruner, _ = build_meta_pruner(wrapper, tp_global(), args, vis, ir)
        current = {}
        for spec in collect_root_specs(model, pruner, args.layer_end, args.include_neck):
            channels = pruner.DG.get_out_channels(spec.module)
            if channels is not None:
                current[spec.name] = int(channels)
        return current
    finally:
        wrapper.restore()
        if pruner is not None:
            del pruner


def prune_one_round(
    model: nn.Module,
    args,
    device: torch.device,
    round_idx: int,
    target_ratio: float,
    initial_channels_map: dict[str, int],
) -> tuple[int, int]:
    vis, ir = create_example_inputs(args.batch_size, args.imgsz, device)

    root_pruned = 0
    channel_pruned = 0
    chunk_size = max(1, int(args.round_to))
    target_total_pruned = ceil_round_prune_count(sum(initial_channels_map.values()), target_ratio, chunk_size)
    failed_roots: set[str] = set()

    while True:
        step_wrapper = ModelWrapper(model)
        pruner = None
        try:
            pruner, _ = build_meta_pruner(step_wrapper, tp_global(), args, vis, ir)
            current_specs = collect_root_specs(model, pruner, args.layer_end, args.include_neck)
            if not current_specs:
                break
            spatial_map = collect_output_spatial_map(model, current_specs, vis, ir)

            current_total_pruned = 0
            for spec in current_specs:
                channels = pruner.DG.get_out_channels(spec.module)
                if channels is None:
                    continue
                current_total_pruned += max(initial_channels_map.get(spec.name, channels) - channels, 0)

            remaining_total = target_total_pruned - current_total_pruned
            if remaining_total <= 0:
                break

            best_candidate = None
            for spec in current_specs:
                if spec.name in failed_roots:
                    continue

                channels = pruner.DG.get_out_channels(spec.module)
                if channels is None:
                    continue

                if channels - args.min_channels < chunk_size:
                    continue
                n_prune = chunk_size if remaining_total >= chunk_size else 0
                if n_prune <= 0:
                    continue

                try:
                    scores = l1_scores_for_conv_out_channels(spec.module)
                except Exception as exc:
                    log(f"第 {round_idx} 轮跳过 {spec.name}: {exc}")
                    failed_roots.add(spec.name)
                    continue
                idxs = pick_pruning_indices(scores, n_prune)
                if len(idxs) != n_prune:
                    continue
                raw_score = float(scores[idxs].mean().item())
                flops_gain = estimate_conv_pruning_gain(spec.module, spatial_map.get(spec.name, (0, 0)), n_prune)
                if flops_gain <= 0:
                    continue
                effective_score = raw_score / (flops_gain ** max(args.flops_bias, 0.0))
                candidate = (effective_score, raw_score, flops_gain, spec, idxs)
                if best_candidate is None or candidate[0] < best_candidate[0]:
                    best_candidate = candidate

            if best_candidate is None:
                log(f"第 {round_idx} 轮没有更多可安全剪枝的层，提前结束当前轮。")
                break

            _, raw_score, flops_gain, spec, idxs = best_candidate
            log(
                f"第 {round_idx} 轮评估 root: {spec.name} (layer={spec.layer_idx}, n_prune={len(idxs)}, "
                f"raw={raw_score:.4e}, gain={flops_gain:.4e})"
            )
            try:
                group = pruner.DG.get_pruning_group(spec.module, spec.pruning_fn, idxs=idxs)
                if not pruner.DG.check_pruning_group(group):
                    log(f"第 {round_idx} 轮跳过 {spec.name}: 过度剪枝保护触发，仅忽略当前层")
                    failed_roots.add(spec.name)
                    continue
                group.prune()
                root_pruned += 1
                channel_pruned += len(idxs)
                log(f"第 {round_idx} 轮剪枝: {spec.name} -> {len(idxs)} channels")
            except Exception as exc:
                log(f"第 {round_idx} 轮跳过 {spec.name}: {exc}")
                failed_roots.add(spec.name)
                continue
        finally:
            step_wrapper.restore()
            if pruner is not None:
                del pruner
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return root_pruned, channel_pruned


def taylor_scores_for_linear_in_features(linear: nn.Linear) -> torch.Tensor:
    w = linear.weight.detach()
    if linear.weight.grad is None:
        return w.abs().sum(0)
    dw = linear.weight.grad.detach()
    return (w * dw).abs().sum(0)


def taylor_scores_for_linear_out_features(linear: nn.Linear) -> torch.Tensor:
    w = linear.weight.detach()
    if linear.weight.grad is None:
        return w.abs().sum(1)
    dw = linear.weight.grad.detach()
    return (w * dw).abs().sum(1)


def taylor_scores_for_mha_hidden(mha: nn.MultiheadAttention) -> torch.Tensor:
    dim = int(mha.embed_dim)
    in_proj = mha.in_proj_weight.detach()
    out_proj = mha.out_proj.weight.detach()
    if mha.in_proj_weight.grad is None or mha.out_proj.weight.grad is None:
        scores = in_proj.abs().sum(0)
        row_scores = in_proj.abs().sum(1).view(3, dim).mean(0)
        scores = scores + row_scores
        scores = scores + out_proj.abs().sum(0)
        scores = scores + out_proj.abs().sum(1)
        return scores
    scores = (in_proj * mha.in_proj_weight.grad.detach()).abs().sum(0)
    row_scores = (in_proj * mha.in_proj_weight.grad.detach()).abs().sum(1).view(3, dim).mean(0)
    scores = scores + row_scores
    scores = scores + (out_proj * mha.out_proj.weight.grad.detach()).abs().sum(0)
    scores = scores + (out_proj * mha.out_proj.weight.grad.detach()).abs().sum(1)
    return scores


def pick_keep_indices(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    if keep_count <= 0 or keep_count > scores.numel():
        raise ValueError(f"Invalid keep_count={keep_count} for {scores.numel()} scores.")
    keep = torch.argsort(scores, descending=True)[:keep_count]
    return torch.sort(keep).values.to(dtype=torch.long)


def round_keep_count(total: int, keep_ratio: float, round_to: int, min_keep: int) -> int:
    keep = max(min_keep, int(total * keep_ratio))
    if round_to > 1:
        keep = max(round_to, (keep // round_to) * round_to)
    keep = min(total, keep)
    if total >= round_to and round_to > 1 and keep == 0:
        keep = round_to
    return max(min_keep, min(total, keep))


def prune_raw_conv2d_out_channels(conv: nn.Conv2d, keep_idx: torch.Tensor) -> None:
    conv.weight = nn.Parameter(conv.weight.data[keep_idx])
    if conv.bias is not None:
        conv.bias = nn.Parameter(conv.bias.data[keep_idx])
    conv.out_channels = int(keep_idx.numel())


def prune_raw_conv2d_in_channels(conv: nn.Conv2d, keep_idx: torch.Tensor) -> None:
    if conv.groups == 1:
        conv.weight = nn.Parameter(conv.weight.data[:, keep_idx])
    else:
        conv.weight = nn.Parameter(conv.weight.data[keep_idx])
        conv.groups = int(keep_idx.numel())
        conv.out_channels = int(keep_idx.numel())
    conv.in_channels = int(keep_idx.numel())


def prune_batchnorm_channels(bn: nn.BatchNorm2d, keep_idx: torch.Tensor) -> None:
    bn.weight = nn.Parameter(bn.weight.data[keep_idx])
    bn.bias = nn.Parameter(bn.bias.data[keep_idx])
    bn.running_mean = bn.running_mean[keep_idx]
    bn.running_var = bn.running_var[keep_idx]
    bn.num_features = int(keep_idx.numel())


def prune_linear_in_features(linear: nn.Linear, keep_idx: torch.Tensor) -> None:
    linear.weight = nn.Parameter(linear.weight.data[:, keep_idx])
    linear.in_features = int(keep_idx.numel())


def prune_linear_out_features(linear: nn.Linear, keep_idx: torch.Tensor) -> None:
    linear.weight = nn.Parameter(linear.weight.data[keep_idx])
    if linear.bias is not None:
        linear.bias = nn.Parameter(linear.bias.data[keep_idx])
    linear.out_features = int(keep_idx.numel())


def prune_conv_module_out_channels(module, keep_idx: torch.Tensor) -> None:
    conv = module.conv
    conv.weight = nn.Parameter(conv.weight.data[keep_idx])
    if conv.bias is not None:
        conv.bias = nn.Parameter(conv.bias.data[keep_idx])
    conv.out_channels = int(keep_idx.numel())
    if hasattr(module, "bn") and isinstance(module.bn, nn.BatchNorm2d):
        prune_batchnorm_channels(module.bn, keep_idx)


def prune_conv_module_in_channels(module, keep_idx: torch.Tensor) -> None:
    conv = module.conv
    if conv.groups == 1:
        conv.weight = nn.Parameter(conv.weight.data[:, keep_idx])
    else:
        conv.weight = nn.Parameter(conv.weight.data[keep_idx])
        conv.groups = int(keep_idx.numel())
        conv.out_channels = int(keep_idx.numel())
        if hasattr(module, "bn") and isinstance(module.bn, nn.BatchNorm2d):
            prune_batchnorm_channels(module.bn, keep_idx)
    conv.in_channels = int(keep_idx.numel())


def conv_in_scores_or_magnitude(conv: nn.Conv2d) -> torch.Tensor:
    w = conv.weight.detach().transpose(0, 1).flatten(1)
    if conv.weight.grad is None:
        return w.abs().sum(1)
    dw = conv.weight.grad.detach().transpose(0, 1).flatten(1)
    return (w * dw).abs().sum(1)


def aggregate_c2f_hidden_scores(module: nn.Module) -> torch.Tensor:
    old_c = int(module.c)
    device = module.cv1.conv.weight.device
    scores = torch.zeros(old_c, device=device)

    cv1_scores = conv_out_scores_or_magnitude(module.cv1.conv)
    if cv1_scores.numel() >= 2 * old_c:
        scores += 0.5 * (cv1_scores[:old_c] + cv1_scores[old_c:2 * old_c])

    cv2_scores = conv_in_scores_or_magnitude(module.cv2.conv)
    if cv2_scores.numel() >= old_c:
        chunks = cv2_scores.split(old_c)
        if chunks:
            scores += torch.stack(chunks, dim=0).mean(0)

    for block in getattr(module, "m", []):
        final_conv = getattr(block, "final_conv", None)
        if final_conv is not None and hasattr(final_conv, "conv"):
            block_scores = conv_out_scores_or_magnitude(final_conv.conv)
            if block_scores.numel() == old_c:
                scores += block_scores
    return scores


def pick_partial_conv_keep_indices(scores: torch.Tensor, keep_count: int, n_div: int) -> torch.Tensor:
    total = int(scores.numel())
    old_proc = total // max(n_div, 1)
    old_passthrough = total - old_proc
    new_proc = keep_count // max(n_div, 1)
    new_passthrough = keep_count - new_proc

    proc_keep = torch.empty(0, dtype=torch.long, device=scores.device)
    if new_proc > 0 and old_proc > 0:
        proc_keep = torch.sort(torch.argsort(scores[:old_proc], descending=True)[:new_proc])[0].to(dtype=torch.long)

    passthrough_keep = torch.empty(0, dtype=torch.long, device=scores.device)
    if new_passthrough > 0 and old_passthrough > 0:
        local = torch.argsort(scores[old_proc:], descending=True)[:new_passthrough]
        passthrough_keep = torch.sort(local)[0].to(dtype=torch.long) + old_proc

    if proc_keep.numel() + passthrough_keep.numel() != keep_count:
        raise RuntimeError(
            f"PartialConv keep mismatch: requested {keep_count}, got {proc_keep.numel() + passthrough_keep.numel()}."
        )
    return torch.cat([proc_keep, passthrough_keep], dim=0)


def prune_partial_conv3_module(module: nn.Module, proc_keep_idx: torch.Tensor, new_hidden: int) -> None:
    conv = module.partial_conv3
    if proc_keep_idx.numel() > 0:
        conv.weight = nn.Parameter(conv.weight.data[proc_keep_idx][:, proc_keep_idx])
        if conv.bias is not None:
            conv.bias = nn.Parameter(conv.bias.data[proc_keep_idx])
    else:
        conv.weight = nn.Parameter(conv.weight.data[:0, :0])
        if conv.bias is not None:
            conv.bias = nn.Parameter(conv.bias.data[:0])
    conv.in_channels = int(proc_keep_idx.numel())
    conv.out_channels = int(proc_keep_idx.numel())
    module.dim_conv3 = int(proc_keep_idx.numel())
    module.dim_untouched = int(new_hidden - proc_keep_idx.numel())


def prune_bottleneck_pconv_block(block: nn.Module, keep_hidden: torch.Tensor) -> None:
    old_proc = int(block.cv1.partial_conv3.in_channels)
    proc_keep = keep_hidden[keep_hidden < old_proc].to(dtype=torch.long)
    new_hidden = int(keep_hidden.numel())
    prune_partial_conv3_module(block.cv1, proc_keep, new_hidden)
    prune_partial_conv3_module(block.cv2, proc_keep, new_hidden)


def slim_c3k_pconv_block(block: nn.Module, keep_outer_hidden: torch.Tensor, round_to: int) -> None:
    old_outer = int(block.cv1.conv.in_channels)
    old_inner = int(block.cv1.conv.out_channels)
    new_outer = int(keep_outer_hidden.numel())
    keep_ratio = new_outer / max(old_outer, 1)
    keep_inner_count = round_keep_count(old_inner, keep_ratio, max(1, round_to), max(1, min(round_to, old_inner)))
    n_div = int(getattr(block.m[0].cv1, "n_div", 4)) if len(block.m) else 4

    inner_scores = conv_out_scores_or_magnitude(block.cv1.conv) + conv_out_scores_or_magnitude(block.cv2.conv)
    cv3_in_scores = conv_in_scores_or_magnitude(block.cv3.conv)
    if cv3_in_scores.numel() >= 2 * old_inner:
        inner_scores += 0.5 * (cv3_in_scores[:old_inner] + cv3_in_scores[old_inner:2 * old_inner])

    keep_inner = pick_partial_conv_keep_indices(inner_scores, keep_inner_count, n_div)

    prune_conv_module_in_channels(block.cv1, keep_outer_hidden)
    prune_conv_module_out_channels(block.cv1, keep_inner)
    prune_conv_module_in_channels(block.cv2, keep_outer_hidden)
    prune_conv_module_out_channels(block.cv2, keep_inner)

    for sub_block in block.m:
        if type(sub_block).__name__ != "Bottleneck_PConv":
            raise RuntimeError(f"Unsupported C3k_PConv sub-block for slimming: {type(sub_block).__name__}")
        prune_bottleneck_pconv_block(sub_block, keep_inner)

    keep_cv3_in = torch.cat([keep_inner, keep_inner + old_inner], dim=0).to(dtype=torch.long)
    prune_conv_module_in_channels(block.cv3, keep_cv3_in)
    prune_conv_module_out_channels(block.cv3, keep_outer_hidden)


def prune_msie_block_internal(block: nn.Module, keep_hidden: torch.Tensor) -> None:
    old_c = int(block.local_conv.conv.in_channels)
    new_c = int(keep_hidden.numel())
    num_bins = len(block.features)
    if num_bins <= 0 or new_c % num_bins != 0 or old_c % num_bins != 0:
        raise RuntimeError(f"MSIE channel count must be divisible by bin count: old={old_c}, new={new_c}, bins={num_bins}")

    old_branch = old_c // num_bins
    new_branch = new_c // num_bins

    prune_conv_module_in_channels(block.local_conv, keep_hidden)
    prune_conv_module_out_channels(block.local_conv, keep_hidden)

    branch_keeps: list[torch.Tensor] = []
    for branch_idx, feature_seq in enumerate(block.features):
        conv1 = feature_seq[1]
        conv2 = feature_seq[2]
        branch_scores = conv_out_scores_or_magnitude(conv1.conv)
        branch_keep = pick_keep_indices(branch_scores, new_branch)
        branch_keeps.append(branch_keep)

        prune_conv_module_in_channels(conv1, keep_hidden)
        prune_conv_module_out_channels(conv1, branch_keep)
        prune_conv_module_in_channels(conv2, branch_keep)

        edge_enhancer = block.ees[branch_idx]
        prune_conv_module_in_channels(edge_enhancer.out_conv, branch_keep)
        prune_conv_module_out_channels(edge_enhancer.out_conv, branch_keep)

    keep_final_chunks = [keep_hidden]
    keep_final_chunks.extend(old_c + branch_idx * old_branch + branch_keep for branch_idx, branch_keep in enumerate(branch_keeps))
    keep_final_in = torch.cat(keep_final_chunks, dim=0).to(dtype=torch.long)
    prune_conv_module_in_channels(block.final_conv, keep_final_in)
    prune_conv_module_out_channels(block.final_conv, keep_hidden)


def slim_csp_msie_module(module: nn.Module, keep_hidden: torch.Tensor) -> None:
    old_c = int(module.c)
    keep_hidden = keep_hidden.to(dtype=torch.long)
    keep_cv1_out = torch.cat([keep_hidden, keep_hidden + old_c], dim=0)
    prune_conv_module_out_channels(module.cv1, keep_cv1_out)

    for block in module.m:
        if type(block).__name__ != "MutilScaleEdgeInformationEnhance":
            raise RuntimeError(f"Unsupported CSP internal block for slimming: {type(block).__name__}")
        prune_msie_block_internal(block, keep_hidden)

    keep_cv2_chunks = [keep_hidden + i * old_c for i in range(2 + len(module.m))]
    keep_cv2_in = torch.cat(keep_cv2_chunks, dim=0).to(dtype=torch.long)
    prune_conv_module_in_channels(module.cv2, keep_cv2_in)
    module.c = int(keep_hidden.numel())


def slim_c3k2_pconv_module(module: nn.Module, keep_hidden: torch.Tensor, round_to: int) -> None:
    old_c = int(module.c)
    keep_hidden = keep_hidden.to(dtype=torch.long)
    keep_cv1_out = torch.cat([keep_hidden, keep_hidden + old_c], dim=0)
    prune_conv_module_out_channels(module.cv1, keep_cv1_out)

    for block in module.m:
        cls_name = type(block).__name__
        if cls_name == "Bottleneck_PConv":
            prune_bottleneck_pconv_block(block, keep_hidden)
            continue
        if cls_name == "C3k_PConv":
            slim_c3k_pconv_block(block, keep_hidden, round_to=max(1, round_to))
            continue
        raise RuntimeError(f"Unsupported C3k2_PConv internal block for slimming: {cls_name}")

    keep_cv2_chunks = [keep_hidden + i * old_c for i in range(2 + len(module.m))]
    keep_cv2_in = torch.cat(keep_cv2_chunks, dim=0).to(dtype=torch.long)
    prune_conv_module_in_channels(module.cv2, keep_cv2_in)
    module.c = int(keep_hidden.numel())


def slim_dattention_ffn_module(module: nn.Module, keep_idx: torch.Tensor) -> None:
    prune_raw_conv2d_out_channels(module.fc1, keep_idx)
    prune_raw_conv2d_in_channels(module.fc2, keep_idx)


def slim_large_modules(model: nn.Module, args, device: torch.device) -> list[str]:
    if not getattr(args, "enable_internal_slimming", False):
        return []

    keep_ratio = max(0.0, 1.0 - float(args.prune_ratio))
    if keep_ratio >= 1.0:
        return []

    vis, ir = create_example_inputs(args.batch_size, args.imgsz, device)
    grad_wrapper = ModelWrapper(model)
    try:
        collect_taylor_gradients(grad_wrapper, vis, ir, args.grad_samples)
    finally:
        grad_wrapper.restore()

    changes: list[str] = []
    for idx, layer in enumerate(model.model):
        cls_name = type(layer).__name__

        if cls_name == "CSP_MutilScaleEdgeInformationEnhance":
            old_c = int(layer.c)
            keep_c = round_keep_count(old_c, keep_ratio, args.round_to, args.min_channels)
            if keep_c < old_c:
                scores = aggregate_c2f_hidden_scores(layer)
                keep_hidden = pick_keep_indices(scores, keep_c)
                slim_csp_msie_module(layer, keep_hidden)
                changes.append(f"model.{idx}:{cls_name} c {old_c}->{keep_c}")
            continue

        if cls_name == "C3k2_PConv":
            old_c = int(layer.c)
            keep_c = round_keep_count(old_c, keep_ratio, args.round_to, args.min_channels)
            if keep_c < old_c:
                n_div = int(getattr(layer.m[0].cv1, "n_div", 4)) if len(layer.m) else 4
                scores = aggregate_c2f_hidden_scores(layer)
                keep_hidden = pick_partial_conv_keep_indices(scores, keep_c, n_div)
                slim_c3k2_pconv_module(layer, keep_hidden, args.round_to)
                changes.append(f"model.{idx}:{cls_name} c {old_c}->{keep_c}")
            continue

        if cls_name == "TransformerEncoderLayer_DAttention":
            old_cm = int(layer.fc1.out_channels)
            keep_cm = round_keep_count(old_cm, keep_ratio, args.round_to, args.round_to)
            if keep_cm < old_cm:
                scores = conv_out_scores_or_magnitude(layer.fc1) + conv_in_scores_or_magnitude(layer.fc2)
                keep_idx = pick_keep_indices(scores, keep_cm)
                slim_dattention_ffn_module(layer, keep_idx)
                changes.append(f"model.{idx}:{cls_name} ffn {old_cm}->{keep_cm}")

    return changes


def prune_layernorm_features(norm: nn.LayerNorm, keep_idx: torch.Tensor) -> None:
    if norm.elementwise_affine:
        norm.weight = nn.Parameter(norm.weight.data[keep_idx])
        norm.bias = nn.Parameter(norm.bias.data[keep_idx])
    norm.normalized_shape = (int(keep_idx.numel()),)


def prune_embedding_features(embedding: nn.Embedding, keep_idx: torch.Tensor) -> None:
    embedding.weight = nn.Parameter(embedding.weight.data[:, keep_idx])
    embedding.embedding_dim = int(keep_idx.numel())


def prune_multihead_attention_hidden(mha: nn.MultiheadAttention, keep_idx: torch.Tensor) -> None:
    old_dim = int(mha.embed_dim)
    expanded = torch.cat([keep_idx, keep_idx + old_dim, keep_idx + 2 * old_dim])
    mha.in_proj_weight = nn.Parameter(mha.in_proj_weight.data[expanded][:, keep_idx])
    if mha.in_proj_bias is not None:
        mha.in_proj_bias = nn.Parameter(mha.in_proj_bias.data[expanded])
    mha.out_proj.weight = nn.Parameter(mha.out_proj.weight.data[keep_idx][:, keep_idx])
    if mha.out_proj.bias is not None:
        mha.out_proj.bias = nn.Parameter(mha.out_proj.bias.data[keep_idx])
    mha.embed_dim = int(keep_idx.numel())
    mha.kdim = int(keep_idx.numel())
    mha.vdim = int(keep_idx.numel())
    mha.head_dim = int(keep_idx.numel()) // int(mha.num_heads)
    mha.out_proj.in_features = int(keep_idx.numel())
    mha.out_proj.out_features = int(keep_idx.numel())
    mha._qkv_same_embed_dim = True


def prune_msdeformattn_hidden(attn: nn.Module, keep_idx: torch.Tensor) -> None:
    prune_linear_in_features(attn.sampling_offsets, keep_idx)
    prune_linear_in_features(attn.attention_weights, keep_idx)
    prune_linear_in_features(attn.value_proj, keep_idx)
    prune_linear_out_features(attn.value_proj, keep_idx)
    prune_linear_in_features(attn.output_proj, keep_idx)
    prune_linear_out_features(attn.output_proj, keep_idx)
    attn.d_model = int(keep_idx.numel())


def slim_bbox_mlp_hidden(mlp: nn.Module, keep_idx: torch.Tensor) -> None:
    if len(mlp.layers) != 3:
        return
    prune_linear_in_features(mlp.layers[0], keep_idx)
    prune_linear_out_features(mlp.layers[0], keep_idx)
    prune_linear_in_features(mlp.layers[1], keep_idx)
    prune_linear_out_features(mlp.layers[1], keep_idx)
    prune_linear_in_features(mlp.layers[2], keep_idx)


def slim_decoder_hidden(model: nn.Module, args, device: torch.device) -> tuple[int, int] | None:
    keep_ratio = float(getattr(args, "decoder_keep_ratio", 1.0))
    if not (0.0 < keep_ratio < 1.0):
        return None

    head = model.model[-1]
    if type(head).__name__ != "RTDETRDecoder":
        return None

    vis, ir = create_example_inputs(args.batch_size, args.imgsz, device)
    grad_wrapper = ModelWrapper(model)
    try:
        collect_taylor_gradients(grad_wrapper, vis, ir, args.grad_samples)
    finally:
        grad_wrapper.restore()

    old_hd = int(head.hidden_dim)
    round_base = max(int(args.round_to), int(head.nhead))
    keep_hd = round_keep_count(old_hd, keep_ratio, round_base, int(head.nhead))
    keep_hd = max(int(head.nhead), (keep_hd // int(head.nhead)) * int(head.nhead))
    if keep_hd >= old_hd:
        return None

    device0 = next(model.parameters()).device
    hidden_scores = torch.zeros(old_hd, device=device0)
    for proj in head.input_proj:
        hidden_scores += conv_out_scores_or_magnitude(proj[0])
    hidden_scores += taylor_scores_for_linear_in_features(head.enc_score_head)
    hidden_scores += taylor_scores_for_linear_in_features(head.enc_output[0])
    hidden_scores += taylor_scores_for_linear_out_features(head.enc_output[0])
    hidden_scores += taylor_scores_for_linear_in_features(head.enc_bbox_head.layers[0])
    hidden_scores += taylor_scores_for_linear_out_features(head.enc_bbox_head.layers[0])
    hidden_scores += taylor_scores_for_linear_in_features(head.enc_bbox_head.layers[1])
    hidden_scores += taylor_scores_for_linear_out_features(head.enc_bbox_head.layers[1])
    hidden_scores += taylor_scores_for_linear_in_features(head.enc_bbox_head.layers[2])

    for layer in head.decoder.layers:
        hidden_scores += taylor_scores_for_mha_hidden(layer.self_attn)
        hidden_scores += taylor_scores_for_linear_in_features(layer.linear1)
        hidden_scores += taylor_scores_for_linear_out_features(layer.linear2)
        hidden_scores += taylor_scores_for_linear_in_features(layer.cross_attn.sampling_offsets)
        hidden_scores += taylor_scores_for_linear_in_features(layer.cross_attn.attention_weights)
        hidden_scores += taylor_scores_for_linear_in_features(layer.cross_attn.value_proj)
        hidden_scores += taylor_scores_for_linear_out_features(layer.cross_attn.value_proj)
        hidden_scores += taylor_scores_for_linear_in_features(layer.cross_attn.output_proj)
        hidden_scores += taylor_scores_for_linear_out_features(layer.cross_attn.output_proj)

    for head_linear in head.dec_score_head:
        hidden_scores += taylor_scores_for_linear_in_features(head_linear)
    for bbox_mlp in head.dec_bbox_head:
        hidden_scores += taylor_scores_for_linear_in_features(bbox_mlp.layers[0])
        hidden_scores += taylor_scores_for_linear_out_features(bbox_mlp.layers[0])
        hidden_scores += taylor_scores_for_linear_in_features(bbox_mlp.layers[1])
        hidden_scores += taylor_scores_for_linear_out_features(bbox_mlp.layers[1])
        hidden_scores += taylor_scores_for_linear_in_features(bbox_mlp.layers[2])

    keep_hd_idx = pick_keep_indices(hidden_scores, keep_hd)

    old_q_mid = int(head.query_pos_head.layers[0].out_features)
    q_mid_scores = taylor_scores_for_linear_out_features(head.query_pos_head.layers[0])
    q_mid_scores += taylor_scores_for_linear_in_features(head.query_pos_head.layers[1])
    keep_q_mid = round_keep_count(old_q_mid, keep_hd / max(old_hd, 1), round_base, round_base)
    keep_q_mid_idx = pick_keep_indices(q_mid_scores, keep_q_mid)

    for proj in head.input_proj:
        prune_raw_conv2d_out_channels(proj[0], keep_hd_idx)
        prune_batchnorm_channels(proj[1], keep_hd_idx)

    prune_embedding_features(head.denoising_class_embed, keep_hd_idx)
    if getattr(head, "learnt_init_query", False) and hasattr(head, "tgt_embed"):
        prune_embedding_features(head.tgt_embed, keep_hd_idx)

    prune_linear_out_features(head.query_pos_head.layers[0], keep_q_mid_idx)
    prune_linear_in_features(head.query_pos_head.layers[1], keep_q_mid_idx)
    prune_linear_out_features(head.query_pos_head.layers[1], keep_hd_idx)

    prune_linear_in_features(head.enc_output[0], keep_hd_idx)
    prune_linear_out_features(head.enc_output[0], keep_hd_idx)
    prune_layernorm_features(head.enc_output[1], keep_hd_idx)
    prune_linear_in_features(head.enc_score_head, keep_hd_idx)
    slim_bbox_mlp_hidden(head.enc_bbox_head, keep_hd_idx)

    for layer in head.decoder.layers:
        prune_multihead_attention_hidden(layer.self_attn, keep_hd_idx)
        prune_layernorm_features(layer.norm1, keep_hd_idx)
        prune_msdeformattn_hidden(layer.cross_attn, keep_hd_idx)
        prune_layernorm_features(layer.norm2, keep_hd_idx)
        old_ffn = int(layer.linear1.out_features)
        ffn_scores = taylor_scores_for_linear_out_features(layer.linear1) + taylor_scores_for_linear_in_features(layer.linear2)
        keep_ffn = round_keep_count(old_ffn, keep_hd / max(old_hd, 1), round_base, round_base)
        keep_ffn_idx = pick_keep_indices(ffn_scores, keep_ffn)
        prune_linear_in_features(layer.linear1, keep_hd_idx)
        prune_linear_out_features(layer.linear1, keep_ffn_idx)
        prune_linear_in_features(layer.linear2, keep_ffn_idx)
        prune_linear_out_features(layer.linear2, keep_hd_idx)
        prune_layernorm_features(layer.norm3, keep_hd_idx)

    for head_linear in head.dec_score_head:
        prune_linear_in_features(head_linear, keep_hd_idx)
    for bbox_mlp in head.dec_bbox_head:
        slim_bbox_mlp_hidden(bbox_mlp, keep_hd_idx)

    head.hidden_dim = keep_hd
    head.decoder.hidden_dim = keep_hd
    return old_hd, keep_hd


def validate_forward(model: nn.Module, device: torch.device, batch_size: int, imgsz: int) -> torch.Size:
    vis, ir = create_example_inputs(batch_size, imgsz, device)
    wrapper = ModelWrapper(model)
    try:
        wrapper.eval()
        with torch.no_grad():
            out = wrapper(vis, ir)
        return out.shape
    finally:
        wrapper.restore()


def compute_model_stats(model: nn.Module, device: torch.device, batch_size: int, imgsz: int) -> tuple[float, int]:
    vis, ir = create_example_inputs(batch_size, imgsz, device)
    wrapper = ModelWrapper(model).to(device)
    try:
        flops, params = tp_global().utils.count_ops_and_params(wrapper, (vis, ir))
        return flops / 1e9, int(params)
    finally:
        wrapper.restore()


def measure_fps(model: nn.Module, device: torch.device, batch_size: int, imgsz: int, warmup: int, iters: int) -> float:
    if device.type != "cuda":
        return 0.0
    vis, ir = create_example_inputs(batch_size, imgsz, device)
    wrapper = ModelWrapper(model).to(device)
    try:
        wrapper.eval()
        with torch.no_grad():
            for _ in range(max(warmup, 0)):
                wrapper(vis, ir)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(max(iters, 1)):
                wrapper(vis, ir)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
        return iters / max(elapsed, 1e-8)
    finally:
        wrapper.restore()


def save_pruned_checkpoint(model: nn.Module, orig_ckpt: dict | None, output_path: Path) -> None:
    sanitized = sanitize_checkpoint_for_save(orig_ckpt, model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sanitized, output_path)


def print_resume_command(saved_path: Path, args) -> None:
    cmd = (
        f"conda run -n MM python trainMM.py --model \"{saved_path}\" "
        f"--data \"{args.data}\" --epochs {args.resume_epochs} --batch {args.resume_batch} "
        f"--imgsz {args.imgsz} --device {args.device} --name finetune_{saved_path.stem}"
    )
    print(cmd)


def main() -> int:
    args = parse_args()
    if not (0.0 < args.prune_ratio < 1.0):
        raise ValueError("--prune-ratio 必须在 (0, 1) 之间")
    if args.iterations < 1:
        raise ValueError("--iterations 必须 >= 1")
    if args.grad_samples < 1:
        raise ValueError("--grad-samples 必须 >= 1")

    if not (0.0 < args.decoder_keep_ratio <= 1.0):
        raise ValueError("--decoder-keep-ratio must be in (0, 1].")

    tp = tp_global()
    set_seed(args.seed)
    device = resolve_device(args.device)
    weights = Path(args.weights)
    output_path = make_output_path(args.output, args.prune_ratio)

    log(f"torch-pruning 版本: {tp.__version__}")
    log(f"运行设备: {device}")
    model, orig_ckpt = load_model(weights, device)
    validate_expected_architecture(model)

    initial_shape = validate_forward(model, device, args.batch_size, args.imgsz)
    log(f"初始前向输出: {tuple(initial_shape)}")

    seed_vis, seed_ir = create_example_inputs(args.batch_size, args.imgsz, device)
    seed_wrapper = ModelWrapper(model)
    seed_pruner = None
    initial_channels_map: dict[str, int] = {}
    try:
        seed_pruner, _ = build_meta_pruner(seed_wrapper, tp, args, seed_vis, seed_ir)
        for spec in collect_root_specs(model, seed_pruner, args.layer_end, args.include_neck):
            ch = seed_pruner.DG.get_in_channels(spec.module) if spec.mode == "pair_in" else seed_pruner.DG.get_out_channels(spec.module)
            if ch is None:
                continue
            initial_channels_map[spec.name] = ch // 2 if spec.mode == "pair_in" else ch
    finally:
        seed_wrapper.restore()
        if seed_pruner is not None:
            del seed_pruner
    initial_candidate_channels = sum(initial_channels_map.values())
    log(f"候选 backbone Conv 通道总数: {initial_candidate_channels}")

    schedule = build_cumulative_schedule(args.prune_ratio, args.iterations)
    for round_idx, target_ratio in enumerate(schedule, start=1):
        local_ratio = target_ratio
        log(f"开始第 {round_idx}/{args.iterations} 轮剪枝，增量比例 {local_ratio:.4f}")
        roots, channels = prune_one_round(model, args, device, round_idx, target_ratio, initial_channels_map)
        shape = validate_forward(model, device, args.batch_size, args.imgsz)
        log(f"第 {round_idx} 轮完成: roots={roots}, channels={channels}, output={tuple(shape)}")

    internal_changes = slim_large_modules(model, args, device)
    if internal_changes:
        shape = validate_forward(model, device, args.batch_size, args.imgsz)
        log(f"Internal slimming applied to {len(internal_changes)} modules, output={tuple(shape)}")
        for item in internal_changes:
            log(f"  {item}")

    decoder_slim = slim_decoder_hidden(model, args, device)
    if decoder_slim is not None:
        old_hd, new_hd = decoder_slim
        shape = validate_forward(model, device, args.batch_size, args.imgsz)
        log(f"RTDETRDecoder hidden_dim: {old_hd} -> {new_hd}, output={tuple(shape)}")

    flops_g = 0.0
    params = sum(p.numel() for p in model.parameters())
    try:
        flops_g, params = compute_model_stats(model, device, args.batch_size, args.imgsz)
    except Exception as exc:
        log(f"GFLOPs/Params 统计失败，回退到参数量统计: {exc}")

    fps = 0.0
    try:
        fps = measure_fps(model, device, args.batch_size, args.imgsz, args.fps_warmup, args.fps_iters)
    except Exception as exc:
        log(f"FPS 统计失败: {exc}")

    final_channels_map = collect_current_root_channel_map(model, args, device)
    final_candidate_channels = sum(final_channels_map.get(name, 0) for name in initial_channels_map)
    achieved_channel_ratio = 0.0
    if initial_candidate_channels > 0:
        achieved_channel_ratio = max(initial_candidate_channels - final_candidate_channels, 0) / initial_candidate_channels

    save_pruned_checkpoint(model, orig_ckpt, output_path)
    log(f"模型已保存: {output_path}")

    reload_shape = None
    try:
        from ultralytics import RTDETRMM

        reloaded = RTDETRMM(str(output_path)).model.to(device).float().eval()
        reload_shape = validate_forward(reloaded, device, args.batch_size, args.imgsz)
        log(f"保存后回载验证通过: {tuple(reload_shape)}")
    except Exception as exc:
        log(f"保存后回载验证失败: {exc}")

    print("\n=== 剪枝结果 ===")
    print(f"GFLOPs : {flops_g:.3f}")
    print(f"Params : {params:,}")
    print(f"FPS    : {fps:.2f}")
    print(f"RootCh : {initial_candidate_channels} -> {final_candidate_channels} ({achieved_channel_ratio:.2%})")
    print(f"Output : {output_path}")
    print("PIAFusionBlock 当前无需额外调用 tp.register_custom_layer；本版脚本会尽量让 backbone 内所有可追踪 Conv 参与剪枝，仅保留直接连接融合/解码器的桥接卷积。")
    print("可直接继续恢复训练的命令:")
    print_resume_command(output_path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
