#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Random-only structured iterative pruning for RTDETRMM VIF models.

This script is intentionally independent from the downloaded pruning scripts.
It keeps the same model assumptions as the current project:
- RGB branch: layers 0-4
- IR branch: layers 5-9
- Fusion: layer 10 (PIAFusionBlock)
- Decoder/attention blocks are protected

Pruning strategy:
- torch-pruning MetaPruner + DependencyGraph
- random channel importance only
- iterative channel pruning without gradient-based ranking
- no extra FLOPs-biased ranking
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


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def parse_args() -> argparse.Namespace:
    pia_default = (
        Path("ReTest")
        / "aifi-dattention-CSP-MutilScaleEdgeInformationEnhance-ASF-P2-lite-PIAFusionBlock"
        / "weights"
        / "best.pt"
    )
    default_weights = pia_default if pia_default.exists() else Path("MODEL") / "pruned_finetune_best.pt"

    parser = argparse.ArgumentParser(description="Random-only iterative pruning for RTDETRMM VIF models")
    parser.add_argument("--weights", type=str, default=str(default_weights), help="Input .pt checkpoint")
    parser.add_argument("--output", type=str, default="prune_outputs_random_only", help="Output .pt path or directory")
    parser.add_argument("--data", type=str, default=r"D:\BaiduNetdiskDownload\M3FD\M3FD_split\data.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prune-ratio", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--grad-samples", type=int, default=2)
    parser.add_argument("--min-channels", type=int, default=8)
    parser.add_argument("--round-to", type=int, default=8)
    parser.add_argument(
        "--include-neck",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include curated neck roots in addition to the backbone roots.",
    )
    parser.add_argument(
        "--layer-end",
        "--backbone-end",
        dest="layer_end",
        type=int,
        default=16,
        help="Exclusive layer end index for pruning roots.",
    )
    parser.add_argument("--fps-warmup", type=int, default=10)
    parser.add_argument("--fps-iters", type=int, default=50)
    parser.add_argument("--resume-epochs", type=int, default=30)
    parser.add_argument("--resume-batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_torch_pruning():
    try:
        import torch_pruning as tp
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("torch-pruning is required. Please install it in the MM environment.") from exc
    return tp


_TP_CACHE = None


def tp_global():
    global _TP_CACHE
    if _TP_CACHE is None:
        _TP_CACHE = ensure_torch_pruning()
    return _TP_CACHE


def resolve_device(device: str) -> torch.device:
    ds = str(device).strip().lower()
    if ds == "cpu":
        return torch.device("cpu")
    if ds.isdigit():
        ds = f"cuda:{ds}"
    if ds.startswith("cuda") and torch.cuda.is_available():
        return torch.device(ds)
    if torch.cuda.is_available():
        log(f"Requested device {device} is not available, fallback to cuda:0")
        return torch.device("cuda:0")
    log("CUDA is not available, fallback to CPU. FPS will not represent GPU performance.")
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
    return out / f"pruned_random_r{ratio_str}_{timestamp}.pt"


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
    ckpt["prune_method"] = "random_only"
    return ckpt


def load_model(weights: Path, device: torch.device):
    from ultralytics import RTDETRMM
    from ultralytics.nn.tasks import torch_safe_load

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    log(f"Loading model: {weights}")
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
        raise RuntimeError("Unexpected model depth for the current RTDETRMM PIAFusionBlock architecture.")
    fusion = model.model[10]
    fusion_type = type(fusion).__name__
    fusion_from = list(getattr(fusion, "f", [])) if isinstance(getattr(fusion, "f", None), (list, tuple)) else []
    compress_type = type(model.model[11]).__name__
    if fusion_type != "PIAFusionBlock" or fusion_from != [4, 9] or compress_type != "Conv":
        raise RuntimeError(
            f"Unexpected architecture: layer10={fusion_type}, from={fusion_from}, layer11={compress_type}"
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
    importance = tp.importance.TaylorImportance(group_reduction="mean", normalizer="mean")
    pruner = tp.pruner.MetaPruner(
        wrapper,
        example_inputs=(vis, ir),
        importance=importance,
        pruning_ratio=args.prune_ratio,
        iterative_steps=args.iterations,
        ignored_layers=find_ignored_layers(wrapper.model),
        root_module_types=[nn.Conv2d],
        round_to=args.round_to,
    )
    return pruner


def collect_root_specs(model: nn.Module, pruner, layer_end: int, include_neck: bool) -> list[RootSpec]:
    specs: list[RootSpec] = []
    name_map = {module: name for name, module in model.named_modules()}
    protected_idxs = find_protected_layer_indices(model)
    seen_modules: set[int] = set()
    prune_fn = tp_global().pruner.function.prune_conv_out_channels

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
                    pruning_fn=prune_fn,
                )
            )
    return specs


def ceil_round_prune_count(channels: int, ratio: float, round_to: int) -> int:
    if channels <= 0:
        return 0
    n_prune = int(torch.ceil(torch.tensor(channels * ratio)).item())
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
        scores = scores.clone()
        scores[~finite] = torch.finfo(scores.dtype).max
    return torch.argsort(scores)[:n_prune].tolist()


def random_scores_for_conv_out_channels(conv: nn.Conv2d) -> torch.Tensor:
    return torch.rand(conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype)


def collect_current_root_channel_map(model: nn.Module, args, device: torch.device) -> dict[str, int]:
    vis, ir = create_example_inputs(args.batch_size, args.imgsz, device)
    wrapper = ModelWrapper(model)
    pruner = None
    try:
        pruner = build_meta_pruner(wrapper, tp_global(), args, vis, ir)
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
            pruner = build_meta_pruner(step_wrapper, tp_global(), args, vis, ir)
            current_specs = collect_root_specs(model, pruner, args.layer_end, args.include_neck)
            if not current_specs:
                break

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
                    scores = random_scores_for_conv_out_channels(spec.module)
                except Exception as exc:
                    log(f"Round {round_idx}: skip {spec.name} because {exc}")
                    failed_roots.add(spec.name)
                    continue

                idxs = pick_pruning_indices(scores, n_prune)
                if len(idxs) != n_prune:
                    continue

                raw_score = float(scores[idxs].mean().item())
                candidate = (raw_score, spec, idxs)
                if best_candidate is None or candidate[0] < best_candidate[0]:
                    best_candidate = candidate

            if best_candidate is None:
                log(f"Round {round_idx}: no more safe Taylor pruning candidates.")
                break

            raw_score, spec, idxs = best_candidate
            log(
                f"Round {round_idx}: prune {spec.name} "
                f"(layer={spec.layer_idx}, n_prune={len(idxs)}, random={raw_score:.4e})"
            )
            try:
                group = pruner.DG.get_pruning_group(spec.module, spec.pruning_fn, idxs=idxs)
                if not pruner.DG.check_pruning_group(group):
                    log(f"Round {round_idx}: skip {spec.name} because pruning protection was triggered.")
                    failed_roots.add(spec.name)
                    continue
                group.prune()
                root_pruned += 1
                channel_pruned += len(idxs)
            except Exception as exc:
                log(f"Round {round_idx}: skip {spec.name} because {exc}")
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
        raise ValueError("--prune-ratio must be in (0, 1)")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    if args.grad_samples < 1:
        raise ValueError("--grad-samples must be >= 1")

    tp = tp_global()
    set_seed(args.seed)
    device = resolve_device(args.device)
    weights = Path(args.weights)
    output_path = make_output_path(args.output, args.prune_ratio)

    log(f"torch-pruning version: {tp.__version__}")
    log(f"device: {device}")
    model, orig_ckpt = load_model(weights, device)
    validate_expected_architecture(model)

    initial_shape = validate_forward(model, device, args.batch_size, args.imgsz)
    log(f"Initial forward output: {tuple(initial_shape)}")

    seed_vis, seed_ir = create_example_inputs(args.batch_size, args.imgsz, device)
    seed_wrapper = ModelWrapper(model)
    seed_pruner = None
    initial_channels_map: dict[str, int] = {}
    try:
        seed_pruner = build_meta_pruner(seed_wrapper, tp, args, seed_vis, seed_ir)
        for spec in collect_root_specs(model, seed_pruner, args.layer_end, args.include_neck):
            ch = seed_pruner.DG.get_out_channels(spec.module)
            if ch is not None:
                initial_channels_map[spec.name] = int(ch)
    finally:
        seed_wrapper.restore()
        if seed_pruner is not None:
            del seed_pruner

    initial_candidate_channels = sum(initial_channels_map.values())
    log(f"Initial candidate root channels: {initial_candidate_channels}")

    schedule = build_cumulative_schedule(args.prune_ratio, args.iterations)
    for round_idx, target_ratio in enumerate(schedule, start=1):
        log(f"Start round {round_idx}/{args.iterations}, cumulative target={target_ratio:.4f}")
        roots, channels = prune_one_round(model, args, device, round_idx, target_ratio, initial_channels_map)
        shape = validate_forward(model, device, args.batch_size, args.imgsz)
        log(f"Finish round {round_idx}: roots={roots}, channels={channels}, output={tuple(shape)}")

    flops_g = 0.0
    params = sum(p.numel() for p in model.parameters())
    try:
        flops_g, params = compute_model_stats(model, device, args.batch_size, args.imgsz)
    except Exception as exc:
        log(f"Failed to compute GFLOPs/Params with torch-pruning utils: {exc}")

    fps = 0.0
    try:
        fps = measure_fps(model, device, args.batch_size, args.imgsz, args.fps_warmup, args.fps_iters)
    except Exception as exc:
        log(f"Failed to measure FPS: {exc}")

    final_channels_map = collect_current_root_channel_map(model, args, device)
    final_candidate_channels = sum(final_channels_map.get(name, 0) for name in initial_channels_map)
    achieved_channel_ratio = 0.0
    if initial_candidate_channels > 0:
        achieved_channel_ratio = max(initial_candidate_channels - final_candidate_channels, 0) / initial_candidate_channels

    save_pruned_checkpoint(model, orig_ckpt, output_path)
    log(f"Saved pruned model to: {output_path}")

    try:
        from ultralytics import RTDETRMM

        reloaded = RTDETRMM(str(output_path)).model.to(device).float().eval()
        reload_shape = validate_forward(reloaded, device, args.batch_size, args.imgsz)
        log(f"Reload check passed: {tuple(reload_shape)}")
    except Exception as exc:
        log(f"Reload check failed: {exc}")

    print("\n=== Random-only pruning summary ===")
    print(f"GFLOPs : {flops_g:.3f}")
    print(f"Params : {params:,}")
    print(f"FPS    : {fps:.2f}")
    print(f"RootCh : {initial_candidate_channels} -> {final_candidate_channels} ({achieved_channel_ratio:.2%})")
    print(f"Output : {output_path}")
    print("Note   : this script uses DepGraph + random channel selection only, without extra FLOPs-biased ranking.")
    print("Resume command:")
    print_resume_command(output_path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
