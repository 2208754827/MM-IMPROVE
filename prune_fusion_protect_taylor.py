#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fusion-protected Taylor structured iterative pruning for RTDETRMM RGB/IR models.

This script is intentionally standalone and does NOT modify the existing pruning scripts.
It reuses the project's working DepGraph + Taylor pipeline while adding a soft protection
mechanism for roots that are close to multimodal fusion and decoder-critical paths.

Key ideas:
- Keep the existing hard protection from the project's Taylor pipeline.
- Add structure-aware soft protection instead of treating all remaining roots equally.
- Prefer preserving roots near fusion/decoder paths while still allowing them to compete.
- Keep recovery-related interfaces (internal slimming / decoder slimming / FPS / stats)
  aligned with the current project workflow for fair effect-first comparison.
"""

from __future__ import annotations

import argparse
import copy
import gc
from collections import deque
from datetime import datetime
from pathlib import Path

import torch

import prune_vif_v10 as base


DEFAULT_OUTPUT = "prune_outputs_fusion_protect_taylor"


def parse_args() -> argparse.Namespace:
    pia_default = Path("ReTest") / "A-DWConv-C3k2_PConv" / "weights" / "best.pt"
    default_weights = pia_default if pia_default.exists() else Path("MODEL") / "pruned_finetune_best.pt"
    parser = argparse.ArgumentParser(description="Fusion-protected Taylor iterative pruning for RTDETRMM VIF models")
    parser.add_argument("--weights", type=str, default=str(default_weights), help="Input .pt checkpoint")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output .pt path or directory")
    parser.add_argument(
        "--data",
        type=str,
        default=r"D:\BaiduNetdiskDownload\M3FD\M3FD_split\data.yaml",
        help="Dataset yaml for the printed resume command",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0, 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size")
    parser.add_argument("--prune-ratio", type=float, default=0.5, help="Target total pruning ratio")
    parser.add_argument("--iterations", type=int, default=8, help="Number of iterative pruning rounds")
    parser.add_argument("--grad-samples", type=int, default=2, help="Dummy backward passes per round")
    parser.add_argument("--min-channels", type=int, default=8, help="Minimum channels to keep for any root module")
    parser.add_argument("--round-to", type=int, default=8, help="Round pruned channels to multiples of N")
    parser.add_argument(
        "--include-neck",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the curated safe neck layers in addition to the backbone roots.",
    )
    parser.add_argument(
        "--flops-bias",
        type=float,
        default=1.25,
        help="Bias pruning selection toward channels that save more FLOPs. Larger values prefer high-resolution layers.",
    )
    parser.add_argument(
        "--layer-end",
        "--backbone-end",
        dest="layer_end",
        type=int,
        default=39,
        help="Exclusive layer end index for pruning roots. 39 covers the current CMX neck roots.",
    )
    parser.add_argument("--fps-warmup", type=int, default=10, help="Warmup iterations for FPS")
    parser.add_argument("--fps-iters", type=int, default=50, help="Benchmark iterations for FPS")
    parser.add_argument("--resume-epochs", type=int, default=30, help="Suggested epochs for the printed resume command")
    parser.add_argument("--resume-batch", type=int, default=1, help="Suggested batch size for the printed resume command")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--enable-internal-slimming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Slim large internals such as CSP_MutilScaleEdgeInformationEnhance, C3k2_PConv and DAttention FFN.",
    )
    parser.add_argument(
        "--decoder-keep-ratio",
        type=float,
        default=1.0,
        help="Keep ratio for RTDETRDecoder hidden width. Default 1.0 keeps decoder width unchanged.",
    )
    parser.add_argument(
        "--fusion-protect-weight",
        type=float,
        default=1.35,
        help="Maximum soft-protection multiplier for roots close to fusion-critical paths.",
    )
    parser.add_argument(
        "--head-protect-weight",
        type=float,
        default=1.20,
        help="Maximum soft-protection multiplier for roots close to decoder input paths.",
    )
    parser.add_argument(
        "--compound-protect-weight",
        type=float,
        default=1.10,
        help="Base protection multiplier for compound blocks such as CSP/C3k2/RepC3 roots.",
    )
    parser.add_argument(
        "--protect-hops",
        type=int,
        default=2,
        help="Graph hop radius used for both fusion and decoder soft protection.",
    )
    parser.add_argument(
        "--protect-decay",
        type=float,
        default=0.60,
        help="Distance decay for soft protection in [0, 1]. Larger means faster decay with hops.",
    )
    parser.add_argument(
        "--prune-style",
        type=str,
        choices=("balanced", "aggressive", "extreme"),
        default="balanced",
        help="Balanced keeps the current protection-first policy. Aggressive/Extreme weaken protection and also push extra slimming.",
    )
    return parser.parse_args()


def apply_prune_style(args: argparse.Namespace) -> argparse.Namespace:
    if args.prune_style == "balanced":
        return args

    if args.prune_style == "aggressive":
        args.prune_ratio = max(float(args.prune_ratio), 0.60)
        args.min_channels = min(int(args.min_channels), 4)
        args.fusion_protect_weight = min(float(args.fusion_protect_weight), 1.15)
        args.head_protect_weight = min(float(args.head_protect_weight), 1.08)
        args.compound_protect_weight = min(float(args.compound_protect_weight), 1.03)
        args.protect_hops = min(int(args.protect_hops), 1)
        args.enable_internal_slimming = True
        if float(args.decoder_keep_ratio) >= 1.0:
            args.decoder_keep_ratio = 0.90
        return args

    args.prune_ratio = max(float(args.prune_ratio), 0.70)
    args.min_channels = min(int(args.min_channels), 4)
    args.fusion_protect_weight = min(float(args.fusion_protect_weight), 1.05)
    args.head_protect_weight = min(float(args.head_protect_weight), 1.02)
    args.compound_protect_weight = min(float(args.compound_protect_weight), 1.00)
    args.protect_hops = 0
    args.enable_internal_slimming = True
    if float(args.decoder_keep_ratio) >= 1.0:
        args.decoder_keep_ratio = 0.85
    return args


def make_output_path(output_arg: str, ratio: float) -> Path:
    out = Path(output_arg)
    if out.suffix.lower() == ".pt":
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    ratio_str = f"{ratio:.2f}".rstrip("0").rstrip(".")
    timestamp = datetime.now().strftime("%m%d_%H%M")
    return out / f"pruned_fusionprotect_taylor_r{ratio_str}_{timestamp}.pt"


def sanitize_checkpoint_for_save(orig_ckpt: dict | None, model) -> dict:
    ckpt = base.sanitize_checkpoint_for_save(orig_ckpt, model)
    ckpt = copy.copy(ckpt)
    ckpt["prune_method"] = "fusion_protect_taylor"
    return ckpt


def save_pruned_checkpoint(model, orig_ckpt: dict | None, output_path: Path) -> None:
    sanitized = sanitize_checkpoint_for_save(orig_ckpt, model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sanitized, output_path)


def build_layer_graph(model) -> dict[int, set[int]]:
    graph = {idx: set() for idx in range(len(model.model))}
    for idx, layer in enumerate(model.model):
        for src in base.get_input_layer_indices(layer):
            if 0 <= src < len(model.model):
                graph[idx].add(src)
                graph[src].add(idx)
    return graph


def multi_source_bfs(graph: dict[int, set[int]], sources: set[int]) -> dict[int, int | None]:
    dist = {idx: None for idx in graph}
    queue = deque()
    for src in sorted(sources):
        if src not in graph:
            continue
        dist[src] = 0
        queue.append(src)
    while queue:
        cur = queue.popleft()
        cur_d = dist[cur]
        if cur_d is None:
            continue
        for nxt in graph[cur]:
            if dist[nxt] is None:
                dist[nxt] = cur_d + 1
                queue.append(nxt)
    return dist


def find_decoder_input_indices(model) -> set[int]:
    inputs: set[int] = set()
    for idx, layer in enumerate(model.model):
        if type(layer).__name__ != "RTDETRDecoder":
            continue
        for src in base.get_input_layer_indices(layer):
            if 0 <= src < len(model.model):
                inputs.add(src)
    return inputs


def build_protection_context(model) -> dict[str, object]:
    fusion_idx, branch_tail_idxs = base.find_fusion_info(model)
    decoder_input_idxs = find_decoder_input_indices(model)
    graph = build_layer_graph(model)

    fusion_sources = set(branch_tail_idxs)
    if fusion_idx is not None:
        fusion_sources.add(fusion_idx)
    for idx, layer in enumerate(model.model):
        inputs = set(base.get_input_layer_indices(layer))
        if fusion_idx is not None and fusion_idx in inputs:
            fusion_sources.add(idx)
        if inputs & branch_tail_idxs:
            fusion_sources.add(idx)

    return {
        "graph": graph,
        "fusion_sources": fusion_sources,
        "decoder_sources": decoder_input_idxs,
        "fusion_dist": multi_source_bfs(graph, fusion_sources),
        "decoder_dist": multi_source_bfs(graph, decoder_input_idxs),
    }


def proximity_multiplier(distance: int | None, max_hops: int, peak_weight: float, decay: float) -> float:
    if distance is None or distance < 0 or distance > max_hops:
        return 1.0
    if peak_weight <= 1.0:
        return 1.0
    span = peak_weight - 1.0
    if max_hops <= 0:
        return peak_weight if distance == 0 else 1.0
    ratio = 1.0 - min(max(distance, 0), max_hops) / max_hops
    shaped = max(0.0, ratio) ** max(float(decay), 1e-6)
    return 1.0 + span * shaped


def compute_protection_weight(model, spec, context: dict[str, object], args) -> tuple[float, list[str]]:
    layer = model.model[spec.layer_idx]
    cls_name = type(layer).__name__
    reasons: list[str] = []
    weight = 1.0

    if any(token in cls_name for token in ("CSP", "C3k2", "RepC3", "C3k_", "C2f")):
        if args.compound_protect_weight > weight:
            weight = args.compound_protect_weight
            reasons.append(f"compound:{cls_name}")

    fusion_dist = context["fusion_dist"].get(spec.layer_idx)
    fusion_weight = proximity_multiplier(
        fusion_dist, args.protect_hops, args.fusion_protect_weight, args.protect_decay
    )
    if fusion_weight > weight:
        weight = fusion_weight
    if fusion_weight > 1.0:
        reasons.append(f"fusion_hop={fusion_dist}")

    decoder_dist = context["decoder_dist"].get(spec.layer_idx)
    decoder_weight = proximity_multiplier(
        decoder_dist, args.protect_hops, args.head_protect_weight, args.protect_decay
    )
    if decoder_weight > weight:
        weight = decoder_weight
    if decoder_weight > 1.0:
        reasons.append(f"decoder_hop={decoder_dist}")

    return weight, reasons


def prune_one_round(
    model,
    args,
    device: torch.device,
    round_idx: int,
    target_ratio: float,
    initial_channels_map: dict[str, int],
    context: dict[str, object],
) -> tuple[int, int]:
    vis, ir = base.create_example_inputs(args.batch_size, args.imgsz, device)

    def refresh_gradients() -> None:
        grad_wrapper = base.ModelWrapper(model)
        try:
            base.collect_taylor_gradients(grad_wrapper, vis, ir, args.grad_samples)
        finally:
            grad_wrapper.restore()

    root_pruned = 0
    channel_pruned = 0
    chunk_size = max(1, int(args.round_to))
    target_total_pruned = base.ceil_round_prune_count(sum(initial_channels_map.values()), target_ratio, chunk_size)
    failed_roots: set[str] = set()

    while True:
        refresh_gradients()
        step_wrapper = base.ModelWrapper(model)
        pruner = None
        try:
            pruner, _ = base.build_meta_pruner(step_wrapper, base.tp_global(), args, vis, ir)
            current_specs = base.collect_root_specs(model, pruner, args.layer_end, args.include_neck)
            if not current_specs:
                break
            spatial_map = base.collect_output_spatial_map(model, current_specs, vis, ir)

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
                    scores = base.taylor_scores_for_conv_out_channels(spec.module)
                except Exception as exc:
                    base.log(f"第 {round_idx} 轮跳过 {spec.name}: {exc}")
                    failed_roots.add(spec.name)
                    continue

                idxs = base.pick_pruning_indices(scores, n_prune)
                if len(idxs) != n_prune:
                    continue

                raw_score = float(scores[idxs].mean().item())
                flops_gain = base.estimate_conv_pruning_gain(spec.module, spatial_map.get(spec.name, (0, 0)), n_prune)
                if flops_gain <= 0:
                    continue

                protect_weight, reasons = compute_protection_weight(model, spec, context, args)
                effective_score = (raw_score * protect_weight) / (flops_gain ** max(args.flops_bias, 0.0))
                candidate = (effective_score, raw_score, protect_weight, flops_gain, spec, idxs, reasons)
                if best_candidate is None or candidate[0] < best_candidate[0]:
                    best_candidate = candidate

            if best_candidate is None:
                base.log(f"第 {round_idx} 轮没有更多可安全剪枝的层，提前结束当前轮。")
                break

            _, raw_score, protect_weight, flops_gain, spec, idxs, reasons = best_candidate
            reason_text = ",".join(reasons) if reasons else "plain"
            base.log(
                f"第 {round_idx} 轮评估 root: {spec.name} "
                f"(layer={spec.layer_idx}, n_prune={len(idxs)}, raw={raw_score:.4e}, "
                f"protect={protect_weight:.3f}, gain={flops_gain:.4e}, reason={reason_text})"
            )
            try:
                group = pruner.DG.get_pruning_group(spec.module, spec.pruning_fn, idxs=idxs)
                if not pruner.DG.check_pruning_group(group):
                    base.log(f"第 {round_idx} 轮跳过 {spec.name}: 过度剪枝保护触发，仅忽略当前层")
                    failed_roots.add(spec.name)
                    continue
                group.prune()
                root_pruned += 1
                channel_pruned += len(idxs)
                base.log(f"第 {round_idx} 轮剪枝: {spec.name} -> {len(idxs)} channels")
            except Exception as exc:
                base.log(f"第 {round_idx} 轮跳过 {spec.name}: {exc}")
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


def print_resume_command(saved_path: Path, args) -> None:
    cmd = (
        f'conda run -n MM python trainMM.py --model "{saved_path}" '
        f'--data "{args.data}" --epochs {args.resume_epochs} --batch {args.resume_batch} '
        f'--imgsz {args.imgsz} --device {args.device} --name finetune_{saved_path.stem}'
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

    args = apply_prune_style(args)

    tp = base.tp_global()
    base.set_seed(args.seed)
    device = base.resolve_device(args.device)
    weights = Path(args.weights)
    output_path = make_output_path(args.output, args.prune_ratio)

    base.log(f"torch-pruning 版本: {tp.__version__}")
    base.log(f"运行设备: {device}")
    model, orig_ckpt = base.load_model(weights, device)
    base.validate_expected_architecture(model)

    initial_shape = base.validate_forward(model, device, args.batch_size, args.imgsz)
    base.log(f"初始前向输出: {tuple(initial_shape)}")

    protection_context = build_protection_context(model)
    base.log(
        "软保护上下文: "
        f"fusion_sources={sorted(protection_context['fusion_sources'])}, "
        f"decoder_sources={sorted(protection_context['decoder_sources'])}"
    )

    seed_vis, seed_ir = base.create_example_inputs(args.batch_size, args.imgsz, device)
    seed_wrapper = base.ModelWrapper(model)
    seed_pruner = None
    initial_channels_map: dict[str, int] = {}
    try:
        seed_pruner, _ = base.build_meta_pruner(seed_wrapper, tp, args, seed_vis, seed_ir)
        for spec in base.collect_root_specs(model, seed_pruner, args.layer_end, args.include_neck):
            ch = seed_pruner.DG.get_in_channels(spec.module) if spec.mode == "pair_in" else seed_pruner.DG.get_out_channels(spec.module)
            if ch is None:
                continue
            initial_channels_map[spec.name] = ch // 2 if spec.mode == "pair_in" else ch
    finally:
        seed_wrapper.restore()
        if seed_pruner is not None:
            del seed_pruner
    initial_candidate_channels = sum(initial_channels_map.values())
    base.log(f"候选 root 通道总数: {initial_candidate_channels}")

    schedule = base.build_cumulative_schedule(args.prune_ratio, args.iterations)
    for round_idx, target_ratio in enumerate(schedule, start=1):
        base.log(f"开始第 {round_idx}/{args.iterations} 轮剪枝，目标累计比例 {target_ratio:.4f}")
        roots, channels = prune_one_round(model, args, device, round_idx, target_ratio, initial_channels_map, protection_context)
        shape = base.validate_forward(model, device, args.batch_size, args.imgsz)
        base.log(f"第 {round_idx} 轮完成: roots={roots}, channels={channels}, output={tuple(shape)}")

    internal_changes = base.slim_large_modules(model, args, device)
    if internal_changes:
        shape = base.validate_forward(model, device, args.batch_size, args.imgsz)
        base.log(f"Internal slimming applied to {len(internal_changes)} modules, output={tuple(shape)}")
        for item in internal_changes:
            base.log(f"  {item}")

    decoder_slim = base.slim_decoder_hidden(model, args, device)
    if decoder_slim is not None:
        old_hd, new_hd = decoder_slim
        shape = base.validate_forward(model, device, args.batch_size, args.imgsz)
        base.log(f"RTDETRDecoder hidden_dim: {old_hd} -> {new_hd}, output={tuple(shape)}")

    flops_g = 0.0
    params = sum(p.numel() for p in model.parameters())
    try:
        flops_g, params = base.compute_model_stats(model, device, args.batch_size, args.imgsz)
    except Exception as exc:
        base.log(f"GFLOPs/Params 统计失败，回退到参数量统计: {exc}")

    fps = 0.0
    try:
        fps = base.measure_fps(model, device, args.batch_size, args.imgsz, args.fps_warmup, args.fps_iters)
    except Exception as exc:
        base.log(f"FPS 统计失败: {exc}")

    final_channels_map = base.collect_current_root_channel_map(model, args, device)
    final_candidate_channels = sum(final_channels_map.get(name, 0) for name in initial_channels_map)
    achieved_channel_ratio = 0.0
    if initial_candidate_channels > 0:
        achieved_channel_ratio = max(initial_candidate_channels - final_candidate_channels, 0) / initial_candidate_channels

    save_pruned_checkpoint(model, orig_ckpt, output_path)
    base.log(f"模型已保存: {output_path}")

    try:
        from ultralytics import RTDETRMM

        reloaded = RTDETRMM(str(output_path)).model.to(device).float().eval()
        reload_shape = base.validate_forward(reloaded, device, args.batch_size, args.imgsz)
        base.log(f"保存后回载验证通过: {tuple(reload_shape)}")
    except Exception as exc:
        base.log(f"保存后回载验证失败: {exc}")

    print("\n=== Fusion-Protected Taylor 剪枝结果 ===")
    print(f"GFLOPs : {flops_g:.3f}")
    print(f"Params : {params:,}")
    print(f"FPS    : {fps:.2f}")
    print(f"RootCh : {initial_candidate_channels} -> {final_candidate_channels} ({achieved_channel_ratio:.2%})")
    print(f"Output : {output_path}")
    print(
        "Protect: "
        f"fusion={args.fusion_protect_weight:.2f}, head={args.head_protect_weight:.2f}, "
        f"compound={args.compound_protect_weight:.2f}, hops={args.protect_hops}"
    )
    print("可直接继续恢复训练的命令:")
    print_resume_command(output_path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
