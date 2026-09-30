#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BN-Slimming with L1 regularization training for RTDETRMM VIF models.

=== 剪枝范式 ===
训练时稀疏化剪枝（Training-time Sparsification Pruning）

=== 方法 ===
Network Slimming (Liu et al., ICCV 2017)

具体做法：
1. 在训练的损失函数中添加 BN gamma 的 L1 正则化项: L_total = L_task + λ × Σ|γ_i|
2. L1 正则化推动不重要的 BN gamma 趋近于零
3. 训练若干 epoch 后，BN gamma 接近零的通道对应的输出趋近于零
4. 按 BN gamma 绝对值从小到大排序，剪掉最小的通道
5. 使用 DepGraph 进行结构化通道移除（保证维度一致性）
6. 微调恢复精度

=== 与 DepGraph 后训练剪枝的根本区别 ===

| 方面           | BN-Slimming（本脚本）         | DepGraph 后训练剪枝            |
|---------------|---------------------------|-------------------------------|
| 剪枝时机       | 训练时（需额外稀疏化训练）      | 训练后（直接剪，不需重训）        |
| 重要性度量     | BN gamma 幅度               | Taylor 一阶梯度近似             |
| 额外训练代价   | ✅ 需要 30 epoch 稀疏化训练   | ❌ 不需要                      |
| FLOPs 感知    | ❌ 无                       | ✅ flops_bias 优先剪高 FLOPs 层 |
| 内部模块剪枝   | ❌ 不剪 CSP/C3k2_PConv 内部  | ✅ internal slimming           |
| 梯度信息       | ❌ 不使用                    | ✅ 使用梯度信息                 |

=== 论文对比意义 ===
证明后训练剪枝（DepGraph）可以在不需要额外稀疏化训练的情况下，
达到甚至超过训练时稀疏化方法的效果。这减少了剪枝的整体时间成本。

Reference:
- Liu et al., "Learning Efficient Convolutional Networks through Network Slimming", ICCV 2017
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def parse_args() -> argparse.Namespace:
    pia_default = Path("ReTest") / "A-DWConv-C3k2_PConv" / "weights" / "best.pt"
    default_weights = pia_default if pia_default.exists() else Path("MODEL") / "pruned_finetune_best.pt"

    parser = argparse.ArgumentParser(description="BN-Slimming training + pruning for RTDETRMM VIF models")
    parser.add_argument("--weights", type=str, default=str(default_weights), help="Input .pt checkpoint")
    parser.add_argument("--output", type=str, default="prune_outputs_bnslim_train", help="Output .pt path or directory")
    parser.add_argument("--data", type=str, default=r"D:\BaiduNetdiskDownload\M3FD\M3FD_split\data.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for sparsification training")
    parser.add_argument("--prune-ratio", type=float, default=0.50, help="Target channel pruning ratio after sparsification")
    parser.add_argument("--sparsify-epochs", type=int, default=30, help="Epochs for L1 sparsification training")
    parser.add_argument("--sparsify-lr", type=float, default=1e-4, help="Learning rate for sparsification")
    parser.add_argument("--l1-coeff", type=float, default=1e-3, help="L1 regularization coefficient on BN gamma")
    parser.add_argument("--min-channels", type=int, default=4)
    parser.add_argument("--round-to", type=int, default=4)
    parser.add_argument("--layer-end", type=int, default=39, help="Prune layers 0 to layer_end-1")
    parser.add_argument("--include-neck", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps-warmup", type=int, default=10)
    parser.add_argument("--fps-iters", type=int, default=50)
    parser.add_argument("--resume-epochs", type=int, default=30)
    parser.add_argument("--resume-batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    ds = str(device).strip().lower()
    if ds == "cpu":
        return torch.device("cpu")
    if ds.isdigit():
        ds = f"cuda:{ds}"
    if ds.startswith("cuda") and torch.cuda.is_available():
        return torch.device(ds)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_bn_l1_loss(model: nn.Module) -> torch.Tensor:
    """Sum of absolute BN gamma values across all BN layers.

    This is the L1 regularization term from Network Slimming (Liu et al., ICCV 2017).
    Minimizing this pushes BN scaling factors toward zero, making the corresponding
    channels unimportant and safe to prune.
    """
    l1_sum = torch.tensor(0.0, device=next(model.parameters()).device)
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if module.affine and module.weight is not None:
                l1_sum = l1_sum + module.weight.abs().sum()
    return l1_sum


def sparsify_training(model, data_path: str, args, device: torch.device) -> None:
    """Fine-tune the model with L1 regularization on BN gamma.

    Network Slimming approach (Liu et al., ICCV 2017):
    - Total loss = task loss + λ × Σ|γ_i|
    - L1 penalty pushes unimportant BN gamma toward zero
    - After sparsification, channels with near-zero gamma are pruned
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ultralytics import RTDETRMM

    log(f"=== BN-Slimming 稀疏化训练 (Network Slimming, Liu et al. ICCV 2017) ===")
    log(f"稀疏化训练: {args.sparsify_epochs} epochs")
    log(f"L1 系数 λ: {args.l1_coeff}")

    rtdetr = RTDETRMM(args.weights)

    # Patch the criterion to add BN L1 penalty
    if hasattr(rtdetr.model, 'criterion') and rtdetr.model.criterion is not None:
        original_forward = rtdetr.model.criterion.forward
        _l1_coeff = args.l1_coeff

        def patched_criterion_forward(self_criterion, preds, targets):
            loss_dict = original_forward(preds, targets)
            bn_l1 = compute_bn_l1_loss(rtdetr.model)
            if isinstance(loss_dict, dict):
                loss_dict["bn_l1_loss"] = bn_l1 * _l1_coeff
            return loss_dict

        rtdetr.model.criterion.forward = patched_criterion_forward.__get__(rtdetr.model.criterion)

    results = rtdetr.train(
        data=data_path,
        epochs=args.sparsify_epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        lr0=args.sparsify_lr,
        cos_lr=True,
        device=args.device,
        project=str(Path(args.output) / "sparsify_run"),
        name="bn_slimming_sparsify",
        exist_ok=True,
        save_period=-1,
        verbose=True,
    )

    log(f"稀疏化训练完成")


def prune_by_bn_gamma(model: nn.Module, args, device: torch.device) -> dict:
    """Prune channels with smallest BN gamma after sparsification training.

    Uses DepGraph for structural dependency tracking (to avoid dimension mismatches),
    but the importance criterion is purely BN gamma magnitude (from Network Slimming).
    """
    import torch_pruning as tp
    from prune_vif_v10 import (
        ModelWrapper, collect_root_specs, find_protected_layer_indices,
        ceil_round_prune_count, SUPPORTED_FUSION_TYPES, CONV_LIKE_TYPES,
    )

    log(f"=== BN-Gamma 结构化剪枝 (目标比例: {args.prune_ratio:.1%}) ===")

    vis = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    ir = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    wrapper = ModelWrapper(model)
    pruner = tp.pruner.MetaPruner(
        wrapper,
        example_inputs=(vis, ir),
        importance=tp.importance.BNScaleImportance(group_reduction="mean", normalizer="mean"),
        pruning_ratio=args.prune_ratio,
        iterative_steps=1,
        ignored_layers=[],
        root_module_types=[nn.Conv2d],
        round_to=args.round_to,
    )

    specs = collect_root_specs(model, pruner, args.layer_end, args.include_neck)
    log(f"候选根模块: {len(specs)}")

    initial_channels = {}
    for spec in specs:
        ch = pruner.DG.get_out_channels(spec.module)
        if ch is not None:
            initial_channels[spec.name] = ch

    total_target = ceil_round_prune_count(
        sum(initial_channels.values()), args.prune_ratio, args.round_to
    )
    log(f"目标剪枝通道数: {total_target} / {sum(initial_channels.values())}")

    chunk_size = max(1, args.round_to)
    current_total_pruned = 0
    failed_roots = set()
    round_idx = 0

    while current_total_pruned < total_target:
        wrapper = ModelWrapper(model)
        pruner = tp.pruner.MetaPruner(
            wrapper,
            example_inputs=(vis, ir),
            importance=tp.importance.BNScaleImportance(group_reduction="mean", normalizer="mean"),
            pruning_ratio=args.prune_ratio,
            iterative_steps=1,
            ignored_layers=[],
            root_module_types=[nn.Conv2d],
            round_to=args.round_to,
        )
        specs = collect_root_specs(model, pruner, args.layer_end, args.include_neck)

        best_candidate = None
        for spec in specs:
            if spec.name in failed_roots:
                continue
            channels = pruner.DG.get_out_channels(spec.module)
            if channels is None:
                continue
            if channels - args.min_channels < chunk_size:
                continue

            remaining = total_target - current_total_pruned
            n_prune = min(chunk_size, remaining)
            if n_prune <= 0:
                continue

            bn_module = None
            for parent in model.modules():
                if hasattr(parent, "conv") and parent.conv is spec.module and hasattr(parent, "bn"):
                    bn_module = parent.bn
                    break
            if bn_module is None or not isinstance(bn_module, nn.modules.batchnorm._BatchNorm):
                continue
            if not bn_module.affine:
                continue

            scores = bn_module.weight.detach().abs()
            sorted_indices = torch.argsort(scores)
            idxs = sorted_indices[:n_prune].tolist()

            raw_score = float(scores[idxs].mean().item())
            candidate = (raw_score, spec, idxs)
            if best_candidate is None or candidate[0] < best_candidate[0]:
                best_candidate = candidate

        if best_candidate is None:
            log(f"没有更多可安全剪枝的层")
            break

        raw_score, spec, idxs = best_candidate
        try:
            group = pruner.DG.get_pruning_group(spec.module, spec.pruning_fn, idxs=idxs)
            if not pruner.DG.check_pruning_group(group):
                failed_roots.add(spec.name)
                continue
            group.prune()
            current_total_pruned += len(idxs)
            round_idx += 1
            if round_idx % 10 == 0:
                log(f"  已剪 {current_total_pruned}/{total_target} 通道")
        except Exception as exc:
            log(f"跳过 {spec.name}: {exc}")
            failed_roots.add(spec.name)
            continue
        finally:
            wrapper.restore()
            del pruner
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    log(f"BN-Gamma 剪枝完成: 共剪 {current_total_pruned} 通道")
    return {"channels_pruned": current_total_pruned, "total_target": total_target}


def make_output_path(output_arg: str, ratio: float) -> Path:
    out = Path(output_arg)
    if out.suffix.lower() == ".pt":
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    ratio_str = f"{ratio:.2f}".rstrip("0").rstrip(".")
    timestamp = datetime.now().strftime("%m%d_%H%M")
    return out / f"pruned_bnslim_train_r{ratio_str}_{timestamp}.pt"


def sanitize_checkpoint_for_save(orig_ckpt: dict | None, model: nn.Module) -> dict:
    ckpt = copy.copy(orig_ckpt) if isinstance(orig_ckpt, dict) else {}
    ckpt.pop("train_args", None)
    for key in ("optimizer", "ema", "updates", "best_fitness"):
        ckpt.pop(key, None)

    save_model = copy.deepcopy(model).eval().half()
    if hasattr(save_model, "criterion"):
        save_model.criterion = None
    if hasattr(save_model, "model") and hasattr(save_model.model, "criterion"):
        save_model.model.criterion = None

    for p in save_model.parameters():
        p.requires_grad_(False)

    ckpt["epoch"] = -1
    ckpt["model"] = save_model
    ckpt["pruned"] = True
    ckpt["date"] = datetime.now().isoformat()
    ckpt["prune_method"] = "bn_slimming_train_l1"
    return ckpt


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    log(f"=== BN-Slimming 训练时稀疏化剪枝 (Network Slimming, Liu et al. ICCV 2017) ===")
    log(f"权重: {args.weights}")
    log(f"目标剪枝比例: {args.prune_ratio:.1%}")
    log(f"稀疏化训练: {args.sparsify_epochs} epochs, L1系数 λ={args.l1_coeff}")

    # Step 1: Sparsification training with L1 regularization on BN gamma
    sparsify_training(None, args.data, args, device)

    # Load the sparsified model (best checkpoint from sparsification run)
    sparsify_best = Path(args.output) / "sparsify_run" / "bn_slimming_sparsify" / "weights" / "best.pt"
    if not sparsify_best.exists():
        log(f"错误: 稀疏化训练产出未找到: {sparsify_best}")
        return 1

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ultralytics import RTDETRMM
    model = RTDETRMM(str(sparsify_best))
    orig_ckpt = None
    try:
        from ultralytics.nn.tasks import torch_safe_load
        orig_ckpt = torch_safe_load(sparsify_best, device=device)
    except Exception:
        pass

    # Step 2: Prune by BN gamma magnitude
    stats = prune_by_bn_gamma(model.model, args, device)

    # Stats
    import torch_pruning as tp
    from prune_vif_v10 import ModelWrapper
    vis = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    ir = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    wrapper = ModelWrapper(model.model)
    try:
        flops, params = tp.utils.count_ops_and_params(wrapper, (vis, ir))
        log(f"剪枝后: GFLOPs={flops/1e9:.1f}, Params={params:,}")
    finally:
        wrapper.restore()

    # Save
    output_path = make_output_path(args.output, args.prune_ratio)
    ckpt = sanitize_checkpoint_for_save(orig_ckpt, model.model)
    torch.save(ckpt, str(output_path))
    log(f"已保存: {output_path}")

    log(f"\n微调命令:")
    log(f"  python finetune_pruned_r03.py --model {output_path} --data {args.data} "
        f"--epochs {args.resume_epochs} --batch {args.resume_batch} --name bnslim-train-r{args.prune_ratio}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
