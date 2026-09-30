#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unstructured magnitude-based pruning for RTDETRMM VIF models.

=== 剪枝范式 ===
非结构化剪枝（Unstructured Pruning）

=== 方法 ===
Global Magnitude Pruning（全局幅度剪枝）

具体做法：
1. 收集模型中所有 Conv2d 和 Linear 的权重
2. 计算所有权重的绝对值
3. 全局排序，找到第 k 小的幅度值作为阈值（k = 总权重数 × 目标稀疏率）
4. 将所有幅度 ≤ 阈值的权重直接置零
5. 保存为密集格式（稀疏权重用 0 填充）

=== 与 DepGraph 结构化剪枝的根本区别 ===

| 方面         | 非结构化（本脚本）            | 结构化（DepGraph）              |
|-------------|---------------------------|-------------------------------|
| 剪枝粒度     | 单个权重                   | 整个通道                       |
| 依赖追踪     | 不需要                     | 需要 DepGraph 追踪层间依赖       |
| GFLOPs 变化  | 不变（密集推理仍计算零权重）    | 下降（通道被移除）               |
| 参数量变化   | 不变（零仍占存储）            | 下降（通道参数被删除）            |
| 模型体积     | 不变                       | 下降                           |
| 推理加速     | 需要稀疏推理引擎/硬件          | 直接加速（密集推理即可）           |
| 理论稀疏率   | 可达很高（逐权重）            | 受通道粒度限制                   |
| 实际部署价值  | 低（缺稀疏硬件时无收益）        | 高（任何设备都能加速）             |

=== 论文对比意义 ===
证明结构化剪枝对于实际部署是必要的：非结构化剪枝虽然理论稀疏率高，
但在没有稀疏推理硬件的情况下无法获得任何加速，GFLOPs/参数量/模型体积
均不变。结构化剪枝移除整个通道，在标准硬件上即可获得实际加速。

Reference:
- Han et al., "Learning both Weights and Connections for Efficient Neural Networks", NIPS 2015
- 本方法采用全局幅度排序（global magnitude ranking），是最经典的非结构化剪枝方法
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import os
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

    parser = argparse.ArgumentParser(description="Unstructured magnitude pruning for RTDETRMM VIF models")
    parser.add_argument("--weights", type=str, default=str(default_weights), help="Input .pt checkpoint")
    parser.add_argument("--output", type=str, default="prune_outputs_unstructured", help="Output .pt path or directory")
    parser.add_argument("--data", type=str, default=r"D:\BaiduNetdiskDownload\M3FD\M3FD_split\data.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--prune-ratio", type=float, default=0.50, help="Target weight sparsity ratio")
    parser.add_argument("--iterations", type=int, default=1, help="Number of pruning iterations (1 = one-shot)")
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


def apply_unstructured_pruning(model: nn.Module, sparsity: float) -> dict[str, int]:
    """Global Magnitude Pruning: zero out the smallest-magnitude weights.

    Method: Global Magnitude Ranking (Han et al., NIPS 2015)
    - Collect all Conv2d/Linear weight tensors
    - Concatenate their absolute values into a single vector
    - Find the k-th smallest value (k = total_weights × sparsity)
    - Zero out all weights whose magnitude ≤ this threshold
    """
    weight_info = []
    for name, param in model.named_parameters():
        if not param.requires_grad or param.dim() < 2:
            continue
        if "weight" not in name:
            continue
        weight_info.append((name, param))

    if not weight_info:
        return {"total_weights": 0, "zero_weights": 0, "sparsity": 0.0}

    all_magnitudes = []
    for name, param in weight_info:
        all_magnitudes.append(param.data.abs().flatten())
    all_magnitudes = torch.cat(all_magnitudes)

    total_weights = all_magnitudes.numel()
    n_prune = int(total_weights * sparsity)
    if n_prune <= 0:
        return {"total_weights": total_weights, "zero_weights": 0, "sparsity": 0.0}

    threshold = torch.kthvalue(all_magnitudes, n_prune).values.item()

    already_zero = 0
    newly_zero = 0
    for name, param in weight_info:
        mask = param.data.abs() <= threshold
        already_zero += (param.data == 0).sum().item()
        param.data[mask] = 0.0
        newly_zero += mask.sum().item()

    total_zero = sum((param.data == 0).sum().item() for _, param in weight_info)
    actual_sparsity = total_zero / total_weights if total_weights > 0 else 0.0

    log(f"  目标稀疏率: {sparsity:.1%}, 实际稀疏率: {actual_sparsity:.1%}")
    log(f"  总权重: {total_weights:,}, 置零权重: {total_zero:,}")
    log(f"  全局阈值: {threshold:.6f}")

    return {
        "total_weights": total_weights,
        "zero_weights": total_zero,
        "sparsity": actual_sparsity,
        "threshold": threshold,
    }


def make_output_path(output_arg: str, ratio: float) -> Path:
    out = Path(output_arg)
    if out.suffix.lower() == ".pt":
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    ratio_str = f"{ratio:.2f}".rstrip("0").rstrip(".")
    timestamp = datetime.now().strftime("%m%d_%H%M")
    return out / f"pruned_unstructured_r{ratio_str}_{timestamp}.pt"


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
    ckpt["prune_method"] = "unstructured_magnitude"
    ckpt["prune_sparsity"] = "unstructured"
    return ckpt


def compute_model_stats(model, device, batch_size, imgsz):
    """Compute GFLOPs and params (same as dense model for unstructured)."""
    import torch_pruning as tp
    from prune_vif_v10 import ModelWrapper

    vis = torch.randn(batch_size, 3, imgsz, imgsz)
    ir = torch.randn(batch_size, 3, imgsz, imgsz)

    wrapper = ModelWrapper(model)
    try:
        wrapper.eval()
        flops, params = tp.utils.count_ops_and_params(wrapper, (vis, ir))
        return flops / 1e9, int(params)
    finally:
        wrapper.restore()


def compute_fps(model, device, batch_size, imgsz, warmup, iters):
    """Benchmark FPS on CPU."""
    from prune_vif_v10 import ModelWrapper
    vis = torch.randn(batch_size, 3, imgsz, imgsz)
    ir = torch.randn(batch_size, 3, imgsz, imgsz)

    wrapper = ModelWrapper(model)
    try:
        wrapper.eval()
        with torch.no_grad():
            for _ in range(warmup):
                wrapper(vis, ir)
            t0 = time.perf_counter()
            for _ in range(iters):
                wrapper(vis, ir)
            elapsed = time.perf_counter() - t0
        return iters / elapsed
    finally:
        wrapper.restore()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    log(f"=== 非结构化幅度剪枝 (Global Magnitude Pruning) ===")
    log(f"方法: 全局幅度排序 — Han et al., NIPS 2015")
    log(f"权重: {args.weights}")
    log(f"目标稀疏率: {args.prune_ratio:.1%}")

    from ultralytics import RTDETRMM
    model = RTDETRMM(args.weights)

    # Unfreeze parameters — loaded checkpoints may have requires_grad=False
    for p in model.model.parameters():
        if p.dtype.is_floating_point:
            p.requires_grad_(True)

    # Move to CPU to avoid GPU tensor issues with torch_pruning op counter
    model.model.to("cpu")

    orig_ckpt = None
    try:
        from ultralytics.nn.tasks import torch_safe_load
        orig_ckpt = torch_safe_load(Path(args.weights), device=device)
    except Exception:
        pass

    gflops_before, params_before = compute_model_stats(model.model, device, args.imgsz, 1)
    log(f"剪枝前: GFLOPs={gflops_before:.1f}, Params={params_before:,}")

    per_iter_sparsity = args.prune_ratio / args.iterations
    for i in range(args.iterations):
        cumulative = per_iter_sparsity * (i + 1)
        log(f"--- 迭代 {i+1}/{args.iterations} (目标累计稀疏率: {cumulative:.1%}) ---")
        stats = apply_unstructured_pruning(model.model, per_iter_sparsity)
        log(f"  当前总稀疏率: {stats['sparsity']:.1%}")

    gflops_after, params_after = compute_model_stats(model.model, device, args.imgsz, 1)
    log(f"剪枝后: GFLOPs={gflops_after:.1f}, Params={params_after:,}")
    log(f"  ⚠ 非结构化剪枝 GFLOPs/参数量不变，需要稀疏推理引擎才能加速")

    total_params = 0
    zero_params = 0
    for name, param in model.model.named_parameters():
        if param.dim() >= 2 and "weight" in name:
            total_params += param.numel()
            zero_params += (param.data == 0).sum().item()
    final_sparsity = zero_params / total_params if total_params > 0 else 0.0
    log(f"最终权重稀疏率: {final_sparsity:.1%} ({zero_params:,}/{total_params:,})")

    try:
        fps = compute_fps(model.model, device, 1, args.imgsz, args.fps_warmup, args.fps_iters)
        log(f"FPS: {fps:.1f} (密集推理，无加速)")
    except Exception as e:
        log(f"FPS 测量失败: {e}")
        fps = 0.0

    output_path = make_output_path(args.output, args.prune_ratio)
    ckpt = sanitize_checkpoint_for_save(orig_ckpt, model.model)
    torch.save(ckpt, str(output_path))
    log(f"已保存: {output_path}")

    log(f"\n微调命令:")
    log(f"  python finetune_pruned_r03.py --model {output_path} --data {args.data} "
        f"--epochs {args.resume_epochs} --batch {args.resume_batch} --name unstructured-r{args.prune_ratio}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
