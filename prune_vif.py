#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VIF (Visible-Infrared Fusion) 双分支目标检测模型剪枝脚本 v9

核心策略：
1. 跨平台路径清理：保存前删除 train_args 中的 Linux 路径信息，防止 Windows 微调报错
2. 精确保护：只保护 CSP cv1+m 和被 head 引用的层，放开 cv2.conv
3. 迭代剪枝：每轮重新建图，积累剪枝率
4. group.exec() → 自动同步 BN + 下游 Conv in_channels
5. 强制 model(vis, ir) 前向自检 → 通过才保存
6. 结束时打印 Markdown 表格剪枝指标

使用方法:
    python prune_vif.py --weights D:/JiQI/MM-experiment/ResTest/xxx/weights/best.pt --prune-ratio 0.3
    python prune_vif.py --weights D:/JiQI/MM-experiment/ResTest/xxx/weights/best.pt --prune-ratio 0.3 --iterations 3
"""

import argparse
import gc
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


# ═══════════════════════════════════════════════════════════════
#  基础工具
# ═══════════════════════════════════════════════════════════════

def check_torch_pruning():
    try:
        import torch_pruning as tp
        print(f"[环境] torch_pruning 版本: {tp.__version__}")
        return tp
    except ImportError:
        print("错误: 未安装 torch-pruning，请执行: pip install torch-pruning")
        sys.exit(1)


def load_vif_model(weights_path: Path, device: str = "cuda"):
    from ultralytics import RTDETRMM
    from ultralytics.nn.tasks import torch_safe_load
    print(f"[加载] 从权重文件加载: {weights_path}")
    uw = RTDETRMM(str(weights_path))
    model = uw.model.to(device).eval()
    # 同时保留原始 ckpt（含 train_args / date / epoch 等字段），保存时需要
    ckpt, _ = torch_safe_load(str(weights_path))
    return model, uw, ckpt


def create_example_inputs(batch_size: int = 1, img_size: int = 640, device: str = "cuda"):
    x_rgb = torch.randn(batch_size, 3, img_size, img_size, device=device)
    x_ir  = torch.randn(batch_size, 3, img_size, img_size, device=device)
    return x_rgb, x_ir


def count_params_and_flops(model: nn.Module, img_size: int = 640) -> Tuple[int, float]:
    from ultralytics.utils.torch_utils import get_flops
    params = sum(p.numel() for p in model.parameters())
    try:
        flops = get_flops(model, imgsz=[img_size, img_size])
    except Exception as e:
        print(f"  [警告] FLOPs 计算失败: {e}")
        flops = 0.0
    return params, flops


def measure_speed(model: nn.Module, example_inputs, warmup: int = 10,
                  iters: int = 50, device: str = "cuda") -> Tuple[float, float]:
    merged = torch.cat(example_inputs, dim=1)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            try:
                model(merged)
            except Exception:
                pass
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            try:
                model(merged)
                if device == "cuda":
                    torch.cuda.synchronize()
            except Exception:
                pass
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000, iters / (t1 - t0)


# ═══════════════════════════════════════════════════════════════
#  PruningModelWrapper
# ═══════════════════════════════════════════════════════════════

class PruningModelWrapper(nn.Module):
    """
    包装模型，使 forward 只返回第一个 Tensor，屏蔽 None。
    同时设置 head.export=True，使 RTDETRDecoder 只输出 y（单一 Tensor），
    确保 DependencyGraph 能成功追踪。
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._head = model.model[-1]
        self._old_export = getattr(self._head, 'export', False)
        self._head.export = True
        print(f"[Wrapper] head.export 已设为 True，forward 只返回第一个 Tensor")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            for item in out:
                if isinstance(item, torch.Tensor):
                    return item
        if isinstance(out, torch.Tensor):
            return out
        raise RuntimeError(f"[Wrapper] 无法从输出中提取 Tensor，类型: {type(out)}")

    def restore(self):
        self._head.export = self._old_export
        print(f"[Wrapper] head.export 已恢复为 {self._old_export}")


# ═══════════════════════════════════════════════════════════════
#  受保护模块识别
# ═══════════════════════════════════════════════════════════════

# 受保护的类名：含这些类的依赖组不执行剪枝
_PROTECTED_CLASS_NAMES = {
    "RTDETRDecoder",
    "DAttention",
}


def _build_protected_module_ids(model: nn.Module) -> Set[int]:
    """
    收集模型中所有受保护模块（RTDETRDecoder / DAttention 及其所有子模块）的 id。
    用于在依赖组中快速判断是否含受保护模块。
    """
    protected_ids: Set[int] = set()
    for m in model.modules():
        if type(m).__name__ in _PROTECTED_CLASS_NAMES:
            # 将该模块及其所有子模块都加入保护集合
            for sub in m.modules():
                protected_ids.add(id(sub))
    return protected_ids


def _build_backbone_conv_ids(model: nn.Module, backbone_end: int = 16) -> Set[int]:
    """
    收集 backbone（model.model[0..backbone_end-1]）中所有 Conv2d 的 id。
    只有属于 backbone 的 Conv2d 才会作为剪枝起点。
    """
    conv_ids: Set[int] = set()
    seq = model.model
    for i in range(min(backbone_end, len(seq))):
        layer = seq[i]
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                conv_ids.add(id(m))
    return conv_ids


# ═══════════════════════════════════════════════════════════════
#  诊断：打印 backbone 每层信息
# ═══════════════════════════════════════════════════════════════

def diagnose_backbone(model: nn.Module, backbone_end: int = 16,
                      protected_ids: Set[int] = None):
    """
    遍历 backbone 层，打印每层的类型、Conv2d 数量、保护状态。
    """
    print("\n" + "=" * 72)
    print("  Backbone 层诊断")
    print("=" * 72)
    print(f"  {'层':<5} {'类型':<42} {'Conv2d数':<10} {'状态'}")
    print("-" * 72)

    seq = model.model
    n_layers = min(backbone_end, len(seq))

    # 找融合层索引（f 字段为列表的多输入层）
    fusion_idxs = set()
    branch_tail_idxs = set()
    for i in range(n_layers):
        m = seq[i]
        if isinstance(m.f, (list, tuple)) and len(m.f) >= 2:
            fusion_idxs.add(i)
            for j in m.f:
                if isinstance(j, int) and j >= 0:
                    branch_tail_idxs.add(j)

    # backbone 最后一层输出进入 head
    branch_tail_idxs.add(n_layers - 1)

    for i in range(n_layers):
        layer = seq[i]
        layer_type = type(layer).__name__
        convs = [m for m in layer.modules() if isinstance(m, nn.Conv2d)]

        has_protected = (protected_ids is not None and
                         any(id(m) in protected_ids for m in layer.modules()))

        if has_protected:
            status = "🔒 含受保护模块（DAttention/Decoder）"
        elif i in fusion_idxs:
            status = "⚡ 融合层（多输入）"
        elif i in branch_tail_idxs:
            status = "🛡 分支末端（输出受保护）"
        else:
            status = "✅ 可剪枝"

        print(f"  {i:<5} {layer_type:<42} {len(convs):<10} {status}")

    print("-" * 72)
    print(f"  融合层: {sorted(fusion_idxs)}")
    print(f"  分支末端保护: {sorted(branch_tail_idxs)}")
    print("=" * 72)

    return fusion_idxs, branch_tail_idxs


# ═══════════════════════════════════════════════════════════════
#  Taylor 梯度收集
# ═══════════════════════════════════════════════════════════════

def collect_taylor_gradients(wrapper: PruningModelWrapper, example_inputs,
                              num_samples: int = 8):
    """通过 wrapper 的多次 forward+backward 积累 Taylor 梯度。"""
    print(f"\n[梯度收集] {num_samples} 次 forward+backward（Taylor 重要性）...")
    wrapper.eval()

    if isinstance(example_inputs, tuple):
        merged = torch.cat(example_inputs, dim=1)
    else:
        merged = example_inputs

    wrapper.zero_grad()
    for i in range(num_samples):
        x = torch.randn_like(merged)
        try:
            out = wrapper(x)
            loss = out.sum() if isinstance(out, torch.Tensor) else \
                   sum(o.sum() for o in out if isinstance(o, torch.Tensor))
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                loss.backward()
            if (i + 1) % 4 == 0:
                print(f"  {i+1}/{num_samples} 次完成")
        except Exception as e:
            print(f"  [警告] 第 {i+1} 次失败: {e}")
    print("[梯度收集] 完成")


def _get_importance(conv: nn.Conv2d) -> torch.Tensor:
    """Taylor 重要性（优先），fallback 到 L1。返回形状 (out_channels,)。"""
    w = conv.weight
    if w.grad is not None:
        return (w * w.grad).abs().sum(dim=(1, 2, 3)).detach()
    with torch.no_grad():
        return w.abs().sum(dim=(1, 2, 3))


def _calc_prune_n(n_channels: int, ratio: float,
                  align: int = 8, min_remain: int = 4) -> int:
    n_prune = int(n_channels * ratio)
    n_prune = (n_prune // align) * align
    if n_channels - n_prune < min_remain:
        n_prune = max(0, ((n_channels - min_remain) // align) * align)
    return n_prune


# ═══════════════════════════════════════════════════════════════
#  核心：DepGraph + 手动循环扫描
# ═══════════════════════════════════════════════════════════════


class _StubMMRouter:
    """
    哑路由器（Stub）：替换真实 mm_router 用于 DepGraph tracing。

    RTDETRDetectionModel._predict_once 在每层调用 route_layer_input，
    根据 module._mm_input_source 和 module._mm_new_input_start 切分双模态输入：
      - _mm_input_source='RGB' → 取 6ch 输入的前 3ch（RGB 分支）
      - _mm_new_input_start=True + _mm_input_source='X' → 取后 3ch（IR 分支）
      - _mm_input_source='NONE' → 不路由，返回 None（使用上层输出）

    setup_multimodal_routing 返回 (True, input_sources)，启用逐层路由，
    确保 model[0]（RGB起点）和 model[5]（IR起点）都收到正确的 3ch 输入，
    让 DepGraph tracing 能正常追踪到所有 Conv2d 节点。
    """
    def setup_multimodal_routing(self, x, profile=False):
        """返回 (True, input_sources)，启用路由，input_sources 按通道切分。"""
        if isinstance(x, torch.Tensor) and x.shape[1] == 6:
            input_sources = {
                'RGB': x[:, :3, :, :],   # 前 3ch
                'X':   x[:, 3:, :, :],   # 后 3ch
            }
        else:
            # 通道数非 6（如已是单分支），不切分
            input_sources = {'RGB': x, 'X': x}
        return True, input_sources

    def route_layer_input(self, x, module, input_sources, profile=False):
        """根据层属性路由输入，与真实 router 逻辑对齐。"""
        if not hasattr(module, '_mm_input_source'):
            return None
        mm_src = module._mm_input_source
        new_start = getattr(module, '_mm_new_input_start', False)

        if new_start:
            # IR 分支起点：强制切换到 X 模态
            return input_sources.get('X', None)
        if mm_src in ('RGB', 'X'):
            return input_sources.get(mm_src, None)
        # 'NONE' 或未知 → 不路由
        return None

    def reset_spatial_input(self, x, m, mm_input_sources, profile=False):
        return x

    def update_dataset_config(self, *a, **kw):
        pass

    def set_runtime_params(self, *a, **kw):
        pass


def _get_backbone_end(model: nn.Module) -> int:
    """backbone 末端索引（不含），YAML 中 backbone=0~15。"""
    return 16


def _get_backbone_direct_output_idxs(model: nn.Module, backbone_end: int) -> Set[int]:
    """
    获取 backbone 中输出被 head 直接引用的层索引集合。
    包含：
    1. head 层中明确用正整数索引引用的 backbone 层（如 [11, ...]  [2, ...] 等）
    2. 已知 backbone 末端层（backbone_end - 1）的输出直接进入 head（相对索引 -1）
    """
    seq = model.model
    protected_backbone_idxs: Set[int] = set()
    for i in range(backbone_end, len(seq)):
        m = seq[i]
        f = m.f
        refs = [f] if isinstance(f, int) else (list(f) if isinstance(f, (list, tuple)) else [])
        for ref in refs:
            if isinstance(ref, int):
                # 将相对索引转绝对索引
                abs_ref = ref if ref >= 0 else i + ref
                if 0 <= abs_ref < backbone_end:
                    protected_backbone_idxs.add(abs_ref)
    return protected_backbone_idxs


def build_dependency_graph(model: nn.Module, example_inputs, tp):
    """
    直接对原始 model 建图。

    核心思路：
    1. _StubMMRouter 替换真实 mm_router，确保双分支路由正确
    2. 只 hook backbone 最后一层（backbone[15]），返回其输出 tensor
       — backbone[15] 的输出包含了所有前面层的计算图路径（顺序执行）
       — 返回单个 tensor（非 sum 累加），DepGraph 遍历不会指数爆炸
    3. head 中直接引用 backbone[2,11,13] 的 FPN conv 通过保护机制处理：
       这些 backbone 层的输出 Conv 加入 protected_out_conv_ids，不被剪枝
       → head FPN 的 in_channels 永远不变，前向通过
    4. RTDETRDecoder 不在追踪路径中，get_pruning_group 瞬时完成
    """
    print("\n[DepGraph] 构建依赖图（StubMMRouter + backbone[last] hook）...")

    backbone_end = _get_backbone_end(model)

    if isinstance(example_inputs, tuple):
        merged = torch.cat(example_inputs, dim=1)
    else:
        merged = example_inputs

    # ── 用 Stub 替换 mm_router ──
    old_mm_router = getattr(model, 'mm_router', None)
    model.mm_router = _StubMMRouter()
    print(f"[DepGraph] mm_router 已替换为 StubMMRouter（原值: {'有' if old_mm_router is not None else '无'}）")

    # ── 只 hook backbone 最后一层（backbone_end-1），返回其输出 tensor ──
    # 返回单个 feature map（非 sum 标量），DepGraph 遍历路径有限，不会卡死
    # backbone 是顺序执行的，最后一层的 grad_fn 树包含所有前面层的路径
    last_bb_idx = backbone_end - 1  # = 15
    _bb_last_out = []

    def _bb_hook(m, inp, out):
        _bb_last_out.clear()
        t = out
        if isinstance(t, (list, tuple)):
            for o in t:
                if isinstance(o, torch.Tensor):
                    t = o; break
            else:
                return
        if isinstance(t, torch.Tensor):
            _bb_last_out.append(t)

    seq = model.model
    hook_handle = seq[last_bb_idx].register_forward_hook(_bb_hook)

    was_training = model.training
    model.train()

    def _forward_fn(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        _bb_last_out.clear()
        try:
            m(x)
        except Exception:
            pass
        if not _bb_last_out:
            raise RuntimeError("[DepGraph] backbone 最后一层无输出")
        t = _bb_last_out[0]
        # 确保 tensor 有 grad_fn（train 模式 + requires_grad 输入保证）
        if not t.requires_grad:
            t = t.detach().requires_grad_(True)
        return t

    try:
        DG = tp.DependencyGraph()
        merged_grad = merged.detach().requires_grad_(True)
        DG.build_dependency(model, example_inputs=merged_grad, forward_fn=_forward_fn)

        n_nodes = len(DG.module2node)
        n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d) and m in DG.module2node)
        print(f"[DepGraph] 依赖图中共有 {n_nodes} 个节点，其中 Conv2d: {n_conv} 个")
        if n_nodes == 0:
            print("[DepGraph] 警告：节点数为 0！")
            return None
        for chk_i in [2, 11, 13, 15]:
            if chk_i < len(seq):
                chk_layer = seq[chk_i]
                chk_conv = None
                if hasattr(chk_layer, 'conv') and isinstance(chk_layer.conv, nn.Conv2d):
                    chk_conv = chk_layer.conv
                elif hasattr(chk_layer, 'cv1') and hasattr(chk_layer.cv1, 'conv'):
                    chk_conv = chk_layer.cv1.conv
                if chk_conv is not None:
                    print(f"  backbone[{chk_i}].conv in graph: {chk_conv in DG.module2node}")
        print("[DepGraph] 构建成功！")
        return DG

    except Exception as e:
        print(f"[DepGraph] 构建失败: {e}")
        traceback.print_exc()
        return None

    finally:
        hook_handle.remove()
        model.mm_router = old_mm_router
        if not was_training:
            model.eval()
        print("[DepGraph] 清理完毕，mm_router 已恢复")


def _group_contains_protected(group, protected_ids: Set[int]) -> bool:
    """检查依赖组内是否含有受保护模块（RTDETRDecoder / DAttention）。"""
    for dep in group:
        # torch_pruning 依赖项：dep.target.module
        try:
            m = dep.target.module
            if id(m) in protected_ids:
                return True
        except Exception:
            pass
    return False


def _get_named_backbone_convs(model: nn.Module, backbone_end: int,
                               backbone_conv_ids: Set[int],
                               dg) -> List[Tuple[str, nn.Conv2d]]:
    """
    收集所有在 DepGraph 中的 backbone Conv2d，并生成名称。
    只返回 DG.module2node 中存在的 conv，其余跳过（防止 get_pruning_group 报错）。
    """
    named_convs = []
    seq = model.model
    for i in range(min(backbone_end, len(seq))):
        layer = seq[i]
        for subname, m in layer.named_modules():
            if isinstance(m, nn.Conv2d) and id(m) in backbone_conv_ids:
                if m in dg.module2node:  # 必须在图中
                    full_name = f'model.{i}.{subname}' if subname else f'model.{i}'
                    named_convs.append((full_name, m))
    return named_convs


def _build_protected_output_conv_ids(model: nn.Module, backbone_end: int) -> Set[int]:
    """
    精确保护策略，最大化可剪枝空间：

    1. 被 head 直接引用的 backbone 层（[2, 11, 13]）内部所有 Conv2d
       — 防止 head FPN in_channels 失步

    2. CSP 块（C2f/C2fVariantBase）：只保护 cv1 + m 内部 Conv2d，
       放开 cv2.conv（出口 Conv，out_channels = 整个块的输出维度）
       — cv1：out_channels/2 = self.c，与 MSIE local_conv.in_channels 硬绑定，不能剪
       — m 内部：chunk/concat 路径，DepGraph 追踪不完整，不能剪
       — cv2：out_channels = 整个块输出维度，DepGraph 能正确传播依赖，可剪

    3. PIAFusionBlock：整个内部全保护
       — element-wise 操作，DepGraph 会错误追踪到下游 in_channels

    4. 其他未知复杂块：全保护（保守策略）

    最终可剪枝目标：
      简单 Conv 层（0,1,3,5,6,8,12,14）的 .conv
      CSP 层（4,7,9,13,15）的 cv2.conv（未被 head 引用的 CSP 层）
    """
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.modules.block import C2f
    from ultralytics.nn.extraction.c2f_base import C2fVariantBase
    from ultralytics.nn.modules.fusion.piafusion import PIAFusionBlock

    seq = model.model
    protected: Set[int] = set()

    # ── 类型1：被 head 直接引用的 backbone 层（[2,11,13]）内部所有 Conv2d ──
    backbone_fpn_idxs = _get_backbone_direct_output_idxs(model, backbone_end)
    print(f"[\u4fdd\u62a4] backbone \u4e2d\u88ab head \u76f4\u63a5\u5f15\u7528\u7684\u5c42: {sorted(backbone_fpn_idxs)}")
    for i in backbone_fpn_idxs:
        if i < len(seq):
            for m in seq[i].modules():
                if isinstance(m, nn.Conv2d):
                    protected.add(id(m))

    # ── 类型2/3/4：backbone 中的复杂块，按类型精确保护 ──
    # 额外保护：输出直接流入 PIAFusionBlock 的层（层 4 和 层 9）
    # PIAFusionBlock 内部 illum_net[1].out_channels = c1//4，与输入通道硬绑定，
    # DepGraph 会错误地将 illum_net[1] 的 out_channels 剪成 0
    pia_input_idxs: Set[int] = set()
    for i in range(backbone_end, len(seq)):
        m = seq[i]
        if type(m).__name__ == 'PIAFusionBlock':
            f = m.f
            refs = [f] if isinstance(f, int) else (list(f) if isinstance(f, (list, tuple)) else [])
            for ref in refs:
                if isinstance(ref, int) and 0 <= ref < backbone_end:
                    pia_input_idxs.add(ref)
        # 层 10 是 backbone 层，也要检查 backbone 中是否有层的 f 指向 PIAFusion
    # 同时检查 backbone 内部，找到层 10（PIAFusionBlock）的输入
    for i in range(backbone_end):
        m = seq[i]
        if type(m).__name__ == 'PIAFusionBlock':
            f = m.f
            refs = [f] if isinstance(f, int) else (list(f) if isinstance(f, (list, tuple)) else [])
            for ref in refs:
                if isinstance(ref, int) and 0 <= ref < backbone_end:
                    pia_input_idxs.add(ref)
    if pia_input_idxs:
        print(f"  [保护] PIAFusionBlock 输入层: {sorted(pia_input_idxs)}（cv2.conv 不可剪）")
        for i in pia_input_idxs:
            if i < len(seq):
                for m in seq[i].modules():
                    if isinstance(m, nn.Conv2d):
                        protected.add(id(m))

    n_csp_cv2_free = 0
    for i in range(min(backbone_end, len(seq))):
        layer = seq[i]
        if i in backbone_fpn_idxs:
            continue  # 已被类型1全保护，跳过
        if isinstance(layer, Conv):
            pass  # 简单 Conv，不保护，可剪
        elif isinstance(layer, PIAFusionBlock):
            # 整个 PIAFusionBlock 内部全保护
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    protected.add(id(m))
        elif isinstance(layer, (C2f, C2fVariantBase)):
            # CSP 块：保护 cv1 + 所有 m 内部 Conv2d，放开 cv2.conv
            for m in layer.cv1.modules():
                if isinstance(m, nn.Conv2d):
                    protected.add(id(m))
            for block in layer.m:
                for m in block.modules():
                    if isinstance(m, nn.Conv2d):
                        protected.add(id(m))
            # 若该层输出直接流入 PIAFusionBlock，则 cv2.conv 已被保护，不再打印"放开"
            if i in pia_input_idxs:
                pass  # cv2.conv 已在上面加入保护，跳过
            else:
                # cv2.conv 不加入保护 -> 可剪枝
                n_csp_cv2_free += 1
                cv2_conv = layer.cv2.conv if hasattr(layer.cv2, 'conv') else None
                if cv2_conv is not None:
                    print(f"  [放开] backbone[{i}].cv2.conv: out_channels={cv2_conv.out_channels} (可剪)")
        else:
            # 未知复杂块，全保护（保守）
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    protected.add(id(m))
            print(f"  [\u4fdd\u62a4] backbone[{i}] ({type(layer).__name__}): \u672a\u77e5\u590d\u6742\u5757\uff0c\u5168\u4fdd\u62a4")

    print(f"[\u4fdd\u62a4] CSP cv2.conv \u653e\u5f00\u53ef\u526a: {n_csp_cv2_free} \u4e2a")
    return protected


def execute_depgraph_pruning(DG, model: nn.Module, tp,
                              backbone_conv_ids: Set[int],
                              protected_ids: Set[int],
                              prune_ratio: float) -> int:
    """
    手动循环扫描 backbone 中在 DepGraph 里的每个 Conv2d：
      1. 跳过被 head 直接引用的 backbone 输出层（防止 head FPN 失步）
      2. 用 dg.get_pruning_group(conv, tp.prune_conv_out_channels, idxs) 获取完整依赖组
      3. 检查组内是否含受保护模块，有则跳过
      4. group.exec() 执行剪枝（自动同步 BN + 下游 Conv in_channels）

    返回成功剪枝的 Conv 数量。
    """
    print("\n" + "=" * 70)
    print("  手动循环扫描 Backbone Conv2d → DepGraph 依赖组剪枝")
    print("=" * 70)

    success_count = 0
    skip_protected = 0
    skip_bounds = 0
    skip_error = 0
    skip_not_in_graph = 0

    backbone_end = 16
    # 获取被 head 直接引用的 backbone 层的 Conv id（这些层的 out_channels 不能剪）
    protected_out_conv_ids = _build_protected_output_conv_ids(model, backbone_end)
    print(f"[保护] 输出被 head 引用的 backbone Conv2d 共 {len(protected_out_conv_ids)} 个")

    # 获取所有在图中的 backbone Conv2d
    backbone_convs = _get_named_backbone_convs(model, backbone_end, backbone_conv_ids, DG)
    print(f"[扫描] Backbone Conv2d 在图中共 {len(backbone_convs)} 个\n")

    for name, conv in backbone_convs:
        # 检查是否属于 head 直接引用的 backbone 输出层（out_channels 不能剪）
        if id(conv) in protected_out_conv_ids:
            # 不打印此层的跳过信息（会有很多），只在最后统计
            skip_protected += 1
            continue

        n_out = conv.out_channels
        n_prune = _calc_prune_n(n_out, prune_ratio)

        if n_prune == 0:
            print(f"  [跳过] {name}: 通道数 {n_out} 太少，跳过")
            continue

        # 选通道重要性最低的索引
        importance = _get_importance(conv)
        if importance.numel() != n_out:
            print(f"  [跳过] {name}: 重要性向量长度 {importance.numel()} != out_channels {n_out}")
            continue

        pruning_idxs = torch.argsort(importance)[:n_prune].tolist()

        # 边界安全检查
        max_idx = max(pruning_idxs)
        if max_idx >= n_out:
            print(f"  [跳过] {name}: 索引越界 max_idx={max_idx} >= out={n_out}")
            skip_bounds += 1
            continue

        # 获取完整依赖组
        try:
            group = DG.get_pruning_group(
                conv,
                tp.prune_conv_out_channels,
                idxs=pruning_idxs
            )
        except Exception as e:
            print(f"  [跳过] {name}: get_pruning_group 失败: {e}")
            skip_error += 1
            continue

        if group is None:
            print(f"  [跳过] {name}: group 为 None")
            skip_error += 1
            continue

        # 检查组内是否含受保护模块
        if _group_contains_protected(group, protected_ids):
            print(f"  [保护] {name}: 依赖组含受保护模块（RTDETRDecoder/DAttention），跳过")
            skip_protected += 1
            continue

        # 执行剪枝（DepGraph 自动处理 BN + 下游 Conv）
        try:
            group.exec()
            success_count += 1
            n_after = conv.out_channels
            print(f"  [成功] {name}: {n_out} → {n_after} (剪 {n_out - n_after})")
        except Exception as e:
            print(f"  [失败] {name}: group.exec() 异常: {e}")
            skip_error += 1

    print("\n" + "-" * 70)
    print(f"[结果] 成功: {success_count}  |  跳过(保护): {skip_protected}"
          f"  |  跳过(越界): {skip_bounds}  |  跳过(错误): {skip_error}")
    return success_count


# ═══════════════════════════════════════════════════════════════
#  前向传播自检
# ═══════════════════════════════════════════════════════════════

def verify_forward_pass(model: nn.Module, x_vis: torch.Tensor,
                         x_ir: torch.Tensor) -> bool:
    """
    强制自检：model(vis_ir_merged) 前向传播。
    vis 和 ir 合并为 6ch 输入（与训练时一致）。
    返回 True 表示通过。
    """
    print("\n" + "=" * 60)
    print("  前向传播自检（model(vis, ir) → 必须通过）")
    print("=" * 60)

    model.eval()
    merged = torch.cat([x_vis, x_ir], dim=1)  # (1, 6, H, W)

    try:
        with torch.no_grad():
            out = model(merged)

        # 提取有效 Tensor
        if isinstance(out, torch.Tensor):
            valid_tensors = [out]
        elif isinstance(out, (tuple, list)):
            valid_tensors = [o for o in out if isinstance(o, torch.Tensor)]
        else:
            valid_tensors = []

        if not valid_tensors:
            print(f"[失败] 输出无有效 Tensor: {type(out).__name__}")
            return False

        # 检查是否全为 0 / NaN
        all_ok = True
        for t in valid_tensors:
            if torch.isnan(t).any():
                print(f"[失败] 输出含 NaN，形状: {t.shape}")
                all_ok = False
            elif t.abs().sum().item() == 0:
                print(f"[警告] 输出全为 0，形状: {t.shape}（可能正常，取决于随机输入）")
            else:
                shapes = str(tuple(t.shape))
                print(f"\n✅ 验证通过：模型结构完整，无维度冲突")
                print(f"   输出 Tensor 形状: {shapes}，sum={t.abs().sum().item():.4f}")

        return all_ok

    except Exception as e:
        print(f"[失败] 前向传播异常: {e}")
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════
#  保存模型
# ═══════════════════════════════════════════════════════════════

def _sanitize_train_args(orig_ckpt: dict) -> dict:
    """
    跨平台路径清理：删除或重置 ckpt['train_args'] 中的 Linux 路径信息。
    防止在 Windows 上微调时去寻找 /home/zhizi/... 等平台相关路径。
    """
    if orig_ckpt is None:
        return orig_ckpt
    cleaned = dict(orig_ckpt)
    # 删除整个 train_args，论文重训练时 ultralytics 会重新生成
    cleaned.pop('train_args', None)
    # 清理其他可能包含路径的字段
    for key in ('data', 'project', 'name', 'save_dir'):
        cleaned.pop(key, None)
    print("[清理] 已删除 ckpt 中的 Linux 路径信息（train_args 等），避免跨平台路径报错")
    return cleaned


def save_pruned_model(model: nn.Module, output_path: str,
                      forward_ok: bool, orig_ckpt: dict = None) -> bool:
    if not forward_ok:
        print("\n[拒绝保存] 前向传播未通过，模型可能已损坏！")
        return False

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())

    # 必须以完整 nn.Module 对象保存（而非 state_dict），
    # 否则 attempt_load_one_weight 中 ckpt["model"].to(device) 会报 AttributeError
    save_dict = {
        'model': model,          # 完整模型对象
        'pruned': True,
        'pruned_params': n_params,
    }
    # 保留原始 ckpt 中 ultralytics 加载所需字段，并清理跨平台路径
    if orig_ckpt is not None:
        cleaned_ckpt = _sanitize_train_args(orig_ckpt)
        for key in ('date', 'epoch', 'best_fitness', 'yaml'):
            if key in cleaned_ckpt:
                save_dict[key] = cleaned_ckpt[key]

    torch.save(save_dict, out)
    print(f"\n[保存] 模型已保存至: {out}")
    print(f"[保存] 剪枝后参数量: {n_params:,} ({n_params / 1e6:.2f} M)")
    return True


# ═══════════════════════════════════════════════════════════════
#  打印对比
# ═══════════════════════════════════════════════════════════════

def print_comparison(bp, ap, bf, af, bt=0, at=0, bfps=0, afps=0):
    print("\n" + "=" * 70)
    print("                    剪枝前后对比")
    print("=" * 70)
    print(f"  {'指标':<20} {'剪枝前':>14} {'剪枝后':>14} {'变化':>14}")
    print("  " + "-" * 64)
    print(f"  {'参数量 (M)':<20} {bp / 1e6:>14.2f} {ap / 1e6:>14.2f} {(ap / bp - 1) * 100:>+13.1f}%")
    if bf > 0 and af > 0:
        print(f"  {'GFLOPs':<20} {bf:>14.2f} {af:>14.2f} {(af / bf - 1) * 100:>+13.1f}%")
    if bt > 0 and at > 0:
        print(f"  {'推理时间 (ms)':<20} {bt:>14.2f} {at:>14.2f} {(at / bt - 1) * 100:>+13.1f}%")
        print(f"  {'FPS':<20} {bfps:>14.2f} {afps:>14.2f} {(afps / bfps - 1) * 100:>+13.1f}%")
    print("=" * 70)


def print_markdown_table(bp, ap, bf, af):
    """输出 Markdown 格式的剪枝指标汇总表。"""
    print("\n## 剪枝指标汇总\n")
    print("| 指标 | 剪枝前 | 剪枝后 | 下降 |")
    print("|------|--------|--------|------|")  
    param_drop = (1 - ap / bp) * 100
    print(f"| 参数量 | {bp/1e6:.2f} M | {ap/1e6:.2f} M | {param_drop:.1f}% |")
    if bf > 0 and af > 0:
        flop_drop = (1 - af / bf) * 100
        print(f"| GFLOPs | {bf:.2f} | {af:.2f} | {flop_drop:.1f}% |")
    print()


# ═══════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VIF 双分支模型剪枝脚本 v9（DepGraph + 手动循环扫描 + 迭代剪枝 + 跨平台清理）"
    )
    parser.add_argument("--weights", type=str, required=True,
                        help="预训练权重路径 (.pt)")
    parser.add_argument("--prune-ratio", type=float, default=0.3,
                        help="剪枝比例（默认 0.3）")
    parser.add_argument("--iterations", type=int, default=1,
                        help="迭代剪枝轮数（默认 1），建议剪不够时设为 3")
    parser.add_argument("--output", type=str, default="D:/BaiduNetdiskDownload/MutilModel_3398475911/prune_outputs",
                        help="输出路径：可传文件路径(.pt)或文件夹路径，"
                             "传文件夹时自动命名为 pruned_<原文件名>.pt")
    parser.add_argument("--device", type=str, default="cuda",
                        help="运行设备（默认 cuda）")
    parser.add_argument("--speed-test", action="store_true",
                        help="是否测试推理速度")
    parser.add_argument("--img-size", type=int, default=640,
                        help="输入图像尺寸（默认 640）")
    parser.add_argument("--backbone-end", type=int, default=16,
                        help="backbone 末端层索引（不含），默认 16（层 0-15）")
    args = parser.parse_args()

    print("=" * 70)
    print("  VIF 双分支模型剪枝工具 v9")
    print("  策略：DepGraph + 手动循环扫描 + 迭代剪枝 + CSP cv2 解锁 + 跨平台清理")
    print("=" * 70)
    print(f"  权重文件  : {args.weights}")
    print(f"  剪枝比例  : {args.prune_ratio:.1%}")
    print(f"  迭代轮数  : {args.iterations}")
    print(f"  输出路径  : {args.output}")
    print(f"  设备      : {args.device}")
    print(f"  图像尺寸  : {args.img_size}")
    print(f"  backbone  : 层 0 ~ {args.backbone_end - 1}")
    print()
    print("核心策略：")
    print("  1. 跨平台清理：删除 train_args 中的 Linux 路径，防止 Windows 微调报错")
    print("  2. 精确保护：只保护 CSP cv1+m 和被 head 引用的层，放开 cv2.conv")
    print("  3. 迭代剪枝：每轮重新建图，积累剪枝率")
    print("  4. group.exec() → 自动同步 BN + 下游 Conv in_channels")
    print("  5. 强制 model(vis, ir) 前向自检 → 通过才保存")

    # ── 检查依赖 ──
    tp = check_torch_pruning()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"\n[错误] 权重文件不存在: {weights_path}")
        sys.exit(1)

    # 生成时间戳，格式：MMDD_HHMM
    timestamp = datetime.now().strftime("%m%d_%H%M")
    ratio_str = f"r{args.prune_ratio}"

    # 输出路径处理：自动加入剪枝比例 + 时间戳，避免覆盖
    output_path = Path(args.output)
    if output_path.suffix == ".pt":
        # 用户传了完整文件路径，在文件名末尾插入时间戳
        stem = output_path.stem
        output_path = output_path.parent / f"{stem}_{ratio_str}_{timestamp}.pt"
    else:
        # 视为目录，自动构造文件名
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / f"pruned_{ratio_str}_{timestamp}.pt"
    args.output = str(output_path.resolve())
    print(f"[输出] 本次保存路径: {args.output}")

    # ── 加载模型 ──
    model, _, orig_ckpt = load_vif_model(weights_path, args.device)

    # ── 创建示例输入 ──
    x_vis, x_ir = create_example_inputs(
        batch_size=1, img_size=args.img_size, device=args.device
    )
    print(f"\n[输入] vis: {x_vis.shape}, ir: {x_ir.shape}")

    # ── 剪枝前评估 ──
    print("\n" + "-" * 40)
    print("[评估] 剪枝前模型指标")
    print("-" * 40)
    before_params, before_flops = count_params_and_flops(model, args.img_size)
    print(f"  参数量: {before_params:,} ({before_params / 1e6:.2f} M)")
    print(f"  GFLOPs: {before_flops:.2f}")
    if args.speed_test:
        before_time, before_fps = measure_speed(
            model, (x_vis, x_ir), device=args.device
        )
        print(f"  推理时间: {before_time:.2f} ms  FPS: {before_fps:.2f}")
    else:
        before_time = before_fps = 0.0

    # ── 构建受保护模块集合 & backbone Conv 集合 ──
    protected_ids = _build_protected_module_ids(model)
    backbone_conv_ids = _build_backbone_conv_ids(model, args.backbone_end)
    print(f"\n[保护] 受保护模块子节点数: {len(protected_ids)}")
    print(f"[Backbone] Conv2d 总数: {len(backbone_conv_ids)}")

    # ── 打印 backbone 诊断 ──
    fusion_idxs, branch_tail_idxs = diagnose_backbone(
        model, args.backbone_end, protected_ids
    )

    # ── 创建 Wrapper（仅用于 Taylor 梁度收集）──
    print("\n" + "=" * 60)
    print("  创建 PruningModelWrapper（仅用于 Taylor 梯度收集）")
    print("=" * 60)
    wrapper = PruningModelWrapper(model)

    # ── 收集 Taylor 梯度（只在第 1 轮之前做一次）──
    collect_taylor_gradients(wrapper, (x_vis, x_ir), num_samples=8)

    # ── 恢复 wrapper 设置，然后直接对原始模型建图 ──
    wrapper.restore()

    # ── 迭代剪枝主循环 ──
    total_success = 0
    for iteration in range(1, args.iterations + 1):
        print(f"\n{'=' * 70}")
        print(f"  迭代剪枝第 {iteration}/{args.iterations} 轮")
        print(f"{'=' * 70}")

        # 每轮重建 backbone_conv_ids（结构已变）
        backbone_conv_ids = _build_backbone_conv_ids(model, args.backbone_end)

        # 每轮重新建图
        DG = build_dependency_graph(model, (x_vis, x_ir), tp)
        if DG is None:
            print(f"\n[错误] 第 {iteration} 轮 DepGraph 构建失败，停止迭代！")
            break

        success_count = execute_depgraph_pruning(
            DG, model, tp,
            backbone_conv_ids=backbone_conv_ids,
            protected_ids=protected_ids,
            prune_ratio=args.prune_ratio,
        )
        total_success += success_count

        if success_count == 0:
            print(f"\n[提示] 第 {iteration} 轮无新剪枝（已收敛），提前结束迭代")
            break

        # 轮间评估（轻量）
        cur_params = sum(p.numel() for p in model.parameters())
        print(f"\n[轮间] 第 {iteration} 轮后参数量: {cur_params:,} ({cur_params/1e6:.2f} M)")

        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    if total_success == 0:
        print("\n[失败] 没有成功剪枝任何组！请检查模型结构或降低剪枝比例。")
        sys.exit(1)

    # ── 剪枝后评估 ──
    print("\n" + "-" * 40)
    print("[评估] 剪枝后模型指标")
    print("-" * 40)
    after_params, after_flops = count_params_and_flops(model, args.img_size)
    print(f"  参数量: {after_params:,} ({after_params / 1e6:.2f} M)")
    print(f"  GFLOPs: {after_flops:.2f}")
    if args.speed_test:
        after_time, after_fps = measure_speed(
            model, (x_vis, x_ir), device=args.device
        )
        print(f"  推理时间: {after_time:.2f} ms  FPS: {after_fps:.2f}")
    else:
        after_time = after_fps = 0.0

    if after_flops == 0.0:
        print("[警告] GFLOPs 为 0，可能模型结构已损坏！")

    print_comparison(before_params, after_params, before_flops, after_flops,
                     before_time, after_time, before_fps, after_fps)

    # ── 强制前向传播自检 ──
    forward_ok = verify_forward_pass(model, x_vis, x_ir)

    # ── 保存（仅自检通过） ──
    saved = save_pruned_model(model, args.output, forward_ok, orig_ckpt=orig_ckpt)
    if saved:
        # 写 results.txt，与 .pt 同名
        results_path = Path(args.output).with_suffix(".txt")
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(f"剪枝结果报告\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"权重来源     : {weights_path}\n")
            f.write(f"剪枝比例     : {args.prune_ratio:.1%}\n")
            f.write(f"迭代轮数     : {args.iterations}\n")
            f.write(f"时间戳       : {timestamp}\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"剪枝前参数量 : {before_params:,} ({before_params/1e6:.2f} M)\n")
            f.write(f"剪枝后参数量 : {after_params:,} ({after_params/1e6:.2f} M)\n")
            f.write(f"参数量变化   : {(after_params/before_params - 1)*100:+.1f}%\n")
            if before_flops > 0 and after_flops > 0:
                f.write(f"剪枝前GFLOPs : {before_flops:.2f}\n")
                f.write(f"剪枝后GFLOPs : {after_flops:.2f}\n")
                f.write(f"GFLOPs变化   : {(after_flops/before_flops - 1)*100:+.1f}%\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"模型保存路径 : {args.output}\n")

        abs_pt_path = str(Path(args.output).resolve())
        abs_txt_path = str(results_path.resolve())

        # Markdown 表格写入 results.txt
        with open(results_path, "a", encoding="utf-8") as f:
            f.write("\n## 剪枝指标汇总\n\n")
            f.write("| 指标 | 剪枝前 | 剪枝后 | 下降 |\n")
            f.write("|------|--------|--------|------|\n")
            f.write(f"| 参数量 | {before_params/1e6:.2f} M | {after_params/1e6:.2f} M | {(1-after_params/before_params)*100:.1f}% |\n")
            if before_flops > 0 and after_flops > 0:
                f.write(f"| GFLOPs | {before_flops:.2f} | {after_flops:.2f} | {(1-after_flops/before_flops)*100:.1f}% |\n")

        print("\n[完成] 剪枝流程结束，模型已安全保存 ✓")

        # 终端打印 Markdown 表格
        print_markdown_table(before_params, after_params, before_flops, after_flops)

        print("=" * 70)
        print("  保存路径（可直接复制）")
        print("=" * 70)
        print(f"  模型文件 : {abs_pt_path}")
        print(f"  结果报告 : {abs_txt_path}")
        print("=" * 70)
        print("\n微调提示：")
        print(f"  from ultralytics import RTDETRMM")
        print(f"  model = RTDETRMM(r'{abs_pt_path}')")
        print(f"  model.train(data='data.yaml', epochs=100, lr0=0.001)")
    else:
        print("\n[完成] 剪枝流程结束，模型未保存（前向自检失败）")
        sys.exit(1)


if __name__ == "__main__":
    main()
