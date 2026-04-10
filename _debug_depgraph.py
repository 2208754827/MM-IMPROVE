"""verify depgraph with requires_grad=True"""
import torch, sys, torch.nn as nn
sys.path.insert(0, 'd:/BaiduNetdiskDownload/MutilModel_3398475911')
import torch_pruning as tp

WEIGHTS = 'ResTest/aifi-dattention-CSP-MutilScaleEdgeInformationEnhance-ASF-P2-lite-PIAFusionBlock/weights/best.pt'
ckpt = torch.load(WEIGHTS, map_location='cpu')
model = ckpt['model'].float().train()

class StubRouter:
    def setup_multimodal_routing(self, x, profile=False):
        if isinstance(x, torch.Tensor) and x.shape[1] == 6:
            return True, {'RGB': x[:, :3], 'X': x[:, 3:]}
        return True, {'RGB': x, 'X': x}
    def route_layer_input(self, x, module, input_sources, profile=False):
        if not hasattr(module, '_mm_input_source'): return None
        mm_src = module._mm_input_source
        if getattr(module, '_mm_new_input_start', False): return input_sources.get('X')
        if mm_src in ('RGB', 'X'): return input_sources.get(mm_src)
        return None
    def reset_spatial_input(self, x, m, s, p=False): return x

model.mm_router = StubRouter()

backbone_last = model.model[15]
_out = []
def _hook(m, inp, out):
    _out.clear()
    if isinstance(out, torch.Tensor): _out.append(out)
    elif isinstance(out, (list, tuple)):
        for o in out:
            if isinstance(o, torch.Tensor): _out.append(o); break
hook = backbone_last.register_forward_hook(_hook)

def fwd(m, x):
    _out.clear()
    m(x)
    if _out: return _out[0]
    raise RuntimeError('hook failed')

# KEY: requires_grad=True
x = torch.zeros(1, 6, 640, 640).requires_grad_(True)
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=x, forward_fn=fwd)
hook.remove()

n = len(DG.module2node)
n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d) and m in DG.module2node)
print(f'[result] nodes={n}, conv2d={n_conv}')
if model.model[0].conv in DG.module2node:
    print('[OK] model.0.conv found in graph')
else:
    print('[FAIL] model.0.conv NOT in graph')

import torch, sys, torch.nn as nn
sys.path.insert(0, 'd:/BaiduNetdiskDownload/MutilModel_3398475911')

WEIGHTS = 'ResTest/aifi-dattention-CSP-MutilScaleEdgeInformationEnhance-ASF-P2-lite-PIAFusionBlock/weights/best.pt'
ckpt = torch.load(WEIGHTS, map_location='cpu')
model = ckpt['model'].float()

class StubRouter:
    def setup_multimodal_routing(self, x, profile=False):
        if isinstance(x, torch.Tensor) and x.shape[1] == 6:
            return True, {'RGB': x[:, :3], 'X': x[:, 3:]}
        return True, {'RGB': x, 'X': x}
    def route_layer_input(self, x, module, input_sources, profile=False):
        if not hasattr(module, '_mm_input_source'): return None
        mm_src = module._mm_input_source
        if getattr(module, '_mm_new_input_start', False): return input_sources.get('X')
        if mm_src in ('RGB', 'X'): return input_sources.get(mm_src)
        return None
    def reset_spatial_input(self, x, m, s, p=False): return x

model.mm_router = StubRouter()
model.train()

# record all conv2d output grad_fns via hook
gfd2mod = {}
hooks = []
for name, m in model.named_modules():
    if isinstance(m, nn.Conv2d):
        def make_h(n, mod):
            def h(module, inp, out):
                gfd2mod[id(out.grad_fn)] = (n, mod)
            return h
        hooks.append(m.register_forward_hook(make_h(name, m)))

x = torch.zeros(1, 6, 640, 640, requires_grad=True)

# capture backbone[15] output
bb_out = []
def _bb_hook(m, inp, out):
    bb_out.clear()
    if isinstance(out, torch.Tensor): bb_out.append(out)
    elif isinstance(out, (list, tuple)):
        for o in out:
            if isinstance(o, torch.Tensor): bb_out.append(o); break
hh = model.model[15].register_forward_hook(_bb_hook)

model(x)
hh.remove()
for h in hooks: h.remove()

print(f'[info] hook 记录到 {len(gfd2mod)} 个 Conv2d 的 grad_fn')
print(f'[info] backbone[15] 输出: {bb_out[0].shape if bb_out else None}')
if bb_out:
    out_t = bb_out[0]
    print(f'[info] backbone out grad_fn: {out_t.grad_fn}')
    # 递归查 grad_fn 链是否可达某个 conv2d 的 hook
    found = []
    visited = set()
    def walk(fn, depth=0):
        if fn is None or id(fn) in visited or depth > 30: return
        visited.add(id(fn))
        if id(fn) in gfd2mod:
            found.append(gfd2mod[id(fn)][0])
        for nf, _ in getattr(fn, 'next_functions', []):
            walk(nf, depth+1)
    walk(out_t.grad_fn)
    print(f'[info] 从 backbone[15] 输出反向可达的 Conv2d 数: {len(found)}')
    print('  前5个:', found[:5])

import torch, sys, torch.nn as nn
sys.path.insert(0, 'd:/BaiduNetdiskDownload/MutilModel_3398475911')
import torch_pruning as tp

WEIGHTS = 'ResTest/aifi-dattention-CSP-MutilScaleEdgeInformationEnhance-ASF-P2-lite-PIAFusionBlock/weights/best.pt'

ckpt = torch.load(WEIGHTS, map_location='cpu')
model = ckpt['model'].float()

class StubRouter:
    def setup_multimodal_routing(self, x, profile=False):
        if isinstance(x, torch.Tensor) and x.shape[1] == 6:
            return True, {'RGB': x[:, :3], 'X': x[:, 3:]}
        return True, {'RGB': x, 'X': x}
    def route_layer_input(self, x, module, input_sources, profile=False):
        if not hasattr(module, '_mm_input_source'): return None
        mm_src = module._mm_input_source
        if getattr(module, '_mm_new_input_start', False): return input_sources.get('X')
        if mm_src in ('RGB', 'X'): return input_sources.get(mm_src)
        return None
    def reset_spatial_input(self, x, m, s, p=False): return x

model.mm_router = StubRouter()
model.train()

# hook 截取 backbone 最后一层输出
backbone_last = model.model[15]
_out = []
def _hook(m, inp, out):
    _out.clear()
    if isinstance(out, torch.Tensor): _out.append(out)
    elif isinstance(out, (list, tuple)):
        for o in out:
            if isinstance(o, torch.Tensor): _out.append(o); break
hook = backbone_last.register_forward_hook(_hook)

def fwd(m, x):
    _out.clear()
    m(x)
    if _out: return _out[0]
    raise RuntimeError('hook failed')

x = torch.zeros(1, 6, 640, 640)
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=x, forward_fn=fwd)
hook.remove()

n = len(DG.module2node)
n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d) and m in DG.module2node)
print(f'[OK] 节点总数: {n}, 其中 Conv2d: {n_conv}')

# 测试一个 backbone conv 是否在图中
c = model.model[0].conv
if c in DG.module2node:
    print('[OK] model.0.conv 在图中')
else:
    print('[FAIL] model.0.conv 不在图中')

import torch, sys, torch.nn as nn
sys.path.insert(0, 'd:/BaiduNetdiskDownload/MutilModel_3398475911')
import torch_pruning as tp

WEIGHTS = 'ResTest/aifi-dattention-CSP-MutilScaleEdgeInformationEnhance-ASF-P2-lite-PIAFusionBlock/weights/best.pt'

ckpt = torch.load(WEIGHTS, map_location='cpu')
model = ckpt['model'].float()

class StubRouter:
    def setup_multimodal_routing(self, x, profile=False):
        if isinstance(x, torch.Tensor) and x.shape[1] == 6:
            return True, {'RGB': x[:, :3], 'X': x[:, 3:]}
        return True, {'RGB': x, 'X': x}
    def route_layer_input(self, x, module, input_sources, profile=False):
        if not hasattr(module, '_mm_input_source'): return None
        mm_src = module._mm_input_source
        if getattr(module, '_mm_new_input_start', False): return input_sources.get('X')
        if mm_src in ('RGB', 'X'): return input_sources.get(mm_src)
        return None
    def reset_spatial_input(self, x, m, s, p=False): return x

model.mm_router = StubRouter()
model.model[-1].export = True
model.train()

x = torch.zeros(1, 6, 640, 640)

# ---- 注册钩子，记录哪些模块实际被调用 ----
called_modules = []
hooks = []
for name, m in model.named_modules():
    if isinstance(m, nn.Conv2d):
        def make_hook(n):
            def hook(mod, inp, out):
                called_modules.append(n)
            return hook
        hooks.append(m.register_forward_hook(make_hook(name)))

with torch.no_grad():
    out = model(x)
for h in hooks:
    h.remove()

print(f"\n[诊断] forward 实际调用了 {len(called_modules)} 个 Conv2d")
print("  前10个:", called_modules[:10])

# ---- 带梯度再跑一次，检查 grad_fn 链 ----
model.train()
x2 = torch.zeros(1, 6, 640, 640, requires_grad=True)
out2 = model(x2)
print(f"\n[诊断] out2 类型: {type(out2)}")
if isinstance(out2, torch.Tensor):
    print(f"  out2 shape: {out2.shape}, grad_fn: {out2.grad_fn}")
    # 追踪 grad_fn 链深度
    fn = out2.grad_fn
    depth = 0
    while fn is not None and depth < 5:
        print(f"  depth={depth}: {type(fn).__name__}")
        if hasattr(fn, 'next_functions') and fn.next_functions:
            fn = fn.next_functions[0][0]
        else:
            fn = None
        depth += 1
elif isinstance(out2, (list, tuple)):
    for i, o in enumerate(out2):
        if isinstance(o, torch.Tensor):
            print(f"  [out2][{i}] shape={o.shape}, grad_fn={o.grad_fn}")
