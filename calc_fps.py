# -*- coding: utf-8 -*-
"""
多模态模型 FPS 计算脚本
参考 RTDETR-main/get_FPS.py 的方式
"""

import warnings
warnings.filterwarnings('ignore')
import torch, time, os
import numpy as np
from tqdm import tqdm

# ─── 配置 ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = r"D:\JiQI\MM-experiment\ResTest\aifi-dattention-CSP-MutilScaleEdgeInformationEnhance\weights\best.pt"
BATCH_SIZE  = 8
IMG_SIZE    = 640
DEVICE      = 'cuda:0'
WARMUP      = 200   # 预热次数
TEST_TIME   = 1000  # 测试次数
HALF        = True # 是否用FP16半精度（开了快很多）

# ─── 加载模型 ─────────────────────────────────────────────────────────────────
print(f"[1/3] 加载模型: {MODEL_PATH}")
from ultralytics import YOLO
import torch
device = torch.device(DEVICE)
model = YOLO(MODEL_PATH)
net = model.model.to(device).eval()
net.fuse()
print(f"      模型加载完成")

# ─── 随机输入（6通道：RGB+IR）─────────────────────────────────────────────────
example_inputs = torch.randn((BATCH_SIZE, 6, IMG_SIZE, IMG_SIZE)).to(device)

if HALF:
    net = net.half()
    example_inputs = example_inputs.half()
    print(f"      使用FP16半精度")

# ─── 预热 ─────────────────────────────────────────────────────────────────────
print(f"[2/3] 预热 {WARMUP} 次…")
for i in tqdm(range(WARMUP), desc='warmup'):
    with torch.no_grad():
        net(example_inputs)

# ─── 测试FPS ────────────────────────────────────────────────────────────────────
print(f"[3/3] 测试 {TEST_TIME} 次…")
time_arr = []

for i in tqdm(range(TEST_TIME), desc='test'):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        net(example_inputs)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    time_arr.append(end_time - start_time)

# ─── 结果 ─────────────────────────────────────────────────────────────────────
std_time = np.std(time_arr)
infer_time_per_image = np.sum(time_arr) / (TEST_TIME * BATCH_SIZE)
fps = 1 / infer_time_per_image

print(f"\n{'='*50}")
print(f"模型: {MODEL_PATH}")
print(f"Batch size: {BATCH_SIZE}")
print(f"输入尺寸: {IMG_SIZE}x{IMG_SIZE} (6通道)")
print(f"预热次数: {WARMUP}")
print(f"测试次数: {TEST_TIME}")
print(f"单张推理时间: {infer_time_per_image*1000:.2f} ms ± {std_time*1000:.2f} ms")
print(f"FPS: {fps:.1f}")
print(f"{'='*50}")
