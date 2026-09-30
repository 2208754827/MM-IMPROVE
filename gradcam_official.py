# -*- coding: utf-8 -*-
"""
官方 Grad-CAM 热力图可视化 v2（多模态版）
用项目自带的 VisualizationManager + pytorch_grad_cam
自动输出到指定目录，RGB和IR各一张叠加图
"""

import os, shutil, cv2
from pathlib import Path
from ultralytics import YOLO

# ─── 配置 ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = r"D:\JiQI\MM-experiment\ResTest\aifi-dattention-CSP-MutilScaleEdgeInformationEnhance-ASF-P2-lite-CDDFusion2\weights\best.pt"
IMG_DIR      = r"D:\BaiduNetdiskDownload\M3FD_split\images\test"
IR_DIR       = r"D:\BaiduNetdiskDownload\M3FD_split\images_ir\test"
SAVE_RGB     = r"D:\JiQI\MM-experiment\heatmap\UP_rgb"
SAVE_IR      = r"D:\JiQI\MM-experiment\heatmap\UP_ir"
LAYER        = '9'   # backbone 最后一层
NUM_IMGS     = 1

os.makedirs(SAVE_RGB, exist_ok=True)
os.makedirs(SAVE_IR, exist_ok=True)

# ─── 加载模型 ─────────────────────────────────────────────────────────────────
print(f"[1/3] 加载模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
net = model.model.to('cuda').eval()
print(f"      模型加载完成")

# ─── 初始化官方可视化管理器 ───────────────────────────────────────────────────
from ultralytics.models.yolo.multimodal.visualize import VisualizationManager
vis_manager = VisualizationManager(net)
print(f"      可视化管理器初始化完成")

# ─── 跑图 ────────────────────────────────────────────────────────────────────
print(f"[2/3] 生成热力图（{NUM_IMGS}张）…")
img_files = sorted([f for f in Path(IMG_DIR).iterdir() if f.suffix.lower() in {".jpg",".jpeg",".png"}])[:NUM_IMGS]

ok = 0
for i, img_path in enumerate(img_files):
    stem = img_path.stem
    ir_path = Path(IR_DIR) / img_path.name
    if not ir_path.exists():
        continue

    # 读图片转numpy
    rgb_np = cv2.imread(str(img_path))
    ir_np = cv2.imread(str(ir_path))

    # 调用官方API（自动保存到 runs/visualize/expX）
    try:
        result = vis_manager(
            {'rgb': rgb_np, 'ir': ir_np},
            method='heatmap',
            alg='gradcam++',
            layers=[LAYER],
        )
    except Exception as e:
        print(f"      [{i+1}/{len(img_files)}] {stem} 失败: {e}")
        continue

    # 找到最新生成的exp目录，复制overlay图到目标目录
    runs_dir = Path("runs/visualize")
    exps = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime)
    if not exps:
        continue
    latest_exp = exps[-1]

    # 找rgb_overlay和ir_overlay
    for f in latest_exp.glob("*_overlay.png"):
        if "rgb" in f.name:
            shutil.copy(f, os.path.join(SAVE_RGB, f"{stem}_heatmap.png"))
        elif "ir" in f.name:
            shutil.copy(f, os.path.join(SAVE_IR, f"{stem}_heatmap.png"))

    ok += 1
    if (i+1) % 50 == 0 or i == 0:
        print(f"      [{i+1}/{len(img_files)}] {stem} 完成")

print(f"[3/3] 完成! {ok}/{len(img_files)} 张")
print(f"      RGB: {SAVE_RGB}")
print(f"      IR:  {SAVE_IR}")
