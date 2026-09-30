"""
轻量推理脚本：仅对图片目录做预测，输出带识别框的标注图
用法: python run_predict.py
"""

import os
import time
from pathlib import Path
from ultralytics import YOLOMM

# ─── 配置 ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = r"D:\JiQI\MM-experiment\ResTest\aifi-dattention-CSP-MutilScaleEdgeInformationEnhance-ASF-P2-lite-CMXFusion2\weights\best.pt"
IMG_DIR     = r"D:\BaiduNetdiskDownload\M3FD_split\images\test"
SAVE_DIR    = r"D:\JiQI\MM-experiment\ResTest\Predict\predict_result"
DEVICE      = 0
CONF        = 0.25    # 置信度阈值
IOU         = 0.7     # NMS IoU 阈值
IMG_SIZE    = 640
BATCH_SIZE  = 16

# ─── 运行 ─────────────────────────────────────────────────────────────────────
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"模型: {MODEL_PATH}")
print(f"图片: {IMG_DIR}")
print(f"输出: {SAVE_DIR}")
print("=" * 60)

model = YOLOMM(MODEL_PATH)
device = str(DEVICE)
model.model.half() if device.startswith("cuda") else model.model.float()
model.model.eval()
model.model.warmup(imgsz=(1, 6, IMG_SIZE, IMG_SIZE))

img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
files = sorted([
    f for f in Path(IMG_DIR).iterdir()
    if f.suffix.lower() in img_exts and f.is_file()
])
print(f"共 {len(files)} 张待处理图片\n")

total_detections = 0
total_time = 0.0
start = time.time()

for idx, img_path in enumerate(files):
    t0 = time.time()
    results = model.predict(
        source=str(img_path),
        conf=CONF,
        iou=IOU,
        imgsz=IMG_SIZE,
        device=device,
        batch=BATCH_SIZE,
        save=True,
        save_txt=True,
        project=SAVE_DIR,
        name="images",
        exist_ok=True,
        verbose=False,
    )
    elapsed = time.time() - t0
    total_time += elapsed

    n = len(results[0].boxes) if results and len(results[0].boxes) > 0 else 0
    total_detections += n
    print(f"  [{idx+1}/{len(files)}] {img_path.name}  "
          f"检测到 {n} 个目标  ({elapsed*1000:.0f}ms)")

avg_ms = total_time / len(files) * 1000 if files else 0
print(f"\n{'='*60}")
print(f"完成!  {len(files)} 张  共检测到 {total_detections} 个目标")
print(f"平均耗时: {avg_ms:.1f} ms/张")
print(f"结果: {Path(SAVE_DIR) / 'images'}")
print(f"{'='*60}")

# 汇总报告
summary = Path(SAVE_DIR) / "summary.txt"
with open(summary, "w", encoding="utf-8") as f:
    f.write(f"预测结果汇总\n{'='*50}\n")
    f.write(f"模型: {MODEL_PATH}\n")
    f.write(f"数据集: {IMG_DIR}\n")
    f.write(f"CONF={CONF}  IoU={IOU}  imgsz={IMG_SIZE}\n")
    f.write(f"图片数: {len(files)}\n")
    f.write(f"总检测数: {total_detections}\n")
    f.write(f"平均耗时: {avg_ms:.1f} ms/张\n")
    f.write(f"{'='*50}\n")
    f.write(f"输出目录: {Path(SAVE_DIR) / 'images'}\n")
print(f"汇总: {summary}")
