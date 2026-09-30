"""
预测诊断：对测试集推理，对比 GT 标注，分类每个框为 TP / FP / FN
输出:
  diagnostic_test.json   完整分类数据
  diagnostic_test.txt    可读报告
  diagnostic_test_fp.txt 仅 FP 列表
  diagnostic_test_fn.txt 仅 FN 列表
  diagnostic_test_imgs/  FP/FN 可视化图片
"""

import os, json, time
from pathlib import Path
import numpy as np
import torch

# matplotlib 用无界面后端，避免循环出图时 Tkinter 崩溃（必须在导入 pyplot 前设置）
import matplotlib
matplotlib.use("Agg")

# ─── 配置 ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = r"D:\JiQI\MM-experiment\ResTest\RTDETR-mid\weights\best.pt"
IMG_DIR      = r"D:\BaiduNetdiskDownload\M3FD_split\images\test"
IR_DIR       = r"D:\BaiduNetdiskDownload\M3FD_split\images_ir\test"   # IR/X 模态图
LABEL_DIR    = r"D:\BaiduNetdiskDownload\M3FD_split\labels\test"
SAVE_DIR     = r"D:\JiQI\MM-experiment\test_img\base_ir"
SAVE_RGB     = r"D:\JiQI\MM-experiment\test_img\base_RGB"
DEVICE       = 0
CONF         = 0.25   # 置信度阈值
IOU_NMS      = 0.7    # NMS IoU 阈值
IOU_THRESH   = 0.5    # TP 判定 IoU 阈值
IMG_SIZE     = 640
SHOW_VIZ     = True

os.makedirs(SAVE_DIR, exist_ok=True)

# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def compute_iou(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """gt (M,4), pred (N,4) → IoU (M,N)"""
    M, N = len(gt), len(pred)
    iou = np.zeros((M, N))
    for m in range(M):
        for n in range(N):
            ix1 = max(gt[m,0], pred[n,0]); iy1 = max(gt[m,1], pred[n,1])
            ix2 = min(gt[m,2], pred[n,2]); iy2 = min(gt[m,3], pred[n,3])
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            agt   = max(0, gt[m,2]-gt[m,0]) * max(0, gt[m,3]-gt[m,1])
            apred = max(0, pred[n,2]-pred[n,0]) * max(0, pred[n,3]-pred[n,1])
            union = agt + apred - inter + 1e-7
            iou[m, n] = inter / union
    return iou

def load_gt(label_path: str, img_w: int, img_h: int):
    """YOLO txt → [(class, [x1,y1,x2,y2]), ...]"""
    gt = []
    if not os.path.exists(label_path):
        return gt
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            gt.append((cls, [
                max(0, (xc - w/2) * img_w),
                max(0, (yc - h/2) * img_h),
                min(img_w, (xc + w/2) * img_w),
                min(img_h, (yc + h/2) * img_h),
            ]))
    return gt

# ─── 加载模型 ─────────────────────────────────────────────────────────────────

from ultralytics import YOLOMM
from ultralytics.utils.torch_utils import select_device

print(f"[1/4] 加载模型: {MODEL_PATH}")
model = YOLOMM(MODEL_PATH)
device = select_device(DEVICE)
is_gpu = str(device).startswith("cuda")
model.model.half() if is_gpu else model.model.float()
model.model.eval()
try:
    model.model.warmup(imgsz=(1, 6, IMG_SIZE, IMG_SIZE))
except AttributeError:
    pass  # RTDETR 等模型没有 warmup
names = model.names
print(f"      类别: {names}")

# ─── 图片列表 ─────────────────────────────────────────────────────────────────

img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
img_files = sorted([
    f for f in Path(IMG_DIR).iterdir()
    if f.suffix.lower() in img_exts and f.is_file()
])
print(f"[2/4] 共 {len(img_files)} 张图片  推理中 …")

# ─── 推理 + 分类 ──────────────────────────────────────────────────────────────

results = []
for idx, img_path in enumerate(img_files):
    img_path = str(img_path)
    from PIL import Image
    with Image.open(img_path) as im:
        img_w, img_h = im.size

    ir_path = str(Path(IR_DIR) / Path(img_path).name) if Path(IR_DIR).exists() else None
    res = model.predict(
        rgb_source=img_path,
        x_source=ir_path,
        conf=CONF, iou=IOU_NMS,
        imgsz=IMG_SIZE, device=device, verbose=False,
    )[0]

    # 预测框提取（兼容 RTDETR Boxes 和 numpy.ndarray）
    boxes_raw = res.boxes
    if boxes_raw is None or len(boxes_raw) == 0:
        preds = []
    elif isinstance(boxes_raw, np.ndarray):
        # RTDETR: res.boxes 本身就是 (N,6) numpy array, 列 [x1,y1,x2,y2,conf,cls]
        preds = [
            {"class": int(boxes_raw[i, 5]),
             "conf":  float(boxes_raw[i, 4]),
             "box":   boxes_raw[i, :4].tolist()}
            for i in range(len(boxes_raw))
        ]
    else:
        # 标准 YOLO Boxes 对象
        data = boxes_raw.data
        if hasattr(data, 'numpy'):
            data = data.numpy()
        preds = [
            {"class": int(data[i, 5]),
             "conf":  float(data[i, 4]),
             "box":   data[i, :4].tolist()}
            for i in range(len(data))
        ]
    p_cls  = np.array([p["class"] for p in preds], dtype=int) if preds else np.array([], dtype=int)
    p_xyxy = np.array([p["box"] for p in preds]) if preds else np.zeros((0, 4))

    # GT 框
    label_path = str(Path(LABEL_DIR) / (Path(img_path).stem + ".txt"))
    gts = load_gt(label_path, img_w, img_h)
    g_cls  = np.array([g[0] for g in gts], dtype=int) if gts else np.array([], dtype=int)
    g_xyxy = np.array([g[1] for g in gts]) if gts else np.zeros((0, 4))

    # IoU 矩阵
    iou_mat = compute_iou(g_xyxy, p_xyxy) if (len(gts)>0 and len(preds)>0) else np.zeros((len(gts), len(preds)))

    # TP/FP: 按置信度降序贪心匹配，每个 GT 只能被一个预测占用（避免重复匹配）
    best_gt = np.full(len(preds), -1, dtype=int)
    best_iou_val = np.zeros(len(preds))
    p_conf_arr = np.array([p["conf"] for p in preds]) if preds else np.array([])
    pred_order = np.argsort(-p_conf_arr) if len(preds) else []
    gt_used = set()
    for pi in pred_order:
        best_gi = -1
        best_iou = 0.0
        for gi in range(len(gts)):
            if gi in gt_used:
                continue
            if g_cls[gi] == p_cls[pi] and iou_mat[gi, pi] >= IOU_THRESH:
                if iou_mat[gi, pi] > best_iou:
                    best_iou = iou_mat[gi, pi]
                    best_gi = gi
        if best_gi >= 0:
            best_gt[pi] = best_gi
            best_iou_val[pi] = best_iou
            gt_used.add(best_gi)
    covered = gt_used

    tp_list, fp_list, fn_list = [], [], []
    for pi in range(len(preds)):
        entry = {
            "class": int(p_cls[pi]),
            "name":  names.get(int(p_cls[pi]), str(p_cls[pi])),
            "conf":  float(preds[pi]["conf"]),
            "box":   preds[pi]["box"],
        }
        if best_gt[pi] >= 0:
            gi = int(best_gt[pi])
            tp_list.append({"pred": entry, "gt_id": gi,
                            "gt_box": g_xyxy[gi].tolist(),
                            "gt_class": int(g_cls[gi]),
                            "iou": round(float(best_iou_val[pi]), 4)})
        else:
            fp_list.append(entry)

    for gi in range(len(gts)):
        if gi not in covered:
            fn_list.append({
                "gt_id": gi, "box": g_xyxy[gi].tolist(),
                "class": int(g_cls[gi]),
                "name":  names.get(int(g_cls[gi]), str(g_cls[gi])),
            })

    results.append({
        "image": img_path, "gt": len(gts), "pred": len(preds),
        "tp": len(tp_list), "fp": len(fp_list), "fn": len(fn_list),
        "tp_list": tp_list, "fp_list": fp_list, "fn_list": fn_list,
    })

    if (idx + 1) % 100 == 0 or idx == 0:
        print(f"      [{idx+1}/{len(img_files)}]  "
              f"TP={len(tp_list)} FP={len(fp_list)} FN={len(fn_list)}")

# ─── 统计 ─────────────────────────────────────────────────────────────────────

total_tp = sum(r["tp"] for r in results)
total_fp = sum(r["fp"] for r in results)
total_fn = sum(r["fn"] for r in results)
prec = total_tp / (total_tp + total_fp + 1e-7)
rec  = total_tp / (total_tp + total_fn + 1e-7)

print(f"[3/4] 分类完成  TP={total_tp}  FP={total_fp}  FN={total_fn}")

# ─── 保存 ─────────────────────────────────────────────────────────────────────

# JSON
json_path = os.path.join(SAVE_DIR, "diagnostic_test.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "config": {"model": MODEL_PATH, "img_dir": IMG_DIR, "label_dir": LABEL_DIR,
                   "iou_threshold": IOU_THRESH, "conf_threshold": CONF},
        "total_images": len(results), "total_tp": total_tp,
        "total_fp": total_fp, "total_fn": total_fn,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "per_image": results,
    }, f, ensure_ascii=False, indent=2)
print(f"      → {json_path}")

# 按类别统计
tp_cls = {}; fp_cls = {}; fn_cls = {}
for r in results:
    for e in r["tp_list"]:
        n = e["pred"]["name"]; tp_cls[n] = tp_cls.get(n, 0) + 1
    for e in r["fp_list"]:
        n = e["name"]; fp_cls[n] = fp_cls.get(n, 0) + 1
    for e in r["fn_list"]:
        n = e["name"]; fn_cls[n] = fn_cls.get(n, 0) + 1

# TXT
txt_path = os.path.join(SAVE_DIR, "diagnostic_test.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(f"预测诊断报告  (IoU≥{IOU_THRESH}, CONF≥{CONF})\n")
    f.write(f"{'='*65}\n")
    f.write(f"模型: {MODEL_PATH}\n图片: {IMG_DIR}\n标注: {LABEL_DIR}\n")
    f.write(f"图片数: {len(results)}\n")
    f.write(f"{'='*65}\n")
    f.write(f"TP={total_tp}   FP={total_fp}   FN={total_fn}\n")
    f.write(f"Precision={prec:.4f}   Recall={rec:.4f}\n")
    f.write(f"{'='*65}\n\n")

    f.write("按类别统计\n")
    f.write(f"{'类别':<15} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Recall':>7}\n")
    f.write("-"*50 + "\n")
    for c in sorted(set(list(tp_cls) + list(fp_cls) + list(fn_cls))):
        t=tp_cls.get(c,0); p=fp_cls.get(c,0); fn=fn_cls.get(c,0)
        f.write(f"{c:<15} {t:>5} {p:>5} {fn:>5} "
                f"{t/(t+p+1e-7):>7.4f} {t/(t+fn+1e-7):>7.4f}\n")
    f.write(f"{'='*50}\n\n")

    f.write("按图片统计  (仅列出 FP>0 或 FN>0)\n")
    f.write(f"{'文件名':<40} {'GT':>3} {'Pred':>4} {'TP':>3} {'FP':>3} {'FN':>3}\n")
    f.write("-"*65 + "\n")
    for r in results:
        if r["fp"] > 0 or r["fn"] > 0:
            f.write(f"{Path(r['image']).name:<40} {r['gt']:>3} {r['pred']:>4} "
                    f"{r['tp']:>3} {r['fp']:>3} {r['fn']:>3}\n")
    f.write(f"{'='*65}\n\n")

    f.write("详细分类\n")
    f.write(f"{'='*65}\n\n")
    for r in results:
        fn_name = Path(r["image"]).name
        f.write(f"【{fn_name}】  GT={r['gt']}  Pred={r['pred']}  "
                f"TP={r['tp']}  FP={r['fp']}  FN={r['fn']}\n")
        for e in r["tp_list"]:
            x1,y1,x2,y2 = e["pred"]["box"]
            f.write(f"  ✓ TP  [{e['pred']['name']}] conf={e['pred']['conf']:.3f}  "
                    f"IoU={e['iou']:.3f}  ({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})\n")
        for e in r["fp_list"]:
            x1,y1,x2,y2 = e["box"]
            f.write(f"  ✗ FP  [{e['name']}] conf={e['conf']:.3f}  "
                    f"({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})\n")
        for e in r["fn_list"]:
            x1,y1,x2,y2 = e["box"]
            f.write(f"  ✗ FN  [{e['name']}]  ({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})\n")
        f.write("\n")
print(f"      → {txt_path}")

# FP / FN 独立文件
fp_path = os.path.join(SAVE_DIR, "diagnostic_test_fp.txt")
fn_path = os.path.join(SAVE_DIR, "diagnostic_test_fn.txt")
with open(fp_path, "w", encoding="utf-8") as f:
    f.write(f"FP (误检) 共 {total_fp} 个\n{'='*60}\n")
    for r in results:
        for e in r["fp_list"]:
            x1,y1,x2,y2 = e["box"]
            f.write(f"{Path(r['image']).name}  [{e['name']}] "
                    f"conf={e['conf']:.3f}  ({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})\n")
with open(fn_path, "w", encoding="utf-8") as f:
    f.write(f"FN (漏检) 共 {total_fn} 个\n{'='*60}\n")
    for r in results:
        for e in r["fn_list"]:
            x1,y1,x2,y2 = e["box"]
            f.write(f"{Path(r['image']).name}  [{e['name']}]  "
                    f"({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})\n")
print(f"      → {fp_path}")
print(f"      → {fn_path}")

# ─── 可视化 ───────────────────────────────────────────────────────────────────

if SHOW_VIZ:
    print(f"[4/4] 生成 FP/FN 可视化（IR + RGB 双版本）…")
    from matplotlib import patches, pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    viz_dir_ir  = os.path.join(SAVE_DIR, "diagnostic_test_imgs")
    viz_dir_rgb = os.path.join(SAVE_RGB, "diagnostic_test_imgs")
    os.makedirs(viz_dir_ir, exist_ok=True)
    os.makedirs(viz_dir_rgb, exist_ok=True)

    def draw_one(img_path, save_path, r):
        from PIL import Image
        img = np.array(Image.open(img_path).convert("RGB"))
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.imshow(img)
        ax.axis("off")
        # 正常检测 TP: 绿色实线框 + 标签
        for e in r["tp_list"]:
            x1,y1,x2,y2 = e["pred"]["box"]
            ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,
                fill=False,edgecolor="#00ff00",lw=2))
            ax.text(x1+2,y1-4,f"{e['pred']['name']} {e['pred']['conf']:.2f}",
                     fontsize=7,color="#00ff00",weight="bold",
                     bbox=dict(facecolor="black",alpha=0.55,pad=1.2,edgecolor="none"))
        # 误检 FP: 红色实线框（只画框）
        for e in r["fp_list"]:
            x1,y1,x2,y2 = e["box"]
            ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,
                fill=False,edgecolor="#ff0000",lw=2))
        # 漏检 FN: 黄色虚线粗框（只画框）
        for e in r["fn_list"]:
            x1,y1,x2,y2 = e["box"]
            ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,
                fill=False,edgecolor="#ffff00",lw=3,linestyle="--"))
        plt.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    cnt = 0
    for r in results:
        try:
            stem = Path(r["image"]).stem
            ir_path = str(Path(IR_DIR) / Path(r["image"]).name)
            draw_one(ir_path, os.path.join(viz_dir_ir, stem + "_viz.png"), r)
            draw_one(r["image"], os.path.join(viz_dir_rgb, stem + "_viz.png"), r)
            cnt += 1
        except Exception as ex:
            print(f"      [WARN] {Path(r['image']).name}: {ex}")
    print(f"      → IR:  {viz_dir_ir}  ({cnt} 张)")
    print(f"      → RGB: {viz_dir_rgb}  ({cnt} 张)")
else:
    print(f"[4/4] 跳过可视化")

# ─── 最终统计 ─────────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"完成!  {len(results)} 张图片")
print(f"  TP: {total_tp}   FP: {total_fp}   FN: {total_fn}")
print(f"  Precision: {prec:.4f}   Recall: {rec:.4f}")
print(f"结果目录: {SAVE_DIR}")
print(f"{'='*55}")




