# -*- coding: utf-8 -*-
"""
多模型对比筛选 v2
找：只有少数模型完美(FP=0,FN=0)，其他模型都有漏检/误检的图。
重点：体现模型间差异 —— 好的模型全对，差的模型全错。
"""
import json, os
from pathlib import Path

MODELS = {
    "RTDETR-mid":   r"D:\JiQI\MM-experiment\test_img\base_ir",
    "CFT":          r"D:\JiQI\MM-experiment\test_img\CFT_ir",
    "ICAFusion":    r"D:\JiQI\MM-experiment\test_img\Icafusion_ir",
    "MMI-Det":      r"D:\JiQI\MM-experiment\test_img\MMIdet_ir",
    "yolo26m-mm":   r"D:\JiQI\MM-experiment\test_img\yolo26_ir",
    "yolov8m-mm":   r"D:\JiQI\MM-experiment\test_img\yolo8_ir",
    "yolov9m-mm":   r"D:\JiQI\MM-experiment\test_img\yolo9_ir",
    "yolov5m-mm":   r"D:\JiQI\MM-experiment\test_img\yolo5_ir",
    "up":           r"D:\JiQI\MM-experiment\test_img\up_ir",
}

# ─── 读取 ──────────────────────────────────────────────────────────────────
model_data = {}
for name, path in MODELS.items():
    jf = os.path.join(path, "diagnostic_test.json")
    if not os.path.exists(jf): continue
    with open(jf, encoding="utf-8") as f:
        j = json.load(f)
    d = {}
    for item in j["per_image"]:
        img_name = Path(item["image"]).name
        d[img_name] = {"tp": item["tp"], "fp": item["fp"], "fn": item["fn"]}
    model_data[name] = d

all_imgs = set()
for d in model_data.values():
    all_imgs.update(d.keys())
all_imgs = sorted(all_imgs)
mnames = list(model_data.keys())

# ─── 逐图统计 ──────────────────────────────────────────────────────────────
rows = []
for img in all_imgs:
    row = {"img": img}
    perfect = []
    error = []
    for m in mnames:
        if img not in model_data[m]: continue
        s = model_data[m][img]
        row[m] = s
        if s["fp"] == 0 and s["fn"] == 0:
            perfect.append(m)
        else:
            error.append((m, s["fp"], s["fn"]))
    row["perfect"] = perfect
    row["error"] = error
    row["perfect_n"] = len(perfect)
    row["error_n"] = len(error)
    row["total_err"] = sum(e[1]+e[2] for e in error)
    rows.append(row)

# ─── 分类找图 ──────────────────────────────────────────────────────────────

# 类型1：只有1个模型完美，其他6个都有错（最能体现阶梯差异）
type1 = [r for r in rows if r["perfect_n"] == 1 and r["error_n"] >= 5]
# 类型2：只有2个模型完美，其他都有错
type2 = [r for r in rows if r["perfect_n"] == 2 and r["error_n"] >= 5]
# 类型3：所有模型都有错误（最差的图）
type3 = [r for r in rows if r["perfect_n"] == 0]
# 类型4：所有模型都完美（最好的图）
type4 = [r for r in rows if r["perfect_n"] == len(mnames)]

# 排序：按总错误数降序
type1.sort(key=lambda r: -r["total_err"])
type2.sort(key=lambda r: -r["total_err"])
type3.sort(key=lambda r: -r["total_err"])

print(f"{'='*80}")
print(f"模型对比筛选结果")
print(f"{'='*80}")
print(f"全部模型都完美的图:  {len(type4)} 张")
print(f"所有模型都有错的图:  {len(type3)} 张")
print(f"只有1个模型完美的图: {len(type1)} 张  ← 最能体现差异")
print(f"只有2个模型完美的图: {len(type2)} 张")
print()

# ─── 详细输出类型1（只有1个完美，其他都错）───────────────────────────────────
print(f"{'='*80}")
print(f"【重点】只有1个模型完美、其他模型都有错的图（共{len(type1)}张，按错误数排序）")
print(f"{'='*80}\n")

out_path = r"D:\BaiduNetdiskDownload\MutilModel_3398475911\model_compare_picks.txt"
with open(out_path, "w", encoding="utf-8") as fo:
    fo.write("多模型对比筛选报告\n")
    fo.write("="*80 + "\n")
    fo.write(f"模型列表: {', '.join(mnames)}\n\n")

    fo.write(f"【第一类】只有1个模型完美、其他模型都有错（共{len(type1)}张）\n")
    fo.write("-"*80 + "\n")
    for r in type1[:30]:
        perfect_name = r["perfect"][0]
        line = f"{r['img']}  | 完美: {perfect_name}  | 出错({r['error_n']}个,共{r['total_err']}个错误): "
        for m, fp, fn in r["error"]:
            line += f"{m}(FP={fp},FN={fn}) "
        fo.write(line + "\n")
        print(line)

    fo.write("\n" + "="*80 + "\n")
    fo.write(f"【第二类】只有2个模型完美、其他模型都有错（共{len(type2)}张，前20张）\n")
    fo.write("-"*80 + "\n")
    for r in type2[:20]:
        line = f"{r['img']}  | 完美: {', '.join(r['perfect'])}  | 出错: "
        for m, fp, fn in r["error"]:
            line += f"{m}(FP={fp},FN={fn}) "
        fo.write(line + "\n")

print(f"\n完整报告已保存: {out_path}")
print(f"\n建议：从【第一类】里挑图，比如 {type1[0]['img']}，")
print(f"这张图上 {type1[0]['perfect'][0]} 完全正常，其他 {type1[0]['error_n']} 个模型都有漏检/误检。")

