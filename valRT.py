import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
from prettytable import PrettyTable
from pathlib import Path
from ultralytics import RTDETRMM
from ultralytics.utils.torch_utils import model_info


def get_weight_size(path):
    stats = os.stat(path)
    return f"{stats.st_size / 1024 / 1024:.1f}MB"


def banner(msg="论文上的数据以以下结果为准"):
    for _ in range(5):
        print("-" * 20 + msg + "-" * 20)


if __name__ == "__main__":
    model_name = "rtdetr-r18-mm-mid-aifi-dattention"
    model_path = Path(f"ResTest/{model_name}/weights/best.pt")

    model = RTDETRMM(str(model_path))

    result = model.val(
        data=r"D:\BaiduNetdiskDownload\M3FD\M3FD_split\data.yaml",
        split="test",
        device=0,
        project="val",
        name="RTDETRval",
        workers=4,
    )

    banner()

    preprocess_ms = float(result.speed.get("preprocess", 0.0))
    inference_ms = float(result.speed.get("inference", 0.0))
    postprocess_ms = float(result.speed.get("postprocess", 0.0))
    total_ms = preprocess_ms + inference_ms + postprocess_ms

    fps_total = 0.0 if total_ms == 0 else 1000.0 / total_ms
    fps_infer = 0.0 if inference_ms == 0 else 1000.0 / inference_ms

    _, n_p, _, flops = model_info(model.model)

    model_info_table = PrettyTable()
    model_info_table.title = "Model Info"
    model_info_table.field_names = [
        "GFLOPs",
        "Parameters",
        "前处理时间/一张图",
        "推理时间/一张图",
        "后处理时间/一张图",
        "FPS(前处理+模型推理+后处理)",
        "FPS(推理)",
        "Model File Size",
    ]

    model_info_table.add_row(
        [
            f"{flops:.1f}",
            f"{n_p:,}",
            f"{preprocess_ms/1000:.6f}s",
            f"{inference_ms/1000:.6f}s",
            f"{postprocess_ms/1000:.6f}s",
            f"{fps_total:.2f}",
            f"{fps_infer:.2f}",
            get_weight_size(str(model_path)),
        ]
    )

    print(model_info_table)

    if model.task == "detect":
        length = result.box.p.size
        model_names = list(result.names.values())

        model_metrics_table = PrettyTable()
        model_metrics_table.title = "Model Metrics"
        model_metrics_table.field_names = [
            "Class Name",
            "Precision",
            "Recall",
            "F1-Score",
            "mAP50",
            "mAP75",
            "mAP50-95",
        ]

        for idx in range(length):
            model_metrics_table.add_row(
                [
                    model_names[idx],
                    f"{result.box.p[idx]:.4f}",
                    f"{result.box.r[idx]:.4f}",
                    f"{result.box.f1[idx]:.4f}",
                    f"{result.box.ap50[idx]:.4f}",
                    f"{result.box.all_ap[idx, 5]:.4f}",
                    f"{result.box.ap[idx]:.4f}",
                ]
            )

        model_metrics_table.add_row(
            [
                "all(平均数据)",
                f"{result.results_dict.get('metrics/precision(B)', 0):.4f}",
                f"{result.results_dict.get('metrics/recall(B)', 0):.4f}",
                f"{np.mean(result.box.f1[:length]):.4f}",
                f"{result.results_dict.get('metrics/mAP50(B)', 0):.4f}",
                f"{np.mean(result.box.all_ap[:length, 5]):.4f}",
                f"{result.results_dict.get('metrics/mAP50-95(B)', 0):.4f}",
            ]
        )

        print(model_metrics_table)

        save_path = result.save_dir / "paper_data.txt"
        with open(save_path, "w+", errors="ignore", encoding="utf-8") as f:
            f.write(str(model_info_table))
            f.write("\n\n")
            f.write(str(model_metrics_table))

        banner(f"结果已保存至 {save_path} ...")

    else:
        print(f"当前模型任务是: {model.task}，不是 detect，所以没有输出 box 指标表格。")

