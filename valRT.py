import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
from prettytable import PrettyTable
from pathlib import Path
from ultralytics import RTDETRMM
from ultralytics.models.rtdetrmm.val import RTDETRMMValidator
from ultralytics.utils.torch_utils import model_info

BASE_MODEL_PATH = Path(r"D:\JiQI\MM-experiment\ResTest\only-asf2\weights\best.pt")
STAGE1_MODEL_PATH = Path(r"D:\JiQI\MM-experiment\ResTest\FLIR-bicycle-oversample-x2\weights\best.pt")
STAGE2_MODEL_PATH = Path(r"D:\JiQI\MM-experiment\ResTest\FLIR-bicycle-x3-640-stage2\weights\best.pt")
# 验证 only-backbone 模型
DEFAULT_MODEL_PATH = BASE_MODEL_PATH
# 也可通过 RTDETRMM_MODEL_PATH 指定任意 checkpoint。
MODEL_PATH = os.environ.get("RTDETRMM_MODEL_PATH", str(DEFAULT_MODEL_PATH))
# 默认按训练分辨率 640 验证；可用 RTDETRMM_IMGSZ=800 显式覆盖。
DEFAULT_IMGSZ = 640
VAL_IMGSZ = int(os.environ.get("RTDETRMM_IMGSZ", DEFAULT_IMGSZ))
DATA_PATH = r"D:\BaiduNetdiskDownload\M3FD_split\data.yaml"
DEVICE = 0


def load_mm_model(path):
    """根据 checkpoint 内模型类自动选择加载器：RT-DETR 系用 RTDETRMM，YOLO 系用 YOLOMM。"""
    import torch
    from ultralytics import RTDETRMM, YOLOMM

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cls_name = type(ckpt["model"]).__name__
    print(f"[val] checkpoint model class: {cls_name}")
    if "RTDETR" in cls_name:
        return RTDETRMM(str(path))
    return YOLOMM(str(path))


def get_weight_size(path):
    stats = os.stat(path)
    return f"{stats.st_size / 1024 / 1024:.1f}MB"


def banner(msg="论文上的数据以以下结果为准"):
    for _ in range(5):
        print("-" * 20 + msg + "-" * 20)


if __name__ == "__main__":
    if not MODEL_PATH.strip():
        raise FileNotFoundError("请先在 MODEL_PATH 里填入要验证的模型路径。")

    model_path = Path(MODEL_PATH).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")

    # This checkpoint is an RT-DETR multimodal model. Do not infer the loader
    # from the serialized Python class name, which may incorrectly select YOLOMM.
    print("[val] force model loader: RTDETRMM")
    model = RTDETRMM(str(model_path))

    result = model.val(
        validator=RTDETRMMValidator,
        data=DATA_PATH,
        split="val",
        imgsz=VAL_IMGSZ,
        batch=4,
        device=DEVICE,
        conf=0.001,
        rect=False,
        plots=True,
        project="val",
        name="RTDETRval",
        exist_ok=True,
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
        "FPS(前处理+推理+后处理)",
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
        print(f"当前模型任务是 {model.task}，不是 detect，所以没有输出 box 指标表格。")
