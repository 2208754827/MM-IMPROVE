#!/usr/bin/env python
"""
KAIST8 续跑脚本（从 last.pt 恢复训练）
与 trainRT.py 的 RESUME 模式对齐。

- checkpoint: D:\\JiQI\\MM-experiment\\KAIST8\\KAIST8\\weights\\last.pt
- 已训到 epoch 79 / 150，resume=True 会自动接着从 epoch 80 跑
- data / epochs / 优化器 / lr 调度 / batch 等全部从 checkpoint 自动恢复，无需传参
- 用法:   python resume_kaist8.py
"""

import os
import gc
import warnings

warnings.filterwarnings("ignore", message=".*deterministic.*")
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")


def main():
    print("[boot] resume_kaist8.py started", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    os.environ.setdefault("YOLO_TQDM_RICH", "false")
    os.environ.setdefault("YOLO_VERBOSE", "true")
    os.environ.setdefault("YOLO_MM_QUIET_PROGRESS", "true")

    import torch
    from ultralytics import RTDETRMM

    # 续跑起点（last.pt）
    LAST_PT = r"D:\JiQI\MM-experiment\KAIST8\KAIST8\weights\last.pt"

    print(f"[resume] resume KAIST8 from {LAST_PT}", flush=True)
    model = RTDETRMM(LAST_PT)
    model.train(resume=True)

    print("[resume] Training completed")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
