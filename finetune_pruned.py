#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
from pathlib import Path
from shutil import copy2

# 1. Environment settings before importing other libraries
os.environ["TQDM_DISABLE"] = "True"
os.environ["YOLO_VERBOSE"] = "False"
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"


def utf8_print(msg):
    sys.stdout.buffer.write((msg + '\n').encode('utf-8'))
    sys.stdout.flush()

from ultralytics import RTDETRMM

# ============ 閰嶇疆鍖哄煙 ============
MODEL_PATH = r"prune_outputs\pruned_r0.5_0410_1911.pt"
DATA_PATH = r"D:\BaiduNetdiskDownload\M3FD\M3FD_split\data.yaml"
EPOCHS = 30
BATCH = 1
IMGSZ = 640
LR0 = 0.0001
DEVICE = 0
PROJECT = r"D:\JiQI\MM-experiment\ResTest"
NAME = "pruned_finetune"
# =================================

# 3. 瀹氫箟鍥炶皟锛氫笉浠呮瘡杞粨鏉熸墦鍗帮紝姣?500 姝ヤ篃鎵撳嵃涓€娆★紝璁╀綘鐭ラ亾瀹冭繕娲荤潃
def on_train_batch_end(trainer):
    # Print lightweight progress every 500 iterations.
    if trainer.ni % 500 == 0:
        epoch = trainer.epoch + 1
        steps = trainer.ni % trainer.nb
        utf8_print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch} progress: {steps}/{trainer.nb} steps")

def on_train_epoch_end(trainer):
    epoch = trainer.epoch + 1
    loss = trainer.loss_items
    utf8_print(f"--- [{time.strftime('%H:%M:%S')}] Epoch {epoch} 瀹屾垚锛丩oss鎬荤粨: {loss} ---")

if __name__ == "__main__":
    utf8_print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [finetune] training started...")
    
    model = RTDETRMM(MODEL_PATH)
    
    # 娉ㄥ唽鍥炶皟
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    
    utf8_print("[finetune] progress log enabled: every 500 steps.")

    model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        lr0=LR0,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        verbose=False, 
        plots=False,
        save_period=-1
    )

    # Only keep best checkpoint after training.
    weights_dir = Path(PROJECT) / NAME / "weights"
    for ckpt in weights_dir.glob("epoch*.pt"):
        if ckpt.exists():
            ckpt.unlink()
    last_pt = weights_dir / "last.pt"
    if last_pt.exists():
        last_pt.unlink()

    best_pt = weights_dir / "best.pt"
    model_dir = Path(__file__).resolve().parent / "MODEL"
    model_dir.mkdir(parents=True, exist_ok=True)
    export_path = model_dir / f"{NAME}_best.pt"
    if best_pt.exists():
        copy2(best_pt, export_path)
        utf8_print(f"[finetune] best model copied to: {export_path}")
    else:
        utf8_print("[finetune] best.pt not found, skip export.")

