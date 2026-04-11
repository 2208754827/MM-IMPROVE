#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time

# 1. 在导入任何库之前，先解决环境报错和进度条问题
os.environ["TQDM_DISABLE"] = "True"                 # 禁用进度条
os.environ["YOLO_VERBOSE"] = "False"                # 禁用冗余日志
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1" # 禁用那个 SSL 检查报错

# 2. 彻底解决中文乱码：不管系统环境，强制 print 使用 utf-8 并刷新
def utf8_print(msg):
    sys.stdout.buffer.write((msg + '\n').encode('utf-8'))
    sys.stdout.flush()

from ultralytics import RTDETRMM

# ============ 配置区域 ============
MODEL_PATH = r"prune_outputs\pruned_r0.5_0410_1911.pt"
DATA_PATH = r"D:\BaiduNetdiskDownload\M3FD\M3FD_split\data.yaml"
EPOCHS = 100
BATCH = 1
IMGSZ = 640
LR0 = 0.0001
DEVICE = 0
PROJECT = r"D:\JiQI\MM-experiment\ResTest"
NAME = "pruned_finetune"
# =================================

# 3. 定义回调：不仅每轮结束打印，每 500 步也打印一次，让你知道它还活着
def on_train_batch_end(trainer):
    # 每 500 步打印一次小进度
    if trainer.ni % 500 == 0:
        epoch = trainer.epoch + 1
        steps = trainer.ni % trainer.nb
        utf8_print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch} 进度: {steps}/{trainer.nb} 步")

def on_train_epoch_end(trainer):
    epoch = trainer.epoch + 1
    loss = trainer.loss_items
    utf8_print(f"--- [{time.strftime('%H:%M:%S')}] Epoch {epoch} 完成！Loss总结: {loss} ---")

if __name__ == "__main__":
    utf8_print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [微调] 任务正式启动...")
    
    model = RTDETRMM(MODEL_PATH)
    
    # 注册回调
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    
    utf8_print(f"[微调] 已开启每 500 步自动汇报进度。")

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
        save_period=5
    )
