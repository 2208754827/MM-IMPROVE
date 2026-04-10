#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
剪枝模型微调脚本
用法: D:\Anaconda\envs\mm\python.exe finetune_pruned.py
"""

from ultralytics import RTDETRMM

# ============ 配置区域 ============
MODEL_PATH = r"prune_outputs\pruned_r0.5_0410_1911.pt"  # 剪枝后的模型路径
DATA_PATH = r"D:\BaiduNetdiskDownload\M3FD\M3FD_split\data.yaml"  # 数据集配置
EPOCHS = 100          # 训练轮数
BATCH =   1           # 批大小
IMGSZ = 640           # 图像尺寸
LR0 = 0.0001          # 初始学习率（剪枝模型建议小一点）
DEVICE = 0            # GPU 设备
PROJECT = "ResTest"   # 输出目录
NAME = "pruned_finetune"  # 实验名称
# =================================

if __name__ == "__main__":
    print(f"[微调] 加载剪枝模型: {MODEL_PATH}")
    model = RTDETRMM(MODEL_PATH)
    
    print(f"[微调] 数据集: {DATA_PATH}")
    print(f"[微调] 训练 {EPOCHS} epochs, batch={BATCH}, lr={LR0}")
    
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
    )
    
    print(f"\n[完成] 模型已保存至: {PROJECT}/{NAME}/weights/best.pt")
