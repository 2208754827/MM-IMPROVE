#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OBC pruning entry for A-DWConv-C3k2_PConv aligned to the Taylor setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

WEIGHTS = Path(r"D:\JiQI\MM-experiment\ResTest\A-DWConv-C3k2_PConv\weights\best.pt")
DATA = Path(r"D:\BaiduNetdiskDownload\m4FD\M3FD_split\data.yaml")
OUTPUT = ROOT / "prune_outputs_obc_compare_c3k2"

DEVICE = "0"
IMGSZ = 640
BATCH_SIZE = 1
PRUNE_RATIO = 0.40
ITERATIONS = 8
GRAD_SAMPLES = 2
MIN_CHANNELS = 8
ROUND_TO = 8
INCLUDE_NECK = True
LAYER_END = 39
FLOPS_BIAS = 1.25
DECODER_KEEP_RATIO = 1.0
ENABLE_INTERNAL_SLIMMING = True
FPS_WARMUP = 10
FPS_ITERS = 50
RESUME_EPOCHS = 30
RESUME_BATCH = 1
SEED = 42


def main() -> int:
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("YOLO_TQDM_RICH", "false")
    os.environ.setdefault("YOLO_VERBOSE", "true")

    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}")
    if not DATA.exists():
        raise FileNotFoundError(f"Data yaml not found: {DATA}")

    import prune_obc_taylorlike

    sys.argv = [
        "prune_obc_taylorlike.py",
        "--weights", str(WEIGHTS),
        "--output", str(OUTPUT),
        "--data", str(DATA),
        "--device", DEVICE,
        "--imgsz", str(IMGSZ),
        "--batch-size", str(BATCH_SIZE),
        "--prune-ratio", str(PRUNE_RATIO),
        "--iterations", str(ITERATIONS),
        "--grad-samples", str(GRAD_SAMPLES),
        "--min-channels", str(MIN_CHANNELS),
        "--round-to", str(ROUND_TO),
        "--layer-end", str(LAYER_END),
        "--flops-bias", str(FLOPS_BIAS),
        "--enable-internal-slimming",
        "--decoder-keep-ratio", str(DECODER_KEEP_RATIO),
        "--fps-warmup", str(FPS_WARMUP),
        "--fps-iters", str(FPS_ITERS),
        "--resume-epochs", str(RESUME_EPOCHS),
        "--resume-batch", str(RESUME_BATCH),
        "--seed", str(SEED),
    ]
    if INCLUDE_NECK:
        sys.argv.append("--include-neck")

    return prune_obc_taylorlike.main()


if __name__ == "__main__":
    raise SystemExit(main())
