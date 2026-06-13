#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Direct-run pruning entry for A-DWConv-C3k2_PConv.

This wrapper launches the project's DepGraph + Taylor pruning script with
settings that also include the neck C3k2_PConv blocks in pruning.
Run it directly:

    python run_prune_a_dwconv_c3k2_pconv.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import prune_vif_v10


ROOT = Path(__file__).resolve().parent

# Core paths
WEIGHTS = Path(r"D:\JiQI\MM-experiment\ResTest\A-DWConv-C3k2_PConv\weights\best.pt")
DATA = Path(r"D:\BaiduNetdiskDownload\m4FD\M3FD_split\data.yaml")
OUTPUT = ROOT / "prune_outputs_depgraph_taylor_c3k2"

# Runtime options
DEVICE = "0"
IMGSZ = 640
BATCH_SIZE = 1

# Pruning options
PRUNE_RATIO = 0.60
ITERATIONS = 8
GRAD_SAMPLES = 2
MIN_CHANNELS = 8
ROUND_TO = 8
INCLUDE_NECK = True
LAYER_END = 39  # Exclusive; 39 covers neck C3k2_PConv roots at 21/26/29/32/38.
FLOPS_BIAS = 1.25
# Keep decoder width unchanged when comparing prune_ratio sweeps.
# This makes the effect come mainly from the recommended backbone/neck roots.
DECODER_KEEP_RATIO = 1.0
ENABLE_INTERNAL_SLIMMING = True
# Report / benchmark options
FPS_WARMUP = 10
FPS_ITERS = 50
RESUME_EPOCHS = 30
RESUME_BATCH = 1
SEED = 42


def build_argv() -> list[str]:
    argv = [
        "prune_vif_v10.py",
        "--weights",
        str(WEIGHTS),
        "--output",
        str(OUTPUT),
        "--data",
        str(DATA),
        "--device",
        DEVICE,
        "--imgsz",
        str(IMGSZ),
        "--batch-size",
        str(BATCH_SIZE),
        "--prune-ratio",
        str(PRUNE_RATIO),
        "--iterations",
        str(ITERATIONS),
        "--grad-samples",
        str(GRAD_SAMPLES),
        "--min-channels",
        str(MIN_CHANNELS),
        "--round-to",
        str(ROUND_TO),
        "--layer-end",
        str(LAYER_END),
        "--flops-bias",
        str(FLOPS_BIAS),
        "--enable-internal-slimming",
        "--decoder-keep-ratio",
        str(DECODER_KEEP_RATIO),
        "--fps-warmup",
        str(FPS_WARMUP),
        "--fps-iters",
        str(FPS_ITERS),
        "--resume-epochs",
        str(RESUME_EPOCHS),
        "--resume-batch",
        str(RESUME_BATCH),
        "--seed",
        str(SEED),
    ]
    if INCLUDE_NECK:
        argv.append("--include-neck")
    return argv


def main() -> int:
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("YOLO_TQDM_RICH", "false")
    os.environ.setdefault("YOLO_VERBOSE", "true")

    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}")
    if not DATA.exists():
        raise FileNotFoundError(f"Data yaml not found: {DATA}")

    sys.argv = build_argv()
    return prune_vif_v10.main()


if __name__ == "__main__":
    raise SystemExit(main())
