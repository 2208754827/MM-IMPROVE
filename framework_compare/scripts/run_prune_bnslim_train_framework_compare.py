#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BN-Slimming training-time sparsification entry for framework comparison experiments.

BASELINE: BN-Slimming with L1 regularization during training.
Adds L1 penalty on BN gamma during training to push them toward zero,
then prunes channels with smallest BN gamma.

Key differences from DepGraph structured pruning:
- Training-time: requires extra sparsification training epochs
- Post-training vs training-time paradigm difference
- No FLOPs bias in channel selection
- No internal slimming
- BN gamma as importance (not gradient-aware)

This baseline shows that post-training pruning (DepGraph) can match or exceed
training-time sparsification without requiring model retraining before pruning.
"""

from __future__ import annotations

import os
import sys

from compare_common import DEFAULT_DATA, DEFAULT_WEIGHTS, ROOT, build_prune_output_dir

WEIGHTS = DEFAULT_WEIGHTS
DATA = DEFAULT_DATA

DEVICE = "0"
IMGSZ = 640
BATCH_SIZE = 4
PRUNE_RATIO = 0.50
SPARSIFY_EPOCHS = 30
SPARSIFY_LR = 1e-4
L1_COEFF = 1e-3
MIN_CHANNELS = 4
ROUND_TO = 4
LAYER_END = 39
INCLUDE_NECK = True
FPS_WARMUP = 10
FPS_ITERS = 50
RESUME_EPOCHS = 30
RESUME_BATCH = 1
SEED = 42
OUTPUT = build_prune_output_dir("bnslim_train", PRUNE_RATIO)


def main() -> int:
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("YOLO_TQDM_RICH", "false")
    os.environ.setdefault("YOLO_VERBOSE", "true")

    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}")
    if not DATA.exists():
        raise FileNotFoundError(f"Data yaml not found: {DATA}")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "framework_compare"))
    import prune_bnslim_train_l1

    sys.argv = [
        "prune_bnslim_train_l1.py",
        "--weights", str(WEIGHTS),
        "--output", str(OUTPUT),
        "--data", str(DATA),
        "--device", DEVICE,
        "--imgsz", str(IMGSZ),
        "--batch-size", str(BATCH_SIZE),
        "--prune-ratio", str(PRUNE_RATIO),
        "--sparsify-epochs", str(SPARSIFY_EPOCHS),
        "--sparsify-lr", str(SPARSIFY_LR),
        "--l1-coeff", str(L1_COEFF),
        "--min-channels", str(MIN_CHANNELS),
        "--round-to", str(ROUND_TO),
        "--layer-end", str(LAYER_END),
        "--fps-warmup", str(FPS_WARMUP),
        "--fps-iters", str(FPS_ITERS),
        "--resume-epochs", str(RESUME_EPOCHS),
        "--resume-batch", str(RESUME_BATCH),
        "--seed", str(SEED),
    ]
    if INCLUDE_NECK:
        sys.argv.append("--include-neck")

    return prune_bnslim_train_l1.main()


if __name__ == "__main__":
    raise SystemExit(main())
