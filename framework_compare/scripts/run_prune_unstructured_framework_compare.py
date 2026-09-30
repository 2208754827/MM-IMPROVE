#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unstructured magnitude pruning entry for framework comparison experiments.

BASELINE: Unstructured (weight-level) pruning via magnitude thresholding.
Zeroes out the smallest-magnitude individual weights globally.

Key differences from DepGraph structured pruning:
- Unstructured: zeros individual weights, not entire channels
- No dependency graph needed
- No FLOPs reduction without sparse inference hardware
- No internal slimming
- Model size / GFLOPs / dense inference speed unchanged

This baseline shows that structured pruning is necessary for practical deployment.
"""

from __future__ import annotations

import os
import sys

from compare_common import DEFAULT_DATA, DEFAULT_WEIGHTS, ROOT, build_prune_output_dir

WEIGHTS = DEFAULT_WEIGHTS
DATA = DEFAULT_DATA

DEVICE = "0"
IMGSZ = 640
PRUNE_RATIO = 0.50
ITERATIONS = 1
FPS_WARMUP = 10
FPS_ITERS = 50
RESUME_EPOCHS = 30
RESUME_BATCH = 1
SEED = 42
OUTPUT = build_prune_output_dir("unstructured", PRUNE_RATIO)


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
    import prune_unstructured_magnitude

    sys.argv = [
        "prune_unstructured_magnitude.py",
        "--weights", str(WEIGHTS),
        "--output", str(OUTPUT),
        "--data", str(DATA),
        "--device", DEVICE,
        "--imgsz", str(IMGSZ),
        "--prune-ratio", str(PRUNE_RATIO),
        "--iterations", str(ITERATIONS),
        "--fps-warmup", str(FPS_WARMUP),
        "--fps-iters", str(FPS_ITERS),
        "--resume-epochs", str(RESUME_EPOCHS),
        "--resume-batch", str(RESUME_BATCH),
        "--seed", str(SEED),
    ]

    return prune_unstructured_magnitude.main()


if __name__ == "__main__":
    raise SystemExit(main())
