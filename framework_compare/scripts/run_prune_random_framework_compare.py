#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Random structured pruning entry for framework comparison experiments.

BASELINE: Random channel selection as a lower-bound reference.
Randomly selects channels to prune, proving that informed selection matters.

Key differences from DepGraph structured pruning:
- Random importance: no learning signal at all
- No FLOPs bias
- No internal slimming
- Same DepGraph dependency tracking for structural correctness only

This baseline establishes the floor: any method that cannot beat random
is not learning useful structure.
"""

from __future__ import annotations

import os
import sys

from compare_common import DEFAULT_DATA, DEFAULT_WEIGHTS, ROOT, build_prune_output_dir

WEIGHTS = DEFAULT_WEIGHTS
DATA = DEFAULT_DATA

DEVICE = "0"
IMGSZ = 640
BATCH_SIZE = 1
PRUNE_RATIO = 0.50
ITERATIONS = 8
GRAD_SAMPLES = 2
MIN_CHANNELS = 4
ROUND_TO = 4
INCLUDE_NECK = True
LAYER_END = 39
FPS_WARMUP = 10
FPS_ITERS = 50
RESUME_EPOCHS = 30
RESUME_BATCH = 1
SEED = 42
OUTPUT = build_prune_output_dir("random", PRUNE_RATIO)


def main() -> int:
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("YOLO_TQDM_RICH", "false")
    os.environ.setdefault("YOLO_VERBOSE", "true")

    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}")
    if not DATA.exists():
        raise FileNotFoundError(f"Data yaml not found: {DATA}")

    sys.path.insert(0, str(ROOT))
    import prune_random_only

    sys.argv = [
        "prune_random_only.py",
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
        "--fps-warmup", str(FPS_WARMUP),
        "--fps-iters", str(FPS_ITERS),
        "--resume-epochs", str(RESUME_EPOCHS),
        "--resume-batch", str(RESUME_BATCH),
        "--seed", str(SEED),
    ]
    if INCLUDE_NECK:
        sys.argv.append("--include-neck")

    return prune_random_only.main()


if __name__ == "__main__":
    raise SystemExit(main())
