#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DepGraph + Taylor pruning entry for framework comparison experiments.

PROPOSED METHOD: DepGraph-based structured pruning with Taylor importance,
FLOPs-biased greedy selection, and internal module slimming.

Key characteristics that set this apart from baselines:
- Structured: removes entire channels → immediate dense-model speedup
- FLOPs-aware: prioritizes pruning computationally expensive layers
- Internal slimming: prunes inside CSP/C3k2_PConv/DAttention blocks
- Gradient-aware: Taylor importance captures actual loss sensitivity
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
FLOPS_BIAS = 1.25
DECODER_KEEP_RATIO = 1.0
ENABLE_INTERNAL_SLIMMING = True
FPS_WARMUP = 10
FPS_ITERS = 50
RESUME_EPOCHS = 30
RESUME_BATCH = 1
SEED = 42
OUTPUT = build_prune_output_dir("depgraph", PRUNE_RATIO)


def main() -> int:
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("YOLO_TQDM_RICH", "false")
    os.environ.setdefault("YOLO_VERBOSE", "true")

    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}")
    if not DATA.exists():
        raise FileNotFoundError(f"Data yaml not found: {DATA}")

    sys.path.insert(0, str(ROOT))
    import prune_vif_v10

    sys.argv = [
        "prune_vif_v10.py",
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
        "--decoder-keep-ratio", str(DECODER_KEEP_RATIO),
        "--fps-warmup", str(FPS_WARMUP),
        "--fps-iters", str(FPS_ITERS),
        "--resume-epochs", str(RESUME_EPOCHS),
        "--resume-batch", str(RESUME_BATCH),
        "--seed", str(SEED),
    ]
    if ENABLE_INTERNAL_SLIMMING:
        sys.argv.append("--enable-internal-slimming")
    if INCLUDE_NECK:
        sys.argv.append("--include-neck")

    return prune_vif_v10.main()


if __name__ == "__main__":
    raise SystemExit(main())
